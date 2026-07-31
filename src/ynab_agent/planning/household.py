"""Prepare bounded household cash-flow and survivorship state for simulation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from ynab_agent.planning.household_models import LongevityMode, age_months_on
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.social_security import (
    EARNINGS_TEST_FRA_YEAR_LIMIT,
    EARNINGS_TEST_UNDER_FRA_LIMIT,
    EntitlementCreditMonths,
    full_retirement_age_months,
    person_social_security,
    survivor_full_retirement_age_months,
)

if TYPE_CHECKING:
    import numpy as np


HOUSEHOLD_LONGEVITY_SEED_XOR = 0x535341


@dataclass(frozen=True)
class HouseholdState:
    inputs_sha256: str
    inflation_factors_sha256: str
    person_ids: tuple[str, ...]
    person_alive: np.ndarray
    person_age_months: np.ndarray
    work_income: np.ndarray
    ordinary_income: np.ndarray
    social_security_income: np.ndarray
    tax_free_income: np.ndarray
    spending_fraction: np.ndarray
    household_alive: np.ndarray
    joint_filing: np.ndarray
    audit: list[dict[str, object]]


def household_state_inputs_sha256(
    scenario: WealthScenario,
) -> str:
    """Fingerprint every scenario input used to prepare household state."""
    payload = {
        "household": (
            scenario.household.model_dump(mode="json")
            if scenario.household is not None
            else None
        ),
        "seed": scenario.seed,
        "current_age": scenario.current_age,
        "end_age": scenario.end_age,
        "trials": scenario.trials,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def household_inflation_factors_sha256(
    inflation_factors: np.ndarray,
) -> str:
    """Fingerprint inflation paths consumed by household cash flows."""
    import numpy as np

    values = np.asarray(
        inflation_factors,
        dtype="<f8",
        order="C",
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            values.shape,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def validate_household_state(
    scenario: WealthScenario,
    state: HouseholdState | None,
    *,
    inflation_factors: np.ndarray,
) -> None:
    """Reject prepared survivorship state from another scenario or path."""
    household = scenario.household
    if household is None:
        if state is not None:
            raise ValueError(
                "prepared_household_state requires household assumptions"
            )
        return
    if state is None:
        raise ValueError(
            "household assumptions require prepared household state"
        )
    years = scenario.end_age - scenario.current_age
    people = len(household.people)
    expected_shape = (years, scenario.trials)
    expected_people_shape = (people, years, scenario.trials)
    expected_age_shape = (people, years)
    if (
        state.person_alive.shape != expected_people_shape
        or state.person_age_months.shape != expected_age_shape
        or any(
            values.shape != expected_shape
            for values in (
                state.work_income,
                state.ordinary_income,
                state.social_security_income,
                state.tax_free_income,
                state.spending_fraction,
                state.household_alive,
                state.joint_filing,
            )
        )
    ):
        raise ValueError(
            "prepared household state does not match scenario shape"
        )
    expected_people = tuple(person.id for person in household.people)
    if state.person_ids != expected_people:
        raise ValueError(
            "prepared household state does not match household people"
        )
    if state.inputs_sha256 != household_state_inputs_sha256(scenario):
        raise ValueError(
            "prepared household state does not match scenario assumptions"
        )
    if (
        state.inflation_factors_sha256
        != household_inflation_factors_sha256(inflation_factors)
    ):
        raise ValueError(
            "prepared household state does not match inflation paths"
        )


def estimate_household_state_bytes(scenario: WealthScenario) -> int:
    """Conservative resident-memory estimate for household trial state."""
    if scenario.household is None:
        return 0
    years = scenario.end_age - scenario.current_age
    people = len(scenario.household.people)
    # Seven household arrays, alive/credit arrays, and temporary per-person flows,
    # plus the retained per-person alive masks and annual ages.
    return (
        years * scenario.trials * (9 + people * 6) * 8
        + people * years * scenario.trials
        + people * years * 8
    )


def _death_ages(scenario: WealthScenario) -> list[np.ndarray]:
    import numpy as np

    household = scenario.household
    if household is None:  # pragma: no cover - caller invariant
        raise ValueError("household state requires a household")
    rng = np.random.default_rng(
        scenario.seed ^ HOUSEHOLD_LONGEVITY_SEED_XOR
    )
    independent = rng.standard_normal((len(household.people), scenario.trials))
    if len(household.people) == 2 and household.longevity_correlation != 0:
        rho = household.longevity_correlation
        independent[1] = rho * independent[0] + (1 - rho**2) ** 0.5 * independent[1]
    result: list[np.ndarray] = []
    for index, person in enumerate(household.people):
        longevity = person.longevity
        if longevity.mode is LongevityMode.DETERMINISTIC:
            if longevity.death_age is None:  # pragma: no cover - model invariant
                raise RuntimeError("deterministic longevity is missing death_age")
            ages = np.full(scenario.trials, longevity.death_age, dtype=float)
        else:
            if (
                longevity.mean_death_age is None
                or longevity.standard_deviation_years is None
            ):  # pragma: no cover - model invariant
                raise RuntimeError("probabilistic longevity parameters are missing")
            ages = (
                longevity.mean_death_age
                + longevity.standard_deviation_years * independent[index]
            )
            ages = np.rint(
                np.clip(
                    ages,
                    longevity.minimum_death_age,
                    longevity.maximum_death_age,
                )
            )
        result.append(ages)
    return result


def _summary(values: np.ndarray) -> dict[str, float]:
    import numpy as np

    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def prepare_household_state(
    scenario: WealthScenario,
    *,
    inflation_factors: np.ndarray,
) -> HouseholdState | None:
    """Prepare identical longevity/cash-flow draws for every strategy evaluation."""
    import numpy as np

    household = scenario.household
    if household is None:
        return None
    years = scenario.end_age - scenario.current_age
    shape = (years, scenario.trials)
    work_income = np.zeros(shape)
    ordinary = np.zeros(shape)
    social_security = np.zeros(shape)
    tax_free = np.zeros(shape)
    spending_fraction = np.zeros(shape)
    joint_filing = np.zeros(shape, dtype=bool)
    death_ages = _death_ages(scenario)
    alive: list[np.ndarray] = []
    ages_months: list[list[int]] = []
    for person, person_death_ages in zip(household.people, death_ages, strict=True):
        start_age_months = age_months_on(person.birth_date, household.plan_start_date)
        person_ages = [start_age_months + offset * 12 for offset in range(years)]
        ages_months.append(person_ages)
        alive.append(
            np.stack(
                [
                    person_death_ages > age_months / 12
                    for age_months in person_ages
                ]
            )
        )

    alive_count = np.sum(np.stack(alive), axis=0)
    household_alive = alive_count > 0
    spending_fraction[alive_count == len(household.people)] = 1.0
    spending_fraction[(alive_count > 0) & (alive_count < len(household.people))] = (
        household.survivor_spending_fraction
    )
    if len(household.people) == 1:
        spending_fraction[alive_count == 1] = 1.0
    else:
        joint_filing = alive[0] & alive[1]

    audit: list[dict[str, object]] = []
    credit_months_by_person = [
        {
            "retirement": np.zeros(scenario.trials, dtype=np.int64),
            "spousal": np.zeros(scenario.trials, dtype=np.int64),
            "survivor": np.zeros(scenario.trials, dtype=np.int64),
            "survivor_pre_62": np.zeros(
                scenario.trials,
                dtype=np.int64,
            ),
        }
        for _ in household.people
    ]
    for offset in range(years):
        inflation = (
            np.full(scenario.trials, inflation_factors[offset])
            if inflation_factors.ndim == 1
            else inflation_factors[offset]
        )
        for person_index, person in enumerate(household.people):
            spouse_index = 1 - person_index if len(household.people) == 2 else None
            spouse = (
                household.people[spouse_index]
                if spouse_index is not None
                else None
            )
            person_alive = alive[person_index][offset]
            spouse_alive = (
                alive[spouse_index][offset]
                if spouse_index is not None
                else np.zeros(scenario.trials, dtype=bool)
            )
            age_months = ages_months[person_index][offset]
            age_years = age_months // 12
            work = sum(
                timeline.annual_covered_earnings
                * (inflation if timeline.inflation_adjusted else 1)
                for timeline in person.work
                if timeline.start_age <= age_years <= timeline.end_age
            )
            person_credits = credit_months_by_person[person_index]
            survivor_fra = survivor_full_retirement_age_months(
                person.birth_date
            )
            applied_survivor_credits = (
                person_credits["survivor"]
                if age_months >= survivor_fra
                else (
                    person_credits["survivor_pre_62"]
                    if age_months >= 62 * 12
                    else np.zeros(scenario.trials, dtype=np.int64)
                )
            )

            def social_security_components(
                *,
                assume_spouse_alive: bool,
            ) -> dict[str, np.ndarray]:
                values = {
                    name: np.zeros(scenario.trials)
                    for name in (
                        "own",
                        "spousal",
                        "survivor",
                        "credit_benefit",
                        "retirement_credit_applied",
                        "spousal_credit_applied",
                        "survivor_credit_applied",
                        "retirement_onset",
                        "spousal_onset",
                        "survivor_onset",
                    )
                }
                spouse_death_months = (
                    np.rint(death_ages[spouse_index] * 12).astype(np.int64)
                    if spouse_index is not None
                    else np.zeros(scenario.trials, dtype=np.int64)
                )
                spouse_retirement_credits = (
                    credit_months_by_person[spouse_index]["retirement"]
                    if spouse_index is not None
                    else np.zeros(scenario.trials, dtype=np.int64)
                )
                contexts = np.column_stack(
                    (
                        person_credits["retirement"],
                        person_credits["spousal"],
                        applied_survivor_credits,
                        spouse_death_months,
                        spouse_retirement_credits,
                    )
                )
                for context in np.unique(contexts, axis=0):
                    selected = np.all(contexts == context, axis=1)
                    benefit = person_social_security(
                        person,
                        spouse=spouse,
                        age_months=age_months,
                        spouse_alive=assume_spouse_alive,
                        person_alive=True,
                        family_maximum_override_monthly=(
                            spouse.family_maximum_monthly
                            if spouse is not None
                            else None
                        ),
                        entitlement_credit_months=(
                            EntitlementCreditMonths(
                                retirement=int(context[0]),
                                spousal=int(context[1]),
                                survivor=int(context[2]),
                            )
                        ),
                        spouse_death_age_months=int(context[3]),
                        spouse_retirement_credit_months_at_death=int(
                            context[4]
                        ),
                        apply_earnings_test=False,
                    )
                    values["own"][selected] = benefit.own
                    values["spousal"][selected] = benefit.spousal
                    values["survivor"][selected] = benefit.survivor
                    values["credit_benefit"][selected] = (
                        benefit.earnings_test_credit_annual
                    )
                    values["retirement_credit_applied"][selected] = (
                        benefit.credit_months_by_entitlement.retirement
                    )
                    values["spousal_credit_applied"][selected] = (
                        benefit.credit_months_by_entitlement.spousal
                    )
                    values["survivor_credit_applied"][selected] = (
                        benefit.credit_months_by_entitlement.survivor
                    )
                    values["retirement_onset"][selected] = (
                        benefit.entitlement_onset_months.retirement
                    )
                    values["spousal_onset"][selected] = (
                        benefit.entitlement_onset_months.spousal
                    )
                    values["survivor_onset"][selected] = (
                        benefit.entitlement_onset_months.survivor
                    )
                return values

            while_spouse_alive = social_security_components(
                assume_spouse_alive=True
            )
            as_survivor = social_security_components(
                assume_spouse_alive=False
            )

            def select_social_security_component(name: str) -> np.ndarray:
                amount = np.where(
                    spouse_alive,
                    while_spouse_alive[name],
                    as_survivor[name],
                )
                selected: np.ndarray = (
                    np.where(person_alive, amount, 0.0) * inflation
                )
                return selected

            own = select_social_security_component("own")
            spousal = select_social_security_component("spousal")
            survivor = select_social_security_component("survivor")
            credit_benefit = select_social_security_component(
                "credit_benefit"
            )
            annual_entitlement = own + spousal + survivor
            if age_months >= full_retirement_age_months(person.birth_date):
                withheld = np.zeros(scenario.trials)
            else:
                limit = (
                    EARNINGS_TEST_FRA_YEAR_LIMIT
                    if age_months + 12
                    >= full_retirement_age_months(person.birth_date)
                    else EARNINGS_TEST_UNDER_FRA_LIMIT
                )
                divisor = 3 if limit == EARNINGS_TEST_FRA_YEAR_LIMIT else 2
                withheld = np.minimum(
                    annual_entitlement,
                    np.maximum(0.0, work - limit) / divisor,
                )
            paid = np.maximum(0.0, own + spousal + survivor - withheld)
            social_security[offset] += paid
            earned_credit_months = np.zeros(scenario.trials, dtype=np.int64)
            positive_entitlement = annual_entitlement > 0
            earned_credit_months[positive_entitlement] = np.ceil(
                np.divide(
                    withheld[positive_entitlement],
                    annual_entitlement[positive_entitlement] / 12,
                )
                - 1e-9
            ).astype(np.int64)
            np.minimum(earned_credit_months, 12, out=earned_credit_months)
            component_values = {
                "retirement": own,
                "spousal": spousal,
                "survivor": survivor,
            }
            onset_values = {
                "retirement": select_social_security_component(
                    "retirement_onset"
                )
                / inflation,
                "spousal": select_social_security_component("spousal_onset")
                / inflation,
                "survivor": select_social_security_component("survivor_onset")
                / inflation,
            }
            fra_values = {
                "retirement": full_retirement_age_months(person.birth_date),
                "spousal": full_retirement_age_months(person.birth_date),
                "survivor": survivor_fra,
            }
            awarded_by_type: dict[str, np.ndarray] = {}
            for entitlement, component in component_values.items():
                active = component > 0
                maximum_credit = np.maximum(
                    0,
                    fra_values[entitlement] - onset_values[entitlement],
                ).astype(np.int64)
                remaining_credit = np.maximum(
                    0,
                    maximum_credit - person_credits[entitlement],
                )
                awarded = np.where(
                    active,
                    np.minimum(earned_credit_months, remaining_credit),
                    0,
                )
                person_credits[entitlement] += awarded
                awarded_by_type[entitlement] = awarded
                if entitlement == "survivor" and age_months < 62 * 12:
                    person_credits["survivor_pre_62"] += awarded

            work_paid = work * person_alive
            work_income[offset] += work_paid
            ordinary[offset] += work_paid
            pension = np.zeros(scenario.trials)
            pension_tax_free = np.zeros(scenario.trials)
            pension_survivor = np.zeros(scenario.trials)
            for stream in person.pensions:
                if age_years < stream.start_age:
                    continue
                if stream.end_age is not None and age_years > stream.end_age:
                    continue
                amount = stream.annual_amount * (
                    inflation if stream.inflation_adjusted else 1
                )
                primary_recipient = person_alive.astype(float)
                survivor_recipient = np.zeros(scenario.trials)
                if spouse_index is not None and stream.survivor_fraction:
                    survivor_recipient = (
                        (~person_alive & spouse_alive).astype(float)
                        * stream.survivor_fraction
                    )
                pension_survivor += amount * survivor_recipient
                recipient = primary_recipient + survivor_recipient
                if stream.tax_free:
                    pension_tax_free += amount * recipient
                else:
                    pension += amount * recipient
            ordinary[offset] += pension
            tax_free[offset] += pension_tax_free
            audit.append(
                {
                    "year": household.plan_start_date.year + offset,
                    "person_id": person.id,
                    "age": age_years,
                    "age_months": age_months,
                    "retired": age_months >= person.retirement_age_months,
                    "alive_probability": float(np.mean(person_alive)),
                    "alive_status": {
                        "alive_probability": float(np.mean(person_alive)),
                        "deceased_probability": float(np.mean(~person_alive)),
                        "spouse_alive_probability": float(
                            np.mean(spouse_alive)
                        ),
                        "household_alive_probability": float(
                            np.mean(household_alive[offset])
                        ),
                    },
                    "work_income_real": _summary(
                        work * person_alive / inflation
                    ),
                    "covered_work_earnings_real": _summary(
                        work * person_alive / inflation
                    ),
                    "pension_taxable_income_real": _summary(
                        pension / inflation
                    ),
                    "pension_tax_free_income_real": _summary(
                        pension_tax_free / inflation
                    ),
                    "pension_survivor_income_real": _summary(
                        pension_survivor / inflation
                    ),
                    "pension_income_real": _summary(
                        (pension + pension_tax_free) / inflation
                    ),
                    "social_security_own_real": _summary(own / inflation),
                    "social_security_spousal_real": _summary(spousal / inflation),
                    "social_security_survivor_real": _summary(survivor / inflation),
                    "social_security_entitlement_real": _summary(
                        (own + spousal + survivor) / inflation
                    ),
                    "earnings_test_withheld_real": _summary(withheld / inflation),
                    "earnings_test_credit_months_earned": _summary(
                        earned_credit_months.astype(float)
                    ),
                    "earnings_test_credit_months_applied": _summary(
                        np.max(
                            np.stack(
                                [
                                    np.where(
                                        person_alive,
                                        np.where(
                                            spouse_alive,
                                            while_spouse_alive[
                                                f"{entitlement}_credit_applied"
                                            ],
                                            as_survivor[
                                                f"{entitlement}_credit_applied"
                                            ],
                                        ),
                                        0,
                                    )
                                    for entitlement in (
                                        "retirement",
                                        "spousal",
                                        "survivor",
                                    )
                                ]
                            ),
                            axis=0,
                        )
                    ),
                    "earnings_test_credit_months_cumulative": _summary(
                        np.max(
                            np.stack(
                                [
                                    person_credits["retirement"],
                                    person_credits["spousal"],
                                    person_credits["survivor"],
                                ]
                            ),
                            axis=0,
                        ).astype(float)
                    ),
                    "earnings_test_credit_months_earned_by_entitlement": {
                        entitlement: _summary(
                            awarded_by_type[entitlement].astype(float)
                        )
                        for entitlement in (
                            "retirement",
                            "spousal",
                            "survivor",
                        )
                    },
                    "earnings_test_credit_months_cumulative_by_entitlement": {
                        entitlement: _summary(
                            person_credits[entitlement].astype(float)
                        )
                        for entitlement in (
                            "retirement",
                            "spousal",
                            "survivor",
                        )
                    },
                    "earnings_test_credit_benefit_real": _summary(
                        credit_benefit / inflation
                    ),
                    "social_security_paid_real": _summary(paid / inflation),
                    "total_income_real": _summary(
                        (work * person_alive + pension + pension_tax_free + paid)
                        / inflation
                    ),
                    "filing_status": (
                        {
                            "married_filing_jointly_probability": float(
                                np.mean(joint_filing[offset])
                            ),
                            "single_probability": float(
                                np.mean((alive_count[offset] == 1))
                            ),
                            "no_filer_probability": float(
                                np.mean((alive_count[offset] == 0))
                            ),
                        }
                    ),
                }
            )

    return HouseholdState(
        inputs_sha256=household_state_inputs_sha256(scenario),
        inflation_factors_sha256=(
            household_inflation_factors_sha256(inflation_factors)
        ),
        person_ids=tuple(person.id for person in household.people),
        person_alive=np.stack(alive),
        person_age_months=np.asarray(ages_months, dtype=np.int64),
        work_income=work_income,
        ordinary_income=ordinary,
        social_security_income=social_security,
        tax_free_income=tax_free,
        spending_fraction=spending_fraction,
        household_alive=household_alive,
        joint_filing=joint_filing,
        audit=audit,
    )
