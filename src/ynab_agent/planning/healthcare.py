"""Seeded household healthcare and long-term-care projection state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from ynab_agent.planning.models import WealthScenario

if TYPE_CHECKING:
    import numpy as np


HEALTHCARE_RANDOM_SEED_XOR = 0x4845414C5448
HEALTHCARE_STATE_SCHEMA_VERSION = 2
IRMAA_LOOKBACK_POLICY = (
    "exact_tax_year_minus_two_supplied_history_over_simulated_magi_fail_closed_v2"
)


@dataclass(frozen=True)
class HealthcareState:
    """Nominal annual healthcare costs aligned to simulation paths."""

    scenario_sha256: str
    inflation_factors_sha256: str
    person_ids: tuple[str, ...]
    pre_medicare_premium: np.ndarray
    medicare_premium: np.ndarray
    out_of_pocket: np.ndarray
    ltc_gross_cost: np.ndarray
    ltc_insurance_benefit: np.ndarray
    ltc_home_equity_used: np.ndarray
    ltc_net_cost: np.ndarray
    total_cost: np.ndarray
    ltc_selected: np.ndarray
    ltc_active_in_plan: np.ndarray
    person_ltc_selected: dict[str, np.ndarray]
    person_ltc_active_in_plan: dict[str, np.ndarray]


def estimate_healthcare_state_bytes(scenario: WealthScenario) -> int:
    """Conservative resident-memory estimate for healthcare state."""
    if scenario.healthcare is None:
        return 0
    years = scenario.end_age - scenario.current_age
    people = len(scenario.healthcare.people)
    matrix_cells = years * scenario.trials
    # Eight prepared float matrices, one execution-time exact-reserve funding
    # matrix, plus conservative preparation peaks for masks, per-person gross
    # costs, residual funding, and final aggregation.
    preparation_peak = matrix_cells * (13 * 8 + 3)
    # Reporting retains eight lifetime component accumulators and one annual
    # conversion temporary; incidence/fingerprint validation is also per trial.
    reporting_peak = scenario.trials * (9 * 8 + (2 * people + 2))
    return preparation_peak + reporting_peak


def healthcare_state_scenario_sha256(scenario: WealthScenario) -> str:
    """Fingerprint every scenario input that can affect prepared care state."""
    canonical = json.dumps(
        scenario.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def healthcare_inflation_factors_sha256(
    inflation_factors: np.ndarray,
) -> str:
    """Fingerprint the exact general-inflation paths funding home equity."""
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


def validate_healthcare_state(
    scenario: WealthScenario,
    state: HealthcareState | None,
    *,
    inflation_factors: np.ndarray,
) -> None:
    """Reject prepared care state that could bypass scenario admission."""
    if scenario.healthcare is None:
        if state is not None:
            raise ValueError(
                "prepared_healthcare_state requires healthcare assumptions"
            )
        return
    if state is None:
        raise ValueError(
            "healthcare assumptions require prepared healthcare state"
        )
    expected_shape = (
        scenario.end_age - scenario.current_age,
        scenario.trials,
    )
    matrices = (
        state.pre_medicare_premium,
        state.medicare_premium,
        state.out_of_pocket,
        state.ltc_gross_cost,
        state.ltc_insurance_benefit,
        state.ltc_home_equity_used,
        state.ltc_net_cost,
        state.total_cost,
    )
    if any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError(
            "prepared healthcare state does not match scenario shape"
        )
    household = scenario.household
    if household is None:  # pragma: no cover - model invariant
        raise ValueError("healthcare assumptions require a household")
    expected_people = tuple(person.id for person in household.people)
    if state.person_ids != expected_people:
        raise ValueError(
            "prepared healthcare state does not match household people"
        )
    expected_trial_shape = (scenario.trials,)
    if (
        state.ltc_selected.shape != expected_trial_shape
        or state.ltc_active_in_plan.shape != expected_trial_shape
        or set(state.person_ltc_selected) != set(expected_people)
        or set(state.person_ltc_active_in_plan) != set(expected_people)
        or any(
            values.shape != expected_trial_shape
            for values in (
                *state.person_ltc_selected.values(),
                *state.person_ltc_active_in_plan.values(),
            )
        )
    ):
        raise ValueError(
            "prepared healthcare state incidence does not match scenario"
        )
    if (
        state.scenario_sha256
        != healthcare_state_scenario_sha256(scenario)
    ):
        raise ValueError(
            "prepared healthcare state does not match scenario assumptions"
        )
    if (
        state.inflation_factors_sha256
        != healthcare_inflation_factors_sha256(inflation_factors)
    ):
        raise ValueError(
            "prepared healthcare state does not match inflation paths"
        )


def healthcare_manifest(scenario: WealthScenario) -> dict[str, object] | None:
    """Return every material care assumption plus replay semantics."""
    assumptions = scenario.healthcare
    if assumptions is None:
        return None
    return {
        "schema_version": HEALTHCARE_STATE_SCHEMA_VERSION,
        "assumptions": assumptions.model_dump(mode="json"),
        "cost_basis": "plan_start_real_dollars",
        "medical_inflation": "separate_fixed_nominal_growth_path",
        "application_window": "retirement_years_while_person_alive",
        "annual_spending_interaction": (
            "healthcare assumptions are incremental and must be excluded from "
            "annual_spending and retirement spending-plan baselines"
        ),
        "pre_medicare_premium": (
            "explicit_aca_or_other_premium_net_of_expected_tax_credit"
        ),
        "irmaa": (
            "progressive_tax_policy_two_year_magi_lookback_and_alive_medicare_count"
        ),
        "ltc_random": {
            "substream": "healthcare_ltc_v1",
            "seed_derivation": "scenario_seed_xor_constant",
            "seed_xor_hex": "0x4845414c5448",
            "incidence": "one_bernoulli_lifetime_draw_per_person_and_trial",
            "onset": "discrete_uniform_inclusive_age_draw",
            "duration": "rounded_bounded_normal_year_draw",
            "severity": "mean_one_lognormal_multiplier_draw",
            "lifetime_selection": (
                "Bernoulli draw independent of whether care becomes active "
                "inside the simulated retirement window"
            ),
            "in_plan_incidence": (
                "selected care with at least one active retirement year "
                "before death and plan end"
            ),
        },
        "funding_order": [
            "ltc_insurance",
            (
                "exact_account_home_equity_reserve"
                if assumptions.ltc_funding_source == "home_equity"
                else "portfolio"
            ),
            "portfolio",
        ],
        "shortfall_order": (
            "base_spending_then_routine_healthcare_then_ltc; "
            "unmet_ltc_is_reported_first"
        ),
        "survivor_semantics": (
            "person_costs_stop_at_death_and_are_not multiplied by "
            "survivor_spending_fraction"
        ),
        "home_equity_inflation": (
            "real funding limit converted by each trial's general inflation "
            "path; actual funding is debited from the exact linked reserve account"
        ),
        "irmaa_lookback": IRMAA_LOOKBACK_POLICY,
        "irmaa_lookback_sources": {
            "precedence": "supplied_exact_history_then_simulated_magi",
            "supplied_exact_history_tax_years": (
                [
                    value.tax_year
                    for value in scenario.tax_assumptions.progressive.irmaa_lookback_magi
                ]
                if scenario.tax_assumptions is not None
                and scenario.tax_assumptions.progressive is not None
                else []
            ),
        },
    }


def prepare_healthcare_state(
    scenario: WealthScenario,
    *,
    household_state: object,
    inflation_factors: np.ndarray,
) -> HealthcareState | None:
    """Draw deterministic care paths and apply shared LTC funding."""
    import numpy as np

    assumptions = scenario.healthcare
    if assumptions is None:
        return None
    if scenario.household is None:  # pragma: no cover - model invariant
        raise ValueError("healthcare assumptions require a household")

    from ynab_agent.planning.household import HouseholdState

    if not isinstance(household_state, HouseholdState):
        raise ValueError("healthcare assumptions require prepared household state")

    years = scenario.end_age - scenario.current_age
    retirement_offset = scenario.retirement_age - scenario.current_age
    retirement_year = np.arange(years)[:, None] >= retirement_offset
    shape = (years, scenario.trials)
    pre_medicare = np.zeros(shape)
    medicare = np.zeros(shape)
    out_of_pocket = np.zeros(shape)
    ltc_gross = np.zeros(shape)
    ltc_insurance = np.zeros(shape)
    person_ltc_selected: dict[str, np.ndarray] = {}
    person_ltc_active: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(
        scenario.seed ^ HEALTHCARE_RANDOM_SEED_XOR
    )
    person_index = {
        person_id: index
        for index, person_id in enumerate(household_state.person_ids)
    }

    for person in assumptions.people:
        index = person_index[person.person_id]
        alive = household_state.person_alive[index]
        ages = household_state.person_age_months[index] // 12
        pre_mask = alive & (ages[:, None] < person.medicare_start_age)
        medicare_mask = alive & (ages[:, None] >= person.medicare_start_age)
        medical_growth = np.asarray(
            [
                (1 + assumptions.medical_inflation_rate) ** offset
                for offset in range(years)
            ],
            dtype=float,
        )[:, None]
        pre_medicare += (
            pre_mask
            * person.pre_medicare_aca_annual_premium_real
            * medical_growth
        )
        medicare += (
            medicare_mask
            * person.medicare_annual_premium_real
            * medical_growth
        )
        out_of_pocket += (
            (
                pre_mask
                * person.pre_medicare_annual_out_of_pocket_real
                + medicare_mask
                * person.medicare_annual_out_of_pocket_real
            )
            * medical_growth
        )

        ltc = person.long_term_care
        if ltc is None:
            person_ltc_selected[person.person_id] = np.zeros(
                scenario.trials,
                dtype=bool,
            )
            person_ltc_active[person.person_id] = np.zeros(
                scenario.trials,
                dtype=bool,
            )
            continue
        selected = (
            rng.random(scenario.trials)
            < ltc.lifetime_incidence_probability
        )
        person_ltc_selected[person.person_id] = selected
        onset = rng.integers(
            ltc.minimum_onset_age,
            ltc.maximum_onset_age + 1,
            scenario.trials,
        )
        if ltc.duration_standard_deviation_years == 0:
            duration = np.full(
                scenario.trials,
                round(ltc.mean_duration_years),
                dtype=int,
            )
        else:
            duration = np.rint(
                rng.normal(
                    ltc.mean_duration_years,
                    ltc.duration_standard_deviation_years,
                    scenario.trials,
                )
            ).astype(int)
        duration = np.clip(duration, 1, ltc.maximum_duration_years)
        sigma = ltc.annual_cost_log_volatility
        severity = (
            np.ones(scenario.trials)
            if sigma == 0
            else rng.lognormal(-0.5 * sigma**2, sigma, scenario.trials)
        )
        active = (
            selected[None, :]
            & alive
            & retirement_year
            & (ages[:, None] >= onset[None, :])
            & (ages[:, None] < onset[None, :] + duration[None, :])
        )
        person_ltc_active[person.person_id] = np.any(active, axis=0)
        annual_gross = (
            active
            * ltc.annual_cost_real
            * severity[None, :]
            * medical_growth
        )
        ltc_gross += annual_gross
        benefit_active = (
            active
            & (
                ages[:, None]
                < onset[None, :] + ltc.insurance_benefit_years
            )
        )
        ltc_insurance += np.minimum(
            annual_gross,
            (
                benefit_active
                * ltc.insurance_annual_benefit_real
                * medical_growth
            ),
        )

    residual = np.maximum(0.0, ltc_gross - ltc_insurance)
    # Home-equity funding is execution state, not a synthetic prepared pool:
    # the simulation debits the exact linked account and reports actual funded
    # and unmet amounts. Prepared care state retains only demand after insurance.
    home_equity_used = np.zeros(shape)
    ltc_net = residual
    return HealthcareState(
        scenario_sha256=healthcare_state_scenario_sha256(scenario),
        inflation_factors_sha256=(
            healthcare_inflation_factors_sha256(inflation_factors)
        ),
        person_ids=household_state.person_ids,
        pre_medicare_premium=pre_medicare,
        medicare_premium=medicare,
        out_of_pocket=out_of_pocket,
        ltc_gross_cost=ltc_gross,
        ltc_insurance_benefit=ltc_insurance,
        ltc_home_equity_used=home_equity_used,
        ltc_net_cost=ltc_net,
        total_cost=pre_medicare + medicare + out_of_pocket + ltc_net,
        ltc_selected=np.any(
            np.stack(tuple(person_ltc_selected.values())),
            axis=0,
        ),
        ltc_active_in_plan=np.any(
            np.stack(tuple(person_ltc_active.values())),
            axis=0,
        ),
        person_ltc_selected=person_ltc_selected,
        person_ltc_active_in_plan=person_ltc_active,
    )
