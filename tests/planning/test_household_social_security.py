from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ynab_agent.planning.household import (
    estimate_household_state_bytes,
    prepare_household_state,
)
from ynab_agent.planning.household_models import Person
from ynab_agent.planning.models import (
    TaxAssumptions,
    ValuationProvenance,
    WealthScenario,
)
from ynab_agent.planning.simulation import simulate
from ynab_agent.planning.social_security import (
    EntitlementCreditMonths,
    earnings_test_withheld_months,
    earnings_test_withholding,
    family_maximum_monthly,
    full_retirement_age_months,
    retired_worker_factor,
    person_social_security,
    spouse_factor,
    survivor_full_retirement_age_months,
    survivor_factor,
)
from ynab_agent.planning.social_security_optimizer import (
    optimize_social_security,
)
from ynab_agent.planning.tax_engine import HouseholdIncomeTaxInput
from ynab_agent.planning.taxes import TaxAwarePortfolio


def _household_scenario(**updates: Any) -> WealthScenario:
    values: dict[str, Any] = {
        "name": "household",
        "current_age": 66,
        "retirement_age": 66,
        "end_age": 72,
        "starting_portfolio": 500_000,
        "annual_spending": 60_000,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "inflation_rate": 0,
        "trials": 100,
        "seed": 44,
        "household": {
            "plan_start_date": "2026-01-02",
            "survivor_spending_fraction": 0.75,
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1960-01-02",
                    "retirement_age_months": 66 * 12,
                    "primary_insurance_amount_monthly": 3_000,
                    "social_security_claim_age_months": 67 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 68,
                    },
                },
                {
                    "id": "blair",
                    "name": "Blair",
                    "birth_date": "1962-01-02",
                    "retirement_age_months": 64 * 12,
                    "primary_insurance_amount_monthly": 1_200,
                    "social_security_claim_age_months": 62 * 12,
                    "survivor_claim_age_months": 60 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 90,
                    },
                },
            ],
        },
    }
    values.update(updates)
    return WealthScenario.model_validate(values)


def test_ssa_age_adjustments_and_fra_follow_monthly_rules() -> None:
    from datetime import date

    assert full_retirement_age_months(date(1960, 2, 1)) == 67 * 12
    assert full_retirement_age_months(date(1960, 1, 1)) == 66 * 12 + 10
    assert full_retirement_age_months(date(1957, 2, 1)) == 66 * 12 + 6
    assert survivor_full_retirement_age_months(date(1960, 2, 1)) == 66 * 12 + 8
    assert retired_worker_factor(62 * 12, 67 * 12) == pytest.approx(0.70)
    assert retired_worker_factor(70 * 12, 67 * 12) == pytest.approx(1.24)
    assert spouse_factor(62 * 12, 67 * 12) == pytest.approx(0.65)
    assert survivor_factor(60 * 12, 67 * 12) == pytest.approx(0.715)
    assert survivor_factor(67 * 12, 67 * 12) == 1


def test_earnings_test_and_family_maximum_are_bounded_policy_inputs() -> None:
    assert earnings_test_withholding(
        annual_benefit=20_000,
        annual_covered_earnings=34_480,
        age_months=63 * 12,
        fra_months=67 * 12,
    ) == 5_000
    assert earnings_test_withholding(
        annual_benefit=20_000,
        annual_covered_earnings=70_000,
        age_months=66 * 12 + 6,
        fra_months=67 * 12,
    ) == pytest.approx((70_000 - 65_160) / 3)
    assert earnings_test_withholding(
        annual_benefit=20_000,
        annual_covered_earnings=100_000,
        age_months=67 * 12,
        fra_months=67 * 12,
    ) == 0
    assert family_maximum_monthly(1_643) == pytest.approx(2_464.5)


def test_spouse_requires_worker_claim_and_survivor_can_switch_independently() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1960-01-02",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 67 * 12,
        }
    )
    spouse = Person.model_validate(
        {
            "id": "spouse",
            "name": "Spouse",
            "birth_date": "1958-01-02",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 1_000,
            "social_security_claim_age_months": 62 * 12,
            "survivor_claim_age_months": 60 * 12,
        }
    )

    before_worker_claim = person_social_security(
        spouse,
        spouse=worker,
        age_months=67 * 12,
        spouse_alive=True,
        person_alive=True,
    )
    after_worker_claim = person_social_security(
        spouse,
        spouse=worker,
        age_months=69 * 12,
        spouse_alive=True,
        person_alive=True,
    )
    as_survivor = person_social_security(
        spouse,
        spouse=worker,
        age_months=69 * 12,
        spouse_alive=False,
        person_alive=True,
    )

    assert before_worker_claim.spousal == 0
    assert after_worker_claim.spousal > 0
    assert as_survivor.survivor > after_worker_claim.spousal


def test_zero_pia_spouse_receives_spousal_and_survivor_benefits() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1962-02-01",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 67 * 12,
        }
    )
    claimant = Person.model_validate(
        {
            "id": "claimant",
            "name": "Claimant",
            "birth_date": "1962-02-01",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 0,
            "social_security_claim_age_months": 62 * 12,
        }
    )

    spouse_benefit = person_social_security(
        claimant,
        spouse=worker,
        age_months=67 * 12,
        spouse_alive=True,
        person_alive=True,
        family_maximum_override_monthly=10_000,
    )
    survivor_benefit = person_social_security(
        claimant,
        spouse=worker,
        age_months=67 * 12,
        spouse_alive=False,
        person_alive=True,
        spouse_death_age_months=67 * 12,
    )

    assert spouse_benefit.own == 0
    assert spouse_benefit.spousal == pytest.approx(1_500 * 12)
    assert survivor_benefit.survivor == pytest.approx(3_000 * 12)


def test_spousal_reduction_uses_later_worker_filing_onset() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1960-02-01",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 67 * 12,
        }
    )
    claimant = Person.model_validate(
        {
            "id": "claimant",
            "name": "Claimant",
            "birth_date": "1960-02-01",
            "retirement_age_months": 62 * 12,
            "primary_insurance_amount_monthly": 1_000,
            "social_security_claim_age_months": 62 * 12,
        }
    )

    before_worker_files = person_social_security(
        claimant,
        spouse=worker,
        age_months=66 * 12,
        spouse_alive=True,
        person_alive=True,
        family_maximum_override_monthly=10_000,
    )
    after_worker_files = person_social_security(
        claimant,
        spouse=worker,
        age_months=67 * 12,
        spouse_alive=True,
        person_alive=True,
        family_maximum_override_monthly=10_000,
    )

    assert before_worker_files.spousal == 0
    assert after_worker_files.own == pytest.approx(1_000 * 12 * 0.70)
    assert after_worker_files.spousal == pytest.approx(500 * 12)


def test_survivor_reduction_freezes_at_actual_death_onset() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1962-02-01",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 67 * 12,
        }
    )
    survivor = Person.model_validate(
        {
            "id": "survivor",
            "name": "Survivor",
            "birth_date": "1962-02-01",
            "retirement_age_months": 67 * 12,
            "survivor_claim_age_months": 60 * 12,
        }
    )
    expected_factor = 0.715 + 0.285 * (60 / 84)

    at_death = person_social_security(
        survivor,
        spouse=worker,
        age_months=65 * 12,
        spouse_alive=False,
        person_alive=True,
        spouse_death_age_months=65 * 12,
    )
    next_year = person_social_security(
        survivor,
        spouse=worker,
        age_months=66 * 12,
        spouse_alive=False,
        person_alive=True,
        spouse_death_age_months=65 * 12,
    )

    assert at_death.survivor == pytest.approx(36_000 * expected_factor)
    assert next_year.survivor == at_death.survivor


@pytest.mark.parametrize(
    ("death_age", "expected_factor"),
    [(65, 1.0), (68, 1.08)],
)
def test_worker_death_before_planned_filing_earns_only_drcs_through_death(
    death_age: int,
    expected_factor: float,
) -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1960-02-01",
            "retirement_age_months": 70 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 70 * 12,
        }
    )
    survivor = Person.model_validate(
        {
            "id": "survivor",
            "name": "Survivor",
            "birth_date": "1960-02-01",
            "retirement_age_months": 67 * 12,
            "survivor_claim_age_months": 67 * 12,
        }
    )

    benefit = person_social_security(
        survivor,
        spouse=worker,
        age_months=max(67, death_age) * 12,
        spouse_alive=False,
        person_alive=True,
        spouse_death_age_months=death_age * 12,
    )

    assert benefit.survivor == pytest.approx(36_000 * expected_factor)


def test_rib_lim_caps_survivor_of_early_retirement_claimant() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1960-02-01",
            "retirement_age_months": 62 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 62 * 12,
        }
    )
    survivor = Person.model_validate(
        {
            "id": "survivor",
            "name": "Survivor",
            "birth_date": "1960-02-01",
            "retirement_age_months": 67 * 12,
            "survivor_claim_age_months": 60 * 12,
        }
    )

    age_60 = person_social_security(
        survivor,
        spouse=worker,
        age_months=60 * 12,
        spouse_alive=False,
        person_alive=True,
        spouse_death_age_months=60 * 12,
    )
    at_fra = person_social_security(
        survivor.model_copy(
            update={"survivor_claim_age_months": 67 * 12}
        ),
        spouse=worker,
        age_months=67 * 12,
        spouse_alive=False,
        person_alive=True,
        spouse_death_age_months=67 * 12,
    )

    assert age_60.survivor == pytest.approx(3_000 * 12 * 0.715)
    assert at_fra.survivor == pytest.approx(3_000 * 12 * 0.825)


def test_adjustment_credits_are_separate_for_each_entitlement_record() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1960-02-01",
            "retirement_age_months": 66 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 66 * 12,
        }
    )
    claimant = Person.model_validate(
        {
            "id": "claimant",
            "name": "Claimant",
            "birth_date": "1960-02-01",
            "retirement_age_months": 62 * 12,
            "primary_insurance_amount_monthly": 1_000,
            "social_security_claim_age_months": 62 * 12,
        }
    )

    benefit = person_social_security(
        claimant,
        spouse=worker,
        age_months=67 * 12,
        spouse_alive=True,
        person_alive=True,
        family_maximum_override_monthly=10_000,
        entitlement_credit_months=EntitlementCreditMonths(
            retirement=36,
            spousal=0,
        ),
    )

    assert benefit.credit_months_by_entitlement == EntitlementCreditMonths(
        retirement=36,
        spousal=0,
        survivor=0,
    )
    assert benefit.own == pytest.approx(
        1_000 * 12 * retired_worker_factor(65 * 12, 67 * 12)
    )
    assert benefit.spousal == pytest.approx(
        500 * 12 * spouse_factor(66 * 12, 67 * 12)
    )


def test_survivor_pre_62_credits_do_not_inflate_retirement_record() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1960-02-01",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 67 * 12,
        }
    )
    claimant = Person.model_validate(
        {
            "id": "claimant",
            "name": "Claimant",
            "birth_date": "1960-02-01",
            "retirement_age_months": 62 * 12,
            "primary_insurance_amount_monthly": 1_000,
            "social_security_claim_age_months": 62 * 12,
            "survivor_claim_age_months": 60 * 12,
        }
    )

    benefit = person_social_security(
        claimant,
        spouse=worker,
        age_months=62 * 12,
        spouse_alive=False,
        person_alive=True,
        spouse_death_age_months=60 * 12,
        entitlement_credit_months=EntitlementCreditMonths(
            retirement=0,
            survivor=24,
        ),
    )

    assert benefit.credit_months_by_entitlement.retirement == 0
    assert benefit.credit_months_by_entitlement.survivor == 24
    assert benefit.own == pytest.approx(1_000 * 12 * 0.70)


def test_earnings_test_honors_work_timeline_inflation_policy() -> None:
    claimant = Person.model_validate(
        {
            "id": "claimant",
            "name": "Claimant",
            "birth_date": "1960-02-01",
            "retirement_age_months": 62 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 62 * 12,
            "work": [
                {
                    "start_age": 62,
                    "end_age": 62,
                    "annual_covered_earnings": 20_000,
                    "inflation_adjusted": True,
                }
            ],
        }
    )
    nominal = person_social_security(
        claimant,
        spouse=None,
        age_months=62 * 12,
        spouse_alive=False,
        person_alive=True,
        work_inflation_factor=2.0,
    )
    fixed = person_social_security(
        claimant.model_copy(
            update={
                "work": [
                    claimant.work[0].model_copy(
                        update={"inflation_adjusted": False}
                    )
                ]
            }
        ),
        spouse=None,
        age_months=62 * 12,
        spouse_alive=False,
        person_alive=True,
        work_inflation_factor=2.0,
    )

    assert nominal.earnings_test_withheld == pytest.approx(7_760)
    assert fixed.earnings_test_withheld == 0


def test_dual_entitlement_reduces_spousal_excess_separately_from_own() -> None:
    worker = Person.model_validate(
        {
            "id": "worker",
            "name": "Worker",
            "birth_date": "1960-01-02",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 62 * 12,
        }
    )
    claimant = Person.model_validate(
        {
            "id": "claimant",
            "name": "Claimant",
            "birth_date": "1960-01-02",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 1_000,
            "social_security_claim_age_months": 62 * 12,
        }
    )

    benefit = person_social_security(
        claimant,
        spouse=worker,
        age_months=62 * 12,
        spouse_alive=True,
        person_alive=True,
        family_maximum_override_monthly=10_000,
    )

    assert benefit.own == pytest.approx(1_000 * 12 * 0.70)
    assert benefit.spousal == pytest.approx((1_500 - 1_000) * 12 * 0.65)


def test_fra_recomputation_applies_fully_withheld_month_credit() -> None:
    claimant = Person.model_validate(
        {
            "id": "claimant",
            "name": "Claimant",
            "birth_date": "1960-01-02",
            "retirement_age_months": 67 * 12,
            "primary_insurance_amount_monthly": 3_000,
            "social_security_claim_age_months": 62 * 12,
        }
    )

    assert earnings_test_withheld_months(
        annual_benefit=25_200,
        annual_withheld=5_000,
    ) == 3
    credited = person_social_security(
        claimant,
        spouse=None,
        age_months=67 * 12,
        spouse_alive=False,
        person_alive=True,
        earnings_test_credit_months=60,
    )

    assert credited.earnings_test_credit_months_applied == 60
    assert credited.own == pytest.approx(36_000)
    assert credited.earnings_test_credit_annual == pytest.approx(10_800)


def test_household_model_preserves_legacy_and_validates_ownership() -> None:
    legacy = WealthScenario.model_validate(
        {
            "name": "legacy",
            "current_age": 60,
            "retirement_age": 65,
            "end_age": 66,
            "starting_portfolio": 100_000,
            "annual_spending": 10_000,
            "trials": 100,
        }
    )
    assert legacy.household is None
    assert simulate(legacy, 100_000).household_cash_flow_audit == []

    with pytest.raises(ValidationError, match="owner_person_id"):
        _household_scenario(
            accounts=[
                {
                    "id": "ira",
                    "role": "retirement",
                    "owner_person_id": "unknown",
                }
            ]
        )


def test_household_simulation_audits_survivor_and_filing_transition() -> None:
    result = simulate(_household_scenario(), 500_000, include_annual_path=False)

    assert len(result.household_cash_flow_audit) == 12
    survivor_row = next(
        row
        for row in result.household_cash_flow_audit
        if row["year"] == 2028 and row["person_id"] == "blair"
    )
    assert survivor_row["alive_probability"] == 1
    assert survivor_row["social_security_survivor_real"]["p50"] > 0  # type: ignore[index]
    assert survivor_row["filing_status"] == {
        "married_filing_jointly_probability": 0.0,
        "single_probability": 1.0,
        "no_filer_probability": 0.0,
    }
    household_manifest = result.reproducibility["household"]
    assert household_manifest["social_security"]["policy_id"] == (  # type: ignore[index]
        "ssa_retirement_household_2026_v3"
    )
    assert household_manifest["longevity_random"] == {  # type: ignore[index]
        "substream": "household_longevity_v1",
        "seed_derivation": "scenario_seed_xor_constant",
        "seed_xor_hex": "0x535341",
        "draw_policy": (
            "one standard-normal draw per person and trial; "
            "second-person correlation applied after draw"
        ),
    }


def test_pre_retirement_pension_and_benefits_are_not_implicit_contributions() -> None:
    early = _household_scenario(
        current_age=62,
        retirement_age=67,
        end_age=68,
        household={
            "plan_start_date": "2022-01-02",
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1960-01-02",
                    "retirement_age_months": 67 * 12,
                    "primary_insurance_amount_monthly": 3_000,
                    "social_security_claim_age_months": 62 * 12,
                    "work": [
                        {
                            "start_age": 62,
                            "end_age": 66,
                            "annual_covered_earnings": 10_000,
                        }
                    ],
                    "pensions": [
                        {
                            "name": "bridge pension",
                            "start_age": 62,
                            "end_age": 66,
                            "annual_amount": 10_000,
                        }
                    ],
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 95,
                    },
                }
            ],
        },
    )
    never_claims = early.model_copy(
        update={
            "household": early.household.model_copy(
                update={
                    "people": [
                        early.household.people[0].model_copy(
                            update={"social_security_claim_age_months": None}
                        )
                    ]
                }
            )
        }
    )
    no_pension = early.model_copy(
        update={
            "household": early.household.model_copy(
                update={
                    "people": [
                        early.household.people[0].model_copy(
                            update={"pensions": []}
                        )
                    ]
                }
            )
        }
    )

    early_result = simulate(early, 500_000, include_annual_path=False)
    never_result = simulate(never_claims, 500_000, include_annual_path=False)
    no_pension_result = simulate(
        no_pension,
        500_000,
        include_annual_path=False,
    )

    assert early_result.ending_balance_real == no_pension_result.ending_balance_real
    # Only the retirement-year benefit offsets retirement spending. Benefits
    # received before the scenario retirement milestone are not auto-invested.
    assert (
        early_result.ending_balance_real["p50"]
        - never_result.ending_balance_real["p50"]
    ) == pytest.approx(3_000 * 12 * 0.70)


def test_household_audit_tracks_withheld_month_credit_through_fra() -> None:
    scenario = _household_scenario(
        current_age=62,
        retirement_age=62,
        end_age=68,
        household={
            "plan_start_date": "2022-01-02",
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1960-01-02",
                    "retirement_age_months": 62 * 12,
                    "primary_insurance_amount_monthly": 3_000,
                    "social_security_claim_age_months": 62 * 12,
                    "work": [
                        {
                            "start_age": 62,
                            "end_age": 66,
                            "annual_covered_earnings": 100_000,
                        }
                    ],
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 95,
                    },
                }
            ],
        },
    )

    result = simulate(scenario, 500_000, include_annual_path=False)
    fra_row = next(
        row
        for row in result.household_cash_flow_audit
        if row["year"] == 2027
    )

    assert fra_row["age_months"] == 67 * 12
    assert fra_row["earnings_test_credit_months_applied"]["p50"] == 54  # type: ignore[index]
    assert fra_row[
        "earnings_test_credit_months_cumulative_by_entitlement"
    ] == {
        "retirement": {"p10": 54.0, "p50": 54.0, "p90": 54.0},
        "spousal": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
        "survivor": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
    }
    assert fra_row["earnings_test_credit_benefit_real"]["p50"] > 0  # type: ignore[index]
    assert fra_row["social_security_paid_real"]["p50"] > 25_200  # type: ignore[index]
    assert fra_row["alive_status"] == {
        "alive_probability": 1.0,
        "deceased_probability": 0.0,
        "spouse_alive_probability": 0.0,
        "household_alive_probability": 1.0,
    }


def test_household_earnings_test_uses_each_work_timeline_inflation_policy() -> None:
    adjusted = _household_scenario(
        current_age=62,
        retirement_age=62,
        end_age=63,
        household={
            "plan_start_date": "2022-01-02",
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1960-01-02",
                    "retirement_age_months": 62 * 12,
                    "primary_insurance_amount_monthly": 3_000,
                    "social_security_claim_age_months": 62 * 12,
                    "work": [
                        {
                            "start_age": 62,
                            "end_age": 62,
                            "annual_covered_earnings": 20_000,
                            "inflation_adjusted": True,
                        }
                    ],
                }
            ],
        },
    )
    fixed = adjusted.model_copy(
        update={
            "household": adjusted.household.model_copy(
                update={
                    "people": [
                        adjusted.household.people[0].model_copy(
                            update={
                                "work": [
                                    adjusted.household.people[0].work[
                                        0
                                    ].model_copy(
                                        update={
                                            "inflation_adjusted": False
                                        }
                                    )
                                ]
                            }
                        )
                    ]
                }
            )
        }
    )
    inflation = np.full((1, adjusted.trials), 2.0)

    adjusted_state = prepare_household_state(
        adjusted,
        inflation_factors=inflation,
    )
    fixed_state = prepare_household_state(
        fixed,
        inflation_factors=inflation,
    )

    assert adjusted_state is not None and fixed_state is not None
    adjusted_audit = adjusted_state.audit[0]
    fixed_audit = fixed_state.audit[0]
    assert adjusted_audit["earnings_test_withheld_real"]["p50"] == (  # type: ignore[index]
        pytest.approx(3_880)
    )
    assert fixed_audit["earnings_test_withheld_real"]["p50"] == 0  # type: ignore[index]
    assert adjusted_audit["covered_work_earnings_real"]["p50"] == 20_000  # type: ignore[index]
    assert fixed_audit["covered_work_earnings_real"]["p50"] == 10_000  # type: ignore[index]


def test_post_death_years_do_not_recover_or_fail_spending_tiers() -> None:
    base: dict[str, Any] = {
        "name": "post-death semantics",
        "current_age": 65,
        "retirement_age": 65,
        "end_age": 68,
        "starting_portfolio": 100,
        "annual_spending": 100,
        "spending_tiers": [{"name": "minimum", "annual_amount": 80}],
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "inflation_rate": 0,
        "trials": 100,
        "household": {
            "plan_start_date": "2026-01-02",
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1961-01-02",
                    "retirement_age_months": 65 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 66,
                    },
                }
            ],
        },
    }
    funded = simulate(WealthScenario.model_validate(base), 100)
    failed_scenario = WealthScenario.model_validate(
        {**base, "starting_portfolio": 0}
    )
    failed = simulate(failed_scenario, 0)

    minimum = next(
        outcome
        for outcome in funded.goal_outcomes
        if outcome.name == "minimum"
    )
    dead_row = next(
        row
        for row in funded.household_cash_flow_audit
        if row["year"] == 2027
    )
    assert minimum.attainment_probability == 1
    assert dead_row["filing_status"] == {
        "married_filing_jointly_probability": 0.0,
        "single_probability": 0.0,
        "no_filer_probability": 1.0,
    }
    assert failed.recovered_trials == 0
    assert failed.recovery_probability == 0


def test_survivor_spending_fraction_scales_spending_tier_targets() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "survivor tier",
            "current_age": 65,
            "retirement_age": 65,
            "end_age": 67,
            "starting_portfolio": 175,
            "annual_spending": 100,
            "spending_tiers": [{"name": "minimum", "annual_amount": 80}],
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "inflation_rate": 0,
            "trials": 100,
            "household": {
                "plan_start_date": "2026-01-02",
                "survivor_spending_fraction": 0.75,
                "people": [
                    {
                        "id": "alex",
                        "name": "Alex",
                        "birth_date": "1961-01-02",
                        "retirement_age_months": 65 * 12,
                        "longevity": {
                            "mode": "deterministic",
                            "death_age": 66,
                        },
                    },
                    {
                        "id": "blair",
                        "name": "Blair",
                        "birth_date": "1961-01-02",
                        "retirement_age_months": 65 * 12,
                        "longevity": {
                            "mode": "deterministic",
                            "death_age": 90,
                        },
                    },
                ],
            },
        }
    )

    result = simulate(scenario, 175)

    minimum = next(
        outcome
        for outcome in result.goal_outcomes
        if outcome.name == "minimum"
    )
    assert result.funded_spending_ratio["p50"] == 1
    assert minimum.attainment_probability == 1


def test_probabilistic_joint_longevity_is_seeded_and_resource_bounded() -> None:
    scenario = _household_scenario(
        household={
            "plan_start_date": "2026-01-02",
            "longevity_correlation": 0.6,
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1960-01-02",
                    "retirement_age_months": 792,
                    "primary_insurance_amount_monthly": 3_000,
                    "social_security_claim_age_months": 804,
                    "longevity": {
                        "mode": "probabilistic",
                        "death_age": None,
                        "mean_death_age": 88,
                        "standard_deviation_years": 5,
                        "minimum_death_age": 70,
                        "maximum_death_age": 105,
                    },
                },
                {
                    "id": "blair",
                    "name": "Blair",
                    "birth_date": "1962-01-02",
                    "retirement_age_months": 768,
                    "primary_insurance_amount_monthly": 1_200,
                    "social_security_claim_age_months": 744,
                    "longevity": {
                        "mode": "probabilistic",
                        "death_age": None,
                        "mean_death_age": 90,
                        "standard_deviation_years": 4,
                        "minimum_death_age": 70,
                        "maximum_death_age": 105,
                    },
                },
            ],
        },
    )
    inflation = np.ones(
        (scenario.end_age - scenario.current_age + 1, scenario.trials)
    )
    first = prepare_household_state(scenario, inflation_factors=inflation)
    second = prepare_household_state(scenario, inflation_factors=inflation)

    assert first is not None and second is not None
    assert first.person_ids == ("alex", "blair")
    assert first.person_alive.shape == (2, 6, scenario.trials)
    assert first.person_age_months.shape == (2, 6)
    np.testing.assert_array_equal(first.person_alive, second.person_alive)
    np.testing.assert_array_equal(
        first.person_age_months,
        second.person_age_months,
    )
    np.testing.assert_array_equal(first.spending_fraction, second.spending_fraction)
    assert first.audit == second.audit
    assert estimate_household_state_bytes(scenario) > (
        first.spending_fraction.nbytes
        + first.person_alive.nbytes
        + first.person_age_months.nbytes
    )


class _RecordingTaxEngine:
    engine_id = "recording_progressive_test_v1"

    def __init__(self) -> None:
        self.joint_filing: list[np.ndarray] = []

    def income_tax(
        self,
        tax_input: HouseholdIncomeTaxInput,
        *,
        assumptions: TaxAssumptions,
    ) -> np.ndarray:
        del assumptions
        self.joint_filing.append(tax_input.joint_filing.copy())
        return np.zeros_like(tax_input.ordinary_income)


def test_tax_engine_port_receives_joint_to_single_transition() -> None:
    scenario = _household_scenario(
        tax_buckets=[
            {
                "tax_treatment": "cash",
                "starting_balance": 500_000,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0.2,
            "long_term_capital_gains_tax_rate": 0.15,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["cash"],
            "retirement_surplus_destination": "cash",
        },
    )
    engine = _RecordingTaxEngine()
    result = simulate(
        scenario,
        500_000,
        tax_engine=engine,
        include_annual_path=False,
    )

    assert np.all(engine.joint_filing[0])
    assert not np.any(engine.joint_filing[2])
    assert result.reproducibility["household"]["tax_engine_port"] == engine.engine_id  # type: ignore[index]


def test_progressive_household_filing_status_changes_income_and_withdrawal_tax() -> None:
    scenario = _household_scenario(
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 500_000,
                "owner_person_id": "alex",
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0.2,
            "long_term_capital_gains_tax_rate": 0.15,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "married_filing_jointly",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1960,
                "spouse_birth_year": 1962,
                "aca_household_size": 2,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred"],
            "retirement_surplus_destination": "tax_deferred",
        },
    )
    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=500_000,
    )
    assumptions = scenario.tax_assumptions
    assert assumptions is not None
    joint_filing = np.zeros(scenario.trials, dtype=bool)
    joint_filing[: scenario.trials // 2] = True
    unmet = portfolio.fund_retirement_spending(
        slice(0, scenario.trials),
        age=66,
        tax_year=2026,
        spending=60_000,
        ordinary_income=20_000,
        social_security_income=12_000,
        tax_free_income=0,
        opening_tax_deferred=portfolio.opening_tax_deferred(
            slice(0, scenario.trials)
        ),
        inflation_factor=1.0,
        assumptions=assumptions,
        joint_filing=joint_filing,
    )

    assert not np.any(unmet > 0.005)
    joint_tax = portfolio.annual_tax_nominal[: scenario.trials // 2]
    single_tax = portfolio.annual_tax_nominal[scenario.trials // 2 :]
    assert np.all(single_tax > joint_tax)
    balances = portfolio.total(slice(0, scenario.trials))
    assert np.all(
        balances[scenario.trials // 2 :] < balances[: scenario.trials // 2]
    )


def test_optimizer_is_bounded_and_ranks_household_portfolio_outcomes() -> None:
    result = optimize_social_security(
        _household_scenario(
            end_age=70,
            tax_buckets=[
                {
                    "tax_treatment": "cash",
                    "starting_balance": 500_000,
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.2,
                "long_term_capital_gains_tax_rate": 0.15,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "married_filing_jointly",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1960,
                    "spouse_birth_year": 1962,
                    "aca_household_size": 2,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["cash"],
                "retirement_surplus_destination": "cash",
            },
        ),
        500_000,
        valuation_provenance=ValuationProvenance(
            source="test",
            account_ids=(),
            source_sha256="test-snapshot",
        ),
        candidate_ages=(62, 70),
    )

    assert result.strategy_count == 4
    assert result.compute_units == 4 * 100 * 4
    assert result.recommended == result.evaluations[0]
    assert "never cumulative benefits" in result.objective
    assert result.manifest["tax_engine_port"] == (
        "progressive_us_indiana_household_v1"
    )
    assert result.manifest["starting_portfolio"] == 500_000
    assert result.manifest["valuation_provenance"] == {
        "source": "test",
        "as_of": None,
        "account_ids": [],
        "source_sha256": "test-snapshot",
    }
    progressive_policy = result.manifest["progressive_tax_policy"]
    assert isinstance(progressive_policy, dict)
    assert progressive_policy["policy_id"] == "us_in_2026_v1"
    assert len(str(progressive_policy["resource_sha256"])) == 64
    assert all(
        evaluation.lifetime_tax_real_p50 is not None
        for evaluation in result.evaluations
    )
