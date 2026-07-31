from __future__ import annotations

import pytest
from pydantic import ValidationError
import numpy as np

from ynab_agent.planning.housing import (
    estimate_housing_state_bytes,
    housing_manifest,
    project_housing_plan,
)
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.paths import (
    MultiAssetSimulationPaths,
    SimulationPaths,
)
from ynab_agent.planning.progressive_tax import calculate_income_tax_component_arrays
from ynab_agent.planning.simulation import (
    canonical_scenario_sha256,
    simulate,
)
from ynab_agent.planning.taxes import TaxAwarePortfolio
from ynab_agent.services.planner_jobs import (
    PlannerExecutionPolicy,
    PlannerSimulationResult,
)


def _scenario(*, decision: dict[str, object], **overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "housing",
        "current_age": 65,
        "retirement_age": 65,
        "end_age": 67,
        "starting_portfolio": 100_000,
        "annual_spending": 1,
        "tax_buckets": [
            {
                "tax_treatment": "cash",
                "starting_balance": 100_000,
                "contribution_fraction": 0,
            }
        ],
        "tax_assumptions": {
            "ordinary_income_tax_rate": 0.2,
            "long_term_capital_gains_tax_rate": 0.2,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["cash"],
            "retirement_surplus_destination": "cash",
        },
        "housing_plan": {
            "home": {
                "current_value": 500_000,
                "cost_basis": 100_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": decision,
            "primary_residence_gain_exclusion": 250_000,
        },
        "inflation_rate": 0,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 19,
    }
    values.update(overrides)
    buckets = [
        {
            **bucket,
            "account_id": bucket.get("account_id", f"account-{index}"),
        }
        for index, bucket in enumerate(values["tax_buckets"])
    ]
    values["tax_buckets"] = buckets
    if "accounts" not in overrides:
        role_by_treatment = {
            "cash": "cash",
            "taxable": "taxable",
        }
        values["accounts"] = [
            {
                "id": bucket["account_id"],
                "role": role_by_treatment.get(
                    str(bucket["tax_treatment"]),
                    "retirement",
                ),
                **(
                    {"owner_person_id": bucket["owner_person_id"]}
                    if bucket.get("owner_person_id") is not None
                    else {}
                ),
            }
            for bucket in buckets
        ]
    return WealthScenario.model_validate(values)


def test_keep_home_never_becomes_spendable() -> None:
    housing = _scenario(decision={"kind": "keep"})
    baseline = housing.model_copy(update={"housing_plan": None})

    keep_result = simulate(housing, 100_000)
    baseline_result = simulate(baseline, 100_000)

    assert keep_result.ending_balance_real == baseline_result.ending_balance_real
    assert keep_result.ending_balance_real["p50"] == pytest.approx(99_998)
    assert keep_result.estate_value_real["p50"] == pytest.approx(599_998)
    assert keep_result.housing_manifest is not None
    assert "unless an explicit" in str(keep_result.housing_manifest["liquidity_invariant"])


def test_sale_deposits_after_tax_proceeds_in_explicit_cash_bucket() -> None:
    result = simulate(
        _scenario(
            decision={
                "kind": "sell",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
            }
        ),
        100_000,
    )

    sale = result.annual_housing[0]
    assert sale["taxable_gain"] == pytest.approx(150_000)
    assert sale["gain_tax_accounting"] == "combined_annual_tax_engine"
    assert sale["liquid_deposit"] == pytest.approx(500_000)
    assert sale["proceeds_destination_account_id"] == "account-0"
    assert result.ending_balance_real["p50"] == pytest.approx(569_998)
    assert result.estate_value_real == result.ending_balance_real


def test_progressive_sale_gain_stacks_on_same_year_ordinary_income() -> None:
    scenario = _scenario(
        decision={
            "kind": "sell",
            "event_age": 65,
            "proceeds_destination_account_id": "account-0",
        },
        income_streams=[
            {
                "name": "pension",
                "start_age": 65,
                "annual_amount": 100_000,
                "end_age": 65,
                "inflation_adjusted": False,
                "tax_treatment": "ordinary",
            }
        ],
        tax_assumptions={
            "tax_model": "progressive_us_indiana",
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
            },
            "withdrawal_order": ["cash"],
            "retirement_surplus_destination": "cash",
            "apply_required_minimum_distributions": False,
        },
    )

    result = simulate(scenario, 100_000)
    audit = result.annual_tax_audit[0]
    assumptions = scenario.tax_assumptions
    assert assumptions is not None and assumptions.progressive is not None
    combined = calculate_income_tax_component_arrays(
        assumptions.progressive,
        tax_year=2026,
        ordinary_income=100_000,
        long_term_capital_gains=150_000,
        social_security_income=0,
    )
    isolated = calculate_income_tax_component_arrays(
        assumptions.progressive,
        tax_year=2026,
        ordinary_income=0,
        long_term_capital_gains=150_000,
        social_security_income=0,
    )
    assert audit["federal_income_tax_nominal"]["p50"] == pytest.approx(
        float(combined.federal_income_tax)
    )
    assert audit["realized_long_term_capital_gains_nominal"]["p50"] == 150_000
    assert float(combined.federal_income_tax) > float(isolated.federal_income_tax)


def test_external_gain_hook_combines_pre_retirement_income_liability() -> None:
    scenario = _scenario(
        decision={"kind": "keep"},
        current_age=55,
        retirement_age=65,
        end_age=66,
        starting_portfolio=300_000,
        tax_buckets=[
            {
                "tax_treatment": "cash",
                "starting_balance": 300_000,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1971,
            },
            "withdrawal_order": ["cash"],
            "retirement_surplus_destination": "cash",
            "apply_required_minimum_distributions": False,
        },
    )
    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=300_000,
    )
    assumptions = scenario.tax_assumptions
    assert assumptions is not None

    unmet = portfolio.fund_retirement_spending(
        slice(0, scenario.trials),
        age=55,
        tax_year=2026,
        spending=0,
        ordinary_income=100_000,
        social_security_income=0,
        tax_free_income=0,
        external_long_term_capital_gains=150_000,
        opening_tax_deferred=portfolio.opening_tax_deferred(
            slice(0, scenario.trials)
        ),
        inflation_factor=1,
        assumptions=assumptions,
    )

    assert np.all(unmet == 0)
    assert portfolio.annual_tax_nominal[0] == pytest.approx(43_015.50)
    assert portfolio.cumulative_tax_real[0] == pytest.approx(43_015.50)


def test_terminal_home_gain_and_portfolio_liquidate_on_one_return() -> None:
    scenario = _scenario(
        decision={"kind": "keep"},
        starting_portfolio=100_000,
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 100_000,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
            },
            "withdrawal_order": ["tax_deferred"],
            "retirement_surplus_destination": "tax_deferred",
            "apply_required_minimum_distributions": False,
        },
    )
    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=100_000,
    )
    assumptions = scenario.tax_assumptions
    assert assumptions is not None and assumptions.progressive is not None
    combined = portfolio.after_tax_estate_value(
        slice(0, scenario.trials),
        assumptions=assumptions,
        tax_year=2027,
        external_disposition_value=500_000,
        external_long_term_capital_gains=174_999,
    )
    portfolio_only = portfolio.after_tax_estate_value(
        slice(0, scenario.trials),
        assumptions=assumptions,
        tax_year=2027,
    )
    isolated_home_tax = calculate_income_tax_component_arrays(
        assumptions.progressive,
        tax_year=2027,
        ordinary_income=0,
        long_term_capital_gains=174_999,
        social_security_income=0,
    ).total_income_tax
    incorrectly_separate = portfolio_only + 500_000 - isolated_home_tax

    assert combined[0] < incorrectly_separate[0]
    assert incorrectly_separate[0] - combined[0] == pytest.approx(
        11_443.01,
        abs=0.01,
    )


def test_terminal_tax_adds_taxable_account_and_home_gains() -> None:
    scenario = _scenario(
        decision={"kind": "keep"},
        starting_portfolio=200_000,
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 100_000,
            },
            {
                "tax_treatment": "taxable",
                "starting_balance": 100_000,
                "taxable_basis": 50_000,
            },
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
            },
            "withdrawal_order": ["taxable", "tax_deferred"],
            "retirement_surplus_destination": "taxable",
            "apply_required_minimum_distributions": False,
        },
    )
    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=200_000,
    )
    assumptions = scenario.tax_assumptions
    assert assumptions is not None

    combined = portfolio.after_tax_estate_value(
        slice(0, scenario.trials),
        assumptions=assumptions,
        tax_year=2027,
        external_disposition_value=500_000,
        external_long_term_capital_gains=400_000,
    )

    assert combined[0] == pytest.approx(604_119)
    assert combined[0] != pytest.approx(676_049)


def test_simulation_reports_combined_terminal_after_tax_estate() -> None:
    scenario = _scenario(
        decision={"kind": "keep"},
        end_age=66,
        starting_portfolio=100_000,
        income_streams=[
            {
                "name": "spending cash",
                "start_age": 65,
                "end_age": 65,
                "annual_amount": 1,
                "inflation_adjusted": False,
                "tax_treatment": "tax_free",
            }
        ],
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 100_000,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
            },
            "withdrawal_order": ["tax_deferred"],
            "retirement_surplus_destination": "tax_deferred",
            "apply_required_minimum_distributions": False,
        },
        housing_plan={
            "home": {
                "current_value": 425_000,
                "cost_basis": 1,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {"kind": "keep"},
            "primary_residence_gain_exclusion": 250_000,
        },
    )
    assumptions = scenario.tax_assumptions
    assert assumptions is not None and assumptions.progressive is not None
    expected_tax = calculate_income_tax_component_arrays(
        assumptions.progressive,
        tax_year=2027,
        ordinary_income=100_000,
        long_term_capital_gains=174_999,
        social_security_income=0,
    ).total_income_tax

    result = simulate(scenario, 100_000)

    assert result.estate_value_real["p50"] == 525_000
    assert result.after_tax_estate_value_real is not None
    assert result.after_tax_estate_value_real["p50"] == pytest.approx(
        525_000 - float(expected_tax)
    )


def test_simulation_adds_mixed_taxable_account_and_home_gains() -> None:
    scenario = _scenario(
        decision={"kind": "keep"},
        end_age=66,
        starting_portfolio=200_001,
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 100_000,
            },
            {
                "tax_treatment": "taxable",
                "starting_balance": 100_000,
                "taxable_basis": 50_000,
            },
            {
                "tax_treatment": "cash",
                "starting_balance": 1,
            },
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
            },
            "withdrawal_order": ["cash", "taxable", "tax_deferred"],
            "retirement_surplus_destination": "cash",
            "apply_required_minimum_distributions": False,
        },
        housing_plan={
            "home": {
                "current_value": 500_000,
                "cost_basis": 100_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {"kind": "keep"},
            "primary_residence_gain_exclusion": 0,
        },
    )

    result = simulate(scenario, 200_001)

    assert result.ending_balance_real["p50"] == 200_000
    assert result.after_tax_estate_value_real is not None
    assert result.after_tax_estate_value_real["p50"] == pytest.approx(604_119)


def test_housing_plan_requires_tax_model_and_matching_destination() -> None:
    raw = _scenario(
        decision={"kind": "keep"},
        current_age=60,
        retirement_age=65,
    ).model_dump(mode="json")
    raw["tax_buckets"] = []
    raw["tax_assumptions"] = None
    with pytest.raises(ValidationError, match="housing_plan requires"):
        WealthScenario.model_validate(raw)

    raw = _scenario(decision={"kind": "keep"}).model_dump(mode="json")
    raw["housing_plan"]["decision"] = {
        "kind": "sell",
        "event_age": 65,
        "proceeds_destination_account_id": "missing-taxable",
    }
    with pytest.raises(ValidationError, match="exactly one linked tax bucket"):
        WealthScenario.model_validate(raw)

    raw = _scenario(
        decision={"kind": "keep"},
        current_age=60,
        retirement_age=65,
    ).model_dump(mode="json")
    raw["housing_plan"]["decision"] = {
        "kind": "sell",
        "event_age": 64,
        "proceeds_destination_account_id": "account-0",
    }
    with pytest.raises(ValidationError, match="before retirement_age"):
        WealthScenario.model_validate(raw)

    raw = _scenario(
        decision={"kind": "keep"},
        current_age=60,
        retirement_age=65,
    ).model_dump(mode="json")
    raw["housing_plan"]["costs_start_age"] = 64
    with pytest.raises(ValidationError, match="costs_start_age before"):
        WealthScenario.model_validate(raw)

    raw = _scenario(
        decision={"kind": "keep"},
        current_age=60,
        retirement_age=65,
    ).model_dump(mode="json")
    raw["housing_plan"]["care"] = {
        "start_age": 64,
        "end_age": 66,
        "annual_cost_real": 10_000,
    }
    with pytest.raises(ValidationError, match="care funding before"):
        WealthScenario.model_validate(raw)

    raw = _scenario(decision={"kind": "keep"}).model_dump(mode="json")
    raw["housing_plan"]["decision"] = {
        "kind": "replace",
        "event_age": 65,
        "proceeds_destination_account_id": "account-0",
        "replacement_home_value": 200_000,
        "replacement_mortgage": {
            "principal": 250_000,
            "annual_interest_rate": 0.05,
            "remaining_years": 30,
        },
    }
    with pytest.raises(ValidationError, match="cannot exceed replacement"):
        WealthScenario.model_validate(raw)

    raw = _scenario(decision={"kind": "keep"}).model_dump(mode="json")
    raw["housing_plan"]["home"]["mortgage"] = {
        "principal": 400_000,
        "annual_interest_rate": 0.05,
        "remaining_years": 20,
    }
    raw["housing_plan"]["decision"] = {
        "kind": "reverse_mortgage",
        "event_age": 65,
        "proceeds_destination_account_id": "account-0",
        "reverse_mortgage_principal": 1_000_000,
    }
    with pytest.raises(ValidationError, match="collateral LTV"):
        WealthScenario.model_validate(raw)

    raw["housing_plan"]["decision"]["reverse_mortgage_principal"] = 400_000
    with pytest.raises(ValidationError, match="pay off the existing mortgage"):
        WealthScenario.model_validate(raw)


def test_housing_state_is_in_shared_resource_admission() -> None:
    housing = _scenario(decision={"kind": "keep"})
    baseline = housing.model_copy(update={"housing_plan": None})
    policy = PlannerExecutionPolicy(
        maximum_working_bytes=64 * 1024 * 1024,
        in_memory_path_bytes=64 * 1024 * 1024,
        maximum_temporary_bytes=64 * 1024 * 1024,
        batch_size=100,
    )

    housing_bytes = policy.required_working_bytes(
        housing,
        paired_historical_inflation=False,
    )
    baseline_bytes = policy.required_working_bytes(
        baseline,
        paired_historical_inflation=False,
    )

    assert housing_bytes > baseline_bytes
    assert housing_bytes - baseline_bytes >= estimate_housing_state_bytes(
        housing.trials,
        True,
    )


def test_projection_audits_replacement_mortgage_care_and_manifest() -> None:
    scenario = _scenario(
        decision={
            "kind": "downsize",
            "event_age": 65,
            "proceeds_destination_account_id": "account-0",
            "replacement_home_value": 300_000,
            "replacement_mortgage": {
                "principal": 100_000,
                "annual_interest_rate": 0,
                "remaining_years": 2,
            },
        },
        housing_plan={
            "home": {
                "current_value": 500_000,
                "cost_basis": 500_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {
                "kind": "downsize",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
                "replacement_home_value": 300_000,
                "replacement_mortgage": {
                    "principal": 100_000,
                    "annual_interest_rate": 0,
                    "remaining_years": 2,
                },
            },
            "care": {
                "start_age": 65,
                "end_age": 66,
                "annual_cost_real": 10_000,
                "funding_account_id": "account-0",
            },
        },
    )

    projection = project_housing_plan(scenario)
    first = projection["annual_housing"][0]
    assert first["action"] == "downsize"
    assert first["liquid_deposit"] == pytest.approx(300_000)
    assert first["mortgage_payment"] == pytest.approx(50_000)
    assert first["care"] == pytest.approx(10_000)
    assert "not complete tax-preparation fidelity" in projection["manifest"]["tax_scope"]


def test_reverse_mortgage_pays_forward_lien_and_is_nonrecourse() -> None:
    scenario = _scenario(
        decision={"kind": "keep"},
        housing_plan={
            "home": {
                "current_value": 500_000,
                "cost_basis": 500_000,
                "annual_appreciation_rate": -0.2,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
                "mortgage": {
                    "principal": 400_000,
                    "annual_interest_rate": 0.05,
                    "remaining_years": 20,
                },
            },
            "decision": {
                "kind": "reverse_mortgage",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
                "reverse_mortgage_principal": 400_000,
                "reverse_mortgage_max_ltv": 0.8,
                "reverse_mortgage_interest_rate": 0.3,
                "reverse_mortgage_origination_rate": 0,
            },
        },
    )

    projection = project_housing_plan(scenario)
    first = projection["annual_housing"][0]

    assert first["liquid_deposit"] == 0
    assert first["ending_mortgage_principal"] == 0
    assert first["mortgage_payment"] == 0
    assert projection["ending_disposition_value"] == 0


def test_unfunded_negative_equity_sale_fails_and_records_shortfall() -> None:
    scenario = _scenario(
        decision={"kind": "keep"},
        starting_portfolio=0,
        tax_buckets=[{"tax_treatment": "cash", "starting_balance": 0}],
        housing_plan={
            "home": {
                "current_value": 100_000,
                "cost_basis": 100_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
                "mortgage": {
                    "principal": 130_000,
                    "annual_interest_rate": 0,
                    "remaining_years": 1,
                },
            },
            "decision": {
                "kind": "sell",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
            },
        },
    )

    result = simulate(scenario, 0)

    assert result.success_rate == 0
    assert result.funded_spending_real["p50"] == 0
    assert result.cumulative_shortfall_real["p50"] == pytest.approx(30_002)
    assert result.annual_housing[0]["forward_lien_payoff"] == 130_000
    assert result.annual_housing[0]["transaction_shortfall"] == 30_000
    assert result.annual_housing[0]["liquid_deposit"] == 0


def test_rent_path_audit_honors_cost_start_and_inflation_paths() -> None:
    scenario = _scenario(
        decision={
            "kind": "rent",
            "event_age": 65,
            "proceeds_destination_account_id": "account-0",
            "annual_rent_real": 12_000,
        },
        housing_plan={
            "home": {
                "current_value": 500_000,
                "cost_basis": 500_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {
                "kind": "rent",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
                "annual_rent_real": 12_000,
            },
            "costs_start_age": 66,
        },
    )
    inflation = np.ones((3, scenario.trials), dtype=float)
    inflation[1, scenario.trials // 2 :] = 2
    inflation[2] = inflation[1]
    paths = SimulationPaths(
        gross_returns=np.ones((2, scenario.trials), dtype=float),
        inflation_factors=inflation,
    )

    result = simulate(scenario, 100_000, prepared_paths=paths)

    assert result.annual_housing[0]["rent"] == 0
    assert result.annual_housing[0]["rent_nominal"] == {
        "p10": 0,
        "p50": 0,
        "p90": 0,
    }
    assert result.annual_housing[1]["rent_nominal"]["p10"] == 12_000
    assert result.annual_housing[1]["rent_nominal"]["p90"] == 24_000


@pytest.mark.parametrize(
    ("decision", "expected_action", "expected_deposit"),
    [
        (
            {
                "kind": "rent",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
                "annual_rent_real": 12_000,
            },
            "rent",
            500_000,
        ),
        (
            {
                "kind": "reverse_mortgage",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
                "reverse_mortgage_principal": 100_000,
                "reverse_mortgage_interest_rate": 0.07,
                "reverse_mortgage_origination_rate": 0.03,
            },
            "reverse_mortgage",
            97_000,
        ),
    ],
)
def test_rent_and_reverse_mortgage_are_explicit_liquidity_events(
    decision: dict[str, object],
    expected_action: str,
    expected_deposit: float,
) -> None:
    projection = project_housing_plan(_scenario(decision=decision))
    first = projection["annual_housing"][0]

    assert first["action"] == expected_action
    assert first["liquid_deposit"] == pytest.approx(expected_deposit)
    if expected_action == "rent":
        assert first["ending_home_value"] == 0
        assert first["rent"] == 12_000
    else:
        assert first["ending_home_value"] == 500_000
        assert first["ending_reverse_mortgage_principal"] == pytest.approx(107_000)
        assert first["reverse_mortgage_gross_proceeds"] == 100_000
        assert first["reverse_mortgage_origination_cost"] == 3_000
        assert first["forward_lien_payoff"] == 0


def test_exact_proceeds_account_never_splits_across_duplicate_owner_buckets() -> None:
    scenario = _scenario(
        decision={
            "kind": "sell",
            "event_age": 65,
            "proceeds_destination_account_id": "blair-taxable",
        },
        starting_portfolio=100_000,
        accounts=[
            {
                "id": "alex-taxable",
                "role": "taxable",
                "owner_person_id": "alex",
            },
            {
                "id": "blair-taxable",
                "role": "taxable",
                "owner_person_id": "blair",
            },
        ],
        household={
            "plan_start_date": "2026-01-02",
            "people": [
                {
                    "id": owner,
                    "name": owner.title(),
                    "birth_date": "1961-01-02",
                    "retirement_age_months": 65 * 12,
                }
                for owner in ("alex", "blair")
            ],
        },
        tax_buckets=[
            {
                "tax_treatment": "taxable",
                "owner_person_id": "alex",
                "account_id": "alex-taxable",
                "starting_balance": 60_000,
                "taxable_basis": 40_000,
            },
            {
                "tax_treatment": "taxable",
                "owner_person_id": "blair",
                "account_id": "blair-taxable",
                "starting_balance": 40_000,
                "taxable_basis": 30_000,
            },
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0.2,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["taxable"],
            "retirement_surplus_destination": "taxable",
        },
    )
    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=100_000,
    )

    destination = portfolio.deposit_external_cash_to_account(
        slice(0, scenario.trials),
        account_id="blair-taxable",
        amount=25_000,
    )
    _, alex_balance, alex_basis = portfolio.account_balance_and_basis(
        slice(0, scenario.trials),
        account_id="alex-taxable",
    )
    _, blair_balance, blair_basis = portfolio.account_balance_and_basis(
        slice(0, scenario.trials),
        account_id="blair-taxable",
    )

    assert destination.owner_person_id == "blair"
    assert alex_balance == pytest.approx(np.full(100, 60_000))
    assert alex_basis == pytest.approx(np.full(100, 40_000))
    assert blair_balance == pytest.approx(np.full(100, 65_000))
    assert blair_basis == pytest.approx(np.full(100, 55_000))

    batched = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=100_000,
    )
    for trial_slice in (slice(0, 37), slice(37, scenario.trials)):
        batched.deposit_external_cash_to_account(
            trial_slice,
            account_id="blair-taxable",
            amount=25_000,
        )
    _, batched_balance, batched_basis = batched.account_balance_and_basis(
        slice(0, scenario.trials),
        account_id="blair-taxable",
    )
    assert batched_balance == pytest.approx(blair_balance)
    assert batched_basis == pytest.approx(blair_basis)

    rerouted_values = scenario.model_dump(mode="json")
    rerouted_values["housing_plan"]["decision"][
        "proceeds_destination_account_id"
    ] = "alex-taxable"
    rerouted = WealthScenario.model_validate(rerouted_values)
    assert canonical_scenario_sha256(rerouted) != canonical_scenario_sha256(
        scenario
    )
    assert housing_manifest(rerouted.housing_plan)[
        "assumptions_sha256"
    ] != housing_manifest(scenario.housing_plan)["assumptions_sha256"]


def test_zero_balance_destination_is_deposited_before_account_return() -> None:
    market = {
        asset: {"expected_return": 0, "volatility": 0}
        for asset in (
            "us_equity",
            "international_equity",
            "bonds",
            "cash",
        )
    }
    market["correlation"] = {
        "values": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    }
    scenario = _scenario(
        decision={
            "kind": "sell",
            "event_age": 65,
            "proceeds_destination_account_id": "destination",
        },
        accounts=[
            {"id": "existing", "role": "taxable"},
            {"id": "destination", "role": "taxable"},
        ],
        tax_buckets=[
            {
                "tax_treatment": "taxable",
                "account_id": "existing",
                "starting_balance": 100_000,
                "taxable_basis": 100_000,
            },
            {
                "tax_treatment": "taxable",
                "account_id": "destination",
                "starting_balance": 0,
                "taxable_basis": 0,
            },
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["taxable"],
            "retirement_surplus_destination": "taxable",
        },
        portfolio_allocation={
            "market": market,
            "accounts": [
                {
                    "account_id": "existing",
                    "portfolio_weight": 1,
                    "target": {
                        "us_equity": 0,
                        "international_equity": 0,
                        "bonds": 0,
                        "cash": 1,
                    },
                },
                {
                    "account_id": "destination",
                    "portfolio_weight": 0,
                    "target": {
                        "us_equity": 1,
                        "international_equity": 0,
                        "bonds": 0,
                        "cash": 0,
                    },
                },
            ],
            "rebalancing": {"frequency_years": 1},
        },
        housing_plan={
            "home": {
                "current_value": 500_000,
                "cost_basis": 500_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {
                "kind": "sell",
                "event_age": 65,
                "proceeds_destination_account_id": "destination",
            },
        },
        end_age=66,
    )
    asset_returns = np.ones((4, 1, scenario.trials), dtype=float)
    asset_returns[0] = 2
    paths = MultiAssetSimulationPaths(
        gross_returns=asset_returns[0],
        inflation_factors=np.ones(2),
        asset_gross_returns=asset_returns,
        source_name="housing-start-year-order",
    )

    result = simulate(scenario, 100_000, prepared_paths=paths)

    housing = result.annual_housing[0]
    returns = result.annual_allocation_real[0]["account_gross_returns"]
    assert housing["destination_balance_before_nominal"]["p50"] == 0
    assert housing["destination_balance_after_nominal"]["p50"] == 500_000
    assert housing["destination_basis_after_nominal"]["p50"] == 500_000
    assert returns["existing"]["p50"] == 1
    assert returns["destination"]["p50"] == 2
    assert result.ending_balance_real["p50"] == pytest.approx(1_099_999)
    persisted = PlannerSimulationResult.model_validate(result.as_dict())
    assert persisted.annual_housing == result.annual_housing
    assert persisted.housing_manifest == result.housing_manifest
    assert persisted.estate_value_real == result.estate_value_real


def test_home_equity_care_port_debits_exact_reserve_without_double_count() -> None:
    scenario = _scenario(
        decision={
            "kind": "sell",
            "event_age": 65,
            "proceeds_destination_account_id": "account-0",
        },
        end_age=66,
        housing_plan={
            "home": {
                "current_value": 500_000,
                "cost_basis": 500_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {
                "kind": "sell",
                "event_age": 65,
                "proceeds_destination_account_id": "account-0",
            },
            "care": {
                "start_age": 65,
                "end_age": 65,
                "annual_cost_real": 30_000,
                "funding_account_id": "account-0",
            },
        },
    )

    result = simulate(scenario, 100_000)
    housing = result.annual_housing[0]

    assert housing["care_funded_from_home_equity_nominal"]["p50"] == 30_000
    assert housing["care_portfolio_fallback_nominal"]["p50"] == 0
    assert housing["destination_balance_after_nominal"]["p50"] == 570_000
    assert result.ending_balance_real["p50"] == pytest.approx(569_999)
    assert result.funded_spending_real["p50"] == 30_001


def test_same_year_home_gain_changes_gain_harvest_decision() -> None:
    tax_buckets = [
        {
            "tax_treatment": "taxable",
            "starting_balance": 50_000,
            "taxable_basis": 0,
        },
        {
            "tax_treatment": "cash",
            "starting_balance": 1,
        },
    ]
    tax_assumptions = {
        "ordinary_income_tax_rate": 0,
        "long_term_capital_gains_tax_rate": 0,
        "tax_model": "progressive_us_indiana",
        "progressive": {
            "filing_status": "single",
            "simulation_start_year": 2026,
            "taxpayer_birth_year": 1961,
        },
        "apply_required_minimum_distributions": False,
        "withdrawal_order": ["cash", "taxable"],
        "retirement_surplus_destination": "cash",
        "strategy": {
            "withdrawal_policy": "ordered",
            "capital_gain_harvest": {
                "start_age": 65,
                "end_age": 65,
                "target_federal_long_term_capital_gains_rate": 0,
                "max_annual_gain_real": 100_000,
            },
        },
    }
    keep = _scenario(
        decision={"kind": "keep"},
        starting_portfolio=50_001,
        end_age=66,
        tax_buckets=tax_buckets,
        tax_assumptions=tax_assumptions,
    )
    sold_values = keep.model_dump(mode="json")
    sold_values["housing_plan"]["decision"] = {
        "kind": "sell",
        "event_age": 65,
        "proceeds_destination_account_id": "account-1",
    }
    sold = WealthScenario.model_validate(sold_values)

    keep_result = simulate(keep, 50_001)
    sold_result = simulate(sold, 50_001)

    keep_harvest = keep_result.annual_tax_strategy_actions[0][
        "harvested_long_term_capital_gains_nominal"
    ]["p50"]
    sold_harvest = sold_result.annual_tax_strategy_actions[0][
        "harvested_long_term_capital_gains_nominal"
    ]["p50"]
    assert keep_harvest > 0
    assert sold_harvest == 0
    assert sold_result.annual_tax_audit[0][
        "realized_long_term_capital_gains_nominal"
    ]["p50"] == pytest.approx(150_000)
