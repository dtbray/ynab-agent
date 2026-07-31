from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ynab_agent.planning.models import TaxTreatment, WealthScenario
from ynab_agent.planning.paths import SimulationPaths
from ynab_agent.planning.simulation import simulate
from ynab_agent.planning.taxes import (
    PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET,
    TaxAwarePortfolio,
    TaxBucketKey,
    estimate_tax_state_bytes,
)
from ynab_agent.services.planner_jobs import PlannerExecutionPolicy


def _progressive_strategy_scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "tax strategy",
        "current_age": 60,
        "retirement_age": 60,
        "end_age": 63,
        "starting_portfolio": 300_000,
        "annual_spending": 1,
        "inflation_rate": 0,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 92,
        "tax_buckets": [
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 200_000,
            },
            {
                "tax_treatment": "roth",
                "starting_balance": 50_000,
            },
            {
                "tax_treatment": "taxable",
                "starting_balance": 50_000,
                "taxable_basis": 0,
            },
        ],
        "tax_assumptions": {
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1966,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["taxable", "tax_deferred", "roth"],
            "retirement_surplus_destination": "taxable",
            "strategy": {
                "withdrawal_policy": "ordered",
                "roth_conversion": {
                    "start_age": 60,
                    "end_age": 61,
                    "target_federal_ordinary_bracket_rate": 0.10,
                    "max_annual_conversion_real": 100_000,
                },
                "capital_gain_harvest": {
                    "start_age": 60,
                    "end_age": 61,
                    "target_federal_long_term_capital_gains_rate": 0,
                    "max_annual_gain_real": 100_000,
                },
                "asset_location_preferences": [
                    {
                        "asset_class": "bonds",
                        "preferred_tax_treatments": ["tax_deferred"],
                    }
                ],
            },
        },
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def _effective_withdrawal_scenario(
    *,
    withdrawal_policy: str,
    tax_deferred: float = 100_000,
    roth: float = 100_000,
) -> WealthScenario:
    strategy: dict[str, object] = {"withdrawal_policy": withdrawal_policy}
    if withdrawal_policy == "proportional":
        strategy["proportional_withdrawal_fractions"] = [
            {"tax_treatment": "tax_deferred", "fraction": 0.5},
            {"tax_treatment": "roth", "fraction": 0.5},
        ]
    return WealthScenario.model_validate(
        {
            "name": withdrawal_policy,
            "current_age": 65,
            "retirement_age": 65,
            "end_age": 66,
            "starting_portfolio": tax_deferred + roth,
            "annual_spending": 50_000,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "tax_buckets": [
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": tax_deferred,
                },
                {
                    "tax_treatment": "roth",
                    "starting_balance": roth,
                },
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["tax_deferred", "roth"],
                "retirement_surplus_destination": "roth",
                "strategy": strategy,
            },
        }
    )


def _action_p50(
    result: object,
    *,
    year_index: int,
    field: str,
) -> float:
    actions = getattr(result, "annual_tax_strategy_actions")
    value = actions[year_index][field]
    assert isinstance(value, dict)
    return float(value["p50"])


def _final_federal_taxable_positions(
    result: object,
    *,
    year_index: int,
) -> tuple[float, float]:
    audits = getattr(result, "annual_tax_audit")
    audit = audits[year_index]
    adjusted_gross_income = float(
        audit["modified_adjusted_gross_income_nominal"]["p50"]
    )
    deduction = float(audit["federal_deduction_nominal"]["p50"])
    gains = float(audit["realized_long_term_capital_gains_nominal"]["p50"])
    taxable_income = max(0.0, adjusted_gross_income - deduction)
    taxable_ordinary = max(0.0, taxable_income - gains)
    return taxable_ordinary, taxable_income


def test_bracket_fill_actions_are_bounded_and_replay_exactly() -> None:
    scenario = _progressive_strategy_scenario()

    first = simulate(scenario, 300_000)
    replay = simulate(scenario, 300_000)

    assert first.annual_tax_strategy_actions == replay.annual_tax_strategy_actions
    # Single 2026: $16,100 deduction + $12,400 top of the 10% bracket.
    assert _action_p50(
        first,
        year_index=0,
        field="roth_conversion_nominal",
    ) == pytest.approx(28_500, abs=0.005)
    # Total taxable income fills the 0% capital-gain band to $49,450.
    assert _action_p50(
        first,
        year_index=0,
        field="harvested_long_term_capital_gains_nominal",
    ) == pytest.approx(36_180.70, abs=0.01)
    taxable_ordinary, taxable_income = _final_federal_taxable_positions(
        first,
        year_index=0,
    )
    assert taxable_ordinary <= 12_400.005
    assert taxable_income <= 49_450.005
    assert first.annual_tax_strategy_actions[0]["decision_information"] == (
        "current_year_only"
    )
    strategy_manifest = first.reproducibility["tax_policy"]["strategy"]
    assert strategy_manifest["schema_version"] == 1
    assert strategy_manifest["future_path_information_used"] is False
    assert strategy_manifest["strategy"] == (
        scenario.tax_assumptions.strategy.model_dump(mode="json")
    )


def test_conversion_reserves_bracket_space_for_same_year_withdrawals() -> None:
    scenario = _progressive_strategy_scenario(
        end_age=61,
        starting_portfolio=300_000,
        annual_spending=20_000,
        tax_buckets=[
            {"tax_treatment": "tax_deferred", "starting_balance": 250_000},
            {"tax_treatment": "roth", "starting_balance": 50_000},
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1966,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred", "roth"],
            "retirement_surplus_destination": "roth",
            "strategy": {
                "withdrawal_policy": "ordered",
                "roth_conversion": {
                    "start_age": 60,
                    "end_age": 60,
                    "target_federal_ordinary_bracket_rate": 0.10,
                    "max_annual_conversion_real": 100_000,
                },
            },
        },
    )

    result = simulate(scenario, 300_000)
    replay = simulate(scenario, 300_000)

    conversion = _action_p50(
        result,
        year_index=0,
        field="roth_conversion_nominal",
    )
    taxable_ordinary, _ = _final_federal_taxable_positions(result, year_index=0)
    assert 0 < conversion < 28_500
    assert taxable_ordinary <= 12_400.005
    assert result.annual_tax_strategy_actions == replay.annual_tax_strategy_actions


def test_forced_withdrawal_income_suppresses_conversion_when_ceiling_is_unavoidable() -> None:
    values = _progressive_strategy_scenario(
        end_age=62,
        annual_spending=60_000,
    ).model_dump(mode="json")
    values["tax_buckets"] = [
        {"tax_treatment": "tax_deferred", "starting_balance": 250_000},
        {"tax_treatment": "roth", "starting_balance": 50_000},
    ]
    values["tax_assumptions"]["withdrawal_order"] = ["tax_deferred", "roth"]
    values["tax_assumptions"]["retirement_surplus_destination"] = "roth"
    values["tax_assumptions"]["strategy"]["capital_gain_harvest"] = None
    scenario = WealthScenario.model_validate(values)

    result = simulate(scenario, 300_000)

    taxable_ordinary, _ = _final_federal_taxable_positions(result, year_index=0)
    assert _action_p50(
        result,
        year_index=0,
        field="roth_conversion_nominal",
    ) == 0
    assert taxable_ordinary > 12_400


def test_rmd_alone_can_unavoidably_exceed_conversion_ceiling() -> None:
    values = _progressive_strategy_scenario().model_dump(mode="json")
    values.update(
        {
            "current_age": 75,
            "retirement_age": 75,
            "end_age": 76,
            "starting_portfolio": 1_100_000,
            "annual_spending": 1,
            "tax_buckets": [
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 1_000_000,
                },
                {"tax_treatment": "roth", "starting_balance": 0},
                {
                    "tax_treatment": "taxable",
                    "starting_balance": 100_000,
                    "taxable_basis": 0,
                },
            ],
        }
    )
    values["tax_assumptions"]["progressive"]["taxpayer_birth_year"] = 1951
    values["tax_assumptions"]["apply_required_minimum_distributions"] = True
    values["tax_assumptions"]["withdrawal_order"] = [
        "tax_deferred",
        "taxable",
        "roth",
    ]
    values["tax_assumptions"]["retirement_surplus_destination"] = "roth"
    values["tax_assumptions"]["strategy"]["roth_conversion"] = {
        "start_age": 75,
        "end_age": 75,
        "target_federal_ordinary_bracket_rate": 0.10,
        "max_annual_conversion_real": 100_000,
    }
    values["tax_assumptions"]["strategy"]["capital_gain_harvest"] = {
        "start_age": 75,
        "end_age": 75,
        "target_federal_long_term_capital_gains_rate": 0.15,
        "max_annual_gain_real": 10_000,
    }
    scenario = WealthScenario.model_validate(values)

    result = simulate(scenario, 1_100_000)

    taxable_ordinary, _ = _final_federal_taxable_positions(result, year_index=0)
    assert _action_p50(
        result,
        year_index=0,
        field="roth_conversion_nominal",
    ) == 0
    assert _action_p50(
        result,
        year_index=0,
        field="harvested_long_term_capital_gains_nominal",
    ) == pytest.approx(10_000, abs=0.005)
    withdrawals = result.annual_tax_strategy_actions[0]["withdrawals_nominal"]
    assert withdrawals["tax_deferred"]["p50"] == pytest.approx(
        40_650.41,
        abs=0.01,
    )
    assert taxable_ordinary > 12_400


def test_inactive_future_conversion_rule_does_not_suppress_current_harvest() -> None:
    scenario = _progressive_strategy_scenario(
        current_age=61,
        retirement_age=61,
        end_age=63,
        starting_portfolio=200_000,
        annual_spending=40_000,
        tax_buckets=[
            {"tax_treatment": "tax_deferred", "starting_balance": 100_000},
            {"tax_treatment": "roth", "starting_balance": 0},
            {
                "tax_treatment": "taxable",
                "starting_balance": 100_000,
                "taxable_basis": 0,
            },
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1965,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred", "taxable", "roth"],
            "retirement_surplus_destination": "taxable",
            "strategy": {
                "withdrawal_policy": "ordered",
                "roth_conversion": {
                    "start_age": 62,
                    "end_age": 62,
                    "target_federal_ordinary_bracket_rate": 0.10,
                    "max_annual_conversion_real": 100_000,
                },
                "capital_gain_harvest": {
                    "start_age": 61,
                    "end_age": 61,
                    "target_federal_long_term_capital_gains_rate": 0,
                    "max_annual_gain_real": 100_000,
                },
            },
        },
    )

    result = simulate(scenario, 200_000)

    assert _action_p50(
        result,
        year_index=0,
        field="roth_conversion_nominal",
    ) == 0
    assert _action_p50(
        result,
        year_index=0,
        field="harvested_long_term_capital_gains_nominal",
    ) == pytest.approx(20_408.83, abs=0.01)
    _, taxable_income = _final_federal_taxable_positions(result, year_index=0)
    assert taxable_income <= 49_450.005


def test_gain_harvest_reserves_band_for_same_year_taxable_withdrawals() -> None:
    scenario = _progressive_strategy_scenario(
        end_age=61,
        starting_portfolio=150_000,
        annual_spending=20_000,
        tax_buckets=[
            {
                "tax_treatment": "taxable",
                "starting_balance": 150_000,
                "taxable_basis": 0,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1966,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["taxable"],
            "retirement_surplus_destination": "taxable",
            "strategy": {
                "withdrawal_policy": "ordered",
                "capital_gain_harvest": {
                    "start_age": 60,
                    "end_age": 60,
                    "target_federal_long_term_capital_gains_rate": 0,
                    "max_annual_gain_real": 100_000,
                },
            },
        },
    )

    result = simulate(scenario, 150_000)
    replay = simulate(scenario, 150_000)

    harvest = _action_p50(
        result,
        year_index=0,
        field="harvested_long_term_capital_gains_nominal",
    )
    _, taxable_income = _final_federal_taxable_positions(result, year_index=0)
    assert 0 < harvest < 65_550
    assert taxable_income <= 49_450.005
    assert result.annual_tax_strategy_actions == replay.annual_tax_strategy_actions


def test_gain_harvest_resets_basis_and_funds_current_tax() -> None:
    scenario = _progressive_strategy_scenario(
        current_age=65,
        retirement_age=65,
        end_age=66,
        starting_portfolio=100_000,
        tax_buckets=[
            {
                "tax_treatment": "taxable",
                "starting_balance": 100_000,
                "taxable_basis": 50_000,
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
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["taxable"],
            "retirement_surplus_destination": "taxable",
            "strategy": {
                "withdrawal_policy": "ordered",
                "capital_gain_harvest": {
                    "start_age": 65,
                    "end_age": 65,
                    "target_federal_long_term_capital_gains_rate": 0.15,
                    "max_annual_gain_real": 50_000,
                },
            },
        },
    )
    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=100_000,
    )
    assert scenario.tax_assumptions is not None

    unmet = portfolio.fund_retirement_spending(
        slice(0, scenario.trials),
        age=65,
        tax_year=2026,
        spending=0,
        ordinary_income=0,
        social_security_income=0,
        tax_free_income=0,
        opening_tax_deferred=portfolio.opening_tax_deferred(
            slice(0, scenario.trials)
        ),
        inflation_factor=1,
        assumptions=scenario.tax_assumptions,
    )

    assert np.max(unmet) == 0
    assert portfolio.taxable_basis is not None
    assert portfolio.annual_harvested_long_term_capital_gains_nominal[0] == (
        pytest.approx(50_000)
    )
    # The sale/rebuy resets basis; the later tax-funding sale reduces basis and
    # balance together, so no embedded gain remains.
    taxable_key = next(iter(portfolio.taxable_basis))
    assert portfolio.taxable_basis[taxable_key][0] == pytest.approx(
        portfolio.balances[taxable_key][0]
    )
    assert portfolio.cumulative_tax_real[0] > 0


def test_ordered_and_proportional_withdrawals_are_distinct_and_fallback() -> None:
    ordered = simulate(
        _effective_withdrawal_scenario(withdrawal_policy="ordered"),
        200_000,
    )
    proportional = simulate(
        _effective_withdrawal_scenario(withdrawal_policy="proportional"),
        200_000,
    )
    fallback = simulate(
        _effective_withdrawal_scenario(
            withdrawal_policy="proportional",
            tax_deferred=10_000,
            roth=90_000,
        ),
        100_000,
    )

    ordered_withdrawals = ordered.annual_tax_strategy_actions[0]["withdrawals_nominal"]
    proportional_withdrawals = proportional.annual_tax_strategy_actions[0][
        "withdrawals_nominal"
    ]
    fallback_withdrawals = fallback.annual_tax_strategy_actions[0]["withdrawals_nominal"]
    assert ordered_withdrawals["tax_deferred"]["p50"] == pytest.approx(62_500)
    assert ordered_withdrawals["roth"]["p50"] == 0
    assert proportional_withdrawals["tax_deferred"]["p50"] == pytest.approx(31_250)
    assert proportional_withdrawals["roth"]["p50"] == pytest.approx(25_000)
    assert fallback_withdrawals["tax_deferred"]["p50"] == pytest.approx(10_000)
    assert fallback_withdrawals["roth"]["p50"] == pytest.approx(42_000)
    assert fallback.success_rate == 1


def test_progressive_proportional_withdrawals_fund_each_net_share_without_discard() -> None:
    values = _effective_withdrawal_scenario(
        withdrawal_policy="proportional"
    ).model_dump(mode="json")
    values["tax_assumptions"].update(
        {
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
            },
        }
    )
    values["household"] = {
        "plan_start_date": "2026-01-02",
        "people": [
            {
                "id": "primary",
                "name": "Primary",
                "birth_date": "1961-01-02",
                "retirement_age_months": 65 * 12,
            }
        ],
    }
    for bucket in values["tax_buckets"]:
        bucket["owner_person_id"] = "primary"
    scenario = WealthScenario.model_validate(values)

    result = simulate(scenario, 200_000)

    withdrawals = result.annual_tax_strategy_actions[0][
        "withdrawals_nominal"
    ]
    deferred = float(withdrawals["tax_deferred"]["p50"])
    roth = float(withdrawals["roth"]["p50"])
    tax = float(
        result.annual_tax_audit[0]["total_income_tax_nominal"]["p50"]
    )
    assert deferred < 100_000
    assert deferred - tax == pytest.approx(25_000, abs=0.005)
    assert roth == pytest.approx(25_000, abs=0.005)
    assert result.ending_balance_real["p50"] == pytest.approx(
        200_000 - deferred - roth,
        abs=0.005,
    )

    owner_flows = {
        row["tax_treatment"]: row
        for row in result.annual_tax_audit[0]["owner_flows"]
    }
    assert {
        row["owner_person_id"]
        for row in result.annual_tax_audit[0]["owner_flows"]
    } == {"primary"}
    assert owner_flows["tax_deferred"]["withdrawal_nominal"]["p50"] == (
        pytest.approx(deferred)
    )
    assert owner_flows["roth"]["withdrawal_nominal"]["p50"] == (
        pytest.approx(roth)
    )


def test_current_year_action_does_not_use_divergent_future_paths() -> None:
    scenario = _progressive_strategy_scenario(end_age=62)
    current = np.ones((2, scenario.trials), dtype=float)
    adverse = current.copy()
    favorable = current.copy()
    adverse[1] = 0.5
    favorable[1] = 2.0
    inflation = np.ones(3, dtype=float)

    adverse_result = simulate(
        scenario,
        300_000,
        prepared_paths=SimulationPaths(adverse, inflation),
    )
    favorable_result = simulate(
        scenario,
        300_000,
        prepared_paths=SimulationPaths(favorable, inflation),
    )

    assert adverse_result.annual_tax_strategy_actions[0] == (
        favorable_result.annual_tax_strategy_actions[0]
    )
    assert adverse_result.ending_balance_real != favorable_result.ending_balance_real


def test_irmaa_exposure_uses_the_two_year_lookback() -> None:
    scenario = _progressive_strategy_scenario(
        current_age=65,
        retirement_age=65,
        end_age=66,
        starting_portfolio=10_000,
        tax_buckets=[{"tax_treatment": "cash", "starting_balance": 10_000}],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
                "irmaa_lookback_magi": [{"tax_year": 2024, "magi": 150_000}],
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["cash"],
            "retirement_surplus_destination": "cash",
        },
    )

    result = simulate(scenario, 10_000)

    expected = 12 * (202.90 + 37.50)
    assert result.annual_tax_audit[0]["irmaa_annual_surcharge_nominal"]["p50"] == (
        pytest.approx(expected)
    )
    assert result.lifetime_irmaa_surcharge_real == pytest.approx(
        {"p10": expected, "p50": expected, "p90": expected}
    )
    assert result.irmaa_exposure_probability == 1


def test_strategy_validation_rejects_unreplayable_or_unsupported_inputs() -> None:
    with pytest.raises(ValidationError, match="fractions must cover every tax bucket"):
        values = _effective_withdrawal_scenario(
            withdrawal_policy="proportional"
        ).model_dump(mode="json")
        values["tax_assumptions"]["strategy"]["proportional_withdrawal_fractions"] = [
            {"tax_treatment": "tax_deferred", "fraction": 1}
        ]
        WealthScenario.model_validate(values)

    values = _effective_withdrawal_scenario(
        withdrawal_policy="ordered"
    ).model_dump(mode="json")
    values["tax_assumptions"]["strategy"]["roth_conversion"] = {
        "start_age": 65,
        "end_age": 65,
        "target_federal_ordinary_bracket_rate": 0.12,
        "max_annual_conversion_real": 10_000,
    }
    with pytest.raises(ValidationError, match="require progressive tax"):
        WealthScenario.model_validate(values)


def test_strategy_resource_estimates_include_bounded_decision_searches() -> None:
    strategy = _progressive_strategy_scenario()
    baseline_values = strategy.model_dump(mode="json")
    baseline_values["tax_assumptions"]["strategy"] = None
    baseline = WealthScenario.model_validate(baseline_values)

    assert estimate_tax_state_bytes(strategy) > estimate_tax_state_bytes(baseline)
    projection_evaluations = (
        len(strategy.tax_assumptions.withdrawal_order)
        * PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET
        + 4
    )
    expected_compute_delta = (
        (strategy.end_age - strategy.current_age)
        * strategy.trials
        * 97
        * projection_evaluations
    )
    assert (
        PlannerExecutionPolicy._compute_units(strategy)
        - PlannerExecutionPolicy._compute_units(baseline)
        == expected_compute_delta
    )

    bucket_count = len(strategy.tax_buckets)
    taxable_count = sum(
        bucket.tax_treatment.value == "taxable"
        for bucket in strategy.tax_buckets
    )
    baseline_arrays = (
        (3 * bucket_count)
        + 17
        + taxable_count
        + 15
        + (3 * bucket_count)
        + (2 * taxable_count)
    )
    strategy_arrays = baseline_arrays + 14
    assert estimate_tax_state_bytes(baseline) == (
        baseline.trials * baseline_arrays * 8
    )
    assert estimate_tax_state_bytes(strategy) == (
        strategy.trials * strategy_arrays * 8
    )


def test_strategy_compute_counts_owner_bucket_visits_and_social_security_matrix() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "owner admission",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 61,
            "starting_portfolio": 200_000,
            "annual_spending": 1,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "household": {
                "plan_start_date": "2026-01-02",
                "people": [
                    {
                        "id": "first",
                        "name": "First",
                        "birth_date": "1966-01-02",
                        "retirement_age_months": 60 * 12,
                    },
                    {
                        "id": "second",
                        "name": "Second",
                        "birth_date": "1966-01-02",
                        "retirement_age_months": 60 * 12,
                    },
                ],
            },
            "tax_buckets": [
                {
                    "tax_treatment": treatment,
                    "owner_person_id": owner,
                    "starting_balance": (
                        100_000 if treatment == "tax_deferred" else 0
                    ),
                }
                for owner in ("first", "second")
                for treatment in ("tax_deferred", "roth")
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "married_filing_jointly",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1966,
                    "spouse_birth_year": 1966,
                    "aca_household_size": 2,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["tax_deferred", "roth"],
                "retirement_surplus_destination": "roth",
                "strategy": {
                    "withdrawal_policy": "ordered",
                    "roth_conversion": {
                        "start_age": 60,
                        "end_age": 60,
                        "target_federal_ordinary_bracket_rate": 0.10,
                        "max_annual_conversion_real": 10_000,
                    },
                },
            },
        }
    )
    baseline_values = scenario.model_dump(mode="json")
    baseline_values["tax_assumptions"]["strategy"] = None
    baseline = WealthScenario.model_validate(baseline_values)
    trial_years = (
        (scenario.end_age - scenario.current_age) * scenario.trials
    )
    action_projections = 48
    owner_bucket_visits = 4
    projection_compute = (
        owner_bucket_visits * PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET
        + 4
    )

    assert (
        PlannerExecutionPolicy._compute_units(scenario)
        - PlannerExecutionPolicy._compute_units(baseline)
        == trial_years * action_projections * projection_compute
    )

    proportional_values = scenario.model_dump(mode="json")
    proportional_values["tax_assumptions"]["strategy"].update(
        {
            "withdrawal_policy": "proportional",
            "proportional_withdrawal_fractions": [
                {"tax_treatment": "tax_deferred", "fraction": 0.5},
                {"tax_treatment": "roth", "fraction": 0.5},
            ],
        }
    )
    proportional = WealthScenario.model_validate(proportional_values)
    extra_execution_compute = owner_bucket_visits * (
        1 + PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET
    )
    extra_projection_compute = (
        action_projections
        * owner_bucket_visits
        * PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET
    )
    assert (
        PlannerExecutionPolicy._compute_units(proportional)
        - PlannerExecutionPolicy._compute_units(scenario)
        == trial_years
        * (extra_execution_compute + extra_projection_compute)
    )

    policy = PlannerExecutionPolicy(
        maximum_working_bytes=1,
        in_memory_path_bytes=0,
        maximum_temporary_bytes=1,
        batch_size=1,
        maximum_compute_units=100_000_000,
    )
    assert policy.social_security_optimization_compute_units(
        proportional,
        strategy_count=4,
    ) == 4 * PlannerExecutionPolicy._compute_units(proportional)


def test_roth_conversion_preserves_owner_and_aggregates_public_schedule() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "owner conversion",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 61,
            "starting_portfolio": 200_000,
            "annual_spending": 1,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "household": {
                "plan_start_date": "2026-01-02",
                "people": [
                    {
                        "id": "first",
                        "name": "First",
                        "birth_date": "1966-01-02",
                        "retirement_age_months": 60 * 12,
                    },
                    {
                        "id": "second",
                        "name": "Second",
                        "birth_date": "1966-01-02",
                        "retirement_age_months": 60 * 12,
                    },
                ],
            },
            "tax_buckets": [
                {
                    "tax_treatment": "tax_deferred",
                    "owner_person_id": "first",
                    "starting_balance": 100_000,
                },
                {
                    "tax_treatment": "roth",
                    "owner_person_id": "first",
                    "starting_balance": 0,
                },
                {
                    "tax_treatment": "tax_deferred",
                    "owner_person_id": "second",
                    "starting_balance": 100_000,
                },
                {
                    "tax_treatment": "roth",
                    "owner_person_id": "second",
                    "starting_balance": 0,
                },
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "married_filing_jointly",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1966,
                    "spouse_birth_year": 1966,
                    "aca_household_size": 2,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["tax_deferred", "roth"],
                "retirement_surplus_destination": "roth",
                "strategy": {
                    "withdrawal_policy": "ordered",
                    "roth_conversion": {
                        "start_age": 60,
                        "end_age": 60,
                        "target_federal_ordinary_bracket_rate": 0.10,
                        "max_annual_conversion_real": 10_000,
                    },
                },
            },
        }
    )
    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=200_000,
    )
    assumptions = scenario.tax_assumptions
    assert assumptions is not None

    portfolio.fund_retirement_spending(
        slice(0, scenario.trials),
        age=60,
        tax_year=2026,
        spending=1,
        ordinary_income=0,
        social_security_income=0,
        tax_free_income=0,
        opening_tax_deferred=portfolio.opening_tax_deferred(
            slice(0, scenario.trials)
        ),
        inflation_factor=1,
        assumptions=assumptions,
        joint_filing=np.ones(scenario.trials, dtype=bool),
        owner_birth_years={"first": 1966, "second": 1966},
    )

    first_roth = TaxBucketKey(TaxTreatment.ROTH, "first")
    second_roth = TaxBucketKey(TaxTreatment.ROTH, "second")
    assert portfolio.annual_roth_conversion_nominal[0] == pytest.approx(
        10_000,
        abs=0.005,
    )
    assert portfolio.balances[first_roth][0] == pytest.approx(
        10_000,
        abs=0.005,
    )
    assert portfolio.balances[second_roth][0] == 0

    result = simulate(scenario, 200_000)
    assert result.annual_tax_strategy_actions[0][
        "roth_conversion_nominal"
    ]["p50"] == pytest.approx(10_000, abs=0.005)
    owner_flows = result.annual_tax_audit[0]["owner_flows"]
    assert len(owner_flows) == 4

    mismatched_values = scenario.model_dump(mode="json")
    mismatched_values["starting_portfolio"] = 100_000
    mismatched_values["tax_buckets"] = [
        mismatched_values["tax_buckets"][0],
        mismatched_values["tax_buckets"][3],
    ]
    mismatched = WealthScenario.model_validate(mismatched_values)
    mismatched_result = simulate(mismatched, 100_000)
    assert mismatched_result.annual_tax_strategy_actions[0][
        "roth_conversion_nominal"
    ]["p50"] == 0
