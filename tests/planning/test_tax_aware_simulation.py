from __future__ import annotations

import pytest
from pydantic import ValidationError

from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.simulation import simulate
from ynab_agent.planning.solver import SolveVariable, solve_scenario
from ynab_agent.services.planner_jobs import PlannerSimulationResult


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "tax-aware",
        "current_age": 65,
        "retirement_age": 65,
        "end_age": 66,
        "starting_portfolio": 100_000,
        "annual_contribution": 0,
        "annual_spending": 50_000,
        "tax_buckets": [
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 50_000,
            },
            {
                "tax_treatment": "roth",
                "starting_balance": 50_000,
            },
        ],
        "tax_assumptions": {
            "ordinary_income_tax_rate": 0.20,
            "long_term_capital_gains_tax_rate": 0.15,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred", "roth"],
            "retirement_surplus_destination": "roth",
        },
        "inflation_rate": 0,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 7,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def test_tax_deferred_withdrawal_pays_tax_before_roth() -> None:
    result = simulate(_scenario(), 100_000)

    assert result.success_rate == 1
    assert result.ending_balance_real["p50"] == pytest.approx(40_000)
    assert result.lifetime_tax_real == pytest.approx(
        {"p10": 10_000, "p50": 10_000, "p90": 10_000}
    )
    assert result.assumptions["tax_model"] == "account_aware_effective_rates_v1"
    assert result.assumptions["tax_bucket_count"] == 2
    persisted = PlannerSimulationResult.model_validate(result.as_dict())
    assert persisted.assumptions["required_minimum_distributions"] is False


def test_withdrawal_order_can_preserve_tax_deferred_balance() -> None:
    result = simulate(
        _scenario(
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["roth", "tax_deferred"],
                "retirement_surplus_destination": "roth",
            }
        ),
        100_000,
    )

    assert result.ending_balance_real["p50"] == pytest.approx(50_000)
    assert result.lifetime_tax_real == pytest.approx(
        {"p10": 0, "p50": 0, "p90": 0}
    )


def test_taxable_withdrawal_taxes_only_embedded_gain() -> None:
    result = simulate(
        _scenario(
            starting_portfolio=100_000,
            tax_buckets=[
                {
                    "tax_treatment": "taxable",
                    "starting_balance": 100_000,
                    "taxable_basis": 60_000,
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.25,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["taxable"],
                "retirement_surplus_destination": "taxable",
            },
        ),
        100_000,
    )

    assert result.ending_balance_real["p50"] == pytest.approx(44_444.444444)
    assert result.lifetime_tax_real is not None
    assert result.lifetime_tax_real["p50"] == pytest.approx(5_555.555556)


def test_taxable_bucket_allows_unrealized_loss_without_a_tax_credit() -> None:
    result = simulate(
        _scenario(
            tax_buckets=[
                {
                    "tax_treatment": "taxable",
                    "starting_balance": 100_000,
                    "taxable_basis": 120_000,
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.25,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["taxable"],
                "retirement_surplus_destination": "taxable",
            },
        ),
        100_000,
    )

    assert result.ending_balance_real["p50"] == pytest.approx(50_000)
    assert result.lifetime_tax_real is not None
    assert result.lifetime_tax_real["p50"] == 0


def test_social_security_tax_creates_a_cash_withdrawal() -> None:
    result = simulate(
        _scenario(
            starting_portfolio=10_000,
            annual_spending=40_000,
            tax_buckets=[
                {
                    "tax_treatment": "cash",
                    "starting_balance": 10_000,
                }
            ],
            income_streams=[
                {
                    "name": "SSA estimate",
                    "start_age": 65,
                    "annual_amount": 42_000,
                    "tax_treatment": "social_security",
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "social_security_taxable_fraction": 0.85,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["cash"],
                "retirement_surplus_destination": "cash",
            },
        ),
        10_000,
    )

    assert result.ending_balance_real["p50"] == pytest.approx(4_860)
    assert result.lifetime_tax_real is not None
    assert result.lifetime_tax_real["p50"] == pytest.approx(7_140)


def test_qualified_hsa_withdrawal_is_tax_free() -> None:
    result = simulate(
        _scenario(
            tax_buckets=[
                {
                    "tax_treatment": "hsa",
                    "starting_balance": 100_000,
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "qualified_hsa_withdrawal_fraction": 1,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["hsa"],
                "retirement_surplus_destination": "hsa",
            },
        ),
        100_000,
    )

    assert result.ending_balance_real["p50"] == pytest.approx(50_000)
    assert result.lifetime_tax_real is not None
    assert result.lifetime_tax_real["p50"] == 0


def test_qualified_hsa_fraction_applies_to_effective_rate_estate_value() -> None:
    result = simulate(
        _scenario(
            annual_spending=1,
            tax_buckets=[
                {
                    "tax_treatment": "hsa",
                    "starting_balance": 100_000,
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "qualified_hsa_withdrawal_fraction": 1,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["hsa"],
                "retirement_surplus_destination": "hsa",
            },
        ),
        100_000,
    )

    assert result.after_tax_ending_balance_real == pytest.approx(
        {"p10": 99_999, "p50": 99_999, "p90": 99_999}
    )


def test_contributions_route_to_default_and_explicit_tax_destinations() -> None:
    result = simulate(
        _scenario(
            current_age=64,
            retirement_age=65,
            end_age=66,
            annual_contribution=7_500,
            annual_spending=1,
            tax_buckets=[
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 70_000,
                    "contribution_fraction": 1,
                },
                {
                    "tax_treatment": "roth",
                    "starting_balance": 20_000,
                },
                {
                    "tax_treatment": "taxable",
                    "starting_balance": 10_000,
                    "taxable_basis": 10_000,
                },
            ],
            cash_flow_streams=[
                {
                    "name": "Freed mortgage payment",
                    "flow_type": "contribution",
                    "start_age": 64,
                    "annual_amount": 10_000,
                    "destination_tax_treatment": "taxable",
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["taxable", "tax_deferred", "roth"],
                "retirement_surplus_destination": "taxable",
            },
        ),
        100_000,
    )

    assert result.retirement_balance_real["p50"] == pytest.approx(117_500)
    assert result.ending_balance_real["p50"] == pytest.approx(117_499)


def test_required_minimum_distribution_reinvests_after_tax_surplus() -> None:
    result = simulate(
        _scenario(
            current_age=75,
            retirement_age=75,
            end_age=76,
            starting_portfolio=24_600,
            annual_spending=100,
            tax_buckets=[
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 24_600,
                },
                {
                    "tax_treatment": "taxable",
                    "starting_balance": 0,
                    "taxable_basis": 0,
                },
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "rmd_start_age": 75,
                "withdrawal_order": ["taxable", "tax_deferred"],
                "retirement_surplus_destination": "taxable",
            },
        ),
        24_600,
    )

    assert result.ending_balance_real["p50"] == pytest.approx(24_300)
    assert result.lifetime_tax_real is not None
    assert result.lifetime_tax_real["p50"] == pytest.approx(200)


def test_tax_model_requires_balanced_explicit_inputs() -> None:
    with pytest.raises(
        ValidationError,
        match="tax bucket balances must equal starting_portfolio",
    ):
        _scenario(
            tax_buckets=[
                {
                    "tax_treatment": "roth",
                    "starting_balance": 90_000,
                }
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "withdrawal_order": ["roth"],
                "retirement_surplus_destination": "roth",
            },
        )

    with pytest.raises(
        ValidationError,
        match="tax-aware scenarios require tax_treatment",
    ):
        _scenario(
            income_streams=[
                {
                    "name": "Unclassified pension",
                    "start_age": 65,
                    "annual_amount": 10_000,
                }
            ]
        )


def test_blended_model_remains_backward_compatible() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "legacy",
            "current_age": 65,
            "retirement_age": 65,
            "end_age": 66,
            "starting_portfolio": 100_000,
            "annual_spending": 50_000,
            "withdrawal_tax_rate": 0.20,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
        }
    )

    result = simulate(scenario, 100_000)

    assert result.ending_balance_real["p50"] == pytest.approx(37_500)
    assert result.lifetime_tax_real is None
    assert result.assumptions["tax_model"] == "blended_withdrawal_rate"


def test_zero_rate_tax_buckets_preserve_legacy_total_balance_math() -> None:
    common: dict[str, object] = {
        "name": "equivalence",
        "current_age": 60,
        "retirement_age": 65,
        "end_age": 90,
        "starting_portfolio": 100_000,
        "annual_contribution": 10_000,
        "annual_spending": 30_000,
        "income_streams": [
            {
                "name": "Tax-free income",
                "start_age": 70,
                "annual_amount": 5_000,
                "tax_treatment": "tax_free",
            }
        ],
        "cash_flow_streams": [
            {
                "name": "Later savings",
                "flow_type": "contribution",
                "start_age": 62,
                "annual_amount": 2_000,
            }
        ],
        "inflation_rate": 0.025,
        "return_mean": 0.06,
        "return_volatility": 0.12,
        "annual_fee_rate": 0.002,
        "trials": 1_000,
        "seed": 42,
    }
    legacy = WealthScenario.model_validate(common)
    tax_aware = WealthScenario.model_validate(
        {
            **common,
            "tax_buckets": [
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 60_000,
                    "contribution_fraction": 1,
                },
                {
                    "tax_treatment": "roth",
                    "starting_balance": 30_000,
                },
                {
                    "tax_treatment": "taxable",
                    "starting_balance": 10_000,
                    "taxable_basis": 10_000,
                },
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "social_security_taxable_fraction": 0,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["taxable", "tax_deferred", "roth"],
                "retirement_surplus_destination": "taxable",
            },
        }
    )

    legacy_result = simulate(legacy, 100_000)
    tax_result = simulate(tax_aware, 100_000)

    assert tax_result.success_rate == legacy_result.success_rate
    assert tax_result.retirement_balance_real == pytest.approx(
        legacy_result.retirement_balance_real
    )
    assert tax_result.ending_balance_real == pytest.approx(
        legacy_result.ending_balance_real
    )


def test_starting_portfolio_solver_scales_tax_buckets_and_basis() -> None:
    result = solve_scenario(
        _scenario(),
        100_000,
        variable=SolveVariable.STARTING_PORTFOLIO,
        lower=50_000,
        upper=100_000,
        target_success_rate=0.90,
        resolution=100,
    )

    assert result.value == pytest.approx(55_600, abs=100)
    assert result.simulation.starting_portfolio == result.value
    assert result.simulation.success_rate == 1
