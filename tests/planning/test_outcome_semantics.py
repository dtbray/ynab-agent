from __future__ import annotations

import pytest
from pydantic import ValidationError

from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.outcomes import GoalKind
from ynab_agent.planning.simulation import simulate


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "outcome semantics",
        "current_age": 60,
        "retirement_age": 60,
        "end_age": 64,
        "starting_portfolio": 50_000,
        "annual_spending": 20_000,
        "inflation_rate": 0,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 7,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def test_shortfall_severity_preserves_success_rate_semantics() -> None:
    result = simulate(_scenario(), 50_000)

    assert result.success_rate == 0
    assert result.depleted_trials == 100
    assert result.median_depletion_age == 62
    assert result.funded_spending_ratio == pytest.approx({"p10": 0.625, "p50": 0.625, "p90": 0.625})
    assert result.funded_spending_real == pytest.approx(
        {"p10": 50_000, "p50": 50_000, "p90": 50_000}
    )
    assert result.cumulative_shortfall_real == pytest.approx(
        {"p10": 30_000, "p50": 30_000, "p90": 30_000}
    )
    assert result.failure_duration_years == pytest.approx({"p10": 2, "p50": 2, "p90": 2})
    assert result.longest_failure_streak_years == pytest.approx({"p10": 2, "p50": 2, "p90": 2})
    assert result.recovered_trials == 0
    assert result.recovery_probability == 0
    assert result.goal_outcomes[0].kind is GoalKind.PLANNED_SPENDING
    assert result.goal_outcomes[0].attainment_probability == 0


def test_failure_can_recover_and_still_attain_a_lower_spending_tier() -> None:
    result = simulate(
        _scenario(
            end_age=63,
            starting_portfolio=20_000,
            annual_spending=10_000,
            cash_flow_streams=[
                {
                    "name": "One-time retirement expense",
                    "flow_type": "expense",
                    "start_age": 60,
                    "end_age": 60,
                    "annual_amount": 20_000,
                }
            ],
            income_streams=[
                {
                    "name": "Pension",
                    "start_age": 61,
                    "annual_amount": 10_000,
                }
            ],
            spending_tiers=[
                {
                    "name": "essential",
                    "annual_amount": 8_000,
                }
            ],
        ),
        20_000,
    )

    assert result.success_rate == 0
    assert result.funded_spending_ratio["p50"] == pytest.approx(0.8)
    assert result.cumulative_shortfall_real["p50"] == 10_000
    assert result.failure_duration_years is not None
    assert result.failure_duration_years["p50"] == 1
    assert result.longest_failure_streak_years is not None
    assert result.longest_failure_streak_years["p50"] == 1
    assert result.recovered_trials == 100
    assert result.recovery_probability == 1
    essential = result.goal_outcomes[1]
    assert essential.kind is GoalKind.SPENDING_TIER
    assert essential.attained_trials == 100
    assert essential.attainment_probability == 1


def test_tax_adjusted_legacy_target_uses_account_character() -> None:
    result = simulate(
        _scenario(
            current_age=65,
            retirement_age=65,
            end_age=66,
            starting_portfolio=100_000,
            annual_spending=1,
            tax_buckets=[
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 50_000,
                },
                {
                    "tax_treatment": "roth",
                    "starting_balance": 50_000,
                },
            ],
            tax_assumptions={
                "ordinary_income_tax_rate": 0.20,
                "long_term_capital_gains_tax_rate": 0.15,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["roth", "tax_deferred"],
                "retirement_surplus_destination": "roth",
            },
            legacy_target_real=89_000,
        ),
        100_000,
    )

    assert result.ending_balance_real["p50"] == 99_999
    assert result.after_tax_ending_balance_real == pytest.approx(
        {"p10": 89_999, "p50": 89_999, "p90": 89_999}
    )
    assert result.legacy_target_probability == 1
    legacy = result.goal_outcomes[-1]
    assert legacy.kind is GoalKind.LEGACY_TARGET
    assert legacy.attained_trials == 100
    assert legacy.evaluation_basis == "tax_adjusted_real_ending_estate"


def test_legacy_target_is_explicitly_unevaluated_without_tax_inputs() -> None:
    result = simulate(
        _scenario(legacy_target_real=10_000),
        50_000,
    )

    assert result.after_tax_ending_balance_real is None
    assert result.legacy_target_probability is None
    legacy = result.goal_outcomes[-1]
    assert legacy.attainment_probability is None
    assert legacy.attained_trials is None
    assert legacy.evaluation_basis == "explicit_tax_inputs_required"


def test_spending_tiers_are_bounded_unique_and_within_the_simulated_plan() -> None:
    with pytest.raises(ValidationError, match="cannot exceed annual_spending"):
        _scenario(spending_tiers=[{"name": "aspirational", "annual_amount": 25_000}])

    with pytest.raises(ValidationError, match="names must be unique"):
        _scenario(
            spending_tiers=[
                {"name": "essential", "annual_amount": 10_000},
                {"name": "Essential", "annual_amount": 15_000},
            ]
        )

    with pytest.raises(ValidationError, match="reserved goal names"):
        _scenario(
            spending_tiers=[
                {"name": "legacy_target", "annual_amount": 10_000},
            ]
        )
