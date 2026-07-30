from __future__ import annotations

import pytest

from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.simulation import simulate
from ynab_agent.planning.spending_guardrails import (
    AnnualSpendingContext,
    FixedRealSpendingPolicy,
    FloorCeilingSpendingPolicy,
    RetirementSpendingPlan,
    SpendingTierAmounts,
    WithdrawalRateGuardrailPolicy,
    evaluate_guardrails,
    spending_plan_manifest,
)


def _plan(
    *,
    policy: object,
) -> RetirementSpendingPlan:
    return RetirementSpendingPlan.model_validate(
        {
            "baseline": {
                "essential": 30_000,
                "lifestyle": 20_000,
                "discretionary": 10_000,
                "one_time": 0,
            },
            "essential_floor": 24_000,
            "policy": policy,
        }
    )


def test_withdrawal_guardrails_reduce_then_restore_and_audit_every_change() -> None:
    plan = _plan(
        policy=WithdrawalRateGuardrailPolicy(
            lower_withdrawal_rate=0.03,
            upper_withdrawal_rate=0.05,
            reduction_rate=0.10,
            restoration_rate=0.10,
        )
    )

    summary = evaluate_guardrails(
        plan,
        [
            AnnualSpendingContext(
                age=60,
                opening_portfolio_real=1_000_000,
            ),
            AnnualSpendingContext(
                age=61,
                opening_portfolio_real=2_000_000,
            ),
            AnnualSpendingContext(
                age=62,
                opening_portfolio_real=2_000_000,
            ),
        ],
    )

    assert [row.action for row in summary.annual_path] == [
        "reduced",
        "restored",
        "restored",
    ]
    assert summary.annual_path[0].applied_spending.total == 54_000
    assert summary.annual_path[0].spending_delta.total == -6_000
    assert summary.annual_path[-1].applied_spending.total == 60_000
    assert summary.reduction_years == 1
    assert summary.restoration_years == 2
    assert summary.minimum_essential_spending_real >= plan.essential_floor


def test_fixed_real_and_floor_ceiling_are_replaceable_policy_objects() -> None:
    fixed = _plan(policy=FixedRealSpendingPolicy())
    bounded = _plan(
        policy=FloorCeilingSpendingPolicy(
            target_withdrawal_rate=0.04,
            maximum_annual_reduction=0.05,
            maximum_annual_restoration=0.05,
        )
    )
    context = [
        AnnualSpendingContext(
            age=60,
            opening_portfolio_real=500_000,
        )
    ]

    fixed_summary = evaluate_guardrails(fixed, context)
    bounded_summary = evaluate_guardrails(bounded, context)

    assert fixed_summary.annual_path[0].applied_spending.total == 60_000
    assert bounded_summary.annual_path[0].applied_spending.total == 57_000
    assert fixed.baseline == bounded.baseline


def test_manifest_changes_with_material_policy_but_not_runtime_state() -> None:
    fixed = _plan(policy=FixedRealSpendingPolicy())
    guardrails = _plan(
        policy=WithdrawalRateGuardrailPolicy(
            lower_withdrawal_rate=0.03,
            upper_withdrawal_rate=0.05,
        )
    )

    fixed_manifest = spending_plan_manifest(fixed)
    guardrail_manifest = spending_plan_manifest(guardrails)

    assert fixed_manifest.plan_sha256 != guardrail_manifest.plan_sha256
    assert guardrail_manifest.policy.kind == "withdrawal_rate_guardrails"
    assert guardrail_manifest.essential_floor == 24_000


def test_guardrails_change_simulated_spending_and_outcomes() -> None:
    plan = _plan(
        policy=WithdrawalRateGuardrailPolicy(
            lower_withdrawal_rate=0.10,
            upper_withdrawal_rate=0.20,
            reduction_rate=0.25,
            restoration_rate=0.25,
        )
    )
    scenario = WealthScenario(
        name="guardrail integration",
        current_age=60,
        retirement_age=60,
        end_age=64,
        starting_portfolio=200_000,
        annual_spending=60_000,
        retirement_spending_plan=plan,
        inflation_rate=0,
        return_mean=0,
        return_volatility=0,
        annual_fee_rate=0,
        trials=100,
        seed=7,
    )
    fixed_scenario = scenario.model_copy(
        update={
            "retirement_spending_plan": plan.model_copy(
                update={"policy": FixedRealSpendingPolicy()}
            )
        }
    )

    guarded = simulate(scenario, 200_000)
    fixed = simulate(fixed_scenario, 200_000)

    assert guarded.ending_balance_real["p50"] > fixed.ending_balance_real["p50"]
    assert guarded.guardrail_metrics is not None
    assert guarded.guardrail_metrics["reduction_events"] == 400
    assert [row["reduced_trials"] for row in guarded.annual_spending_real] == [
        100,
        100,
        100,
        100,
    ]
    assert guarded.annual_spending_real[-1]["essential_floor"] == 24_000
    assert guarded.reproducibility["retirement_spending"] is not None
    assert guarded.funded_spending_ratio["p50"] == 1
    assert guarded.cumulative_shortfall_real["p50"] == 0
    assert guarded.goal_outcomes[0].evaluation_basis == (
        "all_policy_adjusted_retirement_spending_funded"
    )
    assert fixed.guardrail_metrics is not None
    assert fixed.guardrail_metrics["reduction_events"] == 0
    assert fixed.funded_spending_ratio["p50"] == pytest.approx(5 / 6)
    assert fixed.cumulative_shortfall_real["p50"] == 40_000


def test_guardrails_also_drive_account_aware_withdrawals() -> None:
    plan = _plan(
        policy=WithdrawalRateGuardrailPolicy(
            lower_withdrawal_rate=0.10,
            upper_withdrawal_rate=0.20,
            reduction_rate=0.25,
            restoration_rate=0.25,
        )
    )
    scenario = WealthScenario(
        name="tax-aware guardrails",
        current_age=60,
        retirement_age=60,
        end_age=62,
        starting_portfolio=200_000,
        annual_spending=60_000,
        retirement_spending_plan=plan,
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 100_000,
            },
            {
                "tax_treatment": "roth",
                "starting_balance": 100_000,
            },
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred", "roth"],
            "retirement_surplus_destination": "roth",
        },
        inflation_rate=0,
        return_mean=0,
        return_volatility=0,
        annual_fee_rate=0,
        trials=100,
        seed=7,
    )

    result = simulate(scenario, 200_000)

    assert result.guardrail_metrics is not None
    assert result.guardrail_metrics["reduction_events"] == 200
    assert result.annual_spending_real[0]["total"]["p50"] == 45_000
    assert result.ending_balance_real["p50"] == 121_250


def test_essential_floor_cannot_exceed_baseline() -> None:
    with pytest.raises(ValueError, match="essential floor"):
        RetirementSpendingPlan(
            baseline=SpendingTierAmounts(essential=20_000),
            essential_floor=21_000,
        )
