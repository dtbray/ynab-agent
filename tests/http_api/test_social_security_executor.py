from __future__ import annotations

from collections.abc import Callable

import pytest

from ynab_agent.http_api.social_security_executor import (
    BoundedSocialSecurityOptimizationExecutor,
)
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.services.planner_jobs import PlannerExecutionPolicy


class _InlineBoundedWorker:
    def __init__(self) -> None:
        self.required_working_bytes: int | None = None

    async def run_bounded(
        self,
        runner: Callable[[str], str],
        payload_json: str,
        required_working_bytes: int,
    ) -> str:
        self.required_working_bytes = required_working_bytes
        return runner(payload_json)


@pytest.mark.asyncio
async def test_social_security_executor_uses_shared_admission_and_aggregate_compute() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "bounded optimizer",
            "current_age": 62,
            "retirement_age": 62,
            "end_age": 64,
            "starting_portfolio": 100_000,
            "annual_spending": 10_000,
            "return_mean": 0,
            "return_volatility": 0,
            "inflation_rate": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "household": {
                "plan_start_date": "2026-01-02",
                "people": [
                    {
                        "id": "alex",
                        "name": "Alex",
                        "birth_date": "1964-01-02",
                        "retirement_age_months": 62 * 12,
                        "primary_insurance_amount_monthly": 1_000,
                        "longevity": {
                            "mode": "deterministic",
                            "death_age": 90,
                        },
                    }
                ],
            },
            "tax_buckets": [
                {
                    "tax_treatment": "cash",
                    "starting_balance": 100_000,
                }
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0.2,
                "long_term_capital_gains_tax_rate": 0.15,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["cash"],
                "retirement_surplus_destination": "cash",
            },
        }
    )
    policy = PlannerExecutionPolicy(
        maximum_working_bytes=64 * 1024 * 1024,
        in_memory_path_bytes=64 * 1024 * 1024,
        maximum_temporary_bytes=64 * 1024 * 1024,
        batch_size=100,
        maximum_compute_units=10_000_000,
    )
    worker = _InlineBoundedWorker()
    executor = BoundedSocialSecurityOptimizationExecutor(
        worker,
        policy,
        candidate_ages=(62, 70),
    )

    result = await executor.execute(
        scenario,
        100_000,
        ValuationProvenance(
            source="test",
            account_ids=(),
            source_sha256="test-snapshot",
        ),
    )

    expected_compute = (
        policy.social_security_optimization_compute_units(
            scenario,
            strategy_count=2,
        )
    )
    assert result.strategy_count == 2
    assert result.compute_units == expected_compute
    assert result.manifest["maximum_compute_units"] == policy.maximum_compute_units
    assert result.manifest["starting_portfolio"] == 100_000
    assert result.manifest["valuation_provenance"] == {
        "source": "test",
        "as_of": None,
        "account_ids": [],
        "source_sha256": "test-snapshot",
    }
    assert worker.required_working_bytes == (
        policy.required_social_security_optimization_working_bytes(
            scenario,
            strategy_count=2,
        )
    )
