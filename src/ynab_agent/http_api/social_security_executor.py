"""HTTP adapter for Social Security optimization on shared planner capacity."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.planning.social_security_optimizer import (
    SocialSecurityOptimizationResult,
    optimize_social_security,
)
from ynab_agent.services.planner_jobs import PlannerExecutionPolicy


class BoundedPlannerWorker(Protocol):
    """Shared queue, process-pool, and aggregate-memory admission port."""

    async def run_bounded(
        self,
        runner: Callable[[str], str],
        payload_json: str,
        required_working_bytes: int,
    ) -> str: ...


class SocialSecurityOptimizationPayload(BaseModel):
    """Serializable input passed across the planner process boundary."""

    model_config = ConfigDict(frozen=True)

    scenario: WealthScenario
    starting_portfolio: float = Field(ge=0)
    valuation_provenance: ValuationProvenance
    candidate_ages: tuple[int, ...] = Field(min_length=1, max_length=9)
    execution_policy: PlannerExecutionPolicy
    aggregate_compute_units: int = Field(gt=0)
    required_working_bytes: int = Field(gt=0)


def run_social_security_optimization(payload_json: str) -> str:
    """Run one admitted strategy matrix inside the shared process pool."""
    payload = SocialSecurityOptimizationPayload.model_validate_json(payload_json)
    result = optimize_social_security(
        payload.scenario,
        payload.starting_portfolio,
        valuation_provenance=payload.valuation_provenance,
        candidate_ages=payload.candidate_ages,
        run_policy=payload.execution_policy.to_run_policy(),
        maximum_compute_units=payload.execution_policy.maximum_compute_units,
        aggregate_compute_units=payload.aggregate_compute_units,
    )
    return result.model_dump_json()


class BoundedSocialSecurityOptimizationExecutor:
    """Apply aggregate policy accounting before shared worker admission."""

    def __init__(
        self,
        worker: BoundedPlannerWorker,
        execution_policy: PlannerExecutionPolicy,
        *,
        candidate_ages: tuple[int, ...] = tuple(range(62, 71)),
    ) -> None:
        self.worker = worker
        self.execution_policy = execution_policy
        self.candidate_ages = candidate_ages

    async def execute(
        self,
        scenario: WealthScenario,
        starting_portfolio: float,
        valuation_provenance: ValuationProvenance,
    ) -> SocialSecurityOptimizationResult:
        if scenario.household is None:
            raise ValueError("Social Security optimization requires a household")
        if not any(
            person.primary_insurance_amount_monthly > 0
            for person in scenario.household.people
        ):
            raise ValueError("at least one household person requires a positive PIA")
        claimable_people = len(scenario.household.people)
        strategy_count = len(self.candidate_ages) ** claimable_people
        aggregate_compute_units = (
            self.execution_policy.social_security_optimization_compute_units(
                scenario,
                strategy_count=strategy_count,
            )
        )
        required_working_bytes = (
            self.execution_policy.required_social_security_optimization_working_bytes(
                scenario,
                strategy_count=strategy_count,
            )
        )
        payload = SocialSecurityOptimizationPayload(
            scenario=scenario,
            starting_portfolio=starting_portfolio,
            valuation_provenance=valuation_provenance,
            candidate_ages=self.candidate_ages,
            execution_policy=self.execution_policy,
            aggregate_compute_units=aggregate_compute_units,
            required_working_bytes=required_working_bytes,
        )
        result_json = await self.worker.run_bounded(
            run_social_security_optimization,
            payload.model_dump_json(),
            required_working_bytes,
        )
        return SocialSecurityOptimizationResult.model_validate_json(result_json)
