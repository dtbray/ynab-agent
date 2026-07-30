"""Durable planner-job application service and typed port contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from datetime import datetime
from enum import StrEnum
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ynab_agent.planning.historical import (
    HistoricalGapPolicy,
    HistoricalOrderPolicy,
    HistoricalSeries,
)
from ynab_agent.planning.models import ReturnModel, ValuationProvenance, WealthScenario
from ynab_agent.planning.outcomes import (
    GoalKind,
    estimate_outcome_state_bytes,
)
from ynab_agent.planning.paths import (
    PathSpec,
    ProcessResourceBudget,
    ResourceLimitError,
    RunPolicy,
    estimate_path_resources,
)
from ynab_agent.planning.simulation import (
    REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION,
    SIMULATION_ENGINE_VERSION,
    SIMULATION_RESULT_SCHEMA_VERSION,
    simulate,
)
from ynab_agent.planning.spending_guardrails import (
    estimate_guardrail_state_bytes,
)
from ynab_agent.planning.taxes import (
    PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET,
    estimate_tax_state_bytes,
)
from ynab_agent.services.wealth import WealthService


PLANNER_JOB_REQUEST_SCHEMA_VERSION = 3
DEFAULT_MAXIMUM_COMPUTE_UNITS = 200_000_000
MAX_HISTORICAL_OBSERVATIONS = 10_000


class PlannerEngineIdentity(BaseModel):
    """Version identity that invalidates stale durable result reuse."""

    model_config = ConfigDict(frozen=True)

    package_version: str = Field(min_length=1, max_length=64)
    simulation_engine_version: str = Field(min_length=1, max_length=64)
    result_schema_version: int = Field(gt=0)
    manifest_schema_version: int = Field(gt=0)

    @classmethod
    def current(cls) -> PlannerEngineIdentity:
        try:
            package_version = version("ynab-agent")
        except PackageNotFoundError:
            package_version = "development"
        return cls(
            package_version=package_version,
            simulation_engine_version=SIMULATION_ENGINE_VERSION,
            result_schema_version=SIMULATION_RESULT_SCHEMA_VERSION,
            manifest_schema_version=REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION,
        )


class PlannerJobState(StrEnum):
    """Durable lifecycle states for one simulation request."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATES = {
    PlannerJobState.SUCCEEDED,
    PlannerJobState.FAILED,
    PlannerJobState.CANCELLED,
}


class PlannerErrorCode(StrEnum):
    """Stable public failure codes independent of implementation details."""

    HISTORICAL_DATASET_REQUIRED = "historical_dataset_required"
    HISTORICAL_DATASET_NOT_APPLICABLE = "historical_dataset_not_applicable"
    HISTORICAL_DATASET_NOT_FOUND = "historical_dataset_not_found"
    RESOURCE_LIMIT = "resource_limit"
    QUEUE_SATURATED = "queue_saturated"
    JOB_NOT_FOUND = "job_not_found"
    RESULT_NOT_READY = "result_not_ready"
    JOB_NOT_CANCELLABLE = "job_not_cancellable"
    DISPATCH_FAILED = "dispatch_failed"
    EXECUTION_FAILED = "execution_failed"
    INVALID_JOB = "invalid_job"


class PlannerJobServiceError(Exception):
    """Expected service error with a stable public representation."""

    def __init__(self, code: PlannerErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PlannerQueueSaturatedError(PlannerJobServiceError):
    def __init__(self) -> None:
        super().__init__(
            PlannerErrorCode.QUEUE_SATURATED,
            "planner job capacity is currently saturated",
        )


class PlannerResourceLimitError(PlannerJobServiceError):
    def __init__(self) -> None:
        super().__init__(
            PlannerErrorCode.RESOURCE_LIMIT,
            "requested scenario exceeds configured planner resource limits",
        )


class HistoricalDatasetSnapshot(BaseModel):
    """Serializable, path-free snapshot of a registered historical dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(min_length=1, max_length=64)
    years: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_HISTORICAL_OBSERVATIONS,
    )
    nominal_returns: tuple[float, ...] = Field(
        min_length=1,
        max_length=MAX_HISTORICAL_OBSERVATIONS,
    )
    inflation_rates: tuple[float, ...] | None = Field(
        default=None,
        max_length=MAX_HISTORICAL_OBSERVATIONS,
    )
    content_sha256: str = Field(min_length=64, max_length=64)
    observations_sha256: str = Field(min_length=64, max_length=64)
    order_policy: HistoricalOrderPolicy
    gap_policy: HistoricalGapPolicy

    def to_series(self) -> HistoricalSeries:
        return HistoricalSeries(
            years=self.years,
            nominal_returns=self.nominal_returns,
            inflation_rates=self.inflation_rates,
            source=f"registered:{self.dataset_id}",
            sha256=self.content_sha256,
            observations_sha256=self.observations_sha256,
            order_policy=self.order_policy,
            gap_policy=self.gap_policy,
        )


class HistoricalDatasetRegistry(Protocol):
    """Resolve only server-registered dataset identifiers."""

    def resolve(
        self,
        dataset_id: str,
    ) -> HistoricalDatasetSnapshot | None: ...


class PlannerExecutionPolicy(BaseModel):
    """Serializable worker resource policy captured with every request."""

    model_config = ConfigDict(frozen=True)

    maximum_working_bytes: int = Field(gt=0)
    in_memory_path_bytes: int = Field(ge=0)
    maximum_temporary_bytes: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    maximum_compute_units: int = Field(
        default=DEFAULT_MAXIMUM_COMPUTE_UNITS,
        gt=0,
    )

    def to_run_policy(self) -> RunPolicy:
        return RunPolicy(
            maximum_working_bytes=self.maximum_working_bytes,
            in_memory_path_bytes=self.in_memory_path_bytes,
            maximum_temporary_bytes=self.maximum_temporary_bytes,
            batch_size=self.batch_size,
            process_budget=ProcessResourceBudget(maximum_bytes=self.maximum_working_bytes),
        )

    def required_working_bytes(
        self,
        scenario: WealthScenario,
        *,
        paired_historical_inflation: bool,
    ) -> int:
        policy = self.to_run_policy()
        spec = PathSpec.from_scenario(
            scenario,
            paired_historical_inflation=paired_historical_inflation,
        )
        estimate = estimate_path_resources(spec, batch_size=self.batch_size)
        policy.enforce(estimate)
        compute_units = self._compute_units(scenario)
        if compute_units > self.maximum_compute_units:
            raise ResourceLimitError(
                "estimated simulation compute work exceeds the configured limit; "
                "reduce trials, horizon, or income streams"
            )
        required_bytes = (
            policy.required_working_bytes(estimate)
            + estimate_tax_state_bytes(scenario)
            + estimate_outcome_state_bytes(scenario)
            + (
                estimate_guardrail_state_bytes(scenario.trials)
                if scenario.retirement_spending_plan is not None
                else 0
            )
        )
        if required_bytes > self.maximum_working_bytes:
            raise ResourceLimitError(
                "estimated simulation and tax-state memory exceeds the configured limit; "
                "reduce trials, batch size, horizon, or tax buckets"
            )
        return required_bytes

    def required_comparison_working_bytes(
        self,
        scenarios: Collection[WealthScenario],
        *,
        paired_historical_inflation: bool,
    ) -> int:
        """Bound sequential evaluations that reuse one common path matrix."""
        if not scenarios:
            raise ValueError("comparison scenarios must not be empty")
        total_compute_units = sum(
            self._compute_units(scenario) for scenario in scenarios
        )
        if total_compute_units > self.maximum_compute_units:
            raise ResourceLimitError(
                "estimated comparison compute work exceeds the configured limit; "
                "reduce alternatives, trials, horizon, or scenario complexity"
            )
        # Paths remain resident, but evaluation arrays are mutually exclusive.
        return max(
            self.required_working_bytes(
                scenario,
                paired_historical_inflation=paired_historical_inflation,
            )
            for scenario in scenarios
        )

    @staticmethod
    def _compute_units(scenario: WealthScenario) -> int:
        work_per_trial_year = (
            1
            + len(scenario.income_streams)
            + len(scenario.cash_flow_streams)
            + len(scenario.tax_buckets)
            + len(scenario.spending_tiers)
            + int(scenario.retirement_spending_plan is not None)
        )
        progressive = (
            scenario.tax_assumptions.progressive
            if scenario.tax_assumptions is not None
            else None
        )
        if progressive is not None:
            # Every ordered bucket can require a bounded bracket search, a
            # cent-convergent bisection, and a final component calculation.
            work_per_trial_year += (
                len(scenario.tax_buckets)
                * PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET
            )
            # Baseline/final components and the two $1 marginal-rate probes.
            work_per_trial_year += 4
        return (
            (scenario.end_age - scenario.current_age)
            * scenario.trials
            * work_per_trial_year
        )


class PlannerJobSubmission(BaseModel):
    """Validated application-level request before account and data resolution."""

    model_config = ConfigDict(frozen=True)

    scenario: WealthScenario
    historical_dataset_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )


class PlannerJobPayload(BaseModel):
    """Complete serializable worker input persisted before dispatch."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(
        default=PLANNER_JOB_REQUEST_SCHEMA_VERSION,
        ge=1,
    )
    engine_identity: PlannerEngineIdentity | None = None
    scenario: WealthScenario
    starting_portfolio: float = Field(ge=0)
    valuation_provenance: ValuationProvenance
    historical_dataset: HistoricalDatasetSnapshot | None = None
    execution_policy: PlannerExecutionPolicy
    required_working_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_historical_input(self) -> PlannerJobPayload:
        if (
            self.schema_version >= PLANNER_JOB_REQUEST_SCHEMA_VERSION
            and self.engine_identity is None
        ):
            raise ValueError("current planner jobs require a persisted engine identity")
        historical = self.scenario.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
        if historical and self.historical_dataset is None:
            raise ValueError("historical jobs require a registered dataset snapshot")
        if not historical and self.historical_dataset is not None:
            raise ValueError("lognormal jobs cannot include historical data")
        return self


class PlannerGoalOutcome(BaseModel):
    """Persisted goal-attainment primitive."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: GoalKind
    target_real: float
    attainment_probability: float | None
    attained_trials: int | None
    evaluation_basis: str


class PlannerSimulationResult(BaseModel):
    """Persisted typed projection result."""

    model_config = ConfigDict(frozen=True)

    scenario: str
    starting_portfolio: float
    trials: int
    seed: int
    success_rate: float
    success_rate_ci_95: dict[str, float]
    depleted_trials: int
    median_depletion_age: float | None
    retirement_balance_real: dict[str, float]
    ending_balance_real: dict[str, float]
    lifetime_tax_real: dict[str, float] | None = None
    annual_tax_audit: list[dict[str, object]] = Field(default_factory=list)
    annual_balance_real: list[dict[str, float | int]]
    funded_spending_ratio: dict[str, float] = Field(
        default_factory=lambda: {"p10": 1.0, "p50": 1.0, "p90": 1.0}
    )
    funded_spending_real: dict[str, float] = Field(
        default_factory=lambda: {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    )
    cumulative_shortfall_real: dict[str, float] = Field(
        default_factory=lambda: {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    )
    failure_duration_years: dict[str, float] | None = None
    longest_failure_streak_years: dict[str, float] | None = None
    recovered_trials: int = 0
    recovery_probability: float | None = None
    after_tax_ending_balance_real: dict[str, float] | None = None
    legacy_target_probability: float | None = None
    goal_outcomes: list[PlannerGoalOutcome] = Field(default_factory=list)
    annual_spending_real: list[dict[str, object]] = Field(
        default_factory=list,
    )
    guardrail_metrics: dict[str, object] | None = None
    assumptions: dict[str, bool | float | int | str]
    engine: dict[str, str | int]
    reproducibility: dict[str, object]


class PlannerExecutionOutput(BaseModel):
    """Serializable worker output persisted atomically."""

    model_config = ConfigDict(frozen=True)

    result: PlannerSimulationResult
    manifest: dict[str, object]


class PlannerJobPublicError(BaseModel):
    """Stable error persisted for failed jobs."""

    model_config = ConfigDict(frozen=True)

    code: PlannerErrorCode
    message: str


class PlannerJob(BaseModel):
    """Typed durable job record."""

    model_config = ConfigDict(frozen=True)

    id: str
    request_hash: str = Field(min_length=64, max_length=64)
    state: PlannerJobState
    payload: PlannerJobPayload
    cancellation_requested: bool = False
    result: PlannerSimulationResult | None = None
    manifest: dict[str, object] | None = None
    error: PlannerJobPublicError | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class PlannerJobCreateResult(BaseModel):
    """Repository result distinguishing a new row from an idempotent race."""

    model_config = ConfigDict(frozen=True)

    job: PlannerJob
    created: bool


class PlannerJobSubmitResult(BaseModel):
    """Submission result returned to inbound adapters."""

    model_config = ConfigDict(frozen=True)

    job: PlannerJob
    duplicate: bool


class PlannerJobRepository(Protocol):
    async def get_by_request_hash(
        self,
        request_hash: str,
    ) -> PlannerJob | None: ...

    async def create_job(
        self,
        *,
        job_id: str,
        request_hash: str,
        payload: PlannerJobPayload,
    ) -> PlannerJobCreateResult: ...

    async def get_job(self, job_id: str) -> PlannerJob | None: ...

    async def prepare_recovery(self) -> None: ...

    async def list_accepted_jobs(
        self,
        *,
        limit: int,
        exclude_job_ids: Collection[str],
    ) -> tuple[PlannerJob, ...]: ...

    async def mark_running(self, job_id: str) -> PlannerJob | None: ...

    async def mark_succeeded(
        self,
        job_id: str,
        output: PlannerExecutionOutput,
    ) -> PlannerJob | None: ...

    async def mark_failed(
        self,
        job_id: str,
        error: PlannerJobPublicError,
    ) -> PlannerJob | None: ...

    async def request_cancellation(self, job_id: str) -> PlannerJob | None: ...


class PlannerJobDispatcher(Protocol):
    """Bounded process worker admission and notification port."""

    async def reserve(self, job_id: str, required_working_bytes: int) -> None: ...

    async def activate(self, job_id: str) -> None: ...

    async def release(self, job_id: str) -> None: ...


def canonical_job_request_hash(payload: PlannerJobPayload) -> str:
    """Hash the complete versioned worker payload deterministically."""
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def run_planner_job(payload_json: str) -> str:
    """Process-safe simulation entry point with JSON-only inputs and outputs."""
    payload = PlannerJobPayload.model_validate_json(payload_json)
    if (
        payload.engine_identity is None
        or payload.engine_identity != PlannerEngineIdentity.current()
    ):
        raise ValueError("persisted planner job targets a different engine")
    historical_series = (
        payload.historical_dataset.to_series() if payload.historical_dataset is not None else None
    )
    result = simulate(
        payload.scenario,
        payload.starting_portfolio,
        historical_series=historical_series,
        valuation_provenance=payload.valuation_provenance,
        run_policy=payload.execution_policy.to_run_policy(),
    )
    output = PlannerExecutionOutput(
        result=PlannerSimulationResult.model_validate(result.as_dict()),
        manifest=result.reproducibility,
    )
    return output.model_dump_json()


class PlannerJobService:
    """Submit and inspect durable planner work through narrow ports."""

    def __init__(
        self,
        *,
        repository: PlannerJobRepository,
        wealth_service: WealthService,
        historical_datasets: HistoricalDatasetRegistry,
        dispatcher: PlannerJobDispatcher,
        execution_policy: PlannerExecutionPolicy,
    ) -> None:
        self.repository = repository
        self.wealth_service = wealth_service
        self.historical_datasets = historical_datasets
        self.dispatcher = dispatcher
        self.execution_policy = execution_policy

    async def submit(
        self,
        submission: PlannerJobSubmission,
    ) -> PlannerJobSubmitResult:
        dataset = self._resolve_historical_dataset(submission)
        if dataset is not None and submission.scenario.historical_block_size > len(
            dataset.nominal_returns
        ):
            raise PlannerJobServiceError(
                PlannerErrorCode.INVALID_JOB,
                "historical block size exceeds the registered dataset",
            )
        try:
            resolved = await self.wealth_service.resolve_starting_portfolio_with_provenance(
                submission.scenario
            )
        except ValueError as exc:
            raise PlannerJobServiceError(
                PlannerErrorCode.INVALID_JOB,
                "scenario account selection is invalid",
            ) from exc
        try:
            required_working_bytes = self.execution_policy.required_working_bytes(
                submission.scenario,
                paired_historical_inflation=(
                    dataset is not None and dataset.inflation_rates is not None
                ),
            )
        except ResourceLimitError as exc:
            raise PlannerResourceLimitError() from exc

        payload = PlannerJobPayload(
            engine_identity=PlannerEngineIdentity.current(),
            scenario=submission.scenario,
            starting_portfolio=resolved.value,
            valuation_provenance=resolved.provenance,
            historical_dataset=dataset,
            execution_policy=self.execution_policy,
            required_working_bytes=required_working_bytes,
        )
        request_hash = canonical_job_request_hash(payload)
        existing = await self.repository.get_by_request_hash(request_hash)
        if existing is not None:
            return PlannerJobSubmitResult(job=existing, duplicate=True)

        job_id = str(uuid4())
        await self.dispatcher.reserve(job_id, required_working_bytes)
        try:
            created = await self.repository.create_job(
                job_id=job_id,
                request_hash=request_hash,
                payload=payload,
            )
            if not created.created:
                await self.dispatcher.release(job_id)
                return PlannerJobSubmitResult(
                    job=created.job,
                    duplicate=True,
                )
            await self.dispatcher.activate(job_id)
        except BaseException as exc:
            try:
                persisted = await self.repository.get_job(job_id)
                if persisted is not None:
                    await self.repository.mark_failed(
                        job_id,
                        PlannerJobPublicError(
                            code=PlannerErrorCode.DISPATCH_FAILED,
                            message="planner job dispatch failed",
                        ),
                    )
            finally:
                await self.dispatcher.release(job_id)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise PlannerJobServiceError(
                PlannerErrorCode.DISPATCH_FAILED,
                "planner job dispatch failed",
            ) from exc
        return PlannerJobSubmitResult(job=created.job, duplicate=False)

    async def get_status(self, job_id: str) -> PlannerJob:
        job = await self.repository.get_job(job_id)
        if job is None:
            raise PlannerJobServiceError(
                PlannerErrorCode.JOB_NOT_FOUND,
                "planner job was not found",
            )
        return job

    async def get_result(self, job_id: str) -> PlannerJob:
        job = await self.get_status(job_id)
        if job.state is PlannerJobState.FAILED:
            raise PlannerJobServiceError(
                PlannerErrorCode.EXECUTION_FAILED,
                "planner job failed; inspect its status for the stable error code",
            )
        if job.state is not PlannerJobState.SUCCEEDED or job.result is None:
            raise PlannerJobServiceError(
                PlannerErrorCode.RESULT_NOT_READY,
                "planner job result is not available",
            )
        return job

    async def cancel(self, job_id: str) -> PlannerJob:
        current = await self.get_status(job_id)
        if current.state in TERMINAL_JOB_STATES:
            raise PlannerJobServiceError(
                PlannerErrorCode.JOB_NOT_CANCELLABLE,
                "planner job is already in a terminal state",
            )
        updated = await self.repository.request_cancellation(job_id)
        if updated is None:
            raise PlannerJobServiceError(
                PlannerErrorCode.JOB_NOT_FOUND,
                "planner job was not found",
            )
        if updated.state is PlannerJobState.CANCELLED:
            await self.dispatcher.release(job_id)
        return updated

    def _resolve_historical_dataset(
        self,
        submission: PlannerJobSubmission,
    ) -> HistoricalDatasetSnapshot | None:
        is_historical = submission.scenario.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
        dataset_id = submission.historical_dataset_id
        if is_historical and dataset_id is None:
            raise PlannerJobServiceError(
                PlannerErrorCode.HISTORICAL_DATASET_REQUIRED,
                "historical simulations require a registered dataset ID",
            )
        if not is_historical and dataset_id is not None:
            raise PlannerJobServiceError(
                PlannerErrorCode.HISTORICAL_DATASET_NOT_APPLICABLE,
                "historical dataset IDs apply only to historical simulations",
            )
        if dataset_id is None:
            return None
        dataset = self.historical_datasets.resolve(dataset_id)
        if dataset is None:
            raise PlannerJobServiceError(
                PlannerErrorCode.HISTORICAL_DATASET_NOT_FOUND,
                "registered historical dataset was not found",
            )
        return dataset
