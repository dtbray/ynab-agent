"""Saved wealth scenarios and reproducible common-path comparisons."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from typing import NoReturn, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ynab_agent.planning.historical import HistoricalSeries
from ynab_agent.planning.models import (
    ReturnModel,
    ValuationProvenance,
    WealthScenario,
)
from ynab_agent.planning.paths import PathSpec, ResourceLimitError
from ynab_agent.planning.simulation import (
    SimulationResult,
    canonical_scenario_sha256,
    prepare_simulation_paths,
    simulate,
)
from ynab_agent.services.planner_jobs import (
    HistoricalDatasetSnapshot,
    PlannerEngineIdentity,
    PlannerExecutionPolicy,
    PlannerQueueSaturatedError,
    PlannerResourceLimitError,
)
from ynab_agent.services.wealth import ResolvedStartingPortfolio


SCENARIO_REVISION_SCHEMA_VERSION = 1
SCENARIO_COMPARISON_SCHEMA_VERSION = 3
SCENARIO_COMPARISON_POLICY_VERSION = "retirement_outcomes_v2"
MAX_COMPARISON_ALTERNATIVES = 12


def default_comparison_execution_policy() -> PlannerExecutionPolicy:
    """Return the bounded policy used by local comparison commands."""
    return PlannerExecutionPolicy(
        maximum_working_bytes=512 * 1024 * 1024,
        in_memory_path_bytes=128 * 1024 * 1024,
        maximum_temporary_bytes=2 * 1024 * 1024 * 1024,
        batch_size=10_000,
    )


class ScenarioComparisonErrorCode(StrEnum):
    """Stable public error codes for scenario operations."""

    REVISION_NOT_FOUND = "revision_not_found"
    DUPLICATE_ALTERNATIVE = "duplicate_alternative"
    INCOMPATIBLE_PATHS = "incompatible_paths"
    HISTORICAL_DATASET_REQUIRED = "historical_dataset_required"
    HISTORICAL_DATASET_NOT_APPLICABLE = "historical_dataset_not_applicable"
    HISTORICAL_DATASET_MISMATCH = "historical_dataset_mismatch"
    PERSISTENCE_CONFLICT = "persistence_conflict"
    QUEUE_SATURATED = "queue_saturated"
    RESOURCE_LIMIT = "resource_limit"
    PERSISTED_CONTENT_MISMATCH = "persisted_content_mismatch"


class ScenarioComparisonError(Exception):
    """Expected scenario-comparison failure with a stable representation."""

    def __init__(
        self,
        code: ScenarioComparisonErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ScenarioRevisionManifest(BaseModel):
    """Every material input needed to reproduce one saved scenario."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCENARIO_REVISION_SCHEMA_VERSION
    scenario: WealthScenario
    original_scenario_sha256: str = Field(min_length=64, max_length=64)
    resolved_scenario_sha256: str = Field(min_length=64, max_length=64)
    starting_portfolio: float = Field(ge=0)
    valuation_provenance: ValuationProvenance
    historical_dataset: HistoricalDatasetSnapshot | None = None
    engine_identity: PlannerEngineIdentity


class ScenarioRevision(BaseModel):
    """One immutable, numbered revision of a named scenario."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=36, max_length=36)
    scenario_name: str = Field(min_length=1, max_length=200)
    revision: int = Field(gt=0)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    manifest: ScenarioRevisionManifest
    created_at: datetime


class ScenarioOutcomeMetrics(BaseModel):
    """Stable comparison metrics, including nullable richer outcomes."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    success_probability: float = Field(ge=0, le=1)
    funded_spending_ratio_p50: float | None = Field(default=None, ge=0)
    funded_spending_real_p50: float | None = Field(default=None, ge=0)
    cumulative_shortfall_real_p50: float | None = Field(default=None, ge=0)
    guardrail_reduction_real_p50: float | None = Field(default=None, ge=0)
    requested_annual_spending_real: float = Field(gt=0)
    lifetime_tax_real_p50: float | None = Field(default=None, ge=0)
    estate_value_real_p50: float = Field(ge=0)
    estate_value_real_p10: float = Field(ge=0)
    after_tax_estate_value_real_p50: float | None = Field(default=None, ge=0)
    after_tax_estate_value_real_p10: float | None = Field(default=None, ge=0)
    retirement_balance_real_p10: float = Field(ge=0)


class ScenarioMetricDelta(BaseModel):
    """Alternative-minus-baseline deltas with stable field semantics."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    success_probability: float
    funded_spending_ratio_p50: float | None = None
    funded_spending_real_p50: float | None = None
    cumulative_shortfall_real_p50: float | None = None
    guardrail_reduction_real_p50: float | None = None
    requested_annual_spending_real: float
    lifetime_tax_real_p50: float | None = None
    estate_value_real_p50: float
    estate_value_real_p10: float
    after_tax_estate_value_real_p50: float | None = None
    after_tax_estate_value_real_p10: float | None = None
    retirement_balance_real_p10: float


class ChangeKind(StrEnum):
    """Source of a changed material assumption."""

    INPUT = "input"
    SPENDING_POLICY = "spending_policy"
    TAX_POLICY = "tax_policy"
    HOUSEHOLD_POLICY = "household_policy"


class ScenarioInputChange(BaseModel):
    """One changed scenario input or policy value."""

    model_config = ConfigDict(frozen=True)

    path: str
    kind: ChangeKind
    baseline_value: JsonValue
    alternative_value: JsonValue


class TradeoffDirection(StrEnum):
    """Whether one metric moves in a favorable or unfavorable direction."""

    BENEFIT = "benefit"
    COST = "cost"


class ScenarioTradeoff(BaseModel):
    """One material, preference-aware outcome movement."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    metric: str
    direction: TradeoffDirection
    delta: float


class ScenarioEvaluation(BaseModel):
    """Outcome summary for one saved revision."""

    model_config = ConfigDict(frozen=True)

    revision_id: str
    scenario_name: str
    revision: int
    metrics: ScenarioOutcomeMetrics


class ScenarioAlternativeComparison(ScenarioEvaluation):
    """Alternative outcome, deltas, attribution, and dominance."""

    delta: ScenarioMetricDelta
    input_changes: tuple[ScenarioInputChange, ...] = ()
    tradeoffs: tuple[ScenarioTradeoff, ...] = ()
    dominated_by_revision_ids: tuple[str, ...] = ()


class CommonPathManifest(BaseModel):
    """Exact economic-path inputs shared by every alternative."""

    model_config = ConfigDict(frozen=True)

    years: int
    trials: int
    seed: int
    return_model: ReturnModel
    return_mean: float
    return_volatility: float
    inflation_rate: float
    annual_fee_rate: float
    historical_block_size: int
    paired_historical_inflation: bool
    historical_observations_sha256: str | None = None


class ScenarioComparisonManifest(BaseModel):
    """Stored recipe for reproducing a comparison without generated arrays."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCENARIO_COMPARISON_SCHEMA_VERSION
    policy_version: str = SCENARIO_COMPARISON_POLICY_VERSION
    baseline_revision_id: str
    alternative_revision_ids: tuple[str, ...]
    revision_manifest_sha256: dict[str, str]
    common_paths: CommonPathManifest
    execution_policy: PlannerExecutionPolicy
    engine_identity: PlannerEngineIdentity


class ScenarioComparison(BaseModel):
    """Persisted stable result of a common-path comparison."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=36, max_length=36)
    baseline: ScenarioEvaluation
    alternatives: tuple[ScenarioAlternativeComparison, ...]
    manifest_sha256: str = Field(min_length=64, max_length=64)
    result_sha256: str = Field(min_length=64, max_length=64)
    manifest: ScenarioComparisonManifest
    created_at: datetime


class ScenarioRevisionRepository(Protocol):
    """Append-only scenario and comparison persistence port."""

    async def next_revision_number(self, scenario_name: str) -> int: ...

    async def create_revision(self, revision: ScenarioRevision) -> None: ...

    async def get_revision(self, revision_id: str) -> ScenarioRevision | None: ...

    async def create_comparison(self, comparison: ScenarioComparison) -> None: ...

    async def get_comparison(
        self,
        comparison_id: str,
    ) -> ScenarioComparison | None: ...


class ScenarioPortfolioResolver(Protocol):
    """Resolve a scenario's live account inputs at revision time."""

    async def resolve_starting_portfolio_with_provenance(
        self,
        scenario: WealthScenario,
    ) -> ResolvedStartingPortfolio: ...


class BoundedScenarioComparisonExecutor(Protocol):
    """Shared process admission used by jobs and scenario comparisons."""

    async def run_bounded(
        self,
        runner: Callable[[str], str],
        payload_json: str,
        required_working_bytes: int,
    ) -> str: ...


class ScenarioOutcomeExtractor(Protocol):
    """Extension point for richer spending, tax, and estate outcomes."""

    def extract(
        self,
        result: SimulationResult,
        scenario: WealthScenario,
    ) -> ScenarioOutcomeMetrics: ...


class ScenarioChangeClassifier(Protocol):
    """Extension point for later spending, tax, and household policies."""

    def classify(self, path: str) -> ChangeKind: ...


class DefaultScenarioOutcomeExtractor:
    """Map current simulation output into the stable comparison model."""

    def extract(
        self,
        result: SimulationResult,
        scenario: WealthScenario,
    ) -> ScenarioOutcomeMetrics:
        return ScenarioOutcomeMetrics(
            success_probability=result.success_rate,
            funded_spending_ratio_p50=_optional_nested_metric(
                result,
                "funded_spending_ratio",
                "p50",
            ),
            funded_spending_real_p50=_optional_nested_metric(
                result,
                "funded_spending_real",
                "p50",
            ),
            cumulative_shortfall_real_p50=_optional_nested_metric(
                result,
                "cumulative_shortfall_real",
                "p50",
            ),
            guardrail_reduction_real_p50=_optional_mapping_metric(
                result.guardrail_metrics,
                "cumulative_reduction_real",
                "p50",
            ),
            requested_annual_spending_real=scenario.annual_spending,
            lifetime_tax_real_p50=(
                result.lifetime_tax_real["p50"]
                if result.lifetime_tax_real is not None
                else None
            ),
            estate_value_real_p50=result.ending_balance_real["p50"],
            estate_value_real_p10=result.ending_balance_real["p10"],
            after_tax_estate_value_real_p50=(
                result.after_tax_ending_balance_real["p50"]
                if result.after_tax_ending_balance_real is not None
                else None
            ),
            after_tax_estate_value_real_p10=(
                result.after_tax_ending_balance_real["p10"]
                if result.after_tax_ending_balance_real is not None
                else None
            ),
            retirement_balance_real_p10=result.retirement_balance_real["p10"],
        )


class DefaultScenarioChangeClassifier:
    """Classify today's fields while leaving clear extension seams."""

    def classify(self, path: str) -> ChangeKind:
        if "tax_assumptions" in path or "tax_buckets" in path or path.endswith(
            "withdrawal_tax_rate"
        ):
            return ChangeKind.TAX_POLICY
        if (
            "spending" in path
            or "cash_flow_streams" in path
            or "guardrail" in path
        ):
            return ChangeKind.SPENDING_POLICY
        if "household" in path:
            return ChangeKind.HOUSEHOLD_POLICY
        return ChangeKind.INPUT


class ScenarioComparisonService:
    """Save immutable revisions and compare them on one economic experiment."""

    def __init__(
        self,
        *,
        repository: ScenarioRevisionRepository,
        portfolio_resolver: ScenarioPortfolioResolver,
        execution_policy: PlannerExecutionPolicy,
        bounded_executor: BoundedScenarioComparisonExecutor | None = None,
    ) -> None:
        self.repository = repository
        self.portfolio_resolver = portfolio_resolver
        self.execution_policy = execution_policy
        self.bounded_executor = bounded_executor

    async def save_revision(
        self,
        scenario: WealthScenario,
        *,
        historical_dataset: HistoricalDatasetSnapshot | None = None,
    ) -> ScenarioRevision:
        _validate_historical_dataset(scenario, historical_dataset)
        resolved = await self.portfolio_resolver.resolve_starting_portfolio_with_provenance(
            scenario
        )
        resolved_scenario = WealthScenario.model_validate(
            {
                **scenario.model_dump(mode="python"),
                "starting_portfolio": resolved.value,
            }
        )
        manifest = ScenarioRevisionManifest(
            scenario=resolved_scenario,
            original_scenario_sha256=canonical_scenario_sha256(scenario),
            resolved_scenario_sha256=canonical_scenario_sha256(resolved_scenario),
            starting_portfolio=resolved.value,
            valuation_provenance=resolved.provenance,
            historical_dataset=historical_dataset,
            engine_identity=PlannerEngineIdentity.current(),
        )
        manifest_sha256 = _canonical_model_sha256(manifest)
        for _ in range(3):
            revision = ScenarioRevision(
                id=str(uuid4()),
                scenario_name=scenario.name,
                revision=await self.repository.next_revision_number(scenario.name),
                manifest_sha256=manifest_sha256,
                manifest=manifest,
                created_at=_utc_now(),
            )
            try:
                await self.repository.create_revision(revision)
            except ScenarioComparisonError as exc:
                if exc.code is not ScenarioComparisonErrorCode.PERSISTENCE_CONFLICT:
                    raise
                continue
            return revision
        raise ScenarioComparisonError(
            ScenarioComparisonErrorCode.PERSISTENCE_CONFLICT,
            "scenario revision could not be allocated after concurrent writes",
        )

    async def get_revision(self, revision_id: str) -> ScenarioRevision:
        revision = await self.repository.get_revision(revision_id)
        if revision is None:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.REVISION_NOT_FOUND,
                f"scenario revision was not found: {revision_id}",
            )
        verify_scenario_revision(revision)
        return revision

    async def get_comparison(self, comparison_id: str) -> ScenarioComparison:
        comparison = await self.repository.get_comparison(comparison_id)
        if comparison is None:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.REVISION_NOT_FOUND,
                f"scenario comparison was not found: {comparison_id}",
            )
        verify_scenario_comparison(comparison)
        return comparison

    async def compare(
        self,
        *,
        baseline_revision_id: str,
        alternative_revision_ids: Sequence[str],
    ) -> ScenarioComparison:
        alternative_ids = tuple(alternative_revision_ids)
        if len(set(alternative_ids)) != len(alternative_ids) or (
            baseline_revision_id in alternative_ids
        ):
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.DUPLICATE_ALTERNATIVE,
                "comparison revision IDs must be unique",
            )
        if not alternative_ids:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.DUPLICATE_ALTERNATIVE,
                "at least one alternative revision is required",
            )
        if len(alternative_ids) > MAX_COMPARISON_ALTERNATIVES:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.DUPLICATE_ALTERNATIVE,
                f"comparisons accept at most {MAX_COMPARISON_ALTERNATIVES} alternatives",
            )

        revisions = [
            await self.get_revision(revision_id)
            for revision_id in (baseline_revision_id, *alternative_ids)
        ]
        baseline_historical = revisions[0].manifest.historical_dataset
        try:
            required_working_bytes = (
                self.execution_policy.required_comparison_working_bytes(
                    [revision.manifest.scenario for revision in revisions],
                    paired_historical_inflation=(
                        baseline_historical is not None
                        and baseline_historical.inflation_rates is not None
                    ),
                )
            )
        except ResourceLimitError as exc:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.RESOURCE_LIMIT,
                str(exc),
            ) from exc
        request = _build_run_request(revisions, self.execution_policy)
        if self.bounded_executor is None:
            run_result = run_scenario_comparison(request)
        else:
            try:
                result_json = await self.bounded_executor.run_bounded(
                    run_scenario_comparison_json,
                    request.model_dump_json(),
                    required_working_bytes,
                )
            except PlannerQueueSaturatedError as exc:
                raise ScenarioComparisonError(
                    ScenarioComparisonErrorCode.QUEUE_SATURATED,
                    "planner comparison capacity is currently saturated",
                ) from exc
            except (PlannerResourceLimitError, ResourceLimitError) as exc:
                raise ScenarioComparisonError(
                    ScenarioComparisonErrorCode.RESOURCE_LIMIT,
                    "scenario comparison exceeds configured planner resources",
                ) from exc
            run_result = ScenarioComparisonRunResult.model_validate_json(result_json)

        comparison_id = str(uuid4())
        created_at = _utc_now()
        result_sha256 = _comparison_result_sha256(
            comparison_id=comparison_id,
            baseline=run_result.baseline,
            alternatives=run_result.alternatives,
            manifest_sha256=_canonical_model_sha256(request.manifest),
            manifest=request.manifest,
            created_at=created_at,
        )
        comparison = ScenarioComparison(
            id=comparison_id,
            baseline=run_result.baseline,
            alternatives=run_result.alternatives,
            manifest_sha256=_canonical_model_sha256(request.manifest),
            result_sha256=result_sha256,
            manifest=request.manifest,
            created_at=created_at,
        )
        await self.repository.create_comparison(comparison)
        return comparison


class ScenarioComparisonRunRequest(BaseModel):
    """Serializable, process-safe comparison execution input."""

    model_config = ConfigDict(frozen=True)

    revisions: tuple[ScenarioRevision, ...] = Field(min_length=2)
    manifest: ScenarioComparisonManifest


class ScenarioComparisonRunResult(BaseModel):
    """Process-safe output before persistence metadata is attached."""

    model_config = ConfigDict(frozen=True)

    baseline: ScenarioEvaluation
    alternatives: tuple[ScenarioAlternativeComparison, ...]


def historical_snapshot_from_series(
    series: HistoricalSeries,
    *,
    dataset_id: str,
) -> HistoricalDatasetSnapshot:
    """Create a path-free immutable snapshot from a local historical series."""
    return HistoricalDatasetSnapshot(
        dataset_id=dataset_id,
        years=series.years,
        nominal_returns=series.nominal_returns,
        inflation_rates=series.inflation_rates,
        content_sha256=series.sha256,
        observations_sha256=series.observations_sha256,
        order_policy=series.order_policy,
        gap_policy=series.gap_policy,
    )


def run_scenario_comparison_json(request_json: str) -> str:
    """Process-executor entry point using only serializable values."""
    request = ScenarioComparisonRunRequest.model_validate_json(request_json)
    return run_scenario_comparison(request).model_dump_json()


def run_scenario_comparison(
    request: ScenarioComparisonRunRequest,
    *,
    outcome_extractor: ScenarioOutcomeExtractor | None = None,
    change_classifier: ScenarioChangeClassifier | None = None,
) -> ScenarioComparisonRunResult:
    """Evaluate all revisions against one prepared set of return paths."""
    _verify_run_request(request)
    extractor = outcome_extractor or DefaultScenarioOutcomeExtractor()
    classifier = change_classifier or DefaultScenarioChangeClassifier()
    baseline_revision = request.revisions[0]
    baseline_manifest = baseline_revision.manifest
    scenario = baseline_manifest.scenario
    historical = baseline_manifest.historical_dataset
    series = historical.to_series() if historical is not None else None
    run_policy = request.manifest.execution_policy.to_run_policy()

    with prepare_simulation_paths(
        scenario,
        historical_returns=(series.nominal_returns if series is not None else None),
        historical_inflation=(series.inflation_rates if series is not None else None),
        run_policy=run_policy,
    ) as paths:
        evaluations: list[ScenarioEvaluation] = []
        for revision in request.revisions:
            result = simulate(
                revision.manifest.scenario,
                revision.manifest.starting_portfolio,
                historical_series=(
                    revision.manifest.historical_dataset.to_series()
                    if revision.manifest.historical_dataset is not None
                    else None
                ),
                valuation_provenance=revision.manifest.valuation_provenance,
                prepared_paths=paths,
                include_annual_path=False,
            )
            evaluations.append(
                ScenarioEvaluation(
                    revision_id=revision.id,
                    scenario_name=revision.scenario_name,
                    revision=revision.revision,
                    metrics=extractor.extract(result, revision.manifest.scenario),
                )
            )

    baseline = evaluations[0]
    alternatives: list[ScenarioAlternativeComparison] = []
    for revision, evaluation in zip(
        request.revisions[1:],
        evaluations[1:],
        strict=True,
    ):
        delta = _metric_delta(baseline.metrics, evaluation.metrics)
        alternatives.append(
            ScenarioAlternativeComparison(
                **evaluation.model_dump(),
                delta=delta,
                input_changes=_scenario_changes(
                    baseline_manifest.scenario,
                    revision.manifest.scenario,
                    classifier,
                ),
                tradeoffs=_material_tradeoffs(delta),
            )
        )

    all_evaluations = [baseline, *alternatives]
    alternatives = [
        alternative.model_copy(
            update={
                "dominated_by_revision_ids": tuple(
                    other.revision_id
                    for other in all_evaluations
                    if other.revision_id != alternative.revision_id
                    and _dominates(other.metrics, alternative.metrics)
                )
            }
        )
        for alternative in alternatives
    ]
    return ScenarioComparisonRunResult(
        baseline=baseline,
        alternatives=tuple(alternatives),
    )


def _build_run_request(
    revisions: Sequence[ScenarioRevision],
    execution_policy: PlannerExecutionPolicy,
) -> ScenarioComparisonRunRequest:
    baseline = revisions[0]
    baseline_historical = baseline.manifest.historical_dataset
    paired_inflation = (
        baseline_historical is not None
        and baseline_historical.inflation_rates is not None
    )
    expected_spec = PathSpec.from_scenario(
        baseline.manifest.scenario,
        paired_historical_inflation=paired_inflation,
    )
    expected_history_hash = (
        baseline_historical.observations_sha256
        if baseline_historical is not None
        else None
    )
    for revision in revisions[1:]:
        historical = revision.manifest.historical_dataset
        candidate_spec = PathSpec.from_scenario(
            revision.manifest.scenario,
            paired_historical_inflation=(
                historical is not None and historical.inflation_rates is not None
            ),
        )
        if candidate_spec != expected_spec:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.INCOMPATIBLE_PATHS,
                "saved revisions must have identical horizon, trial, seed, "
                "return, inflation, fee, and bootstrap path assumptions",
            )
        candidate_hash = (
            historical.observations_sha256 if historical is not None else None
        )
        if candidate_hash != expected_history_hash:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.HISTORICAL_DATASET_MISMATCH,
                "saved revisions must use the same historical observations",
            )

    manifest = ScenarioComparisonManifest(
        baseline_revision_id=baseline.id,
        alternative_revision_ids=tuple(revision.id for revision in revisions[1:]),
        revision_manifest_sha256={
            revision.id: revision.manifest_sha256 for revision in revisions
        },
        common_paths=CommonPathManifest(
            **{
                **expected_spec.__dict__,
                "historical_observations_sha256": expected_history_hash,
            }
        ),
        execution_policy=execution_policy,
        engine_identity=PlannerEngineIdentity.current(),
    )
    return ScenarioComparisonRunRequest(
        revisions=tuple(revisions),
        manifest=manifest,
    )


def _validate_historical_dataset(
    scenario: WealthScenario,
    historical_dataset: HistoricalDatasetSnapshot | None,
) -> None:
    if (
        scenario.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
        and historical_dataset is None
    ):
        raise ScenarioComparisonError(
            ScenarioComparisonErrorCode.HISTORICAL_DATASET_REQUIRED,
            "historical scenarios require an immutable dataset snapshot",
        )
    if (
        scenario.return_model is not ReturnModel.HISTORICAL_BOOTSTRAP
        and historical_dataset is not None
    ):
        raise ScenarioComparisonError(
            ScenarioComparisonErrorCode.HISTORICAL_DATASET_NOT_APPLICABLE,
            "historical datasets apply only to historical scenarios",
        )
    if (
        historical_dataset is not None
        and scenario.historical_block_size > len(historical_dataset.nominal_returns)
    ):
        raise ScenarioComparisonError(
            ScenarioComparisonErrorCode.HISTORICAL_DATASET_MISMATCH,
            "historical block size exceeds the saved observations",
        )


def _canonical_model_sha256(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_scenario_revision(revision: ScenarioRevision) -> None:
    """Reject a revision whose content no longer matches its stored identity."""
    manifest = revision.manifest
    if revision.manifest_sha256 != _canonical_model_sha256(manifest):
        _raise_persisted_content_mismatch("scenario revision manifest hash mismatch")
    if revision.scenario_name != manifest.scenario.name:
        _raise_persisted_content_mismatch("scenario revision name mismatch")
    if manifest.resolved_scenario_sha256 != canonical_scenario_sha256(
        manifest.scenario
    ):
        _raise_persisted_content_mismatch("resolved scenario hash mismatch")
    if manifest.scenario.starting_portfolio != manifest.starting_portfolio:
        _raise_persisted_content_mismatch(
            "resolved scenario portfolio does not match its manifest"
        )
    historical = manifest.historical_dataset
    if (
        historical is not None
        and historical.observations_sha256
        != _historical_snapshot_sha256(historical)
    ):
        _raise_persisted_content_mismatch(
            "historical observation hash mismatch"
        )


def verify_scenario_comparison(comparison: ScenarioComparison) -> None:
    """Reject a comparison whose manifest or complete result was corrupted."""
    if comparison.manifest_sha256 != _canonical_model_sha256(
        comparison.manifest
    ):
        _raise_persisted_content_mismatch("scenario comparison manifest hash mismatch")
    if comparison.manifest.baseline_revision_id != comparison.baseline.revision_id:
        _raise_persisted_content_mismatch("scenario comparison baseline mismatch")
    alternative_ids = tuple(
        alternative.revision_id for alternative in comparison.alternatives
    )
    if alternative_ids != comparison.manifest.alternative_revision_ids:
        _raise_persisted_content_mismatch("scenario comparison alternatives mismatch")
    expected_result_sha256 = _comparison_result_sha256(
        comparison_id=comparison.id,
        baseline=comparison.baseline,
        alternatives=comparison.alternatives,
        manifest_sha256=comparison.manifest_sha256,
        manifest=comparison.manifest,
        created_at=comparison.created_at,
    )
    if comparison.result_sha256 != expected_result_sha256:
        _raise_persisted_content_mismatch("scenario comparison result hash mismatch")


def _verify_run_request(request: ScenarioComparisonRunRequest) -> None:
    for revision in request.revisions:
        verify_scenario_revision(revision)
    expected_manifest = _build_run_request(
        request.revisions,
        request.manifest.execution_policy,
    ).manifest
    if request.manifest != expected_manifest:
        _raise_persisted_content_mismatch(
            "comparison execution manifest does not match its revisions"
        )


def _historical_snapshot_sha256(
    snapshot: HistoricalDatasetSnapshot,
) -> str:
    payload = json.dumps(
        {
            "inflation_rates": snapshot.inflation_rates,
            "nominal_returns": snapshot.nominal_returns,
            "years": snapshot.years,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _comparison_result_sha256(
    *,
    comparison_id: str,
    baseline: ScenarioEvaluation,
    alternatives: tuple[ScenarioAlternativeComparison, ...],
    manifest_sha256: str,
    manifest: ScenarioComparisonManifest,
    created_at: datetime,
) -> str:
    payload = json.dumps(
        {
            "id": comparison_id,
            "baseline": baseline.model_dump(mode="json"),
            "alternatives": [
                alternative.model_dump(mode="json")
                for alternative in alternatives
            ],
            "manifest_sha256": manifest_sha256,
            "manifest": manifest.model_dump(mode="json"),
            "created_at": created_at.isoformat(),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _raise_persisted_content_mismatch(message: str) -> NoReturn:
    raise ScenarioComparisonError(
        ScenarioComparisonErrorCode.PERSISTED_CONTENT_MISMATCH,
        message,
    )


def _optional_nested_metric(
    result: SimulationResult,
    field_name: str,
    percentile: str,
) -> float | None:
    value = getattr(result, field_name, None)
    if not isinstance(value, Mapping):
        return None
    percentile_value = value.get(percentile)
    return (
        float(percentile_value)
        if isinstance(percentile_value, (float, int))
        else None
    )


def _optional_mapping_metric(
    value: Mapping[str, object] | None,
    field_name: str,
    percentile: str,
) -> float | None:
    if value is None:
        return None
    nested = value.get(field_name)
    if not isinstance(nested, Mapping):
        return None
    percentile_value = nested.get(percentile)
    return (
        float(percentile_value)
        if isinstance(percentile_value, (float, int))
        else None
    )


def _optional_delta(
    baseline: float | None,
    alternative: float | None,
) -> float | None:
    if baseline is None or alternative is None:
        return None
    return alternative - baseline


def _metric_delta(
    baseline: ScenarioOutcomeMetrics,
    alternative: ScenarioOutcomeMetrics,
) -> ScenarioMetricDelta:
    return ScenarioMetricDelta(
        success_probability=(
            alternative.success_probability - baseline.success_probability
        ),
        funded_spending_ratio_p50=_optional_delta(
            baseline.funded_spending_ratio_p50,
            alternative.funded_spending_ratio_p50,
        ),
        funded_spending_real_p50=_optional_delta(
            baseline.funded_spending_real_p50,
            alternative.funded_spending_real_p50,
        ),
        cumulative_shortfall_real_p50=_optional_delta(
            baseline.cumulative_shortfall_real_p50,
            alternative.cumulative_shortfall_real_p50,
        ),
        guardrail_reduction_real_p50=_optional_delta(
            baseline.guardrail_reduction_real_p50,
            alternative.guardrail_reduction_real_p50,
        ),
        requested_annual_spending_real=(
            alternative.requested_annual_spending_real
            - baseline.requested_annual_spending_real
        ),
        lifetime_tax_real_p50=_optional_delta(
            baseline.lifetime_tax_real_p50,
            alternative.lifetime_tax_real_p50,
        ),
        estate_value_real_p50=(
            alternative.estate_value_real_p50 - baseline.estate_value_real_p50
        ),
        estate_value_real_p10=(
            alternative.estate_value_real_p10 - baseline.estate_value_real_p10
        ),
        after_tax_estate_value_real_p50=_optional_delta(
            baseline.after_tax_estate_value_real_p50,
            alternative.after_tax_estate_value_real_p50,
        ),
        after_tax_estate_value_real_p10=_optional_delta(
            baseline.after_tax_estate_value_real_p10,
            alternative.after_tax_estate_value_real_p10,
        ),
        retirement_balance_real_p10=(
            alternative.retirement_balance_real_p10
            - baseline.retirement_balance_real_p10
        ),
    )


def _scenario_changes(
    baseline: WealthScenario,
    alternative: WealthScenario,
    classifier: ScenarioChangeClassifier,
) -> tuple[ScenarioInputChange, ...]:
    baseline_values = _flatten_json(baseline.model_dump(mode="json"))
    alternative_values = _flatten_json(alternative.model_dump(mode="json"))
    changes: list[ScenarioInputChange] = []
    for path in sorted(set(baseline_values) | set(alternative_values)):
        if path == "name":
            continue
        before = baseline_values.get(path)
        after = alternative_values.get(path)
        if before == after:
            continue
        changes.append(
            ScenarioInputChange(
                path=path,
                kind=classifier.classify(path),
                baseline_value=before,
                alternative_value=after,
            )
        )
    return tuple(changes)


def _flatten_json(
    value: JsonValue,
    *,
    prefix: str = "",
) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        flattened: dict[str, JsonValue] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_json(child, prefix=path))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            flattened.update(_flatten_json(child, prefix=path))
        if not value:
            flattened[prefix] = []
        return flattened
    return {prefix: value}


_METRIC_PREFERENCES: dict[str, int] = {
    "success_probability": 1,
    "funded_spending_ratio_p50": 1,
    "funded_spending_real_p50": 1,
    "cumulative_shortfall_real_p50": -1,
    "guardrail_reduction_real_p50": -1,
    "requested_annual_spending_real": 1,
    "lifetime_tax_real_p50": -1,
    "estate_value_real_p50": 1,
    "estate_value_real_p10": 1,
    "after_tax_estate_value_real_p50": 1,
    "after_tax_estate_value_real_p10": 1,
    "retirement_balance_real_p10": 1,
}


def _dominates(
    candidate: ScenarioOutcomeMetrics,
    other: ScenarioOutcomeMetrics,
) -> bool:
    no_worse = True
    strictly_better = False
    candidate_values = candidate.model_dump()
    other_values = other.model_dump()
    for metric, preference in _METRIC_PREFERENCES.items():
        candidate_value = candidate_values[metric]
        other_value = other_values[metric]
        if candidate_value is None or other_value is None:
            continue
        preferred_delta = preference * (float(candidate_value) - float(other_value))
        if preferred_delta < -1e-9:
            no_worse = False
            break
        if preferred_delta > 1e-9:
            strictly_better = True
    return no_worse and strictly_better


def _material_tradeoffs(
    delta: ScenarioMetricDelta,
) -> tuple[ScenarioTradeoff, ...]:
    thresholds = {
        "success_probability": 0.005,
        "funded_spending_ratio_p50": 0.005,
        "funded_spending_real_p50": 500.0,
        "cumulative_shortfall_real_p50": 500.0,
        "guardrail_reduction_real_p50": 500.0,
        "requested_annual_spending_real": 500.0,
        "lifetime_tax_real_p50": 500.0,
        "estate_value_real_p50": 1_000.0,
        "estate_value_real_p10": 1_000.0,
        "after_tax_estate_value_real_p50": 1_000.0,
        "after_tax_estate_value_real_p10": 1_000.0,
        "retirement_balance_real_p10": 1_000.0,
    }
    values = delta.model_dump()
    tradeoffs: list[ScenarioTradeoff] = []
    for metric, preference in _METRIC_PREFERENCES.items():
        value = values[metric]
        if value is None or abs(float(value)) < thresholds[metric]:
            continue
        tradeoffs.append(
            ScenarioTradeoff(
                metric=metric,
                direction=(
                    TradeoffDirection.BENEFIT
                    if preference * float(value) > 0
                    else TradeoffDirection.COST
                ),
                delta=float(value),
            )
        )
    return tuple(tradeoffs)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
