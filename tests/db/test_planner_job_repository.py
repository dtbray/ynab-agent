from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.planner_jobs import SqlPlannerJobRepository
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.services.planner_jobs import (
    PlannerExecutionOutput,
    PlannerExecutionPolicy,
    PlannerEngineIdentity,
    PlannerErrorCode,
    PlannerJobPayload,
    PlannerJobPublicError,
    PlannerJobState,
    PlannerSimulationResult,
    canonical_job_request_hash,
)


def _payload() -> PlannerJobPayload:
    scenario = WealthScenario.model_validate(
        {
            "name": "persisted job",
            "current_age": 40,
            "retirement_age": 60,
            "end_age": 95,
            "starting_portfolio": 100_000,
            "annual_spending": 40_000,
            "trials": 100,
        }
    )
    return PlannerJobPayload(
        engine_identity=PlannerEngineIdentity.current(),
        scenario=scenario,
        starting_portfolio=100_000,
        valuation_provenance=ValuationProvenance(
            source="explicit_scenario_input",
            source_sha256="a" * 64,
        ),
        execution_policy=PlannerExecutionPolicy(
            maximum_working_bytes=16 * 1024 * 1024,
            in_memory_path_bytes=16 * 1024 * 1024,
            maximum_temporary_bytes=16 * 1024 * 1024,
            batch_size=100,
        ),
        required_working_bytes=100_000,
    )


def _output() -> PlannerExecutionOutput:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "scenario": {"sha256": "b" * 64},
    }
    return PlannerExecutionOutput(
        result=PlannerSimulationResult(
            scenario="persisted job",
            starting_portfolio=100_000,
            trials=100,
            seed=42,
            success_rate=0.9,
            success_rate_ci_95={"low": 0.8, "high": 0.95},
            depleted_trials=10,
            median_depletion_age=90,
            retirement_balance_real={"p10": 1.0, "p50": 2.0, "p90": 3.0},
            ending_balance_real={"p10": 1.0, "p50": 2.0, "p90": 3.0},
            annual_balance_real=[{"age": 40, "p50": 100_000.0}],
            assumptions={"current_age": 40},
            engine={"schema_version": 1},
            reproducibility=manifest,
        ),
        manifest=manifest,
    )


@pytest.mark.asyncio
async def test_repository_persists_idempotent_request_state_manifest_and_result(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'planner-jobs.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    payload = _payload()
    request_hash = canonical_job_request_hash(payload)
    try:
        first = await repository.create_job(
            job_id="00000000-0000-4000-8000-000000000001",
            request_hash=request_hash,
            payload=payload,
        )
        duplicate = await repository.create_job(
            job_id="00000000-0000-4000-8000-000000000002",
            request_hash=request_hash,
            payload=payload,
        )

        assert first.created is True
        assert duplicate.created is False
        assert duplicate.job.id == first.job.id
        running = await repository.mark_running(first.job.id)
        assert running is not None
        assert running.state is PlannerJobState.RUNNING

        completed = await repository.mark_succeeded(first.job.id, _output())
        assert completed is not None
        assert completed.state is PlannerJobState.SUCCEEDED
        assert completed.result is not None
        assert completed.result.success_rate == 0.9
        assert completed.manifest == _output().manifest
        assert completed.started_at is not None
        assert completed.completed_at is not None
        assert completed.created_at.tzinfo is not None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_repository_recovers_running_jobs_and_honors_cancellation(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'planner-recovery.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    payload = _payload()
    try:
        created = await repository.create_job(
            job_id="00000000-0000-4000-8000-000000000003",
            request_hash="c" * 64,
            payload=payload,
        )
        assert await repository.mark_running(created.job.id) is not None

        await repository.prepare_recovery()
        recovered = await repository.list_accepted_jobs(
            limit=1,
            exclude_job_ids=(),
        )

        assert [job.id for job in recovered] == [created.job.id]
        assert recovered[0].state is PlannerJobState.ACCEPTED
        assert recovered[0].started_at is None

        rerun = await repository.mark_running(created.job.id)
        assert rerun is not None
        cancellation = await repository.request_cancellation(created.job.id)
        assert cancellation is not None
        assert cancellation.state is PlannerJobState.RUNNING
        assert cancellation.cancellation_requested is True

        discarded = await repository.mark_succeeded(
            created.job.id,
            _output(),
        )
        assert discarded is not None
        assert discarded.state is PlannerJobState.CANCELLED
        assert discarded.result is None
        assert discarded.manifest is None
        assert discarded.completed_at is not None
        assert discarded.completed_at <= datetime.now(timezone.utc)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_terminal_jobs_release_deduplication_for_safe_retries(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'planner-retries.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    payload = _payload()
    request_hash = canonical_job_request_hash(payload)
    try:
        failed = await repository.create_job(
            job_id="00000000-0000-4000-8000-000000000031",
            request_hash=request_hash,
            payload=payload,
        )
        await repository.mark_failed(
            failed.job.id,
            PlannerJobPublicError(
                code=PlannerErrorCode.EXECUTION_FAILED,
                message="planner execution failed",
            ),
        )

        retry_after_failure = await repository.create_job(
            job_id="00000000-0000-4000-8000-000000000032",
            request_hash=request_hash,
            payload=payload,
        )
        concurrent_duplicate = await repository.create_job(
            job_id="00000000-0000-4000-8000-000000000033",
            request_hash=request_hash,
            payload=payload,
        )

        assert retry_after_failure.created is True
        assert retry_after_failure.job.id != failed.job.id
        assert concurrent_duplicate.created is False
        assert concurrent_duplicate.job.id == retry_after_failure.job.id

        cancelled = await repository.request_cancellation(retry_after_failure.job.id)
        assert cancelled is not None
        assert cancelled.state is PlannerJobState.CANCELLED
        retry_after_cancellation = await repository.create_job(
            job_id="00000000-0000-4000-8000-000000000034",
            request_hash=request_hash,
            payload=payload,
        )
        assert retry_after_cancellation.created is True
        assert retry_after_cancellation.job.request_hash == request_hash
    finally:
        await database.close()
