from __future__ import annotations

import asyncio
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pytest

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.planner_jobs import SqlPlannerJobRepository
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.services.planner_jobs import (
    PlannerErrorCode,
    PlannerExecutionOutput,
    PlannerExecutionPolicy,
    PlannerEngineIdentity,
    PlannerJob,
    PlannerJobPayload,
    PlannerJobState,
    PlannerQueueSaturatedError,
    PlannerResourceLimitError,
    PlannerSimulationResult,
)
from ynab_agent.workers.planner import PlannerJobWorker


class RecordingPlannerJobRepository(SqlPlannerJobRepository):
    def __init__(self, database: DatabaseManager) -> None:
        super().__init__(database)
        self.recovery_limits: list[int] = []

    async def list_accepted_jobs(
        self,
        *,
        limit: int,
        exclude_job_ids: Collection[str],
    ) -> tuple[PlannerJob, ...]:
        self.recovery_limits.append(limit)
        return await super().list_accepted_jobs(
            limit=limit,
            exclude_job_ids=exclude_job_ids,
        )


def _payload(
    *,
    name: str,
    required_working_bytes: int = 80,
) -> PlannerJobPayload:
    return PlannerJobPayload(
        engine_identity=PlannerEngineIdentity.current(),
        scenario=WealthScenario.model_validate(
            {
                "name": name,
                "current_age": 40,
                "retirement_age": 60,
                "end_age": 95,
                "starting_portfolio": 100_000,
                "annual_spending": 40_000,
                "trials": 100,
            }
        ),
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
        required_working_bytes=required_working_bytes,
    )


def _successful_output(payload_json: str) -> str:
    payload = PlannerJobPayload.model_validate_json(payload_json)
    manifest: dict[str, object] = {"schema_version": 1}
    return PlannerExecutionOutput(
        result=PlannerSimulationResult(
            scenario=payload.scenario.name,
            starting_portfolio=payload.starting_portfolio,
            trials=payload.scenario.trials,
            seed=payload.scenario.seed,
            success_rate=1,
            success_rate_ci_95={"low": 0.95, "high": 1},
            depleted_trials=0,
            median_depletion_age=None,
            retirement_balance_real={"p10": 1, "p50": 2, "p90": 3},
            ending_balance_real={"p10": 1, "p50": 2, "p90": 3},
            annual_balance_real=[],
            assumptions={"current_age": payload.scenario.current_age},
            engine={"schema_version": 1},
            reproducibility=manifest,
        ),
        manifest=manifest,
    ).model_dump_json()


async def _wait_for_state(
    repository: SqlPlannerJobRepository,
    job_id: str,
    state: PlannerJobState,
) -> None:
    for _ in range(200):
        job = await repository.get_job(job_id)
        if job is not None and job.state is state:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {state}")


@pytest.mark.asyncio
async def test_worker_restart_recovers_running_job(tmp_path: Path) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'worker-restart.db'}")
    await database.initialize()
    repository = RecordingPlannerJobRepository(database)
    payload = _payload(name="recovered")
    created = await repository.create_job(
        job_id="00000000-0000-4000-8000-000000000010",
        request_hash="1" * 64,
        payload=payload,
    )
    second = await repository.create_job(
        job_id="00000000-0000-4000-8000-000000000015",
        request_hash="6" * 64,
        payload=_payload(name="recovered beyond queue"),
    )
    assert await repository.mark_running(created.job.id) is not None
    executor = ThreadPoolExecutor(max_workers=1)
    worker = PlannerJobWorker(
        repository,
        max_workers=1,
        max_pending_jobs=0,
        maximum_total_working_bytes=100,
        executor=executor,
        runner=_successful_output,
    )
    try:
        await worker.start()
        await _wait_for_state(
            repository,
            created.job.id,
            PlannerJobState.SUCCEEDED,
        )
        await _wait_for_state(
            repository,
            second.job.id,
            PlannerJobState.SUCCEEDED,
        )
        recovered = await repository.get_job(created.job.id)
        recovered_second = await repository.get_job(second.job.id)
        assert recovered is not None
        assert recovered.result is not None
        assert recovered.result.scenario == "recovered"
        assert recovered_second is not None
        assert recovered_second.result is not None
        assert recovered_second.result.scenario == "recovered beyond queue"
        assert len(repository.recovery_limits) >= 2
        assert set(repository.recovery_limits) == {1}
    finally:
        await worker.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        await database.close()


@pytest.mark.asyncio
async def test_default_worker_executes_simulation_in_process_pool(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'worker-process.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    created = await repository.create_job(
        job_id="00000000-0000-4000-8000-000000000014",
        request_hash="5" * 64,
        payload=_payload(
            name="process pool",
            required_working_bytes=100_000,
        ),
    )
    worker = PlannerJobWorker(
        repository,
        max_workers=1,
        max_pending_jobs=1,
        maximum_total_working_bytes=16 * 1024 * 1024,
    )
    try:
        await worker.start()
        await _wait_for_state(
            repository,
            created.job.id,
            PlannerJobState.SUCCEEDED,
        )
        completed = await repository.get_job(created.job.id)
        assert completed is not None
        assert completed.result is not None
        assert completed.result.scenario == "process pool"
        assert completed.manifest is not None
        assert completed.manifest["engine_version"] == "wealth_simulation_v8"
    finally:
        await worker.stop()
        await database.close()


@pytest.mark.asyncio
async def test_worker_enforces_concurrency_and_memory_together(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'worker-memory.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    first = await repository.create_job(
        job_id="00000000-0000-4000-8000-000000000011",
        request_hash="2" * 64,
        payload=_payload(name="first"),
    )
    second = await repository.create_job(
        job_id="00000000-0000-4000-8000-000000000012",
        request_hash="3" * 64,
        payload=_payload(name="second"),
    )
    first_entered = Event()
    release_first = Event()
    entered_names: list[str] = []
    entered_lock = Lock()

    def blocking_runner(payload_json: str) -> str:
        payload = PlannerJobPayload.model_validate_json(payload_json)
        with entered_lock:
            entered_names.append(payload.scenario.name)
        if payload.scenario.name == "first":
            first_entered.set()
            if not release_first.wait(timeout=5):
                raise RuntimeError("test runner timed out")
        return _successful_output(payload_json)

    executor = ThreadPoolExecutor(max_workers=2)
    worker = PlannerJobWorker(
        repository,
        max_workers=2,
        max_pending_jobs=2,
        maximum_total_working_bytes=100,
        executor=executor,
        runner=blocking_runner,
    )
    try:
        await worker.start()
        assert await asyncio.to_thread(first_entered.wait, 5)
        await asyncio.sleep(0.05)
        assert worker.reserved_working_bytes == 80
        assert entered_names == ["first"]

        second_before_release = await repository.get_job(second.job.id)
        assert second_before_release is not None
        assert second_before_release.state is PlannerJobState.ACCEPTED

        release_first.set()
        await _wait_for_state(
            repository,
            first.job.id,
            PlannerJobState.SUCCEEDED,
        )
        await _wait_for_state(
            repository,
            second.job.id,
            PlannerJobState.SUCCEEDED,
        )
        for _ in range(100):
            if worker.reserved_working_bytes == 0:
                break
            await asyncio.sleep(0.01)
        assert entered_names == ["first", "second"]
        assert worker.reserved_working_bytes == 0
    finally:
        release_first.set()
        await worker.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        await database.close()


@pytest.mark.asyncio
async def test_worker_admission_has_explicit_saturation_and_resource_errors(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'worker-admission.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    executor = ThreadPoolExecutor(max_workers=1)
    worker = PlannerJobWorker(
        repository,
        max_workers=1,
        max_pending_jobs=0,
        maximum_total_working_bytes=100,
        executor=executor,
        runner=_successful_output,
    )
    try:
        await worker.reserve("first", 80)
        with pytest.raises(PlannerQueueSaturatedError) as saturated:
            await worker.reserve("second", 80)
        assert saturated.value.code is PlannerErrorCode.QUEUE_SATURATED

        await worker.release("first")
        with pytest.raises(PlannerResourceLimitError) as resource:
            await worker.reserve("too-large", 101)
        assert resource.value.code is PlannerErrorCode.RESOURCE_LIMIT
    finally:
        await worker.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        await database.close()


@pytest.mark.asyncio
async def test_bounded_comparison_shares_worker_queue_and_memory_admission(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(
        f"sqlite+aiosqlite:///{tmp_path / 'comparison-admission.db'}"
    )
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    entered = Event()
    release = Event()

    def blocking_runner(payload_json: str) -> str:
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test runner timed out")
        return payload_json

    executor = ThreadPoolExecutor(max_workers=1)
    worker = PlannerJobWorker(
        repository,
        max_workers=1,
        max_pending_jobs=0,
        maximum_total_working_bytes=100,
        executor=executor,
    )
    try:
        comparison = asyncio.create_task(
            worker.run_bounded(blocking_runner, "comparison", 80)
        )
        assert await asyncio.to_thread(entered.wait, 5)
        assert worker.outstanding_jobs == 1
        assert worker.reserved_working_bytes == 80
        with pytest.raises(PlannerQueueSaturatedError):
            await worker.reserve("planner-job", 20)

        release.set()
        assert await comparison == "comparison"
        assert worker.outstanding_jobs == 0
        assert worker.reserved_working_bytes == 0
    finally:
        release.set()
        await worker.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        await database.close()


@pytest.mark.asyncio
async def test_cancelled_bounded_caller_keeps_admission_until_runner_finishes(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(
        f"sqlite+aiosqlite:///{tmp_path / 'comparison-cancellation.db'}"
    )
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    entered = Event()
    release = Event()

    def failing_runner(payload_json: str) -> str:
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test runner timed out")
        raise RuntimeError("expected detached runner failure")

    executor = ThreadPoolExecutor(max_workers=1)
    worker = PlannerJobWorker(
        repository,
        max_workers=1,
        max_pending_jobs=0,
        maximum_total_working_bytes=100,
        executor=executor,
    )
    try:
        caller = asyncio.create_task(
            worker.run_bounded(failing_runner, "comparison", 80)
        )
        assert await asyncio.to_thread(entered.wait, 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert worker.outstanding_jobs == 1
        assert worker.reserved_working_bytes == 80

        release.set()
        for _ in range(100):
            if worker.outstanding_jobs == 0:
                break
            await asyncio.sleep(0.01)
        assert worker.outstanding_jobs == 0
        assert worker.reserved_working_bytes == 0
        assert not worker._bounded_tasks
    finally:
        release.set()
        await worker.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        await database.close()


@pytest.mark.asyncio
async def test_worker_failure_is_stable_and_does_not_leak_local_paths(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'worker-failure.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    created = await repository.create_job(
        job_id="00000000-0000-4000-8000-000000000013",
        request_hash="4" * 64,
        payload=_payload(name="failure"),
    )

    def fail_with_path(payload_json: str) -> str:
        raise RuntimeError("/private/server/dataset.csv")

    executor = ThreadPoolExecutor(max_workers=1)
    worker = PlannerJobWorker(
        repository,
        max_workers=1,
        max_pending_jobs=1,
        maximum_total_working_bytes=100,
        executor=executor,
        runner=fail_with_path,
    )
    try:
        await worker.start()
        await _wait_for_state(
            repository,
            created.job.id,
            PlannerJobState.FAILED,
        )
        failed = await repository.get_job(created.job.id)
        assert failed is not None
        assert failed.error is not None
        assert failed.error.code is PlannerErrorCode.EXECUTION_FAILED
        assert failed.error.message == "planner execution failed"
        assert "/private/" not in failed.error.message
    finally:
        await worker.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        await database.close()


@pytest.mark.asyncio
async def test_running_cancellation_discards_completed_cpu_result(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'worker-cancel-running.db'}")
    await database.initialize()
    repository = SqlPlannerJobRepository(database)
    created = await repository.create_job(
        job_id="00000000-0000-4000-8000-000000000016",
        request_hash="7" * 64,
        payload=_payload(name="cancel running"),
    )
    entered = Event()
    release = Event()

    def blocking_runner(payload_json: str) -> str:
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("/private/server/timeout")
        return _successful_output(payload_json)

    executor = ThreadPoolExecutor(max_workers=1)
    worker = PlannerJobWorker(
        repository,
        max_workers=1,
        max_pending_jobs=1,
        maximum_total_working_bytes=100,
        executor=executor,
        runner=blocking_runner,
    )
    try:
        await worker.start()
        assert await asyncio.to_thread(entered.wait, 5)
        cancellation = await repository.request_cancellation(created.job.id)
        assert cancellation is not None
        assert cancellation.state is PlannerJobState.RUNNING
        assert cancellation.cancellation_requested is True

        release.set()
        await _wait_for_state(
            repository,
            created.job.id,
            PlannerJobState.CANCELLED,
        )
        cancelled = await repository.get_job(created.job.id)
        assert cancelled is not None
        assert cancelled.result is None
        assert cancelled.manifest is None
        assert cancelled.error is None
    finally:
        release.set()
        await worker.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        await database.close()
