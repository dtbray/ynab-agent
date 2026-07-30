"""Bounded process worker for durable retirement simulations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from uuid import uuid4

from ynab_agent.planning.paths import ResourceLimitError
from ynab_agent.services.planner_jobs import (
    PlannerErrorCode,
    PlannerExecutionOutput,
    PlannerJobPublicError,
    PlannerJobRepository,
    PlannerQueueSaturatedError,
    PlannerResourceLimitError,
    run_planner_job,
)


PlannerRunner = Callable[[str], str]


@dataclass
class _Admission:
    required_working_bytes: int
    active: bool = False
    running: bool = False


class PlannerJobWorker:
    """Coordinate durable jobs across bounded process and memory capacity."""

    def __init__(
        self,
        repository: PlannerJobRepository,
        *,
        max_workers: int,
        max_pending_jobs: int,
        maximum_total_working_bytes: int,
        executor: Executor | None = None,
        runner: PlannerRunner = run_planner_job,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("planner max_workers must be positive")
        if max_pending_jobs < 0:
            raise ValueError("planner max_pending_jobs must not be negative")
        if maximum_total_working_bytes <= 0:
            raise ValueError("planner maximum_total_working_bytes must be positive")
        self.repository = repository
        self.max_workers = max_workers
        self.max_outstanding_jobs = max_workers + max_pending_jobs
        self.maximum_total_working_bytes = maximum_total_working_bytes
        self.runner = runner
        self._executor = executor or ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context("spawn"),
        )
        self._owns_executor = executor is None
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._admissions: dict[str, _Admission] = {}
        self._admission_lock = asyncio.Lock()
        self._recovery_lock = asyncio.Lock()
        self._memory_condition = asyncio.Condition()
        self._reserved_working_bytes = 0
        self._consumers: tuple[asyncio.Task[None], ...] = ()
        self._bounded_tasks: set[asyncio.Task[str]] = set()
        self._started = False

    @property
    def outstanding_jobs(self) -> int:
        return len(self._admissions)

    @property
    def reserved_working_bytes(self) -> int:
        return self._reserved_working_bytes

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            await self.repository.prepare_recovery()
            await self._refill_recovery_queue()
            self._consumers = tuple(
                asyncio.create_task(
                    self._consume(),
                    name=f"planner-worker-{index}",
                )
                for index in range(self.max_workers)
            )
        except BaseException:
            self._started = False
            raise

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        for consumer in self._consumers:
            consumer.cancel()
        await asyncio.gather(*self._consumers, return_exceptions=True)
        self._consumers = ()
        if self._bounded_tasks:
            await asyncio.gather(
                *tuple(self._bounded_tasks),
                return_exceptions=True,
            )
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def reserve(
        self,
        job_id: str,
        required_working_bytes: int,
    ) -> None:
        if required_working_bytes > self.maximum_total_working_bytes:
            raise PlannerResourceLimitError()
        async with self._admission_lock:
            if job_id in self._admissions:
                return
            if len(self._admissions) >= self.max_outstanding_jobs:
                raise PlannerQueueSaturatedError()
            self._admissions[job_id] = _Admission(required_working_bytes=required_working_bytes)

    async def activate(self, job_id: str) -> None:
        async with self._admission_lock:
            admission = self._admissions.get(job_id)
            if admission is None:
                raise RuntimeError("planner job has no reserved worker admission")
            if admission.active:
                return
            admission.active = True
        self._queue.put_nowait(job_id)

    async def release(self, job_id: str) -> None:
        released = False
        async with self._admission_lock:
            admission = self._admissions.get(job_id)
            if admission is None or admission.running:
                return
            del self._admissions[job_id]
            released = True
        if released:
            await self._refill_recovery_queue()

    async def run_bounded(
        self,
        runner: PlannerRunner,
        payload_json: str,
        required_working_bytes: int,
    ) -> str:
        """Run non-durable planner work under the shared queue and memory caps."""
        admission_id = f"comparison:{uuid4()}"
        await self.reserve(admission_id, required_working_bytes)
        admission = await self._get_admission(admission_id)
        if admission is None:  # pragma: no cover - reserve invariant
            raise RuntimeError("bounded planner work lost its admission")
        admission.active = True

        async def execute() -> str:
            memory_acquired = False
            try:
                await self._acquire_memory(required_working_bytes)
                memory_acquired = True
                admission.running = True
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    self._executor,
                    runner,
                    payload_json,
                )
            finally:
                if memory_acquired:
                    await self._release_memory(required_working_bytes)
                async with self._admission_lock:
                    self._admissions.pop(admission_id, None)
                await self._refill_recovery_queue()

        task = asyncio.create_task(
            execute(),
            name=f"planner-bounded-{admission_id}",
        )
        self._bounded_tasks.add(task)
        task.add_done_callback(self._finish_bounded_task)
        # Client cancellation must not release admission while process work runs.
        return await asyncio.shield(task)

    def _finish_bounded_task(self, task: asyncio.Task[str]) -> None:
        """Retire shielded work and consume failures after caller cancellation."""
        self._bounded_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _consume(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                admission = await self._get_admission(job_id)
                if admission is None:
                    continue
                await self._acquire_memory(admission.required_working_bytes)
                try:
                    job = await self.repository.mark_running(job_id)
                    if job is None:
                        continue
                    admission.running = True
                    loop = asyncio.get_running_loop()
                    output_json = await loop.run_in_executor(
                        self._executor,
                        self.runner,
                        job.payload.model_dump_json(),
                    )
                    output = PlannerExecutionOutput.model_validate_json(output_json)
                    await self.repository.mark_succeeded(job_id, output)
                except asyncio.CancelledError:
                    raise
                except ResourceLimitError:
                    await self._record_failure(
                        job_id,
                        PlannerErrorCode.RESOURCE_LIMIT,
                        "planner execution exceeded its resource policy",
                    )
                except ValueError:
                    await self._record_failure(
                        job_id,
                        PlannerErrorCode.INVALID_JOB,
                        "planner execution rejected the persisted job input",
                    )
                except Exception:
                    await self._record_failure(
                        job_id,
                        PlannerErrorCode.EXECUTION_FAILED,
                        "planner execution failed",
                    )
                finally:
                    await self._release_memory(admission.required_working_bytes)
                    async with self._admission_lock:
                        self._admissions.pop(job_id, None)
                    await self._refill_recovery_queue()
            finally:
                self._queue.task_done()

    async def _refill_recovery_queue(self) -> None:
        """Page durable accepted work into the bounded in-memory queue."""
        if not self._started:
            return
        async with self._recovery_lock:
            while self._started:
                async with self._admission_lock:
                    available = self.max_outstanding_jobs - len(self._admissions)
                    excluded_job_ids = tuple(self._admissions)
                if available <= 0:
                    return
                jobs = await self.repository.list_accepted_jobs(
                    limit=available,
                    exclude_job_ids=excluded_job_ids,
                )
                if not jobs:
                    return

                failed_jobs: list[str] = []
                queued_jobs: list[str] = []
                async with self._admission_lock:
                    for job in jobs:
                        if len(self._admissions) >= self.max_outstanding_jobs:
                            break
                        if job.id in self._admissions:
                            continue
                        required = job.payload.required_working_bytes
                        if required > self.maximum_total_working_bytes:
                            failed_jobs.append(job.id)
                            continue
                        self._admissions[job.id] = _Admission(
                            required_working_bytes=required,
                            active=True,
                        )
                        queued_jobs.append(job.id)

                for job_id in failed_jobs:
                    await self.repository.mark_failed(
                        job_id,
                        PlannerJobPublicError(
                            code=PlannerErrorCode.RESOURCE_LIMIT,
                            message=("job exceeds the worker resource policy active after restart"),
                        ),
                    )
                for job_id in queued_jobs:
                    self._queue.put_nowait(job_id)

                if not failed_jobs:
                    return

    async def _get_admission(self, job_id: str) -> _Admission | None:
        async with self._admission_lock:
            return self._admissions.get(job_id)

    async def _acquire_memory(self, required_working_bytes: int) -> None:
        async with self._memory_condition:
            await self._memory_condition.wait_for(
                lambda: (
                    self._reserved_working_bytes + required_working_bytes
                    <= self.maximum_total_working_bytes
                )
            )
            self._reserved_working_bytes += required_working_bytes

    async def _release_memory(self, released_working_bytes: int) -> None:
        async with self._memory_condition:
            self._reserved_working_bytes -= released_working_bytes
            if self._reserved_working_bytes < 0:
                self._reserved_working_bytes = 0
                raise RuntimeError("planner worker memory reservation underflow")
            self._memory_condition.notify_all()

    async def _record_failure(
        self,
        job_id: str,
        code: PlannerErrorCode,
        message: str,
    ) -> None:
        await self.repository.mark_failed(
            job_id,
            PlannerJobPublicError(code=code, message=message),
        )
