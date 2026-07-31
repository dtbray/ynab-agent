"""Bounded durable worker for continuous-calibration runs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from datetime import datetime, timezone
import json
from multiprocessing import get_context

from ynab_agent.services.calibration import (
    CalibrationRepository,
    CalibrationRun,
    CalibrationRunState,
)
from ynab_agent.services.calibration_execution import (
    CalibrationResourceLimitError,
    run_calibration_snapshot,
)
from ynab_agent.services.calibration_models import canonical_content_sha256
from ynab_agent.services.calibration_attribution import AttributionReport


CalibrationRunner = Callable[[str], str]


class CalibrationWorker:
    """Execute accepted runs with restart-safe atomic state transitions."""

    def __init__(
        self,
        repository: CalibrationRepository,
        *,
        max_workers: int = 1,
        poll_interval_seconds: float = 30,
        executor: Executor | None = None,
        runner: CalibrationRunner = run_calibration_snapshot,
    ) -> None:
        if max_workers < 1:
            raise ValueError("calibration max_workers must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("calibration poll interval must be positive")
        self.repository = repository
        self.poll_interval_seconds = poll_interval_seconds
        self.runner = runner
        self._executor = executor or ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context("spawn"),
        )
        self._owns_executor = executor is None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            await self.repository.prepare_run_recovery()
            self._task = asyncio.create_task(
                self._serve(),
                name="calibration-worker",
            )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def run_once(self) -> int:
        accepted = await self.repository.list_runs(
            state=CalibrationRunState.ACCEPTED,
            limit=100,
        )
        completed = 0
        for run in accepted:
            if await self._execute(run):
                completed += 1
        return completed

    async def _serve(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self.poll_interval_seconds)

    async def _execute(self, run: CalibrationRun) -> bool:
        claimed = await self.repository.mark_run_running(
            run.id,
            datetime.now(timezone.utc),
        )
        if claimed is None or claimed.state is not CalibrationRunState.RUNNING:
            return False
        snapshot = await self.repository.get_snapshot(run.snapshot_id)
        if snapshot is None:
            await self.repository.mark_run_failed(
                run.id,
                error_code="snapshot_not_found",
                completed_at=datetime.now(timezone.utc),
            )
            return True
        try:
            loop = asyncio.get_running_loop()
            result_json = await loop.run_in_executor(
                self._executor,
                self.runner,
                snapshot.model_dump_json(),
            )
            result = json.loads(result_json)
            if not isinstance(result, dict):
                raise ValueError("calibration runner returned a non-object")
            attribution_value = result.get("attribution")
            if not isinstance(attribution_value, dict):
                raise ValueError("calibration runner omitted its attribution report")
            completed_at = datetime.now(timezone.utc)
            await self.repository.store_attribution(
                run.id,
                AttributionReport.model_validate(attribution_value),
                created_at=completed_at,
            )
            await self.repository.mark_run_succeeded(
                run.id,
                result=result,
                result_sha256=canonical_content_sha256(result),
                completed_at=completed_at,
            )
        except CalibrationResourceLimitError:
            await self.repository.mark_run_failed(
                run.id,
                error_code="resource_limit",
                completed_at=datetime.now(timezone.utc),
            )
        except ValueError:
            await self.repository.mark_run_failed(
                run.id,
                error_code="invalid_snapshot",
                completed_at=datetime.now(timezone.utc),
            )
        except Exception:
            await self.repository.mark_run_failed(
                run.id,
                error_code="execution_failed",
                completed_at=datetime.now(timezone.utc),
            )
        return True
