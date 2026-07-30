"""Scoped SQL adapter for durable planner jobs."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime, timezone
import json
from typing import Protocol, cast

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.planner_jobs import (
    PlannerErrorCode,
    PlannerExecutionOutput,
    PlannerJob,
    PlannerJobCreateResult,
    PlannerJobPayload,
    PlannerJobPublicError,
    PlannerJobState,
    PlannerSimulationResult,
)


_JOB_COLUMNS = """
    id, request_hash, state, request_json, manifest_json, result_json,
    error_code, error_message, cancellation_requested,
    required_working_bytes, created_at, updated_at, started_at, completed_at
"""


class _RowCountResult(Protocol):
    rowcount: int


class SqlPlannerJobRepository:
    """Persist typed jobs with atomic state transitions."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def get_by_request_hash(
        self,
        request_hash: str,
    ) -> PlannerJob | None:
        rows = await self.database.fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM planner_jobs
            WHERE deduplication_key = :request_hash
            """,
            {"request_hash": request_hash},
        )
        return _job_from_row(rows[0]) if rows else None

    async def create_job(
        self,
        *,
        job_id: str,
        request_hash: str,
        payload: PlannerJobPayload,
    ) -> PlannerJobCreateResult:
        now = _utc_now()
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO planner_jobs (
                          id, request_hash, deduplication_key,
                          state, request_json,
                          cancellation_requested, required_working_bytes,
                          created_at, updated_at
                        )
                        VALUES (
                          :id, :request_hash, :request_hash,
                          :state, :request_json,
                          FALSE, :required_working_bytes,
                          :created_at, :updated_at
                        )
                        """
                    ),
                    {
                        "id": job_id,
                        "request_hash": request_hash,
                        "state": PlannerJobState.ACCEPTED.value,
                        "request_json": payload.model_dump_json(),
                        "required_working_bytes": payload.required_working_bytes,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                await session.commit()
        except IntegrityError:
            existing = await self.get_by_request_hash(request_hash)
            if existing is None:
                raise
            return PlannerJobCreateResult(job=existing, created=False)

        created = await self.get_job(job_id)
        if created is None:  # pragma: no cover - defensive persistence invariant
            raise RuntimeError("created planner job could not be read back")
        return PlannerJobCreateResult(job=created, created=True)

    async def get_job(self, job_id: str) -> PlannerJob | None:
        rows = await self.database.fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM planner_jobs
            WHERE id = :job_id
            """,
            {"job_id": job_id},
        )
        return _job_from_row(rows[0]) if rows else None

    async def prepare_recovery(self) -> None:
        """Reset interrupted jobs without collecting the queue into memory."""
        interrupted = await self.database.fetch_all(
            """
            SELECT id
            FROM planner_jobs
            WHERE state = :running
            LIMIT 1
            """,
            {"running": PlannerJobState.RUNNING.value},
        )
        if not interrupted:
            return
        now = _utc_now()
        async with self.database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE planner_jobs
                    SET state = :accepted,
                        updated_at = :updated_at,
                        started_at = NULL
                    WHERE state = :running
                      AND cancellation_requested IS FALSE
                    """
                ),
                {
                    "accepted": PlannerJobState.ACCEPTED.value,
                    "running": PlannerJobState.RUNNING.value,
                    "updated_at": now,
                },
            )
            await session.execute(
                text(
                    """
                    UPDATE planner_jobs
                    SET state = :cancelled,
                        deduplication_key = NULL,
                        updated_at = :updated_at,
                        completed_at = :updated_at
                    WHERE state = :running
                      AND cancellation_requested IS TRUE
                    """
                ),
                {
                    "cancelled": PlannerJobState.CANCELLED.value,
                    "running": PlannerJobState.RUNNING.value,
                    "updated_at": now,
                },
            )
            await session.commit()

    async def list_accepted_jobs(
        self,
        *,
        limit: int,
        exclude_job_ids: Collection[str],
    ) -> tuple[PlannerJob, ...]:
        """Return one bounded recovery page, excluding already admitted jobs."""
        if limit < 1:
            raise ValueError("planner recovery page limit must be positive")
        excluded = tuple(exclude_job_ids)
        params: dict[str, object] = {
            "accepted": PlannerJobState.ACCEPTED.value,
            "limit": limit,
        }
        exclusion_sql = ""
        if excluded:
            placeholders = ", ".join(f":excluded_{index}" for index in range(len(excluded)))
            exclusion_sql = f"AND id NOT IN ({placeholders})"
            params.update({f"excluded_{index}": job_id for index, job_id in enumerate(excluded)})
        rows = await self.database.fetch_all(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM planner_jobs
            WHERE state = :accepted
              {exclusion_sql}
            ORDER BY created_at, id
            LIMIT :limit
            """,
            params,
        )
        return tuple(_job_from_row(row) for row in rows)

    async def mark_running(self, job_id: str) -> PlannerJob | None:
        now = _utc_now()
        async with self.database.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    UPDATE planner_jobs
                    SET state = :running,
                        started_at = :updated_at,
                        updated_at = :updated_at
                    WHERE id = :job_id
                      AND state = :accepted
                      AND cancellation_requested IS FALSE
                    """
                ),
                {
                    "job_id": job_id,
                    "accepted": PlannerJobState.ACCEPTED.value,
                    "running": PlannerJobState.RUNNING.value,
                    "updated_at": now,
                },
            )
            await session.commit()
        if cast(_RowCountResult, result).rowcount != 1:
            return None
        return await self.get_job(job_id)

    async def mark_succeeded(
        self,
        job_id: str,
        output: PlannerExecutionOutput,
    ) -> PlannerJob | None:
        now = _utc_now()
        async with self.database.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    UPDATE planner_jobs
                    SET state = :succeeded,
                        result_json = :result_json,
                        manifest_json = :manifest_json,
                        updated_at = :updated_at,
                        completed_at = :updated_at
                    WHERE id = :job_id
                      AND state = :running
                      AND cancellation_requested IS FALSE
                    """
                ),
                {
                    "job_id": job_id,
                    "running": PlannerJobState.RUNNING.value,
                    "succeeded": PlannerJobState.SUCCEEDED.value,
                    "result_json": output.result.model_dump_json(),
                    "manifest_json": json.dumps(
                        output.manifest,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "updated_at": now,
                },
            )
            if cast(_RowCountResult, result).rowcount != 1:
                await session.execute(
                    text(
                        """
                        UPDATE planner_jobs
                        SET state = :cancelled,
                            deduplication_key = NULL,
                            result_json = NULL,
                            manifest_json = NULL,
                            updated_at = :updated_at,
                            completed_at = :updated_at
                        WHERE id = :job_id
                          AND state = :running
                          AND cancellation_requested IS TRUE
                        """
                    ),
                    {
                        "job_id": job_id,
                        "running": PlannerJobState.RUNNING.value,
                        "cancelled": PlannerJobState.CANCELLED.value,
                        "updated_at": now,
                    },
                )
            await session.commit()
        return await self.get_job(job_id)

    async def mark_failed(
        self,
        job_id: str,
        error: PlannerJobPublicError,
    ) -> PlannerJob | None:
        now = _utc_now()
        async with self.database.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    UPDATE planner_jobs
                    SET state = :failed,
                        deduplication_key = NULL,
                        error_code = :error_code,
                        error_message = :error_message,
                        updated_at = :updated_at,
                        completed_at = :updated_at
                    WHERE id = :job_id
                      AND state IN (:accepted, :running)
                      AND cancellation_requested IS FALSE
                    """
                ),
                {
                    "job_id": job_id,
                    "accepted": PlannerJobState.ACCEPTED.value,
                    "running": PlannerJobState.RUNNING.value,
                    "failed": PlannerJobState.FAILED.value,
                    "error_code": error.code.value,
                    "error_message": error.message,
                    "updated_at": now,
                },
            )
            if cast(_RowCountResult, result).rowcount != 1:
                await session.execute(
                    text(
                        """
                        UPDATE planner_jobs
                        SET state = :cancelled,
                            deduplication_key = NULL,
                            updated_at = :updated_at,
                            completed_at = :updated_at
                        WHERE id = :job_id
                          AND state IN (:accepted, :running)
                          AND cancellation_requested IS TRUE
                        """
                    ),
                    {
                        "job_id": job_id,
                        "accepted": PlannerJobState.ACCEPTED.value,
                        "running": PlannerJobState.RUNNING.value,
                        "cancelled": PlannerJobState.CANCELLED.value,
                        "updated_at": now,
                    },
                )
            await session.commit()
        return await self.get_job(job_id)

    async def request_cancellation(self, job_id: str) -> PlannerJob | None:
        now = _utc_now()
        async with self.database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE planner_jobs
                    SET state = :cancelled,
                        deduplication_key = NULL,
                        cancellation_requested = TRUE,
                        updated_at = :updated_at,
                        completed_at = :updated_at
                    WHERE id = :job_id
                      AND state = :accepted
                    """
                ),
                {
                    "job_id": job_id,
                    "accepted": PlannerJobState.ACCEPTED.value,
                    "cancelled": PlannerJobState.CANCELLED.value,
                    "updated_at": now,
                },
            )
            await session.execute(
                text(
                    """
                    UPDATE planner_jobs
                    SET cancellation_requested = TRUE,
                        updated_at = :updated_at
                    WHERE id = :job_id
                      AND state = :running
                    """
                ),
                {
                    "job_id": job_id,
                    "running": PlannerJobState.RUNNING.value,
                    "updated_at": now,
                },
            )
            await session.commit()
        return await self.get_job(job_id)


def _job_from_row(row: Mapping[str, object]) -> PlannerJob:
    error_code = row["error_code"]
    error = (
        PlannerJobPublicError(
            code=PlannerErrorCode(str(error_code)),
            message=str(row["error_message"] or ""),
        )
        if error_code is not None
        else None
    )
    result_json = row["result_json"]
    manifest_json = row["manifest_json"]
    return PlannerJob(
        id=str(row["id"]),
        request_hash=str(row["request_hash"]),
        state=PlannerJobState(str(row["state"])),
        payload=PlannerJobPayload.model_validate_json(str(row["request_json"])),
        cancellation_requested=bool(row["cancellation_requested"]),
        result=(
            PlannerSimulationResult.model_validate_json(str(result_json))
            if result_json is not None
            else None
        ),
        manifest=(
            cast(dict[str, object], json.loads(str(manifest_json)))
            if manifest_json is not None
            else None
        ),
        error=error,
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        started_at=(
            datetime.fromisoformat(str(row["started_at"]))
            if row["started_at"] is not None
            else None
        ),
        completed_at=(
            datetime.fromisoformat(str(row["completed_at"]))
            if row["completed_at"] is not None
            else None
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
