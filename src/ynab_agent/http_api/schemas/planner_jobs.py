"""Public schemas for durable planner-job operations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ynab_agent.planning.stress import NamedStressName
from ynab_agent.services.planner_jobs import (
    PlannerJob,
    PlannerJobState,
    PlannerSimulationResult,
)

from .wealth import ScenarioValidationRequest


class PlannerJobSubmitRequest(BaseModel):
    """HTTP-safe submission with a registry ID, never a filesystem path."""

    model_config = ConfigDict(extra="forbid")

    scenario: ScenarioValidationRequest
    historical_dataset_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    named_stress: NamedStressName | None = None


class PlannerJobErrorRead(BaseModel):
    code: str
    message: str


class PlannerJobStatusResponse(BaseModel):
    """Stable public status without persisted request internals."""

    job_id: str
    request_hash: str
    state: PlannerJobState
    cancellation_requested: bool
    error: PlannerJobErrorRead | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result_url: str

    @classmethod
    def from_job(cls, job: PlannerJob) -> PlannerJobStatusResponse:
        return cls(
            job_id=job.id,
            request_hash=job.request_hash,
            state=job.state,
            cancellation_requested=job.cancellation_requested,
            error=(
                PlannerJobErrorRead(
                    code=job.error.code.value,
                    message=job.error.message,
                )
                if job.error is not None
                else None
            ),
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result_url=f"/planner/jobs/{job.id}/result",
        )


class PlannerJobAcceptedResponse(BaseModel):
    """Accepted or idempotently reused durable job."""

    job_id: str
    request_hash: str
    state: PlannerJobState
    duplicate: bool
    status_url: str
    result_url: str


class PlannerSimulationResultRead(PlannerSimulationResult):
    """Public typed simulation result."""


class PlannerJobResultResponse(BaseModel):
    """Completed result with the persisted reproducibility manifest."""

    job_id: str
    state: Literal[PlannerJobState.SUCCEEDED]
    result: PlannerSimulationResultRead
    manifest: dict[str, object]

    @classmethod
    def from_job(cls, job: PlannerJob) -> PlannerJobResultResponse:
        if job.result is None or job.manifest is None:
            raise ValueError("successful planner job is missing its result")
        return cls(
            job_id=job.id,
            state=PlannerJobState.SUCCEEDED,
            result=PlannerSimulationResultRead.model_validate(
                job.result.model_dump()
            ),
            manifest=job.manifest,
        )
