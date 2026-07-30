"""Authenticated durable planner-job HTTP operations."""

from __future__ import annotations

from typing import Annotated, NoReturn

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Response,
    status,
)

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.planner_jobs import (
    PlannerErrorCode,
    PlannerJobServiceError,
    PlannerJobSubmission,
)

from ..dependencies import PlannerJobServiceDep, require_api_access
from ..schemas.planner_jobs import (
    PlannerJobAcceptedResponse,
    PlannerJobResultResponse,
    PlannerJobStatusResponse,
    PlannerJobSubmitRequest,
)


router = APIRouter(
    prefix="/planner/jobs",
    tags=["planner-jobs"],
    dependencies=[Depends(require_api_access)],
)

JobIdPath = Annotated[
    str,
    Path(
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
]


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_planner_job(
    request: PlannerJobSubmitRequest,
    response: Response,
    service: PlannerJobServiceDep,
) -> PlannerJobAcceptedResponse:
    """Persist and enqueue one idempotent simulation request."""
    try:
        submitted = await service.submit(
            PlannerJobSubmission(
                scenario=WealthScenario.model_validate(
                    request.scenario.model_dump()
                ),
                historical_dataset_id=request.historical_dataset_id,
            )
        )
    except PlannerJobServiceError as exc:
        _raise_http_error(exc)

    job = submitted.job
    result_url = f"/planner/jobs/{job.id}/result"
    response.headers["Location"] = result_url
    return PlannerJobAcceptedResponse(
        job_id=job.id,
        request_hash=job.request_hash,
        state=job.state,
        duplicate=submitted.duplicate,
        status_url=f"/planner/jobs/{job.id}",
        result_url=result_url,
    )


@router.get("/{job_id}")
async def get_planner_job_status(
    job_id: JobIdPath,
    service: PlannerJobServiceDep,
) -> PlannerJobStatusResponse:
    """Return durable job state and stable failure details."""
    try:
        job = await service.get_status(job_id)
    except PlannerJobServiceError as exc:
        _raise_http_error(exc)
    return PlannerJobStatusResponse.from_job(job)


@router.get("/{job_id}/result")
async def get_planner_job_result(
    job_id: JobIdPath,
    service: PlannerJobServiceDep,
) -> PlannerJobResultResponse:
    """Return a completed typed result and reproducibility manifest."""
    try:
        job = await service.get_result(job_id)
    except PlannerJobServiceError as exc:
        _raise_http_error(exc)
    return PlannerJobResultResponse.from_job(job)


@router.delete("/{job_id}", status_code=status.HTTP_202_ACCEPTED)
async def cancel_planner_job(
    job_id: JobIdPath,
    service: PlannerJobServiceDep,
) -> PlannerJobStatusResponse:
    """Cancel accepted work or request cancellation of a running job."""
    try:
        job = await service.cancel(job_id)
    except PlannerJobServiceError as exc:
        _raise_http_error(exc)
    return PlannerJobStatusResponse.from_job(job)


def _raise_http_error(error: PlannerJobServiceError) -> NoReturn:
    status_code = {
        PlannerErrorCode.JOB_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        PlannerErrorCode.QUEUE_SATURATED: status.HTTP_429_TOO_MANY_REQUESTS,
        PlannerErrorCode.DISPATCH_FAILED: status.HTTP_503_SERVICE_UNAVAILABLE,
        PlannerErrorCode.RESULT_NOT_READY: status.HTTP_409_CONFLICT,
        PlannerErrorCode.JOB_NOT_CANCELLABLE: status.HTTP_409_CONFLICT,
        PlannerErrorCode.EXECUTION_FAILED: status.HTTP_409_CONFLICT,
    }.get(error.code, status.HTTP_422_UNPROCESSABLE_CONTENT)
    headers = (
        {"Retry-After": "5"}
        if error.code is PlannerErrorCode.QUEUE_SATURATED
        else None
    )
    raise HTTPException(
        status_code=status_code,
        detail={"code": error.code.value, "message": error.message},
        headers=headers,
    ) from error
