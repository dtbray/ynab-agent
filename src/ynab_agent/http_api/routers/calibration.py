"""Authenticated continuous-calibration operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from ynab_agent.services.calibration import (
    AlertState,
    CalibrationAlert,
    CalibrationProfile,
    CalibrationProfileState,
)

from ..dependencies import (
    CalibrationRepositoryDep,
    CalibrationServiceDep,
    require_api_access,
)
from ..schemas.calibration import (
    CalibrationCaptureRequest,
    CalibrationCaptureResponse,
    CalibrationProfileCreateRequest,
    CalibrationStatusResponse,
)


router = APIRouter(
    prefix="/wealth/calibration",
    tags=["wealth-calibration"],
    dependencies=[Depends(require_api_access)],
)
Identifier = Annotated[str, Path(min_length=36, max_length=36)]


@router.post("/profiles", status_code=status.HTTP_201_CREATED)
async def create_profile(
    request: CalibrationProfileCreateRequest,
    service: CalibrationServiceDep,
) -> CalibrationProfile:
    """Create one immutable profile bound to a saved scenario revision."""
    try:
        return await service.create_profile(
            scenario_revision_id=request.scenario_revision_id,
            budget_id=request.budget_id,
            policy=request.policy,
            reviewed_allocations=request.reviewed_allocations,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error


@router.post(
    "/profiles/{profile_id}/capture",
    response_model=CalibrationCaptureResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def capture(
    profile_id: Identifier,
    request: CalibrationCaptureRequest,
    service: CalibrationServiceDep,
) -> CalibrationCaptureResponse:
    """Idempotently capture source inputs before admitting automation."""
    try:
        snapshot, run, created = await service.capture_after_sync(
            profile_id,
            source_sync_batch_id=request.source_sync_batch_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    return CalibrationCaptureResponse(
        created=created,
        snapshot_id=snapshot.id,
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        run=run,
    )


@router.get(
    "/profiles/{profile_id}",
    response_model=CalibrationStatusResponse,
)
async def status_view(
    profile_id: Identifier,
    repository: CalibrationRepositoryDep,
) -> CalibrationStatusResponse:
    """Return the same privacy-bounded state exposed by the CLI."""
    profile = await repository.get_profile(profile_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="calibration profile was not found",
        )
    return CalibrationStatusResponse(
        profile=profile,
        profile_state=(await repository.get_profile_state(profile_id))
        or CalibrationProfileState.DISABLED,
        runs=await repository.list_profile_runs(profile_id, limit=100),
        alerts=await repository.list_alerts(profile_id),
        capture_failures=await repository.list_capture_failures(profile_id),
    )


@router.post("/profiles/{profile_id}/disable")
async def disable_profile(
    profile_id: Identifier,
    service: CalibrationServiceDep,
) -> dict[str, str]:
    """Disable an obsolete immutable profile for subsequent syncs."""
    if not await service.disable_profile(profile_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="calibration profile was not found",
        )
    return {"profile_id": profile_id, "state": "disabled"}


@router.post("/profiles/{profile_id}/alerts/{alert_id}/acknowledge")
async def acknowledge(
    profile_id: Identifier,
    alert_id: Identifier,
    repository: CalibrationRepositoryDep,
) -> CalibrationAlert:
    """Append an acknowledgement without exposing financial values."""
    snapshot = await repository.get_latest_snapshot(profile_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="calibration profile has no snapshot",
        )
    try:
        alert = await repository.transition_alert(
            alert_id,
            profile_id,
            snapshot.id,
            AlertState.ACKNOWLEDGED,
            at=datetime.now(timezone.utc),
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    if alert is None or alert.profile_id != profile_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="calibration alert was not found",
        )
    return alert
