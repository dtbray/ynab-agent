"""Authenticated saved-scenario and comparison HTTP operations."""

from __future__ import annotations

from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Path, status

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.scenario_comparison import (
    ScenarioComparison,
    ScenarioComparisonError,
    ScenarioComparisonErrorCode,
    ScenarioRevision,
)

from ..dependencies import (
    HttpRuntimeDep,
    ScenarioComparisonServiceDep,
    require_api_access,
)
from ..schemas.scenario_comparison import (
    ScenarioComparisonCreateRequest,
    ScenarioRevisionCreateRequest,
)


router = APIRouter(
    prefix="/wealth/scenarios",
    tags=["wealth-scenarios"],
    dependencies=[Depends(require_api_access)],
)

RevisionIdPath = Annotated[
    str,
    Path(
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-f-]{36}$",
    ),
]


@router.post("/revisions", status_code=status.HTTP_201_CREATED)
async def create_scenario_revision(
    request: ScenarioRevisionCreateRequest,
    service: ScenarioComparisonServiceDep,
    runtime: HttpRuntimeDep,
) -> ScenarioRevision:
    """Resolve live inputs and append one immutable scenario revision."""
    historical_dataset = (
        runtime.historical_datasets.resolve(request.historical_dataset_id)
        if request.historical_dataset_id is not None
        else None
    )
    if (
        request.historical_dataset_id is not None
        and historical_dataset is None
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "historical_dataset_not_found",
                "message": "registered historical dataset was not found",
            },
        )
    try:
        return await service.save_revision(
            WealthScenario.model_validate(request.scenario.model_dump()),
            historical_dataset=historical_dataset,
        )
    except (ScenarioComparisonError, ValueError) as exc:
        _raise_http_error(exc)


@router.get("/revisions/{revision_id}")
async def get_scenario_revision(
    revision_id: RevisionIdPath,
    service: ScenarioComparisonServiceDep,
) -> ScenarioRevision:
    """Read one immutable scenario revision and its resolved manifest."""
    try:
        return await service.get_revision(revision_id)
    except ScenarioComparisonError as exc:
        _raise_http_error(exc)


@router.post("/comparisons", status_code=status.HTTP_201_CREATED)
async def create_scenario_comparison(
    request: ScenarioComparisonCreateRequest,
    service: ScenarioComparisonServiceDep,
) -> ScenarioComparison:
    """Run and persist one reproducible common-path comparison."""
    try:
        return await service.compare(
            baseline_revision_id=request.baseline_revision_id,
            alternative_revision_ids=request.alternative_revision_ids,
            named_stress=request.named_stress,
        )
    except (ScenarioComparisonError, ValueError, RuntimeError) as exc:
        _raise_http_error(exc)


@router.get("/comparisons/{comparison_id}")
async def get_scenario_comparison(
    comparison_id: RevisionIdPath,
    service: ScenarioComparisonServiceDep,
) -> ScenarioComparison:
    """Read a persisted comparison result and reproducibility recipe."""
    try:
        return await service.get_comparison(comparison_id)
    except ScenarioComparisonError as exc:
        _raise_http_error(exc)


def _raise_http_error(error: Exception) -> NoReturn:
    if isinstance(error, ScenarioComparisonError):
        if error.code is ScenarioComparisonErrorCode.REVISION_NOT_FOUND:
            status_code = status.HTTP_404_NOT_FOUND
        elif error.code is ScenarioComparisonErrorCode.QUEUE_SATURATED:
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif error.code in {
            ScenarioComparisonErrorCode.PERSISTENCE_CONFLICT,
            ScenarioComparisonErrorCode.PERSISTED_CONTENT_MISMATCH,
        }:
            status_code = status.HTTP_409_CONFLICT
        else:
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        detail: object = {
            "code": error.code.value,
            "message": error.message,
        }
    else:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        detail = {"code": "invalid_scenario", "message": str(error)}
    raise HTTPException(status_code=status_code, detail=detail) from error
