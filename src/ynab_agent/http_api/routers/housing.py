"""Authenticated housing decision projection."""

from fastapi import APIRouter, Depends

from ynab_agent.planning.housing import project_housing_plan

from ..dependencies import require_api_access
from ..schemas.housing import HousingProjectionRequest, HousingProjectionResponse


router = APIRouter(
    prefix="/wealth/housing",
    tags=["wealth"],
    dependencies=[Depends(require_api_access)],
)


@router.post("/project")
def project_housing(request: HousingProjectionRequest) -> HousingProjectionResponse:
    """Validate and project the same housing ledger used by simulation."""
    return HousingProjectionResponse.model_validate(
        project_housing_plan(request.scenario)
    )
