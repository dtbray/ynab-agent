"""Read-only HTTP operations backed by the wealth application service."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.progressive_tax import (
    AnnualTaxResult,
    TaxCalculationInput,
    calculate_annual_tax,
)

from ..dependencies import WealthServiceDep, require_api_access
from ..schemas.wealth import (
    AccountFreshnessRead,
    AccountFreshnessResponse,
    MortgageProjectionRead,
    ScenarioValidationRequest,
    ScenarioValidationResponse,
)


router = APIRouter(
    prefix="/wealth",
    tags=["wealth"],
    dependencies=[Depends(require_api_access)],
)


@router.post("/tax/calculate")
def calculate_tax(request: TaxCalculationInput) -> AnnualTaxResult:
    """Calculate one auditable annual federal and Indiana tax return."""
    try:
        return calculate_annual_tax(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get("/accounts/freshness")
async def list_account_freshness(
    service: WealthServiceDep,
    tracking_only: Annotated[
        bool,
        Query(description="Return only off-budget tracking accounts."),
    ] = True,
) -> AccountFreshnessResponse:
    """List active cached accounts and their latest transaction dates."""
    rows = await service.list_account_freshness(tracking_only=tracking_only)
    today = date.today()
    return AccountFreshnessResponse(
        accounts=[
            AccountFreshnessRead.from_service(row, today=today)
            for row in rows
        ]
    )


@router.get("/accounts/{account_id}/mortgage-projection")
async def get_mortgage_projection(
    account_id: Annotated[
        str,
        Path(min_length=1, max_length=128),
    ],
    service: WealthServiceDep,
    current_age: Annotated[
        int,
        Query(ge=0, le=100),
    ],
    as_of: Annotated[date | None, Query()] = None,
) -> MortgageProjectionRead:
    """Project a cached YNAB mortgage and return planner-ready cash flows."""
    try:
        projection = await service.project_mortgage(
            account_id,
            as_of=as_of or date.today(),
            current_age=current_age,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return MortgageProjectionRead.from_domain(
        projection,
        current_age=current_age,
    )


@router.post("/scenarios/validate")
async def validate_scenario(
    request: ScenarioValidationRequest,
    service: WealthServiceDep,
) -> ScenarioValidationResponse:
    """Validate a scenario and resolve its starting portfolio without running it."""
    scenario = WealthScenario.model_validate(request.model_dump())
    try:
        resolved = await service.resolve_starting_portfolio_with_provenance(
            scenario
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return ScenarioValidationResponse.from_service(scenario, resolved)
