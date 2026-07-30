"""Read-only HTTP operations backed by the wealth application service."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.progressive_tax import (
    AnnualTaxResult,
    TaxCalculationInput,
    calculate_annual_tax,
)
from ynab_agent.planning.paths import ResourceLimitError
from ynab_agent.services.planner_jobs import (
    PlannerErrorCode,
    PlannerJobServiceError,
    PlannerJobSubmission,
    PlannerQueueSaturatedError,
    PlannerResourceLimitError,
)

from ..dependencies import (
    PlannerJobServiceDep,
    SocialSecurityOptimizerDep,
    WealthServiceDep,
    require_api_access,
)
from ..schemas.planner_jobs import (
    PlannerJobAcceptedResponse,
    PlannerJobSubmitRequest,
)
from ..schemas.wealth import (
    AccountFreshnessRead,
    AccountFreshnessResponse,
    MortgageProjectionRead,
    ScenarioValidationRequest,
    ScenarioValidationResponse,
    SocialSecurityOptimizationRequest,
    SocialSecurityOptimizationResponse,
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


@router.post(
    "/tax/strategies/jobs",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_tax_strategy_job(
    request: PlannerJobSubmitRequest,
    response: Response,
    service: PlannerJobServiceDep,
) -> PlannerJobAcceptedResponse:
    """Submit a replayable tax-strategy simulation to shared planner admission."""
    scenario = WealthScenario.model_validate(request.scenario.model_dump())
    if scenario.tax_assumptions is None or scenario.tax_assumptions.strategy is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="scenario must configure tax_assumptions.strategy",
        )
    try:
        submitted = await service.submit(
            PlannerJobSubmission(
                scenario=scenario,
                historical_dataset_id=request.historical_dataset_id,
            )
        )
    except PlannerJobServiceError as exc:
        status_code = {
            PlannerErrorCode.QUEUE_SATURATED: status.HTTP_429_TOO_MANY_REQUESTS,
            PlannerErrorCode.DISPATCH_FAILED: status.HTTP_503_SERVICE_UNAVAILABLE,
        }.get(exc.code, status.HTTP_422_UNPROCESSABLE_CONTENT)
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.code.value, "message": exc.message},
            headers=(
                {"Retry-After": "5"}
                if exc.code is PlannerErrorCode.QUEUE_SATURATED
                else None
            ),
        ) from exc
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


@router.post("/social-security/optimize")
async def optimize_household_social_security(
    request: SocialSecurityOptimizationRequest,
    service: WealthServiceDep,
    optimizer: SocialSecurityOptimizerDep,
) -> SocialSecurityOptimizationResponse:
    """Run a bounded strategy matrix without blocking the API event loop."""
    scenario = WealthScenario.model_validate(request.model_dump())
    try:
        resolved = await service.resolve_starting_portfolio_with_provenance(
            scenario
        )
        result = await optimizer.execute(
            scenario,
            resolved.value,
            resolved.provenance,
        )
    except PlannerQueueSaturatedError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.message,
        ) from exc
    except (PlannerResourceLimitError, ResourceLimitError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Social Security optimization exceeds configured planner resources",
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return SocialSecurityOptimizationResponse(result=result)
