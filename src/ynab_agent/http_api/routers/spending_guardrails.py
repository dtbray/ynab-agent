"""YNAB-derived retirement spending tier and guardrail operations."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from ynab_agent.planning.spending_guardrails import evaluate_guardrails
from ynab_agent.services.spending_guardrails import (
    SpendingTierAssignment,
    UnknownSpendingCategoryError,
)

from ..dependencies import SpendingGuardrailServiceDep, require_api_access
from ..schemas.spending_guardrails import (
    GuardrailPreviewRequest,
    GuardrailPreviewResponse,
    SpendingBaselineResponse,
    SpendingTierAssignmentRequest,
    SpendingTierMappingRead,
    SpendingTierMappingsResponse,
)


router = APIRouter(
    prefix="/wealth",
    tags=["wealth"],
    dependencies=[Depends(require_api_access)],
)


@router.put("/spending-tiers/{category_id}")
async def assign_spending_tier(
    category_id: Annotated[str, Path(min_length=1, max_length=64)],
    request: SpendingTierAssignmentRequest,
    service: SpendingGuardrailServiceDep,
) -> SpendingTierMappingRead:
    """Create or replace a stable category-to-tier assignment."""
    try:
        mapping = await service.assign_tier(
            SpendingTierAssignment(
                category_id=category_id,
                **request.model_dump(),
            )
        )
    except UnknownSpendingCategoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return SpendingTierMappingRead.from_service(mapping)


@router.get("/spending-tiers")
async def list_spending_tiers(
    service: SpendingGuardrailServiceDep,
    budget_id: Annotated[
        str,
        Query(min_length=1, max_length=64),
    ],
) -> SpendingTierMappingsResponse:
    """List mappings even when a cached category was renamed or deleted."""
    mappings = await service.list_mappings(budget_id=budget_id)
    return SpendingTierMappingsResponse(
        mappings=[
            SpendingTierMappingRead.from_service(mapping)
            for mapping in mappings
        ]
    )


@router.get("/spending-baseline")
async def get_spending_baseline(
    service: SpendingGuardrailServiceDep,
    budget_id: Annotated[
        str,
        Query(min_length=1, max_length=64),
    ],
    through_month: Annotated[date, Query()],
    lookback_months: Annotated[int, Query(ge=1, le=120)] = 12,
) -> SpendingBaselineResponse:
    """Annualize mapped YNAB spending through an exclusive month."""
    plan = await service.derive_plan(
        budget_id=budget_id,
        through_month=through_month,
        lookback_months=lookback_months,
    )
    return SpendingBaselineResponse(plan=plan)


@router.post("/spending-guardrails/preview")
async def preview_spending_guardrails(
    request: GuardrailPreviewRequest,
) -> GuardrailPreviewResponse:
    """Evaluate reductions and restorations without running Monte Carlo."""
    return GuardrailPreviewResponse(
        summary=evaluate_guardrails(request.plan, request.contexts)
    )
