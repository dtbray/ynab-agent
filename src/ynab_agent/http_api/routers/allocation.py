"""Multi-asset allocation validation operations."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from ynab_agent.planning.allocation import (
    PortfolioAllocationPlan,
    PortfolioAllocationValidation,
    validate_portfolio_allocation,
)
from ynab_agent.planning.stress import (
    NamedStressDefinition,
    named_stress_catalog,
)

from ..dependencies import require_api_access


router = APIRouter(
    prefix="/wealth",
    tags=["wealth"],
    dependencies=[Depends(require_api_access)],
)


class NamedStressCatalogResponse(BaseModel):
    """Bounded catalog accepted by planner-job selectors."""

    model_config = ConfigDict(frozen=True)

    stresses: tuple[NamedStressDefinition, ...]


@router.get("/allocations/stresses")
def list_named_stresses() -> NamedStressCatalogResponse:
    """List every auditable built-in multi-asset stress."""
    return NamedStressCatalogResponse(stresses=named_stress_catalog())


@router.post("/allocations/validate")
def validate_allocation(
    plan: PortfolioAllocationPlan,
) -> PortfolioAllocationValidation:
    """Validate and fingerprint bounded allocation assumptions."""
    return validate_portfolio_allocation(plan)
