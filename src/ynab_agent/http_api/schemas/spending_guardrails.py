"""Public schemas for YNAB-backed retirement spending guardrails."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ynab_agent.planning.spending_guardrails import (
    AnnualSpendingContext,
    GuardrailAuditSummary,
    RetirementSpendingPlan,
    SpendingTier,
)
from ynab_agent.services.spending_guardrails import SpendingTierMapping


class SpendingTierAssignmentRequest(BaseModel):
    """Set one durable category policy using its immutable YNAB identity."""

    model_config = ConfigDict(extra="forbid")

    budget_id: str = Field(min_length=1, max_length=64)
    tier: SpendingTier
    essential_floor_milliunits: int | None = Field(default=None, ge=0)
    note: str | None = Field(default=None, max_length=500)


class SpendingTierMappingRead(BaseModel):
    """Tier identity plus the latest optional category display labels."""

    budget_id: str
    category_id: str
    tier: SpendingTier
    essential_floor_milliunits: int | None
    note: str | None
    category_name: str | None
    category_group_name: str | None
    category_hidden: bool
    category_deleted: bool
    updated_at: str

    @classmethod
    def from_service(
        cls,
        mapping: SpendingTierMapping,
    ) -> SpendingTierMappingRead:
        return cls.model_validate(mapping, from_attributes=True)


class SpendingTierMappingsResponse(BaseModel):
    """All durable mappings for one budget."""

    mappings: list[SpendingTierMappingRead] = Field(default_factory=list)


class SpendingBaselineResponse(BaseModel):
    """Policy-independent YNAB baseline ready for a scenario."""

    plan: RetirementSpendingPlan


class GuardrailPreviewRequest(BaseModel):
    """Bounded deterministic policy preview."""

    model_config = ConfigDict(extra="forbid")

    plan: RetirementSpendingPlan
    contexts: list[AnnualSpendingContext] = Field(
        min_length=1,
        max_length=130,
    )


class GuardrailPreviewResponse(BaseModel):
    """Auditable reductions and restorations for one policy preview."""

    summary: GuardrailAuditSummary
