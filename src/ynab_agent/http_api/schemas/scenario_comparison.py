"""Public request schemas for saved scenario comparison."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .wealth import ScenarioValidationRequest


class ScenarioRevisionCreateRequest(BaseModel):
    """Save a validated scenario and optional registered historical data."""

    model_config = ConfigDict(extra="forbid")

    scenario: ScenarioValidationRequest
    historical_dataset_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )


class ScenarioComparisonCreateRequest(BaseModel):
    """Compare saved alternatives against one baseline revision."""

    model_config = ConfigDict(extra="forbid")

    baseline_revision_id: str = Field(min_length=36, max_length=36)
    alternative_revision_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=12,
    )
