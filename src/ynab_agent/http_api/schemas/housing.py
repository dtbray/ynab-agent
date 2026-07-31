"""Public housing decision schemas."""

from pydantic import BaseModel, ConfigDict, Field

from ynab_agent.planning.models import WealthScenario


class HousingProjectionRequest(BaseModel):
    """One fully validated housing-aware scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario: WealthScenario


class HousingProjectionResponse(BaseModel):
    """Deterministic housing representation shared with the CLI."""

    manifest: dict[str, object]
    annual_housing: list[dict[str, object]] = Field(default_factory=list)
    ending_disposition_value: float
    after_tax_ending_disposition_value: float
