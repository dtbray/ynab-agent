"""HTTP contracts for continuous calibration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ynab_agent.services.calibration import (
    CalibrationRun,
    CalibrationStatus,
    ReviewedAllocation,
    default_calibration_policy,
)
from ynab_agent.services.calibration_models import CalibrationPolicy


class CalibrationProfileCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_revision_id: str = Field(min_length=36, max_length=36)
    budget_id: str = Field(min_length=1, max_length=64)
    policy: CalibrationPolicy = Field(default_factory=default_calibration_policy)
    reviewed_allocations: tuple[ReviewedAllocation, ...] = ()


class CalibrationCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_sync_batch_id: str = Field(min_length=1, max_length=64)


class CalibrationCaptureResponse(BaseModel):
    created: bool
    snapshot_id: str
    snapshot_manifest_sha256: str
    run: CalibrationRun


class CalibrationStatusResponse(CalibrationStatus):
    """HTTP name for the interface-neutral public status contract."""
