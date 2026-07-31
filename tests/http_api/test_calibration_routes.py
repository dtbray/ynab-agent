from __future__ import annotations

from datetime import datetime, timezone

from ynab_agent.http_api.schemas.calibration import (
    CalibrationStatusResponse,
)
from ynab_agent.services.calibration import (
    CalibrationCaptureFailure,
    CalibrationProfile,
    CalibrationProfileState,
    CalibrationStatus,
    default_calibration_policy,
)


def test_http_and_cli_share_the_exact_calibration_status_contract() -> None:
    profile = CalibrationProfile.create(
        id="00000000-0000-4000-8000-000000000097",
        scenario_revision_id="00000000-0000-4000-8000-000000000098",
        budget_id="budget-1",
        policy=default_calibration_policy(),
        created_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
    )
    shared = CalibrationStatus(
        profile=profile,
        profile_state=CalibrationProfileState.DISABLED,
        runs=(),
        alerts=(),
        capture_failures=(
            CalibrationCaptureFailure(
                id="00000000-0000-4000-8000-000000000099",
                profile_id=profile.id,
                source_sync_batch_id="batch-failed",
                error_code="invalid_source",
                created_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
            ),
        ),
    )

    assert CalibrationStatusResponse.model_validate(
        shared.model_dump(mode="python")
    ).model_dump(mode="json") == shared.model_dump(mode="json")
