"""Continuous-calibration CLI commands."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import typer

from ynab_agent.cli_support import print_json, render_rows
from ynab_agent.config import settings
from ynab_agent.db.calibration import SqlCalibrationRepository
from ynab_agent.runtime import open_database
from ynab_agent.services.calibration import (
    AlertState,
    CalibrationProfileState,
    CalibrationService,
    CalibrationStatus,
    ReviewedAllocation,
    default_calibration_policy,
)
from ynab_agent.workers.calibration import CalibrationWorker
from ynab_agent.services.calibration_models import CalibrationPolicy


calibration_app = typer.Typer(
    help="Continuously reconcile a saved plan with YNAB actuals.",
)


class CalibrationProfileInput(BaseModel):
    """Optional policy and reviewed allocations loaded from private JSON."""

    model_config = ConfigDict(extra="forbid")

    policy: CalibrationPolicy = Field(default_factory=default_calibration_policy)
    reviewed_allocations: tuple[ReviewedAllocation, ...] = ()


def _load_profile_input(path: Path | None) -> CalibrationProfileInput:
    if path is None:
        return CalibrationProfileInput()
    try:
        return CalibrationProfileInput.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise typer.BadParameter(
            f"invalid calibration profile file: {error}",
            param_hint="--input",
        ) from error


@calibration_app.command("create")
def create_profile(
    scenario_revision_id: Annotated[str, typer.Option("--scenario-revision")],
    budget_id: Annotated[str, typer.Option("--budget-id")],
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Private JSON with policy and reviewed account allocations.",
        ),
    ] = None,
) -> None:
    """Create one immutable calibration profile."""
    profile_input = _load_profile_input(input_path)

    async def run() -> None:
        async with open_database(settings) as database:
            service = CalibrationService(SqlCalibrationRepository(database))
            profile = await service.create_profile(
                scenario_revision_id=scenario_revision_id,
                budget_id=budget_id,
                policy=profile_input.policy,
                reviewed_allocations=profile_input.reviewed_allocations,
            )
        print_json(profile.model_dump(mode="json"))

    asyncio.run(run())


@calibration_app.command("capture")
def capture(
    profile_id: Annotated[str, typer.Option("--profile")],
    source_sync_batch_id: Annotated[str, typer.Option("--sync-batch")],
) -> None:
    """Capture an immutable snapshot and admit its idempotent run."""

    async def run() -> None:
        async with open_database(settings) as database:
            service = CalibrationService(SqlCalibrationRepository(database))
            snapshot, calibration_run, created = await service.capture_after_sync(
                profile_id,
                source_sync_batch_id=source_sync_batch_id,
            )
        print_json(
            {
                "created": created,
                "snapshot_id": snapshot.id,
                "snapshot_manifest_sha256": snapshot.manifest_sha256,
                "run": calibration_run.model_dump(mode="json"),
            }
        )

    asyncio.run(run())


@calibration_app.command("process")
def process() -> None:
    """Process the bounded durable calibration queue once."""

    async def run() -> None:
        async with open_database(settings) as database:
            repository = SqlCalibrationRepository(database)
            await repository.prepare_run_recovery()
            worker = CalibrationWorker(repository)
            try:
                completed = await worker.run_once()
            finally:
                await worker.stop()
        print_json({"completed_runs": completed})

    asyncio.run(run())


@calibration_app.command("status")
def status(
    profile_id: Annotated[str, typer.Option("--profile")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete public status object."),
    ] = False,
) -> None:
    """Show run state and privacy-bounded material alerts."""

    async def run() -> None:
        async with open_database(settings) as database:
            repository = SqlCalibrationRepository(database)
            profile = await repository.get_profile(profile_id)
            if profile is None:
                raise typer.BadParameter(
                    "calibration profile was not found",
                    param_hint="--profile",
                )
            runs = await repository.list_profile_runs(profile_id, limit=100)
            alerts = await repository.list_alerts(profile_id)
            capture_failures = await repository.list_capture_failures(profile_id)
            profile_state = await repository.get_profile_state(profile_id)
            selected_runs = runs
        public_status = CalibrationStatus(
            profile=profile,
            profile_state=profile_state or CalibrationProfileState.DISABLED,
            runs=selected_runs,
            alerts=alerts,
            capture_failures=capture_failures,
        )
        if json_output:
            print_json(public_status.model_dump(mode="json"))
            return
        render_rows(
            [
                {
                    "run_id": row.id,
                    "state": row.state.value,
                    "snapshot_id": row.snapshot_id,
                    "completed_at": row.completed_at or "",
                }
                for row in selected_runs
            ],
            [
                ("run_id", "Run"),
                ("state", "State"),
                ("snapshot_id", "Snapshot"),
                ("completed_at", "Completed"),
            ],
        )
        render_rows(
            [
                {
                    "sync_batch": row.source_sync_batch_id,
                    "error_code": row.error_code,
                    "created_at": row.created_at,
                }
                for row in capture_failures
            ],
            [
                ("sync_batch", "Sync batch"),
                ("error_code", "Capture failure"),
                ("created_at", "Failed"),
            ],
        )
        render_rows(
            [
                {
                    "kind": row.drift_kind.value,
                    "subject_id": row.subject_id,
                    "state": row.state.value,
                    "changed_at": row.state_changed_at,
                }
                for row in alerts
            ],
            [
                ("kind", "Drift"),
                ("subject_id", "Subject ID"),
                ("state", "State"),
                ("changed_at", "Changed"),
            ],
        )

    asyncio.run(run())


@calibration_app.command("disable")
def disable(
    profile_id: Annotated[str, typer.Option("--profile")],
) -> None:
    """Disable an obsolete immutable profile for future sync captures."""

    async def run() -> None:
        async with open_database(settings) as database:
            service = CalibrationService(SqlCalibrationRepository(database))
            changed = await service.disable_profile(profile_id)
            if not changed:
                raise typer.BadParameter(
                    "calibration profile was not found",
                    param_hint="--profile",
                )
        print_json({"profile_id": profile_id, "state": "disabled"})

    asyncio.run(run())


@calibration_app.command("acknowledge")
def acknowledge(
    profile_id: Annotated[str, typer.Option("--profile")],
    alert_id: Annotated[str, typer.Option("--alert")],
) -> None:
    """Append an acknowledgement to one alert lifecycle."""

    async def run() -> None:
        async with open_database(settings) as database:
            repository = SqlCalibrationRepository(database)
            snapshot = await repository.get_latest_snapshot(profile_id)
            if snapshot is None:
                raise typer.BadParameter(
                    "calibration profile has no snapshot",
                    param_hint="--profile",
                )
            try:
                alert = await repository.transition_alert(
                    alert_id,
                    profile_id,
                    snapshot.id,
                    AlertState.ACKNOWLEDGED,
                    at=datetime.now(timezone.utc),
                )
            except ValueError as error:
                raise typer.BadParameter(
                    str(error),
                    param_hint="--alert",
                ) from error
            if alert is None or alert.profile_id != profile_id:
                raise typer.BadParameter(
                    "calibration alert was not found",
                    param_hint="--alert",
                )
        print_json(alert.model_dump(mode="json"))

    asyncio.run(run())
