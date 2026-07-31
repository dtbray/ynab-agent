"""Checkpoint-safe YNAB synchronization command."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import logging
from typing import Annotated, cast

import typer

from ynab_agent.cli_support import console
from ynab_agent.config import settings
from ynab_agent.db.wealth import SqlWealthRepository
from ynab_agent.db.calibration import SqlCalibrationRepository
from ynab_agent.runtime import open_database
from ynab_agent.services.sync import (
    SyncGateway,
    SyncRequest,
    SyncService,
    SyncStore,
)
from ynab_agent.services.wealth import WealthService
from ynab_agent.services.calibration import CalibrationService


sync_app = typer.Typer()


def _iso_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            "must use YYYY-MM-DD",
            param_hint=option,
        ) from exc


@sync_app.command("sync")
def sync(
    plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            "-p",
            help='YNAB plan ID, "last-used", or "default"',
        ),
    ] = None,
    since: Annotated[
        str,
        typer.Option("--since", help="Sync transactions since ISO date"),
    ] = "2026-01-01",
    month: Annotated[
        str | None,
        typer.Option(
            "--month",
            help="Sync budget month/category data for ISO month date",
        ),
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Ignore stored YNAB server knowledge and fetch full endpoint payloads",
        ),
    ] = False,
    skip_month_detail: Annotated[
        bool,
        typer.Option(
            "--skip-month-detail",
            help="Skip full month category detail refresh; month summaries still delta-sync",
        ),
    ] = False,
) -> None:
    """Sync YNAB data to the configured database using YNAB delta checkpoints."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    today = date.today()
    resolved_month = month or date(today.year, today.month, 1).isoformat()
    request = SyncRequest(
        plan_id=plan_id or settings.ynab_plan_id,
        since=_iso_date(since, option="--since"),
        month=_iso_date(resolved_month, option="--month"),
        full=full,
        skip_month_detail=skip_month_detail,
    )

    async def run() -> None:
        from ynab_agent.api.client import YnabClient

        console.print("🔄 Syncing YNAB data...")
        async with open_database(settings, initialize=True) as database:
            wealth_service = WealthService(SqlWealthRepository(database))
            calibration_service = CalibrationService(SqlCalibrationRepository(database))
            async with YnabClient() as client:
                service = SyncService(
                    cast(SyncGateway, client),
                    cast(SyncStore, database),
                    valuation_capture=wealth_service,
                    post_sync_hook=calibration_service,
                    clock=lambda: datetime.now(timezone.utc),
                )
                summary = await service.run(request)

        if summary.used_plan_alias:
            console.print(
                f"Resolved YNAB plan alias {summary.requested_plan_id!r} "
                f"to {summary.resolved_plan_id}"
            )
        console.print(
            "✅ Synced "
            f"{summary.plan_count} plans, "
            f"{summary.account_count} accounts, "
            f"{summary.payee_count} payees, "
            f"{summary.payee_location_count} payee locations, "
            f"{summary.category_group_count} category groups, "
            f"{summary.category_count} categories, "
            f"{summary.month_count} months, "
            f"{summary.month_category_count} month categories, "
            f"{summary.transaction_count} transactions, "
            f"{summary.scheduled_transaction_count} scheduled transactions, "
            f"{summary.scheduled_subtransaction_count} scheduled splits"
        )

    asyncio.run(run())
