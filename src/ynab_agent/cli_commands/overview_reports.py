"""Typer adapters for cached overview and hygiene reports."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated

import typer

from ynab_agent.cli_support import (
    configure_logging,
    format_milliunits,
    print_json,
    render_rows,
)
from ynab_agent.config import settings
from ynab_agent.db.overview_reports import SqlOverviewReportRepository
from ynab_agent.runtime import open_database
from ynab_agent.services.reports.overview import (
    CachedChange,
    HygieneCheck,
    NetWorthBucket,
    OverviewReportService,
)


overview_reports_app = typer.Typer()


def _iso_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise typer.BadParameter(
            "must use YYYY-MM-DD",
            param_hint=option,
        ) from exc


def _net_worth_row(row: NetWorthBucket) -> dict[str, object]:
    return {
        "bucket": row.bucket,
        "type": row.account_type,
        "accounts": row.account_count,
        "balance": format_milliunits(row.balance_milliunits),
    }


def _change_row(row: CachedChange) -> dict[str, object]:
    return {
        "batch_id": row.batch_id,
        "budget_id": row.budget_id,
        "resource": row.resource,
        "entity_id": row.entity_id,
        "action": row.action,
        "entity_name": row.entity_name,
        "recorded_at": row.recorded_at,
    }


def _hygiene_row(row: HygieneCheck) -> dict[str, object]:
    return {"check_name": row.check_name, "count": row.count}


@overview_reports_app.command("net-worth")
def report_net_worth(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Summarize cached account balances."""
    configure_logging(settings)

    async def run() -> tuple[NetWorthBucket, ...]:
        async with open_database(settings) as database:
            service = OverviewReportService(SqlOverviewReportRepository(database))
            return await service.net_worth()

    rows = [_net_worth_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("bucket", "Bucket"),
            ("type", "Type"),
            ("accounts", "Accounts"),
            ("balance", "Balance"),
        ],
    )


@overview_reports_app.command("changes")
def report_changes(
    plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            "-p",
            help="Limit to one YNAB plan ID",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Show cached changes from the most recent sync batch."""
    configure_logging(settings)

    async def run() -> tuple[CachedChange, ...]:
        async with open_database(settings, initialize=True) as database:
            service = OverviewReportService(SqlOverviewReportRepository(database))
            return await service.latest_changes(plan_id)

    rows = [_change_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("resource", "Resource"),
            ("action", "Action"),
            ("entity_name", "Name"),
            ("entity_id", "ID"),
            ("recorded_at", "Recorded"),
        ],
    )


@overview_reports_app.command("hygiene")
def report_hygiene(
    since: Annotated[
        str,
        typer.Option(
            "--since",
            help="Include transactions since ISO date",
        ),
    ] = "2026-01-01",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Show cached budget cleanup counts."""
    configure_logging(settings)
    since_date = _iso_date(since, option="--since")

    async def run() -> tuple[HygieneCheck, ...]:
        async with open_database(settings) as database:
            service = OverviewReportService(SqlOverviewReportRepository(database))
            return await service.hygiene(since=since_date)

    rows = [_hygiene_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(rows, [("check_name", "Check"), ("count", "Count")])
