"""Typer adapter for cached scheduled obligations."""

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
from ynab_agent.db.obligation_reports import SqlObligationRepository
from ynab_agent.runtime import open_database
from ynab_agent.services.reports.obligations import (
    ObligationRequest,
    ObligationService,
    ScheduledObligation,
)


obligation_reports_app = typer.Typer()


def _iso_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise typer.BadParameter(
            "must use YYYY-MM-DD",
            param_hint=option,
        ) from exc


def _obligation_row(row: ScheduledObligation) -> dict[str, object]:
    return {
        "due_date": row.due_date,
        "payee": row.payee,
        "group_name": row.group_name,
        "category": row.category,
        "account": row.account,
        "frequency": row.frequency,
        "amount": format_milliunits(row.amount_milliunits),
        "memo": row.memo,
    }


@obligation_reports_app.command("obligations")
def report_obligations(
    through: Annotated[
        str | None,
        typer.Option(
            "--through",
            help="Include scheduled transactions due on/before ISO date",
        ),
    ] = None,
    include_inflows: Annotated[
        bool,
        typer.Option(
            "--include-inflows",
            help="Include positive amounts",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            min=1,
            help="Maximum rows to display",
        ),
    ] = 30,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """List upcoming cached scheduled transactions."""
    configure_logging(settings)
    try:
        request = ObligationRequest(
            through=(
                _iso_date(through, option="--through") if through is not None else date.today()
            ),
            include_inflows=include_inflows,
            limit=limit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def run() -> tuple[ScheduledObligation, ...]:
        async with open_database(settings) as database:
            service = ObligationService(SqlObligationRepository(database))
            return await service.list(request)

    rows = [_obligation_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("due_date", "Due"),
            ("payee", "Payee"),
            ("amount", "Amount"),
            ("frequency", "Frequency"),
            ("group_name", "Group"),
            ("category", "Category"),
            ("account", "Account"),
            ("memo", "Memo"),
        ],
    )
