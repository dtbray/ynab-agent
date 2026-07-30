"""Typer adapters for YNAB account inspection."""

from __future__ import annotations

import asyncio
from typing import Annotated, cast

import typer

from ynab_agent.cli_commands.reconciliation import reconciliation_app
from ynab_agent.cli_support import (
    configure_logging,
    format_milliunits,
    print_json,
    render_rows,
)
from ynab_agent.config import settings


accounts_app = typer.Typer(help="Inspect YNAB accounts")
accounts_app.add_typer(reconciliation_app)


def _account_row(row: dict[str, object]) -> dict[str, object]:
    rendered = dict(row)
    for key in ("balance", "cleared_balance", "uncleared_balance"):
        rendered[key] = format_milliunits(cast(int | None, row[key]))
    return rendered


@accounts_app.command("list")
def list_accounts(
    plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            "-p",
            help='YNAB plan ID, "last-used", or "default"',
        ),
    ] = None,
    include_closed: Annotated[
        bool,
        typer.Option(
            "--include-closed",
            help="Include closed accounts",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """List accounts for a YNAB plan."""
    configure_logging(settings)
    resolved_plan_id = plan_id or settings.ynab_plan_id

    async def run() -> list[dict[str, object]]:
        from ynab_agent.api.client import YnabClient

        async with YnabClient() as client:
            raw_rows = await client.get_accounts(plan_id=resolved_plan_id)
        rows = cast(list[dict[str, object]], raw_rows)
        if not include_closed:
            rows = [row for row in rows if not row["closed"] and not row["deleted"]]
        return [_account_row(row) for row in rows]

    rows = asyncio.run(run())
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("name", "Name"),
            ("type", "Type"),
            ("balance", "Balance"),
            ("cleared_balance", "Cleared"),
            ("uncleared_balance", "Uncleared"),
            ("id", "ID"),
        ],
    )
