"""Plan-listing Typer commands."""

from __future__ import annotations

import asyncio
from typing import Annotated, cast

import typer

from ynab_agent.cli_support import (
    configure_logging,
    print_json,
    render_rows,
)
from ynab_agent.config import settings


plans_app = typer.Typer(help="List YNAB plans")


@plans_app.command("list")
def list_plans(
    include_accounts: Annotated[
        bool,
        typer.Option(
            "--include-accounts",
            help="Ask YNAB to include account summaries in the response",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """List accessible YNAB plans."""
    configure_logging(settings)

    async def run() -> None:
        from ynab_agent.api.client import YnabClient

        async with YnabClient() as client:
            raw_rows = await client.get_plans(include_accounts=include_accounts)
        rows = cast(list[dict[str, object]], raw_rows)
        if json_output:
            print_json(rows)
            return
        render_rows(
            rows,
            [
                ("id", "ID"),
                ("name", "Name"),
                ("first_month", "First Month"),
                ("last_month", "Last Month"),
                ("last_modified_on", "Last Modified"),
            ],
        )

    asyncio.run(run())
