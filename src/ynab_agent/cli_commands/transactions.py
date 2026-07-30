"""Transaction Typer commands."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, cast

import typer

from ynab_agent.cli_support import (
    configure_logging,
    console,
    format_milliunits,
    print_json,
    render_rows,
)
from ynab_agent.config import settings


transactions_app = typer.Typer(help="Inspect YNAB transactions")


def _dollars_to_milliunits(amount: str) -> int:
    decimal_amount = Decimal(amount)
    return int(
        (decimal_amount * Decimal("1000")).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


@transactions_app.command("list")
def list_transactions(
    plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            "-p",
            help='YNAB plan ID, "last-used", or "default"',
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Only include transactions on/after ISO date, e.g. 2026-05-18",
        ),
    ] = None,
    transaction_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            help='YNAB transaction filter, such as "unapproved" or "uncategorized"',
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            min=1,
            help="Maximum rows to display",
        ),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """List transactions from YNAB."""
    configure_logging(settings)
    resolved_plan_id = settings.ynab_plan_id if plan_id is None else plan_id

    async def run() -> None:
        from ynab_agent.api.client import SyncResult, YnabClient

        since_date = date.fromisoformat(since) if since else None
        async with YnabClient() as client:
            result = await client.get_transactions(
                plan_id=resolved_plan_id,
                since_date=since_date,
                transaction_type=transaction_type,
            )
        raw_rows = result.data if isinstance(result, SyncResult) else result
        rows = cast(list[dict[str, object]], raw_rows[:limit])
        display_rows = [
            {
                **row,
                "amount": format_milliunits(cast(int | None, row["amount"])),
            }
            for row in rows
        ]
        if json_output:
            print_json(display_rows)
            return
        render_rows(
            display_rows,
            [
                ("date", "Date"),
                ("amount", "Amount"),
                ("memo", "Memo"),
                ("cleared", "Cleared"),
                ("approved", "Approved"),
                ("id", "ID"),
            ],
        )

    asyncio.run(run())


@transactions_app.command("create")
def create_transaction(
    account_id: Annotated[
        str,
        typer.Option("--account-id", help="YNAB account UUID"),
    ],
    amount: Annotated[
        str,
        typer.Option(
            "--amount",
            help='Dollar amount; use negative values for outflows, e.g. "-4.25"',
        ),
    ],
    transaction_date: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="Transaction date as YYYY-MM-DD",
        ),
    ] = None,
    plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            "-p",
            help='YNAB plan ID, "last-used", or "default"',
        ),
    ] = None,
    payee_id: Annotated[
        str | None,
        typer.Option("--payee-id", help="Existing YNAB payee UUID"),
    ] = None,
    payee_name: Annotated[
        str | None,
        typer.Option(
            "--payee-name",
            help="Payee name for a new/manual payee",
        ),
    ] = None,
    category_id: Annotated[
        str | None,
        typer.Option("--category-id", help="YNAB category UUID"),
    ] = None,
    memo: Annotated[
        str | None,
        typer.Option("--memo", help="Transaction memo"),
    ] = None,
    cleared: Annotated[
        str,
        typer.Option(
            "--cleared",
            help="Cleared status: uncleared, cleared, or reconciled",
        ),
    ] = "uncleared",
    approved: Annotated[
        bool,
        typer.Option(
            "--approved/--unapproved",
            help="Mark the transaction approved",
        ),
    ] = False,
    import_id: Annotated[
        str | None,
        typer.Option(
            "--import-id",
            help="Optional import id for deduplication",
        ),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Actually create the transaction in YNAB",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Confirm a real write when using --execute",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Create a manual YNAB transaction, dry-running unless --execute."""
    configure_logging(settings)

    if payee_id and payee_name:
        raise typer.BadParameter("Use either --payee-id or --payee-name, not both")

    resolved_plan_id = settings.ynab_plan_id if plan_id is None else plan_id
    transaction_day = date.fromisoformat(transaction_date or date.today().isoformat())
    milliunits = _dollars_to_milliunits(amount)
    payload: dict[str, object] = {
        "plan_id": resolved_plan_id,
        "account_id": account_id,
        "date": transaction_day.isoformat(),
        "amount": milliunits,
        "amount_display": format_milliunits(milliunits),
        "payee_id": payee_id,
        "payee_name": payee_name,
        "category_id": category_id,
        "memo": memo,
        "cleared": cleared,
        "approved": approved,
        "import_id": import_id,
    }

    if not execute:
        output = {"dry_run": True, "transaction": payload}
        if json_output:
            print_json(output)
            return
        console.print("Dry run only. Re-run with --execute --yes to create this transaction.")
        render_rows(
            [payload],
            [
                ("date", "Date"),
                ("amount_display", "Amount"),
                ("payee_name", "Payee"),
                ("memo", "Memo"),
                ("account_id", "Account"),
            ],
        )
        return

    if not yes:
        raise typer.BadParameter("Real writes require --yes with --execute")

    async def run() -> None:
        from ynab_agent.api.client import YnabClient

        async with YnabClient() as client:
            result = await client.create_transaction(
                plan_id=resolved_plan_id,
                account_id=account_id,
                transaction_date=transaction_day,
                amount=milliunits,
                payee_id=payee_id,
                payee_name=payee_name,
                category_id=category_id,
                memo=memo,
                cleared=cleared,
                approved=approved,
                import_id=import_id,
            )
        if json_output:
            print_json(result)
            return
        transaction_ids = cast(list[object], result["transaction_ids"])
        render_rows(
            [
                {
                    "transaction_ids": ", ".join(str(item) for item in transaction_ids),
                    "server_knowledge": result["server_knowledge"],
                }
            ],
            [
                ("transaction_ids", "Transaction IDs"),
                ("server_knowledge", "Server Knowledge"),
            ],
        )

    asyncio.run(run())
