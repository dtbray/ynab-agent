"""Account reconciliation Typer adapter."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated, cast

import typer

from ynab_agent.api.client import SyncResult, YnabClient
from ynab_agent.cli_support import format_milliunits, print_json, render_rows
from ynab_agent.config import settings
from ynab_agent.services.reconciliation import (
    AccountReconciliation,
    ReconcileBatchResponse,
    ReconciliationApplyResult,
    ReconciliationRequest,
    ReconciliationService,
    Row,
    classify_account,
)


reconciliation_app = typer.Typer()


class _ClientAdapter:
    def __init__(self, client: YnabClient) -> None:
        self._client = client

    async def get_accounts(self, *, plan_id: str) -> list[Row]:
        result = await self._client.get_accounts(plan_id=plan_id)
        rows = result.data if isinstance(result, SyncResult) else result
        return cast(list[Row], rows)

    async def get_transactions(
        self,
        *,
        plan_id: str,
        since_date: date | None,
    ) -> list[Row]:
        result = await self._client.get_transactions(
            plan_id=plan_id,
            since_date=since_date,
        )
        rows = result.data if isinstance(result, SyncResult) else result
        return cast(list[Row], rows)

    async def reconcile_transactions(
        self,
        *,
        plan_id: str,
        transaction_ids: list[str],
    ) -> ReconcileBatchResponse:
        result = await self._client.reconcile_transactions(
            plan_id=plan_id,
            transaction_ids=transaction_ids,
        )
        raw_ids = result.get("transaction_ids", ())
        ids = tuple(str(item) for item in raw_ids) if isinstance(raw_ids, list) else ()
        raw_knowledge = result.get("server_knowledge")
        return ReconcileBatchResponse(
            transaction_ids=ids,
            server_knowledge=(
                int(raw_knowledge) if isinstance(raw_knowledge, int) else None
            ),
        )


def _account_row(
    account: AccountReconciliation,
    *,
    execute: bool,
    result: ReconciliationApplyResult | None,
) -> dict[str, object]:
    reconciled_ids = (
        list(result.reconciled_transaction_ids(account.account_id))
        if result is not None
        else []
    )
    reconciled_count = (
        len(reconciled_ids)
        if reconciled_ids
        else account.reconciled_transaction_count
    )
    return {
        "id": account.account_id,
        "name": account.name,
        "type": account.account_type,
        "status": account.status.value,
        "balance": account.balance,
        "cleared_balance": account.cleared_balance,
        "uncleared_balance": account.uncleared_balance,
        "uncleared_delta": account.uncleared_delta,
        "eligible_transactions": len(account.eligible_transaction_ids),
        "uncleared_transactions": account.uncleared_transaction_count,
        "reconciled_transactions": reconciled_count,
        "last_reconciled_at": account.last_reconciled_at,
        "reasons": list(account.reasons),
        "reason": account.reason,
        "eligible_transaction_ids": list(account.eligible_transaction_ids),
        "dry_run": not execute,
        "reconciled_transaction_ids": reconciled_ids,
    }


def reconciliation_status_compat(
    account: dict[str, object],
    transactions: list[dict[str, object]],
) -> dict[str, object]:
    """Preserve the legacy private helper while callers migrate to the service."""
    classified = classify_account(account, transactions)
    row = _account_row(classified, execute=False, result=None)
    row.pop("dry_run")
    row.pop("reconciled_transaction_ids")
    return row


def _display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **row,
            "balance": format_milliunits(cast(int, row["balance"])),
            "cleared_balance": format_milliunits(
                cast(int, row["cleared_balance"])
            ),
            "uncleared_balance": format_milliunits(
                cast(int, row["uncleared_balance"])
            ),
            "uncleared_delta": format_milliunits(
                cast(int, row["uncleared_delta"])
            ),
        }
        for row in rows
    ]


@reconciliation_app.command("reconcile")
def reconcile(
    plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            "-p",
            help='YNAB plan ID, "last-used", or "default"',
        ),
    ] = None,
    account_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--account-id",
            help="Limit reconciliation to an account UUID; repeatable",
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Only inspect account transactions on/after ISO date",
        ),
    ] = None,
    include_closed: Annotated[
        bool,
        typer.Option("--include-closed", help="Include closed accounts"),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Mark eligible cleared transactions reconciled",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Confirm real writes when using --execute",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Show reconciliation readiness and optionally reconcile ready accounts."""
    if execute and not yes:
        raise typer.BadParameter("Real writes require --yes with --execute")
    try:
        since_date = date.fromisoformat(since) if since else None
    except ValueError as exc:
        raise typer.BadParameter(
            "must use YYYY-MM-DD",
            param_hint="--since",
        ) from exc

    async def run() -> tuple[list[dict[str, object]], bool]:
        from ynab_agent.api.client import YnabClient as RuntimeYnabClient

        async with RuntimeYnabClient() as client:
            adapter = _ClientAdapter(client)
            service = ReconciliationService(adapter, adapter)
            plan = await service.plan(
                ReconciliationRequest(
                    plan_id=plan_id or settings.ynab_plan_id,
                    account_ids=tuple(account_ids or ()),
                    since=since_date,
                    include_closed=include_closed,
                )
            )
            applied = await service.apply(plan) if execute else None
        return (
            [
                _account_row(account, execute=execute, result=applied)
                for account in plan.accounts
            ],
            applied is None or applied.is_complete,
        )

    rows, complete = asyncio.run(run())
    if json_output:
        print_json(rows)
    else:
        if not execute:
            typer.echo(
                "Dry run only. Re-run with --execute --yes to reconcile ready accounts."
            )
        render_rows(
            _display_rows(rows),
            [
                ("name", "Account"),
                ("status", "Status"),
                ("balance", "Balance"),
                ("cleared_balance", "Cleared"),
                ("uncleared_balance", "Uncleared"),
                ("eligible_transactions", "Eligible"),
                ("uncleared_transactions", "Uncleared Txns"),
                ("reason", "Reason"),
            ],
        )
    if not complete:
        raise typer.Exit(code=1)
