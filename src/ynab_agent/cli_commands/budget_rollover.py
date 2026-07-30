"""Budget rollover Typer adapter."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated, cast

import typer

from ynab_agent.api.client import YnabClient
from ynab_agent.cli_support import format_milliunits, print_json, render_rows
from ynab_agent.config import settings
from ynab_agent.db.budget_rollover import SqlBudgetRolloverReader
from ynab_agent.runtime import open_database
from ynab_agent.services.budget_rollover import (
    BudgetRolloverApplyResult,
    BudgetRolloverPlan,
    BudgetRolloverRequest,
    BudgetRolloverService,
    CategoryUpdate,
    MonthCategory,
)


rollover_app = typer.Typer()


class _ClientAdapter:
    def __init__(self, client: YnabClient) -> None:
        self._client = client

    async def get_month_categories(
        self,
        *,
        plan_id: str,
        month: date,
    ) -> tuple[MonthCategory, ...]:
        _, rows = await self._client.get_month(plan_id=plan_id, month=month)
        return tuple(
            MonthCategory(
                category_id=str(row["category_id"]),
                group_name=str(row.get("category_group_name") or "Unknown"),
                name=str(row.get("name") or row["category_id"]),
                budgeted=int(row.get("budgeted") or 0),
                balance=int(row.get("balance") or 0),
                hidden=bool(row.get("hidden")),
                deleted=bool(row.get("deleted")),
            )
            for row in rows
        )

    async def set_category_budget(
        self,
        *,
        plan_id: str,
        month: date,
        category_id: str,
        budgeted: int,
    ) -> CategoryUpdate:
        row = await self._client.update_month_category_budgeted(
            plan_id=plan_id,
            month=month,
            category_id=category_id,
            budgeted=budgeted,
        )
        return CategoryUpdate(
            budgeted=int(row["budgeted"]),
            balance=(
                int(row["balance"])
                if row.get("balance") is not None
                else None
            ),
        )


def _result_rows(
    plan: BudgetRolloverPlan,
    *,
    execute: bool,
    applied: BudgetRolloverApplyResult | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in plan.categories:
        execution = (
            applied.result_for(category.mutation_id)
            if applied is not None
            else None
        )
        source_update = execution.source_update if execution is not None else None
        target_update = execution.target_update if execution is not None else None
        rows.append(
            {
                "source_month": plan.source_month.isoformat(),
                "target_month": plan.target_month.isoformat(),
                "group": category.group_name,
                "category": category.category_name,
                "overspent": category.overspent,
                "source_budgeted_before": category.source_budgeted_before,
                "source_budgeted_after": (
                    source_update.budgeted
                    if source_update is not None
                    else category.source_budgeted_after
                ),
                "source_balance_after": (
                    source_update.balance if source_update is not None else None
                ),
                "target_budgeted_before": category.target_budgeted_before,
                "target_budgeted_after": (
                    target_update.budgeted
                    if target_update is not None
                    else category.target_budgeted_after
                ),
                "target_balance_after": (
                    target_update.balance if target_update is not None else None
                ),
                "dry_run": not execute,
                "status": (
                    execution.status.value
                    if execution is not None
                    else category.status.value
                ),
            }
        )
    return rows


def _display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **row,
            "overspent": format_milliunits(cast(int, row["overspent"])),
            "source_budgeted_before": format_milliunits(
                cast(int, row["source_budgeted_before"])
            ),
            "source_budgeted_after": format_milliunits(
                cast(int, row["source_budgeted_after"])
            ),
            "source_balance_after": format_milliunits(
                cast(int | None, row["source_balance_after"])
            ),
            "target_budgeted_before": format_milliunits(
                cast(int, row["target_budgeted_before"])
            ),
            "target_budgeted_after": format_milliunits(
                cast(int, row["target_budgeted_after"])
            ),
            "target_balance_after": format_milliunits(
                cast(int | None, row["target_balance_after"])
            ),
        }
        for row in rows
    ]


@rollover_app.command("yeet-the-red")
def yeet_the_red(
    plan_id: Annotated[
        str | None,
        typer.Option(
            "--plan-id",
            "-p",
            help='YNAB plan ID, "last-used", or "default"',
        ),
    ] = None,
    month: Annotated[
        str | None,
        typer.Option(
            "--month",
            help="Month with negative balances to push forward",
        ),
    ] = None,
    next_month: Annotated[
        str | None,
        typer.Option(
            "--next-month",
            help="Month to receive the negative budget adjustment; defaults to following month",
        ),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually update YNAB"),
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
    """Push budget category overspending into the next month."""
    if execute and not yes:
        raise typer.BadParameter("Real writes require --yes with --execute")
    today = date.today()
    source_text = month or date(today.year, today.month, 1).isoformat()
    try:
        source_month = date.fromisoformat(source_text[:10])
        target_month = (
            date.fromisoformat(next_month[:10])
            if next_month is not None
            else None
        )
    except ValueError as exc:
        raise typer.BadParameter(
            "months must use YYYY-MM-DD",
            param_hint="--month",
        ) from exc
    request = BudgetRolloverRequest(
        plan_id=plan_id or settings.ynab_plan_id,
        source_month=source_month,
        target_month=target_month,
    )

    async def run() -> tuple[BudgetRolloverPlan, BudgetRolloverApplyResult | None]:
        if execute:
            from ynab_agent.api.client import YnabClient as RuntimeYnabClient

            async with RuntimeYnabClient() as client:
                adapter = _ClientAdapter(client)
                service = BudgetRolloverService(adapter, adapter)
                plan = await service.plan(request)
                return plan, await service.apply(plan)
        async with open_database(settings, initialize=True) as database:
            plan = await BudgetRolloverService(
                SqlBudgetRolloverReader(database)
            ).plan(request)
        return plan, None

    plan, applied = asyncio.run(run())
    rows = _result_rows(plan, execute=execute, applied=applied)
    display_rows = _display_rows(rows)
    if not display_rows:
        display_rows = [
            {
                "source_month": plan.source_month.isoformat(),
                "target_month": plan.target_month.isoformat(),
                "group": "",
                "category": "No budgetable red to yeet",
                "overspent": "0.00",
                "source_budgeted_before": "",
                "source_budgeted_after": "",
                "source_balance_after": "",
                "target_budgeted_before": "",
                "target_budgeted_after": "",
                "target_balance_after": "",
                "dry_run": not execute,
                "status": "nothing_to_do",
            }
        ]
    if json_output:
        print_json(display_rows)
    else:
        render_rows(
            display_rows,
            [
                ("source_month", "From"),
                ("target_month", "To"),
                ("group", "Group"),
                ("category", "Category"),
                ("overspent", "Yeeted"),
                ("source_budgeted_before", "Current Budgeted"),
                ("source_budgeted_after", "Current After"),
                ("target_budgeted_before", "Next Budgeted"),
                ("target_budgeted_after", "After"),
                ("status", "Status"),
            ],
        )
    if applied is not None and not applied.is_complete:
        raise typer.Exit(code=1)
