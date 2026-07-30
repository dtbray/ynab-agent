"""Typer adapters for cached category activity reports."""

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
from ynab_agent.db.budget_activity_reports import (
    SqlBudgetActivityRepository,
)
from ynab_agent.runtime import open_database
from ynab_agent.services.reports.budget_activity import (
    BudgetActivityRequest,
    BudgetActivityService,
    CategoryActivity,
    OverspendingCategory,
)


budget_activity_reports_app = typer.Typer()


def _current_month() -> date:
    today = date.today()
    return date(today.year, today.month, 1)


def _iso_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise typer.BadParameter(
            "must use YYYY-MM-DD",
            param_hint=option,
        ) from exc


def _activity_row(row: CategoryActivity) -> dict[str, object]:
    return {
        "group_name": row.group_name,
        "category": row.category,
        "budgeted": format_milliunits(row.budgeted_milliunits),
        "activity": format_milliunits(row.activity_milliunits),
        "balance": format_milliunits(row.balance_milliunits),
    }


def _overspending_row(
    row: OverspendingCategory,
) -> dict[str, object]:
    return {
        **_activity_row(
            CategoryActivity(
                group_name=row.group_name,
                category=row.category,
                budgeted_milliunits=row.budgeted_milliunits,
                activity_milliunits=row.activity_milliunits,
                balance_milliunits=row.balance_milliunits,
            )
        ),
        "overspent": format_milliunits(row.overspent_milliunits),
    }


def _request(month: str | None, *, limit: int) -> BudgetActivityRequest:
    target_month = _iso_date(month, option="--month") if month is not None else _current_month()
    try:
        return BudgetActivityRequest(month=target_month, limit=limit)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@budget_activity_reports_app.command("overspending")
def report_overspending(
    month: Annotated[
        str | None,
        typer.Option("--month", help="ISO month date"),
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
    """List cached categories with negative available balances for a budget month."""
    configure_logging(settings)
    request = _request(month, limit=limit)

    async def run() -> tuple[OverspendingCategory, ...]:
        async with open_database(settings) as database:
            service = BudgetActivityService(SqlBudgetActivityRepository(database))
            return await service.overspending(request)

    rows = [_overspending_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("group_name", "Group"),
            ("category", "Category"),
            ("overspent", "Overspent"),
            ("balance", "Available"),
            ("activity", "Activity"),
            ("budgeted", "Budgeted"),
        ],
    )


@budget_activity_reports_app.command("hidden-funds")
def report_hidden_funds(
    month: Annotated[
        str | None,
        typer.Option("--month", help="ISO month date"),
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
    """List hidden categories with assigned or available funds for a budget month."""
    configure_logging(settings)
    request = _request(month, limit=limit)

    async def run() -> tuple[CategoryActivity, ...]:
        async with open_database(settings) as database:
            service = BudgetActivityService(SqlBudgetActivityRepository(database))
            return await service.hidden_funds(request)

    rows = [_activity_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("group_name", "Group"),
            ("category", "Category"),
            ("budgeted", "Assigned"),
            ("balance", "Available"),
            ("activity", "Activity"),
        ],
    )


@budget_activity_reports_app.command("month")
def report_month(
    month: Annotated[
        str | None,
        typer.Option("--month", help="ISO month date"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            min=1,
            help="Maximum category rows",
        ),
    ] = 15,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Show cached category activity for one budget month."""
    configure_logging(settings)
    request = _request(month, limit=limit)

    async def run() -> tuple[CategoryActivity, ...]:
        async with open_database(settings) as database:
            service = BudgetActivityService(SqlBudgetActivityRepository(database))
            return await service.month_activity(request)

    rows = [_activity_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("group_name", "Group"),
            ("category", "Category"),
            ("activity", "Activity"),
            ("balance", "Balance"),
            ("budgeted", "Budgeted"),
        ],
    )
