"""Typer adapters for cached cashflow and spending reports."""

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
from ynab_agent.db.spending_reports import SqlSpendingReportRepository
from ynab_agent.runtime import open_database
from ynab_agent.services.reports.spending import (
    BurnRateBucket,
    BurnRateGrouping,
    CashflowMonth,
    DateRangeRequest,
    PayeeSpend,
    SpendingReportService,
)


spending_reports_app = typer.Typer()


def _iso_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise typer.BadParameter(
            "must use YYYY-MM-DD",
            param_hint=option,
        ) from exc


def _grouping(value: str) -> BurnRateGrouping:
    try:
        return BurnRateGrouping(value)
    except ValueError as exc:
        raise typer.BadParameter(
            'group-by must be "category" or "group"',
            param_hint="--group-by",
        ) from exc


def _cashflow_row(row: CashflowMonth) -> dict[str, object]:
    return {
        "month": row.month,
        "inflow": format_milliunits(row.inflow_milliunits),
        "outflow": format_milliunits(row.outflow_milliunits),
        "net": format_milliunits(row.net_milliunits),
        "transactions": row.transaction_count,
    }


def _burn_rate_row(row: BurnRateBucket) -> dict[str, object]:
    return {
        "bucket": row.bucket,
        "outflow": format_milliunits(row.outflow_milliunits),
        "transactions": row.transaction_count,
        "active_months": row.active_months,
        "monthly_burn": format_milliunits(row.monthly_burn_milliunits),
    }


def _payee_row(row: PayeeSpend) -> dict[str, object]:
    return {
        "payee": row.payee,
        "outflow": format_milliunits(row.outflow_milliunits),
        "transactions": row.transaction_count,
    }


@spending_reports_app.command("cashflow")
def report_cashflow(
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
    """Summarize categorized non-transfer cashflow by month."""
    configure_logging(settings)
    request = DateRangeRequest(since=_iso_date(since, option="--since"))

    async def run() -> tuple[CashflowMonth, ...]:
        async with open_database(settings) as database:
            service = SpendingReportService(SqlSpendingReportRepository(database))
            return await service.cashflow(request)

    rows = [_cashflow_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("month", "Month"),
            ("inflow", "Inflow"),
            ("outflow", "Outflow"),
            ("net", "Net"),
            ("transactions", "Txns"),
        ],
    )


@spending_reports_app.command("burn-rate")
def report_burn_rate(
    since: Annotated[
        str,
        typer.Option(
            "--since",
            help="Include transactions since ISO date",
        ),
    ] = "2026-01-01",
    through: Annotated[
        str | None,
        typer.Option(
            "--through",
            help="Include transactions up to ISO date",
        ),
    ] = None,
    group_by: Annotated[
        str,
        typer.Option(
            "--group-by",
            help='Group by "category" or "group"',
        ),
    ] = "category",
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
    """Show cached average monthly outflow by category or category group."""
    configure_logging(settings)
    grouping = _grouping(group_by)
    request = DateRangeRequest(
        since=_iso_date(since, option="--since"),
        through=(_iso_date(through, option="--through") if through is not None else None),
        limit=limit,
    )
    current_date = date.today()

    async def run() -> tuple[BurnRateBucket, ...]:
        async with open_database(settings) as database:
            service = SpendingReportService(SqlSpendingReportRepository(database))
            return await service.burn_rate(
                request,
                grouping=grouping,
                current_date=current_date,
            )

    rows = [_burn_rate_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            (
                "bucket",
                "Category" if grouping is BurnRateGrouping.CATEGORY else "Group",
            ),
            ("monthly_burn", "Monthly Burn"),
            ("outflow", "Outflow"),
            ("transactions", "Txns"),
            ("active_months", "Active Months"),
        ],
    )


@spending_reports_app.command("top-spend")
def report_top_spend(
    since: Annotated[
        str,
        typer.Option(
            "--since",
            help="Include transactions since ISO date",
        ),
    ] = "2026-01-01",
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
    """List top cached outflows by payee."""
    configure_logging(settings)
    request = DateRangeRequest(
        since=_iso_date(since, option="--since"),
        limit=limit,
    )

    async def run() -> tuple[PayeeSpend, ...]:
        async with open_database(settings) as database:
            service = SpendingReportService(SqlSpendingReportRepository(database))
            return await service.top_spend(request)

    rows = [_payee_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("payee", "Payee"),
            ("outflow", "Outflow"),
            ("transactions", "Txns"),
        ],
    )
