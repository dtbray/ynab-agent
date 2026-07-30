"""Typer adapter for the cost-to-be-me report."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated

import typer

from ynab_agent.cli_support import (
    configure_logging,
    format_milliunits,
    print_json,
    render_rows,
)
from ynab_agent.config import settings
from ynab_agent.db.cost_to_be_me import SqlCostToBeMeRepository
from ynab_agent.runtime import open_database
from ynab_agent.services.reports.cost_to_be_me import (
    CostToBeMeReport,
    CostToBeMeRequest,
    CostToBeMeService,
    MissingCostToBeMeMonthError,
)


cost_to_be_me_app = typer.Typer()


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


def _dollars_to_milliunits(amount: str, *, option: str) -> int:
    try:
        decimal_amount = Decimal(amount)
    except Exception as exc:
        raise typer.BadParameter(
            "must be a decimal dollar amount",
            param_hint=option,
        ) from exc
    return int(
        (decimal_amount * Decimal("1000")).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _report_row(report: CostToBeMeReport) -> dict[str, object]:
    return {
        "month": report.month.isoformat(),
        "monthly_funding_plan": format_milliunits(report.monthly_funding_plan_milliunits),
        "assigned": format_milliunits(report.assigned_milliunits),
        "still_underfunded": format_milliunits(report.still_underfunded_milliunits),
        "goal_period_targets": format_milliunits(report.goal_period_targets_milliunits),
        "observed_spending_median": format_milliunits(report.observed_spending_median_milliunits),
        "spending_window": report.spending_window,
        "spending_samples": report.spending_samples,
        "expected_income": format_milliunits(report.expected_income_milliunits),
        "surplus_or_shortfall": format_milliunits(report.surplus_or_shortfall_milliunits),
        "income_source": report.income_source,
        "income_window": report.income_window,
        "income_samples": report.income_samples,
        "funding_category_count": report.funding_category_count,
    }


@cost_to_be_me_app.command("cost-to-be-me")
def report_cost_to_be_me(
    month: Annotated[
        str | None,
        typer.Option("--month", help="ISO month date"),
    ] = None,
    expected_income: Annotated[
        str | None,
        typer.Option(
            "--expected-income",
            help=(
                "Manual expected monthly income in dollars; otherwise "
                "infer from Ready to Assign inflows"
            ),
        ),
    ] = None,
    income_months: Annotated[
        int,
        typer.Option(
            "--income-months",
            min=1,
            help=("Complete prior months to average when inferring expected income"),
        ),
    ] = 6,
    spending_months: Annotated[
        int,
        typer.Option(
            "--spending-months",
            min=1,
            help=("Complete prior months used for the observed median spending baseline"),
        ),
    ] = 12,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Compare current-month funding needs and observed spending with income."""
    configure_logging(settings)
    target_month = _iso_date(month, option="--month") if month is not None else _current_month()
    manual_income = (
        _dollars_to_milliunits(
            expected_income,
            option="--expected-income",
        )
        if expected_income is not None
        else None
    )
    try:
        request = CostToBeMeRequest(
            target_month=target_month,
            expected_income_milliunits=manual_income,
            income_months=income_months,
            spending_months=spending_months,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def run() -> CostToBeMeReport:
        async with open_database(settings) as database:
            service = CostToBeMeService(SqlCostToBeMeRepository(database))
            return await service.build(request)

    try:
        report = asyncio.run(run())
    except MissingCostToBeMeMonthError as exc:
        raise typer.BadParameter(str(exc)) from exc
    row = _report_row(report)
    if json_output:
        print_json([row])
        return
    render_rows(
        [row],
        [
            ("month", "Month"),
            ("monthly_funding_plan", "Funding Plan"),
            ("assigned", "Assigned"),
            ("still_underfunded", "Still Underfunded"),
            ("goal_period_targets", "Goal Period Targets*"),
            ("observed_spending_median", "Observed Median Spend"),
            ("spending_window", "Spending Window"),
            ("spending_samples", "Spending Months"),
            ("expected_income", "Expected Income"),
            ("surplus_or_shortfall", "Surplus/Shortfall"),
            ("income_source", "Income Source"),
            ("income_window", "Income Window"),
            ("income_samples", "Income Months"),
            ("funding_category_count", "Funding Categories"),
        ],
    )
