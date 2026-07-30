"""Typer adapters for the debt report family."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Annotated

import typer

from ynab_agent.cli_support import (
    configure_logging,
    format_milliunits,
    print_json,
    render_rows,
)
from ynab_agent.config import settings
from ynab_agent.debt_planner import load_debt_enrichment
from ynab_agent.db.debt_reports import SqlDebtReportRepository
from ynab_agent.runtime import open_database
from ynab_agent.services.debt_reports import (
    CreditCardCoverage,
    DebtDragSummary,
    DebtPlanRequest,
    DebtProjection,
    DebtReportService,
    DebtStrategy,
)


debt_reports_app = typer.Typer()


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


def _next_month(month: date) -> date:
    if month.month == 12:
        return date(month.year + 1, 1, 1)
    return date(month.year, month.month + 1, 1)


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


def _format_percent(amount: Decimal) -> str:
    return f"{float(amount) * 100:,.2f}%"


def _coverage_row(row: CreditCardCoverage) -> dict[str, object]:
    return {
        "account": row.account,
        "balance": format_milliunits(row.balance_milliunits),
        "cleared_balance": format_milliunits(row.cleared_balance_milliunits),
        "uncleared_balance": format_milliunits(row.uncleared_balance_milliunits),
        "payment_available": format_milliunits(row.payment_available_milliunits),
        "payoff_needed": format_milliunits(row.payoff_needed_milliunits),
        "float_shortfall": format_milliunits(row.float_shortfall_milliunits),
        "minimum_payment_data": row.minimum_payment_data,
    }


def _drag_rows(summary: DebtDragSummary) -> list[dict[str, object]]:
    return [
        {
            "metric": "debt_payoff_needed",
            "amount": format_milliunits(summary.payoff_needed_milliunits),
            "note": "Current negative debt/credit balances",
        },
        {
            "metric": "budgeted_for_payoff",
            "amount": format_milliunits(summary.payment_available_milliunits),
            "note": "Credit-card payment category coverage",
        },
        {
            "metric": "credit_card_float_shortfall",
            "amount": format_milliunits(summary.float_shortfall_milliunits),
            "note": "Payoff need not covered by payment categories",
        },
        {
            "metric": "scheduled_monthly_debt_obligations",
            "amount": format_milliunits(summary.scheduled_obligations_milliunits),
            "note": (f"Debt-like scheduled outflows through {summary.through.isoformat()}"),
        },
        {
            "metric": "known_card_minimums",
            "amount": format_milliunits(summary.known_card_minimums_milliunits),
            "note": "Parsed from YNAB debt metadata when present",
        },
        {
            "metric": "monthly_debt_drag",
            "amount": format_milliunits(summary.monthly_debt_drag_milliunits),
            "note": "Scheduled debt obligations plus known card minimums",
        },
    ]


def _projection_row(row: DebtProjection) -> dict[str, object]:
    return {
        "name": row.name,
        "kind": row.kind,
        "starting_balance": format_milliunits(row.starting_balance_milliunits),
        "minimum_payment": format_milliunits(row.minimum_payment_milliunits),
        "apr": _format_percent(row.apr),
        "monthly_fee": format_milliunits(row.monthly_fee_milliunits),
        "payment_available": format_milliunits(row.payment_available_milliunits),
        "interest_paid": format_milliunits(row.interest_paid_milliunits),
        "fees_paid": format_milliunits(row.fees_paid_milliunits),
        "paid_off_month": row.paid_off_month,
        "months_to_payoff": row.months_to_payoff or "",
        "remaining_balance": format_milliunits(row.remaining_balance_milliunits),
        "promo_end": row.promo_end,
    }


@debt_reports_app.command("credit-cards")
def report_credit_cards(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Show cached credit-card payoff coverage and likely float exposure."""
    configure_logging(settings)

    async def run() -> tuple[CreditCardCoverage, ...]:
        async with open_database(settings) as database:
            service = DebtReportService(SqlDebtReportRepository(database))
            return await service.credit_card_coverage()

    rows = [_coverage_row(row) for row in asyncio.run(run())]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("account", "Card"),
            ("balance", "Balance"),
            ("payment_available", "Payment Available"),
            ("payoff_needed", "Payoff Needed"),
            ("float_shortfall", "Float Shortfall"),
            ("cleared_balance", "Cleared"),
            ("uncleared_balance", "Uncleared"),
            ("minimum_payment_data", "Minimum Payment Data"),
        ],
    )


@debt_reports_app.command("debt-drag")
def report_debt_drag(
    month: Annotated[
        str | None,
        typer.Option("--month", help="ISO month date"),
    ] = None,
    through: Annotated[
        str | None,
        typer.Option(
            "--through",
            help="Include scheduled obligations due on/before ISO date",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Summarize debt balances, payoff coverage, and monthly obligations."""
    configure_logging(settings)
    target_month = _iso_date(month, option="--month") if month is not None else _current_month()
    through_date = (
        _iso_date(through, option="--through") if through is not None else _next_month(target_month)
    )

    async def run() -> DebtDragSummary:
        async with open_database(settings) as database:
            service = DebtReportService(SqlDebtReportRepository(database))
            return await service.debt_drag(through=through_date)

    rows = _drag_rows(asyncio.run(run()))
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [("metric", "Metric"), ("amount", "Amount"), ("note", "Note")],
    )


@debt_reports_app.command("debt-plan")
def report_debt_plan(
    month: Annotated[
        str | None,
        typer.Option("--month", help="ISO month date"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=("JSON file with APRs, minimums, priorities, and pay-over-time balances"),
        ),
    ] = None,
    monthly_payment: Annotated[
        str,
        typer.Option(
            "--monthly-payment",
            help="Total monthly amount available for debts, in dollars",
        ),
    ] = "0",
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help='Payoff order: "avalanche", "snowball", or "priority"',
        ),
    ] = "avalanche",
    max_months: Annotated[
        int,
        typer.Option("--max-months", min=1, help="Projection cap"),
    ] = 360,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Project debt payoff using cached balances plus local metadata."""
    configure_logging(settings)
    try:
        debt_strategy = DebtStrategy(strategy)
    except ValueError as exc:
        raise typer.BadParameter(
            'strategy must be "avalanche", "snowball", or "priority"',
            param_hint="--strategy",
        ) from exc
    start_month = (
        _iso_date(month, option="--month") if month is not None else _current_month()
    ).replace(day=1)
    payment = _dollars_to_milliunits(
        monthly_payment,
        option="--monthly-payment",
    )
    try:
        enrichment = load_debt_enrichment(config)
        request = DebtPlanRequest(
            start_month=start_month,
            monthly_payment_milliunits=payment,
            strategy=debt_strategy,
            max_months=max_months,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def run() -> tuple[DebtProjection, ...]:
        async with open_database(settings) as database:
            service = DebtReportService(SqlDebtReportRepository(database))
            return await service.debt_plan(
                request,
                enrichment=enrichment,
            )

    try:
        projections = asyncio.run(run())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    rows = [_projection_row(row) for row in projections]
    if json_output:
        print_json(rows)
        return
    render_rows(
        rows,
        [
            ("name", "Debt"),
            ("kind", "Kind"),
            ("starting_balance", "Balance"),
            ("minimum_payment", "Min"),
            ("apr", "APR"),
            ("monthly_fee", "Fee"),
            ("payment_available", "YNAB Available"),
            ("months_to_payoff", "Months"),
            ("paid_off_month", "Paid Off"),
            ("interest_paid", "Interest"),
            ("fees_paid", "Fees"),
            ("remaining_balance", "Remaining"),
            ("promo_end", "Promo End"),
        ],
    )
