"""Debt payoff projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from typing import Any, TypedDict


MILLIUNITS = Decimal("1000")


class ActiveDebt(TypedDict):
    name: str
    balance: Decimal
    starting_balance: int
    minimum_payment: int
    apr: Decimal
    monthly_fee: int
    payment_available: int
    kind: str
    priority: int | None
    promo_end: str | None
    interest_paid: Decimal
    fees_paid: Decimal
    paid_off_month: str | None
    months_to_payoff: int | None


@dataclass
class DebtInput:
    """One debt or pay-over-time balance to include in a payoff projection."""

    name: str
    balance: int
    minimum_payment: int
    apr: Decimal = Decimal("0")
    monthly_fee: int = 0
    payment_available: int = 0
    kind: str = "account"
    priority: int | None = None
    promo_end: str | None = None


def load_debt_enrichment(path: Path | None) -> dict[str, Any]:
    """Load optional debt metadata from JSON."""
    if path is None or not path.exists():
        return {"accounts": {}, "pay_over_time": []}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "accounts": data.get("accounts", {}),
        "pay_over_time": data.get("pay_over_time", []),
    }


def debt_inputs_from_rows(
    account_rows: list[dict[str, Any]], enrichment: dict[str, Any]
) -> list[DebtInput]:
    """Build projection inputs from cached YNAB rows and enrichment metadata."""
    account_config = enrichment.get("accounts", {})
    debts: list[DebtInput] = []
    for row in account_rows:
        name = str(row["account"])
        config = account_config.get(name, {})
        balance = int(row.get("payoff_needed") or 0)
        if balance <= 0 and not config.get("include_when_zero", False):
            continue
        minimum_payment = _dollars_to_milliunits(config.get("minimum_payment", 0))
        if minimum_payment == 0:
            minimum_payment = int(row.get("goal_target") or 0)
        if minimum_payment == 0:
            minimum_payment = int(row.get("known_minimum_payment") or 0)
        debts.append(
            DebtInput(
                name=name,
                balance=balance,
                minimum_payment=minimum_payment,
                apr=_decimal_rate(config.get("apr", 0)),
                monthly_fee=_dollars_to_milliunits(config.get("monthly_fee", 0)),
                payment_available=int(row.get("payment_available") or 0),
                kind=str(config.get("kind", "credit_card")),
                priority=_optional_int(config.get("priority")),
                promo_end=config.get("promo_end"),
            )
        )

    for item in enrichment.get("pay_over_time", []):
        debts.append(
            DebtInput(
                name=str(item["name"]),
                balance=_dollars_to_milliunits(item.get("balance", 0)),
                minimum_payment=_dollars_to_milliunits(item.get("minimum_payment", 0)),
                apr=_decimal_rate(item.get("apr", 0)),
                monthly_fee=_dollars_to_milliunits(item.get("monthly_fee", 0)),
                kind="pay_over_time",
                priority=_optional_int(item.get("priority")),
                promo_end=item.get("promo_end"),
            )
        )
    return debts


def project_debt_payoff(
    debts: list[DebtInput],
    *,
    monthly_payment: int,
    strategy: str = "avalanche",
    start_month: str,
    max_months: int = 360,
) -> list[dict[str, Any]]:
    """Project payoff timing, interest, and fees for each debt."""
    active: list[ActiveDebt] = [
        {
            "name": debt.name,
            "balance": Decimal(debt.balance),
            "starting_balance": debt.balance,
            "minimum_payment": debt.minimum_payment,
            "apr": debt.apr,
            "monthly_fee": debt.monthly_fee,
            "payment_available": debt.payment_available,
            "kind": debt.kind,
            "priority": debt.priority,
            "promo_end": debt.promo_end,
            "interest_paid": Decimal("0"),
            "fees_paid": Decimal("0"),
            "paid_off_month": None,
            "months_to_payoff": None,
        }
        for debt in debts
        if debt.balance > 0
    ]
    if not active:
        return []

    minimum_total = sum(int(row["minimum_payment"]) for row in active)
    if monthly_payment < minimum_total:
        raise ValueError(
            f"monthly payment {_format_milliunits(monthly_payment)} is below minimums "
            f"{_format_milliunits(minimum_total)}"
        )

    month = date.fromisoformat(start_month[:10]).replace(day=1)
    for month_index in range(1, max_months + 1):
        unpaid = [row for row in active if row["balance"] > 0]
        if not unpaid:
            break

        for row in unpaid:
            interest = _round_milliunits(row["balance"] * row["apr"] / Decimal("12"))
            fee = Decimal(row["monthly_fee"])
            row["balance"] += interest + fee
            row["interest_paid"] += interest
            row["fees_paid"] += fee

        ordered = _strategy_order(unpaid, strategy)
        remaining_payment = Decimal(monthly_payment)
        payments: dict[str, Decimal] = {}
        for row in unpaid:
            payment = min(Decimal(row["minimum_payment"]), row["balance"])
            payments[row["name"]] = payment
            remaining_payment -= payment

        for row in ordered:
            if remaining_payment <= 0:
                break
            extra = min(remaining_payment, row["balance"] - payments[row["name"]])
            if extra > 0:
                payments[row["name"]] += extra
                remaining_payment -= extra

        for row in unpaid:
            row["balance"] -= payments[row["name"]]
            if row["balance"] <= 0 and row["paid_off_month"] is None:
                row["balance"] = Decimal("0")
                row["paid_off_month"] = month.isoformat()
                row["months_to_payoff"] = month_index

        month = _next_month_date(month)

    return [_projection_row(row) for row in active]


def _strategy_order(rows: list[ActiveDebt], strategy: str) -> list[ActiveDebt]:
    if strategy == "snowball":
        return sorted(rows, key=lambda row: (row["priority"] is None, row["priority"] or 0, row["balance"]))
    if strategy == "priority":
        return sorted(rows, key=lambda row: (row["priority"] is None, row["priority"] or 9999))
    return sorted(
        rows,
        key=lambda row: (
            row["priority"] is None,
            row["priority"] or 0,
            -(row["apr"] + _fee_rate(row)),
            -row["balance"],
        ),
    )


def _fee_rate(row: ActiveDebt) -> Decimal:
    if row["balance"] <= 0:
        return Decimal("0")
    return Decimal(row["monthly_fee"]) / row["balance"]


def _projection_row(row: ActiveDebt) -> dict[str, Any]:
    return {
        "name": row["name"],
        "kind": row["kind"],
        "starting_balance": int(row["starting_balance"]),
        "minimum_payment": int(row["minimum_payment"]),
        "apr": row["apr"],
        "monthly_fee": int(row["monthly_fee"]),
        "payment_available": int(row["payment_available"]),
        "interest_paid": int(row["interest_paid"]),
        "fees_paid": int(row["fees_paid"]),
        "paid_off_month": row["paid_off_month"] or "",
        "months_to_payoff": row["months_to_payoff"] or "",
        "remaining_balance": int(row["balance"]),
        "promo_end": row["promo_end"] or "",
    }


def _dollars_to_milliunits(value: object) -> int:
    decimal_value = Decimal(str(value or 0))
    return int((decimal_value * MILLIUNITS).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _decimal_rate(value: object) -> Decimal:
    rate = Decimal(str(value or 0))
    return rate / Decimal("100") if rate > 1 else rate


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _round_milliunits(value: Decimal) -> Decimal:
    return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def _next_month_date(month: date) -> date:
    if month.month == 12:
        return date(month.year + 1, 1, 1)
    return date(month.year, month.month + 1, 1)


def _format_milliunits(amount: int) -> str:
    return f"{amount / 1000:,.2f}"
