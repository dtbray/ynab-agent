"""Deterministic mortgage projections from cached YNAB loan metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from typing import Mapping


@dataclass(frozen=True)
class MortgageProjection:
    """A replayable payoff estimate based on one cached mortgage snapshot."""

    account_id: str
    as_of: date
    current_balance: float
    annual_interest_rate: float
    monthly_payment: float
    monthly_principal_and_interest: float
    monthly_escrow: float
    remaining_months: int
    estimated_payoff_date: date
    estimated_payoff_age: int
    source_sha256: str

    @property
    def annual_principal_and_interest(self) -> float:
        return self.monthly_principal_and_interest * 12

    @property
    def annual_escrow(self) -> float:
        return self.monthly_escrow * 12


def project_mortgage(
    *,
    account_id: str,
    balance_milliunits: int,
    interest_rates: Mapping[str, int],
    minimum_payments: Mapping[str, int],
    escrow_amounts: Mapping[str, int],
    as_of: date,
    current_age: int,
) -> MortgageProjection:
    """Estimate payoff timing from YNAB's dated loan terms."""
    if current_age < 0 or current_age > 100:
        raise ValueError("current_age must be between 0 and 100")
    balance = abs(balance_milliunits) / 1000
    if balance <= 0:
        raise ValueError("mortgage balance must be non-zero")

    raw_interest_rate = _latest_value(interest_rates, as_of, "interest rate")
    raw_payment = _latest_value(minimum_payments, as_of, "minimum payment")
    raw_escrow = _latest_value(escrow_amounts, as_of, "escrow amount", default=0)

    annual_interest_rate = raw_interest_rate / 100_000
    monthly_payment = raw_payment / 1000
    monthly_escrow = raw_escrow / 1000
    monthly_principal_and_interest = monthly_payment - monthly_escrow
    if monthly_principal_and_interest <= 0:
        raise ValueError(
            "mortgage payment must exceed its escrow amount"
        )

    monthly_rate = annual_interest_rate / 12
    if monthly_rate == 0:
        remaining_months = math.ceil(
            balance / monthly_principal_and_interest
        )
    else:
        first_month_interest = balance * monthly_rate
        if monthly_principal_and_interest <= first_month_interest:
            raise ValueError(
                "mortgage payment does not amortize the current balance"
            )
        remaining_months = math.ceil(
            -math.log(
                1
                - (
                    monthly_rate
                    * balance
                    / monthly_principal_and_interest
                )
            )
            / math.log(1 + monthly_rate)
        )

    payoff_date = _add_months(as_of.replace(day=1), remaining_months)
    payoff_age = current_age + math.ceil(remaining_months / 12)
    if payoff_age > 120:
        raise ValueError("mortgage payoff age exceeds the planner horizon")
    source = {
        "account_id": account_id,
        "as_of": as_of.isoformat(),
        "balance_milliunits": balance_milliunits,
        "interest_rate_milli_percent": raw_interest_rate,
        "minimum_payment_milliunits": raw_payment,
        "escrow_milliunits": raw_escrow,
    }
    source_sha256 = hashlib.sha256(
        json.dumps(
            source,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return MortgageProjection(
        account_id=account_id,
        as_of=as_of,
        current_balance=balance,
        annual_interest_rate=annual_interest_rate,
        monthly_payment=monthly_payment,
        monthly_principal_and_interest=monthly_principal_and_interest,
        monthly_escrow=monthly_escrow,
        remaining_months=remaining_months,
        estimated_payoff_date=payoff_date,
        estimated_payoff_age=payoff_age,
        source_sha256=source_sha256,
    )


def _latest_value(
    history: Mapping[str, int],
    as_of: date,
    label: str,
    *,
    default: int | None = None,
) -> int:
    eligible = [
        (date.fromisoformat(day), int(value))
        for day, value in history.items()
        if date.fromisoformat(day) <= as_of
    ]
    if not eligible:
        if default is not None:
            return default
        raise ValueError(f"mortgage has no {label} on or before {as_of}")
    return max(eligible, key=lambda item: item[0])[1]


def _add_months(month: date, delta: int) -> date:
    month_index = month.year * 12 + month.month - 1 + delta
    year, zero_based_month = divmod(month_index, 12)
    return date(year, zero_based_month + 1, 1)
