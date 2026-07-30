"""Debt-report calculations independent of CLI and database details."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
import json
from typing import Any, Protocol

from ynab_agent.debt_planner import debt_inputs_from_rows, project_debt_payoff


_MONTHLY_FREQUENCY_MULTIPLIERS = {
    "never": Decimal(1),
    "daily": Decimal("365.25") / Decimal(12),
    "weekly": Decimal(52) / Decimal(12),
    "everyotherweek": Decimal(26) / Decimal(12),
    "twiceamonth": Decimal(2),
    "every4weeks": Decimal(13) / Decimal(12),
    "monthly": Decimal(1),
    "everyothermonth": Decimal(1) / Decimal(2),
    "every3months": Decimal(1) / Decimal(3),
    "quarterly": Decimal(1) / Decimal(3),
    "every4months": Decimal(1) / Decimal(4),
    "twiceayear": Decimal(1) / Decimal(6),
    "yearly": Decimal(1) / Decimal(12),
    "annual": Decimal(1) / Decimal(12),
    "everyotheryear": Decimal(1) / Decimal(24),
}


@dataclass(frozen=True)
class DebtAccountSnapshot:
    """Cached account and payment-category facts used by debt reports."""

    account: str
    account_type: str
    balance_milliunits: int
    payment_available_milliunits: int
    cleared_balance_milliunits: int = 0
    uncleared_balance_milliunits: int = 0
    goal_target_milliunits: int = 0
    goal_under_funded_milliunits: int = 0
    goal_overall_left_milliunits: int = 0
    minimum_payment_data: str | None = None


@dataclass(frozen=True)
class ScheduledDebtObligation:
    """One debt-like scheduled outflow included in the drag report."""

    group_name: str
    category: str
    payee: str
    amount_milliunits: int
    frequency: str


class DebtReportRepository(Protocol):
    """Read-only cache access needed by the debt report family."""

    async def list_credit_cards(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]: ...

    async def list_debt_drag_accounts(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]: ...

    async def list_debt_plan_accounts(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]: ...

    async def list_scheduled_debt_obligations(
        self,
        *,
        through: date,
        plan_id: str | None,
    ) -> Sequence[ScheduledDebtObligation]: ...


class UnsupportedScheduledDebtFrequencyError(ValueError):
    """Raised when a scheduled obligation cannot be normalized monthly."""

    frequency: str

    def __init__(self, frequency: str) -> None:
        self.frequency = frequency
        super().__init__(
            f"Unsupported scheduled debt frequency {frequency!r}; "
            "update the cached schedule before calculating monthly debt drag"
        )


@dataclass(frozen=True)
class CreditCardCoverage:
    account: str
    balance_milliunits: int
    cleared_balance_milliunits: int
    uncleared_balance_milliunits: int
    payment_available_milliunits: int
    payoff_needed_milliunits: int
    float_shortfall_milliunits: int
    minimum_payment_data: str | None


@dataclass(frozen=True)
class DebtDragSummary:
    through: date
    payoff_needed_milliunits: int
    payment_available_milliunits: int
    float_shortfall_milliunits: int
    scheduled_obligations_milliunits: int
    known_card_minimums_milliunits: int
    monthly_debt_drag_milliunits: int


class DebtStrategy(StrEnum):
    AVALANCHE = "avalanche"
    SNOWBALL = "snowball"
    PRIORITY = "priority"


@dataclass(frozen=True)
class DebtPlanRequest:
    start_month: date
    monthly_payment_milliunits: int
    strategy: DebtStrategy = DebtStrategy.AVALANCHE
    max_months: int = 360
    plan_id: str | None = None

    def __post_init__(self) -> None:
        if self.start_month.day != 1:
            raise ValueError("debt plan start month must be the first day of a month")
        if self.monthly_payment_milliunits < 0:
            raise ValueError("monthly debt payment cannot be negative")
        if self.max_months < 1:
            raise ValueError("debt plan max months must be at least one")


@dataclass(frozen=True)
class DebtProjection:
    name: str
    kind: str
    starting_balance_milliunits: int
    minimum_payment_milliunits: int
    apr: Decimal
    monthly_fee_milliunits: int
    payment_available_milliunits: int
    interest_paid_milliunits: int
    fees_paid_milliunits: int
    paid_off_month: str
    months_to_payoff: int | None
    remaining_balance_milliunits: int
    promo_end: str


class DebtReportService:
    """Build typed debt reports from a narrow read-only repository."""

    def __init__(self, repository: DebtReportRepository) -> None:
        self.repository = repository

    async def credit_card_coverage(
        self,
        *,
        plan_id: str | None = None,
    ) -> tuple[CreditCardCoverage, ...]:
        snapshots = await self.repository.list_credit_cards(plan_id=plan_id)
        rows = tuple(_credit_card_coverage(snapshot) for snapshot in snapshots)
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    -row.float_shortfall_milliunits,
                    -row.payoff_needed_milliunits,
                    row.account,
                ),
            )
        )

    async def debt_drag(
        self,
        *,
        through: date,
        plan_id: str | None = None,
    ) -> DebtDragSummary:
        accounts = await self.repository.list_debt_drag_accounts(plan_id=plan_id)
        obligations = await self.repository.list_scheduled_debt_obligations(
            through=through,
            plan_id=plan_id,
        )
        coverages = tuple(_credit_card_coverage(account) for account in accounts)
        payoff_needed = sum(row.payoff_needed_milliunits for row in coverages)
        payment_available = sum(row.payment_available_milliunits for row in coverages)
        float_shortfall = sum(row.float_shortfall_milliunits for row in coverages)
        scheduled_monthly = round(
            sum(
                (
                    _monthly_scheduled_amount(obligation)
                    for obligation in obligations
                ),
                start=Decimal(0),
            )
        )
        known_minimums = sum(
            minimum
            for account in accounts
            if (minimum := extract_first_milliunit_amount(account.minimum_payment_data))
            is not None
        )
        return DebtDragSummary(
            through=through,
            payoff_needed_milliunits=payoff_needed,
            payment_available_milliunits=payment_available,
            float_shortfall_milliunits=float_shortfall,
            scheduled_obligations_milliunits=scheduled_monthly,
            known_card_minimums_milliunits=known_minimums,
            monthly_debt_drag_milliunits=scheduled_monthly + known_minimums,
        )

    async def debt_plan(
        self,
        request: DebtPlanRequest,
        *,
        enrichment: dict[str, Any],
    ) -> tuple[DebtProjection, ...]:
        accounts = await self.repository.list_debt_plan_accounts(
            plan_id=request.plan_id
        )
        account_rows = [_planner_account_row(account) for account in accounts]
        debts = debt_inputs_from_rows(account_rows, enrichment)
        projections = project_debt_payoff(
            debts,
            monthly_payment=request.monthly_payment_milliunits,
            strategy=request.strategy.value,
            start_month=request.start_month.isoformat(),
            max_months=request.max_months,
        )
        return tuple(_debt_projection(row) for row in projections)


def extract_first_milliunit_amount(value: str | None) -> int | None:
    """Return the first amount-like number from YNAB debt metadata."""
    if not value:
        return None
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return None
    return _walk_amount(parsed)


def _monthly_scheduled_amount(
    obligation: ScheduledDebtObligation,
) -> Decimal:
    frequency = obligation.frequency.strip().casefold()
    try:
        multiplier = _MONTHLY_FREQUENCY_MULTIPLIERS[frequency]
    except KeyError as exc:
        raise UnsupportedScheduledDebtFrequencyError(
            obligation.frequency
        ) from exc
    return Decimal(max(0, -obligation.amount_milliunits)) * multiplier


def _walk_amount(node: object) -> int | None:
    if isinstance(node, int):
        return node
    if isinstance(node, float):
        return int(node)
    if isinstance(node, dict):
        for key in ("amount", "minimum_payment", "minimumPayment", "payment"):
            found = _walk_amount(node.get(key))
            if found is not None:
                return found
        for nested in node.values():
            found = _walk_amount(nested)
            if found is not None:
                return found
    if isinstance(node, list):
        for nested in node:
            found = _walk_amount(nested)
            if found is not None:
                return found
    return None


def _credit_card_coverage(snapshot: DebtAccountSnapshot) -> CreditCardCoverage:
    payoff_needed = max(0, -snapshot.balance_milliunits)
    payment_available = snapshot.payment_available_milliunits
    return CreditCardCoverage(
        account=snapshot.account,
        balance_milliunits=snapshot.balance_milliunits,
        cleared_balance_milliunits=snapshot.cleared_balance_milliunits,
        uncleared_balance_milliunits=snapshot.uncleared_balance_milliunits,
        payment_available_milliunits=payment_available,
        payoff_needed_milliunits=payoff_needed,
        float_shortfall_milliunits=(
            max(0, payoff_needed - payment_available)
            if snapshot.balance_milliunits < 0
            else 0
        ),
        minimum_payment_data=snapshot.minimum_payment_data,
    )


def _planner_account_row(account: DebtAccountSnapshot) -> dict[str, Any]:
    return {
        "account": account.account,
        "payoff_needed": max(0, -account.balance_milliunits),
        "payment_available": account.payment_available_milliunits,
        "goal_target": account.goal_target_milliunits,
        "known_minimum_payment": (
            extract_first_milliunit_amount(account.minimum_payment_data) or 0
        ),
    }


def _debt_projection(row: Mapping[str, Any]) -> DebtProjection:
    months = row["months_to_payoff"]
    return DebtProjection(
        name=str(row["name"]),
        kind=str(row["kind"]),
        starting_balance_milliunits=int(row["starting_balance"]),
        minimum_payment_milliunits=int(row["minimum_payment"]),
        apr=Decimal(row["apr"]),
        monthly_fee_milliunits=int(row["monthly_fee"]),
        payment_available_milliunits=int(row["payment_available"]),
        interest_paid_milliunits=int(row["interest_paid"]),
        fees_paid_milliunits=int(row["fees_paid"]),
        paid_off_month=str(row["paid_off_month"]),
        months_to_payoff=int(months) if months != "" else None,
        remaining_balance_milliunits=int(row["remaining_balance"]),
        promo_end=str(row["promo_end"]),
    )
