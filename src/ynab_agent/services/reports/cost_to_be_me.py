"""Typed cost-to-be-me report service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Protocol


@dataclass(frozen=True)
class FundingCategory:
    """Current-month category funding facts."""

    group_name: str
    category_name: str
    assigned_milliunits: int
    still_underfunded_milliunits: int
    goal_period_target_milliunits: int


@dataclass(frozen=True)
class MonthlySpending:
    month: str
    spending_milliunits: int


@dataclass(frozen=True)
class MonthlyIncome:
    month: str
    income_milliunits: int


class CostToBeMeRepository(Protocol):
    """Read-only aggregates needed by the cost-to-be-me report."""

    async def list_funding_categories(
        self,
        *,
        month: date,
        plan_id: str | None,
    ) -> Sequence[FundingCategory]: ...

    async def list_monthly_spending(
        self,
        *,
        start_date: date,
        end_date: date,
        plan_id: str | None,
    ) -> Sequence[MonthlySpending]: ...

    async def list_monthly_ready_to_assign_income(
        self,
        *,
        start_date: date,
        end_date: date,
        plan_id: str | None,
    ) -> Sequence[MonthlyIncome]: ...


@dataclass(frozen=True)
class CostToBeMeRequest:
    target_month: date
    expected_income_milliunits: int | None = None
    income_months: int = 6
    spending_months: int = 12
    plan_id: str | None = None

    def __post_init__(self) -> None:
        if self.income_months < 1:
            raise ValueError("income months must be at least one")
        if self.spending_months < 1:
            raise ValueError("spending months must be at least one")


@dataclass(frozen=True)
class CostToBeMeReport:
    month: date
    monthly_funding_plan_milliunits: int
    assigned_milliunits: int
    still_underfunded_milliunits: int
    goal_period_targets_milliunits: int
    observed_spending_median_milliunits: int
    spending_window: str
    spending_samples: int
    expected_income_milliunits: int
    surplus_or_shortfall_milliunits: int
    income_source: str
    income_window: str
    income_samples: int
    funding_category_count: int


class MissingCostToBeMeMonthError(ValueError):
    """Raised when the target month has no cached category detail."""


class CostToBeMeService:
    """Calculate monthly funding need, observed spending, and expected income."""

    def __init__(self, repository: CostToBeMeRepository) -> None:
        self.repository = repository

    async def build(self, request: CostToBeMeRequest) -> CostToBeMeReport:
        target_month = request.target_month.replace(day=1)
        income_end_month = _add_months(target_month, -1)
        income_start_month = _add_months(target_month, -request.income_months)
        spending_start_month = _add_months(
            target_month,
            -request.spending_months,
        )

        categories = await self.repository.list_funding_categories(
            month=target_month,
            plan_id=request.plan_id,
        )
        if not categories:
            raise MissingCostToBeMeMonthError(
                f"No cached category detail exists for {target_month.isoformat()}; "
                f"run `ynab sync --month {target_month.isoformat()}` first"
            )

        assigned = sum(max(0, row.assigned_milliunits) for row in categories)
        still_underfunded = sum(
            max(0, row.still_underfunded_milliunits) for row in categories
        )
        goal_period_targets = sum(
            max(0, row.goal_period_target_milliunits) for row in categories
        )
        funding_category_count = sum(
            1
            for row in categories
            if row.assigned_milliunits > 0
            or row.still_underfunded_milliunits > 0
        )
        monthly_funding_plan = assigned + still_underfunded

        spending = await self.repository.list_monthly_spending(
            start_date=spending_start_month,
            end_date=target_month,
            plan_id=request.plan_id,
        )
        spending_months = _month_labels(
            spending_start_month,
            target_month,
        )
        spending_by_month = {
            row.month: max(0, row.spending_milliunits) for row in spending
        }
        observed_spending = round(
            median(
                spending_by_month.get(month, 0)
                for month in spending_months
            )
        )

        income_source = "manual"
        income_window = ""
        income_samples = 0
        if request.expected_income_milliunits is not None:
            expected_income = request.expected_income_milliunits
        else:
            income_source = (
                f"inferred_{request.income_months}mo_ready_to_assign"
            )
            income_window = _month_window(
                income_start_month,
                income_end_month,
            )
            income = await self.repository.list_monthly_ready_to_assign_income(
                start_date=income_start_month,
                end_date=target_month,
                plan_id=request.plan_id,
            )
            income_months = _month_labels(
                income_start_month,
                target_month,
            )
            income_by_month = {
                row.month: max(0, row.income_milliunits) for row in income
            }
            income_samples = len(income_months)
            expected_income = round(
                sum(
                    income_by_month.get(month, 0)
                    for month in income_months
                )
                / income_samples
            )

        return CostToBeMeReport(
            month=target_month,
            monthly_funding_plan_milliunits=monthly_funding_plan,
            assigned_milliunits=assigned,
            still_underfunded_milliunits=still_underfunded,
            goal_period_targets_milliunits=goal_period_targets,
            observed_spending_median_milliunits=observed_spending,
            spending_window=_month_window(
                spending_start_month,
                income_end_month,
            ),
            spending_samples=len(spending_months),
            expected_income_milliunits=expected_income,
            surplus_or_shortfall_milliunits=(
                expected_income - monthly_funding_plan
            ),
            income_source=income_source,
            income_window=income_window,
            income_samples=income_samples,
            funding_category_count=funding_category_count,
        )


def _add_months(month: date, delta: int) -> date:
    month_index = month.year * 12 + month.month - 1 + delta
    year, zero_based_month = divmod(month_index, 12)
    return date(year, zero_based_month + 1, 1)


def _month_window(start: date, end: date) -> str:
    return f"{start:%Y-%m}..{end:%Y-%m}"


def _month_labels(start: date, end: date) -> tuple[str, ...]:
    months: list[str] = []
    current = start.replace(day=1)
    end_month = end.replace(day=1)
    while current < end_month:
        months.append(f"{current:%Y-%m}")
        current = _add_months(current, 1)
    return tuple(months)
