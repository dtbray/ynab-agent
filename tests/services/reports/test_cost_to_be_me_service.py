from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from ynab_agent.services.reports.cost_to_be_me import (
    CostToBeMeRequest,
    CostToBeMeService,
    FundingCategory,
    MissingCostToBeMeMonthError,
    MonthlyIncome,
    MonthlySpending,
)


class FakeCostToBeMeRepository:
    def __init__(
        self,
        *,
        categories: Sequence[FundingCategory] = (),
        spending: Sequence[MonthlySpending] = (),
        income: Sequence[MonthlyIncome] = (),
    ) -> None:
        self.categories = tuple(categories)
        self.spending = tuple(spending)
        self.income = tuple(income)
        self.calls: list[tuple[str, date, date | None, str | None]] = []

    async def list_funding_categories(
        self,
        *,
        month: date,
        plan_id: str | None,
    ) -> Sequence[FundingCategory]:
        self.calls.append(("funding", month, None, plan_id))
        return self.categories

    async def list_monthly_spending(
        self,
        *,
        start_date: date,
        end_date: date,
        plan_id: str | None,
    ) -> Sequence[MonthlySpending]:
        self.calls.append(("spending", start_date, end_date, plan_id))
        return self.spending

    async def list_monthly_ready_to_assign_income(
        self,
        *,
        start_date: date,
        end_date: date,
        plan_id: str | None,
    ) -> Sequence[MonthlyIncome]:
        self.calls.append(("income", start_date, end_date, plan_id))
        return self.income


def _funding(
    category_name: str,
    *,
    assigned: int = 0,
    underfunded: int = 0,
    goal_target: int = 0,
) -> FundingCategory:
    return FundingCategory(
        group_name="Bills",
        category_name=category_name,
        assigned_milliunits=assigned,
        still_underfunded_milliunits=underfunded,
        goal_period_target_milliunits=goal_target,
    )


@pytest.mark.asyncio
async def test_manual_income_preserves_funding_math_and_skips_income_query() -> None:
    repository = FakeCostToBeMeRepository(
        categories=(
            _funding("Rent", assigned=1_200_000),
            _funding(
                "Long Goal",
                assigned=-10_000,
                underfunded=50_000,
                goal_target=25_800_000,
            ),
            _funding(
                "Negative",
                underfunded=-20_000,
                goal_target=-30_000,
            ),
        ),
        spending=(
            MonthlySpending("2026-03", 200_000),
            MonthlySpending("2026-04", 400_000),
        ),
        income=(MonthlyIncome("2026-04", 99_000_000),),
    )

    report = await CostToBeMeService(repository).build(
        CostToBeMeRequest(
            target_month=date(2026, 5, 20),
            expected_income_milliunits=1_500_000,
            plan_id="plan-1",
        )
    )

    assert report.month == date(2026, 5, 1)
    assert report.assigned_milliunits == 1_200_000
    assert report.still_underfunded_milliunits == 50_000
    assert report.monthly_funding_plan_milliunits == 1_250_000
    assert report.goal_period_targets_milliunits == 25_800_000
    assert report.funding_category_count == 2
    assert report.observed_spending_median_milliunits == 0
    assert report.spending_window == "2025-05..2026-04"
    assert report.spending_samples == 12
    assert report.expected_income_milliunits == 1_500_000
    assert report.surplus_or_shortfall_milliunits == 250_000
    assert report.income_source == "manual"
    assert report.income_window == ""
    assert report.income_samples == 0
    assert repository.calls == [
        ("funding", date(2026, 5, 1), None, "plan-1"),
        (
            "spending",
            date(2025, 5, 1),
            date(2026, 5, 1),
            "plan-1",
        ),
    ]


@pytest.mark.asyncio
async def test_inferred_income_uses_complete_months_and_exact_boundaries() -> None:
    repository = FakeCostToBeMeRepository(
        categories=(_funding("Rent", assigned=1_200_000),),
        spending=(),
        income=(
            MonthlyIncome("2026-03", 3_000_000),
            MonthlyIncome("2026-04", 5_000_000),
        ),
    )

    report = await CostToBeMeService(repository).build(
        CostToBeMeRequest(
            target_month=date(2026, 5, 1),
            income_months=2,
            spending_months=3,
        )
    )

    assert report.observed_spending_median_milliunits == 0
    assert report.spending_window == "2026-02..2026-04"
    assert report.spending_samples == 3
    assert report.expected_income_milliunits == 4_000_000
    assert report.surplus_or_shortfall_milliunits == 2_800_000
    assert report.income_source == "inferred_2mo_ready_to_assign"
    assert report.income_window == "2026-03..2026-04"
    assert report.income_samples == 2
    assert repository.calls == [
        ("funding", date(2026, 5, 1), None, None),
        (
            "spending",
            date(2026, 2, 1),
            date(2026, 5, 1),
            None,
        ),
        ("income", date(2026, 3, 1), date(2026, 5, 1), None),
    ]


@pytest.mark.asyncio
async def test_sparse_windows_include_zero_months_across_year_boundary() -> None:
    repository = FakeCostToBeMeRepository(
        categories=(_funding("Rent", assigned=1_200_000),),
        spending=(
            MonthlySpending("2026-10", 100_000),
            MonthlySpending("2026-12", 300_000),
        ),
        income=(MonthlyIncome("2026-12", 600_000),),
    )

    report = await CostToBeMeService(repository).build(
        CostToBeMeRequest(
            target_month=date(2027, 1, 31),
            income_months=2,
            spending_months=3,
        )
    )

    assert report.spending_window == "2026-10..2026-12"
    assert report.spending_samples == 3
    assert report.observed_spending_median_milliunits == 100_000
    assert report.income_window == "2026-11..2026-12"
    assert report.income_samples == 2
    assert report.expected_income_milliunits == 300_000


@pytest.mark.asyncio
async def test_missing_target_month_stops_before_historical_queries() -> None:
    repository = FakeCostToBeMeRepository()

    with pytest.raises(
        MissingCostToBeMeMonthError,
        match="No cached category detail exists for 2026-05-01",
    ):
        await CostToBeMeService(repository).build(
            CostToBeMeRequest(target_month=date(2026, 5, 15))
        )

    assert repository.calls == [
        ("funding", date(2026, 5, 1), None, None)
    ]


def test_request_requires_positive_history_windows() -> None:
    with pytest.raises(ValueError, match="income months"):
        CostToBeMeRequest(target_month=date(2026, 5, 1), income_months=0)
    with pytest.raises(ValueError, match="spending months"):
        CostToBeMeRequest(target_month=date(2026, 5, 1), spending_months=0)
