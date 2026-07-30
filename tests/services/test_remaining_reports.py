from __future__ import annotations

from datetime import date

import pytest

from ynab_agent.services.reports.budget_activity import (
    BudgetActivityRequest,
)
from ynab_agent.services.reports.obligations import ObligationRequest
from ynab_agent.services.reports.spending import (
    BurnRateGrouping,
    DateRangeRequest,
    RawBurnRateBucket,
    SpendingReportService,
    inclusive_month_count,
)


class StubSpendingRepository:
    def __init__(self) -> None:
        self.request: DateRangeRequest | None = None
        self.grouping: BurnRateGrouping | None = None

    async def list_cashflow(
        self,
        request: DateRangeRequest,
    ) -> tuple[()]:
        return ()

    async def list_burn_rate(
        self,
        request: DateRangeRequest,
        *,
        grouping: BurnRateGrouping,
    ) -> tuple[RawBurnRateBucket, ...]:
        self.request = request
        self.grouping = grouping
        return (
            RawBurnRateBucket(
                bucket="Bills",
                outflow_milliunits=1_000_001,
                transaction_count=3,
                active_months=2,
            ),
        )

    async def list_top_spend(
        self,
        request: DateRangeRequest,
    ) -> tuple[()]:
        return ()


@pytest.mark.parametrize(
    ("since", "through", "expected"),
    (
        (date(2026, 1, 31), date(2026, 1, 31), 1),
        (date(2026, 12, 1), date(2027, 1, 1), 2),
        (date(2027, 2, 1), date(2027, 1, 31), 1),
    ),
)
def test_inclusive_month_count_handles_boundaries(
    since: date,
    through: date,
    expected: int,
) -> None:
    assert inclusive_month_count(since, through) == expected


@pytest.mark.asyncio
async def test_burn_rate_service_owns_monthly_average_policy() -> None:
    repository = StubSpendingRepository()
    service = SpendingReportService(repository)
    request = DateRangeRequest(
        since=date(2026, 12, 15),
        through=date(2027, 1, 10),
        limit=20,
    )

    result = await service.burn_rate(
        request,
        grouping=BurnRateGrouping.GROUP,
        current_date=date(2030, 1, 1),
    )

    assert repository.request is request
    assert repository.grouping is BurnRateGrouping.GROUP
    assert result[0].monthly_burn_milliunits == 500_000


def test_report_requests_reject_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="limit must be at least one"):
        BudgetActivityRequest(month=date(2026, 1, 1), limit=0)
    with pytest.raises(ValueError, match="limit must be at least one"):
        DateRangeRequest(since=date(2026, 1, 1), limit=0)
    with pytest.raises(ValueError, match="limit must be at least one"):
        ObligationRequest(
            through=date(2026, 1, 1),
            include_inflows=False,
            limit=0,
        )
