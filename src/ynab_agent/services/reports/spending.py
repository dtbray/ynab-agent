"""Typed services and calculations for cached spending reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol


class BurnRateGrouping(StrEnum):
    CATEGORY = "category"
    GROUP = "group"


@dataclass(frozen=True)
class DateRangeRequest:
    since: date
    limit: int | None = None
    through: date | None = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least one")


@dataclass(frozen=True)
class CashflowMonth:
    month: str
    inflow_milliunits: int | None
    outflow_milliunits: int | None
    net_milliunits: int | None
    transaction_count: int


@dataclass(frozen=True)
class BurnRateBucket:
    bucket: str
    outflow_milliunits: int
    transaction_count: int
    active_months: int
    monthly_burn_milliunits: int


@dataclass(frozen=True)
class RawBurnRateBucket:
    bucket: str
    outflow_milliunits: int | None
    transaction_count: int
    active_months: int


@dataclass(frozen=True)
class PayeeSpend:
    payee: str
    outflow_milliunits: int | None
    transaction_count: int


class SpendingReportRepository(Protocol):
    async def list_cashflow(
        self,
        request: DateRangeRequest,
    ) -> Sequence[CashflowMonth]: ...

    async def list_burn_rate(
        self,
        request: DateRangeRequest,
        *,
        grouping: BurnRateGrouping,
    ) -> Sequence[RawBurnRateBucket]: ...

    async def list_top_spend(
        self,
        request: DateRangeRequest,
    ) -> Sequence[PayeeSpend]: ...


def inclusive_month_count(since: date, through: date) -> int:
    """Count calendar months inclusively, preserving the legacy lower bound."""
    months = (through.year - since.year) * 12 + through.month - since.month + 1
    return max(1, months)


class SpendingReportService:
    """Apply report calculations to typed spending aggregates."""

    def __init__(self, repository: SpendingReportRepository) -> None:
        self.repository = repository

    async def cashflow(
        self,
        request: DateRangeRequest,
    ) -> tuple[CashflowMonth, ...]:
        return tuple(await self.repository.list_cashflow(request))

    async def burn_rate(
        self,
        request: DateRangeRequest,
        *,
        grouping: BurnRateGrouping,
        current_date: date,
    ) -> tuple[BurnRateBucket, ...]:
        through = request.through or current_date
        month_count = inclusive_month_count(request.since, through)
        rows = await self.repository.list_burn_rate(
            request,
            grouping=grouping,
        )
        return tuple(
            BurnRateBucket(
                bucket=row.bucket,
                outflow_milliunits=row.outflow_milliunits or 0,
                transaction_count=row.transaction_count,
                active_months=row.active_months,
                monthly_burn_milliunits=int((row.outflow_milliunits or 0) / month_count),
            )
            for row in rows
        )

    async def top_spend(
        self,
        request: DateRangeRequest,
    ) -> tuple[PayeeSpend, ...]:
        return tuple(await self.repository.list_top_spend(request))
