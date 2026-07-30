"""Typed services for cached overview and hygiene reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class NetWorthBucket:
    bucket: str
    account_type: str
    account_count: int
    balance_milliunits: int | None


@dataclass(frozen=True)
class CachedChange:
    batch_id: str
    budget_id: str
    resource: str
    action: str
    entity_name: str | None
    entity_id: str
    recorded_at: datetime | str


@dataclass(frozen=True)
class HygieneCheck:
    check_name: str
    count: int


class OverviewReportRepository(Protocol):
    async def list_net_worth(self) -> Sequence[NetWorthBucket]: ...

    async def list_latest_changes(
        self,
        plan_id: str | None,
    ) -> Sequence[CachedChange]: ...

    async def list_hygiene_checks(
        self,
        *,
        since: date,
    ) -> Sequence[HygieneCheck]: ...


class OverviewReportService:
    """Expose read-only overview reports without leaking database rows."""

    def __init__(self, repository: OverviewReportRepository) -> None:
        self.repository = repository

    async def net_worth(self) -> tuple[NetWorthBucket, ...]:
        return tuple(await self.repository.list_net_worth())

    async def latest_changes(
        self,
        plan_id: str | None = None,
    ) -> tuple[CachedChange, ...]:
        return tuple(await self.repository.list_latest_changes(plan_id))

    async def hygiene(self, *, since: date) -> tuple[HygieneCheck, ...]:
        return tuple(await self.repository.list_hygiene_checks(since=since))
