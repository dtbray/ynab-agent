"""Typed services for cached category activity reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class BudgetActivityRequest:
    month: date
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be at least one")


@dataclass(frozen=True)
class CategoryActivity:
    group_name: str
    category: str
    budgeted_milliunits: int | None
    activity_milliunits: int | None
    balance_milliunits: int | None


@dataclass(frozen=True)
class OverspendingCategory:
    group_name: str
    category: str
    budgeted_milliunits: int | None
    activity_milliunits: int | None
    balance_milliunits: int | None
    overspent_milliunits: int | None


class BudgetActivityRepository(Protocol):
    async def list_overspending(
        self,
        request: BudgetActivityRequest,
    ) -> Sequence[OverspendingCategory]: ...

    async def list_hidden_funds(
        self,
        request: BudgetActivityRequest,
    ) -> Sequence[CategoryActivity]: ...

    async def list_month_activity(
        self,
        request: BudgetActivityRequest,
    ) -> Sequence[CategoryActivity]: ...


class BudgetActivityService:
    """Query bounded category reports through typed requests."""

    def __init__(self, repository: BudgetActivityRepository) -> None:
        self.repository = repository

    async def overspending(
        self,
        request: BudgetActivityRequest,
    ) -> tuple[OverspendingCategory, ...]:
        return tuple(await self.repository.list_overspending(request))

    async def hidden_funds(
        self,
        request: BudgetActivityRequest,
    ) -> tuple[CategoryActivity, ...]:
        return tuple(await self.repository.list_hidden_funds(request))

    async def month_activity(
        self,
        request: BudgetActivityRequest,
    ) -> tuple[CategoryActivity, ...]:
        return tuple(await self.repository.list_month_activity(request))
