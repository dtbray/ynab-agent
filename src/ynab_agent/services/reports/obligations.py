"""Typed service for scheduled obligation reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class ObligationRequest:
    through: date
    include_inflows: bool
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be at least one")


@dataclass(frozen=True)
class ScheduledObligation:
    due_date: str
    payee: str
    group_name: str
    category: str | None
    account: str | None
    frequency: str | None
    amount_milliunits: int | None
    memo: str | None


class ObligationRepository(Protocol):
    async def list_obligations(
        self,
        request: ObligationRequest,
    ) -> Sequence[ScheduledObligation]: ...


class ObligationService:
    def __init__(self, repository: ObligationRepository) -> None:
        self.repository = repository

    async def list(
        self,
        request: ObligationRequest,
    ) -> tuple[ScheduledObligation, ...]:
        return tuple(await self.repository.list_obligations(request))
