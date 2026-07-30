"""Typed account-valuation observations used by wealth analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class ValuationSnapshotSource(StrEnum):
    """Provenance of an account balance observation."""

    YNAB_SYNC = "ynab_sync"
    USER_REVIEWED = "user_reviewed"
    IMPORTED_REVIEWED = "imported_reviewed"


@dataclass(frozen=True)
class AccountValuationSnapshot:
    """One dated account balance with explicit observation provenance."""

    account_id: str
    valuation_date: date
    balance_milliunits: int
    source: ValuationSnapshotSource
    observed_at: datetime
    reviewed: bool

    def __post_init__(self) -> None:
        if not self.account_id:
            raise ValueError("valuation snapshot account_id must not be empty")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("valuation snapshot observed_at must include a timezone")


class ValuationSnapshotConflictError(ValueError):
    """Raised for contradictory observations with the same unique identity."""
