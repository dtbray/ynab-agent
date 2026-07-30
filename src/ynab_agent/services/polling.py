"""Persistence port for transaction polling and notification delivery."""

from __future__ import annotations

from typing import Protocol

from ynab_agent.services.sync import Row


class TransactionPollStore(Protocol):
    """Persistence required by transaction polling and notification delivery."""

    async def save_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
    ) -> int: ...

    async def get_unnotified_transactions(self) -> list[Row]: ...

    async def mark_notified(self, transaction_id: str) -> None: ...
