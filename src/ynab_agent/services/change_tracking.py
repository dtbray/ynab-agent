"""Application port for synchronization change logs."""

from __future__ import annotations

from typing import Protocol

from ynab_agent.services.sync import Row


class ChangeLog(Protocol):
    """Record and read user-facing synchronization changes."""

    async def record_changes(
        self,
        batch_id: str,
        plan_id: str,
        resource: str,
        rows: list[Row],
    ) -> int: ...

    async def get_latest_changes(
        self,
        plan_id: str | None = None,
    ) -> list[Row]: ...
