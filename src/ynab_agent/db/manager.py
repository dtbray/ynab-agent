"""Async SQLAlchemy lifecycle with compatibility database forwards."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ynab_agent.db.models import Base
from ynab_agent.services.sync import Row

if TYPE_CHECKING:
    from ynab_agent.db.change_tracking import SqlChangeLog
    from ynab_agent.db.notifications import SqlTransactionNotificationStore
    from ynab_agent.db.sync import SqlSyncStore


class DatabaseManager:
    """Own database lifecycle while legacy methods forward to focused adapters."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: AsyncEngine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._sync_store: SqlSyncStore | None = None
        self._change_log: SqlChangeLog | None = None
        self._notification_store: SqlTransactionNotificationStore | None = None

    async def initialize(self) -> None:
        """Initialize database tables from SQLAlchemy metadata."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Dispose the underlying SQLAlchemy engine."""
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a SQLAlchemy async session."""
        async with self.session_factory() as session:
            yield session

    async def fetch_all(
        self,
        sql: str,
        params: Mapping[str, object] | None = None,
    ) -> list[Row]:
        """Run a SELECT statement and return rows as dictionaries."""
        async with self.session_factory() as session:
            result = await session.execute(text(sql), params or {})
            return [dict(row) for row in result.mappings().all()]

    def _sync(self) -> SqlSyncStore:
        if self._sync_store is None:
            from ynab_agent.db.sync import SqlSyncStore

            self._sync_store = SqlSyncStore(self)
        return self._sync_store

    def _changes(self) -> SqlChangeLog:
        if self._change_log is None:
            from ynab_agent.db.change_tracking import SqlChangeLog

            self._change_log = SqlChangeLog(self)
        return self._change_log

    def _notifications(self) -> SqlTransactionNotificationStore:
        if self._notification_store is None:
            from ynab_agent.db.notifications import (
                SqlTransactionNotificationStore,
            )

            self._notification_store = SqlTransactionNotificationStore(self)
        return self._notification_store

    async def save_transactions(
        self,
        budget_id: str,
        transactions: list[Row],
        change_batch_id: str | None = None,
    ) -> int:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_transactions(
            budget_id,
            transactions,
            change_batch_id=change_batch_id or "",
        )

    async def save_subtransactions(
        self,
        budget_id: str,
        subtransactions: list[Row],
        change_batch_id: str | None = None,
    ) -> int:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_subtransactions(
            budget_id,
            subtransactions,
            change_batch_id=change_batch_id or "",
        )

    async def save_accounts(
        self,
        budget_id: str,
        accounts: list[Row],
        change_batch_id: str | None = None,
    ) -> int:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_accounts(
            budget_id,
            accounts,
            change_batch_id=change_batch_id or "",
        )

    async def save_budgets(
        self,
        budgets: list[Row],
        change_batch_id: str | None = None,
    ) -> int:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_budgets(
            budgets,
            change_batch_id=change_batch_id or "",
        )

    async def save_payees(
        self,
        payees: list[Row],
        change_batch_id: str | None = None,
    ) -> int:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_payees(
            payees,
            change_batch_id=change_batch_id or "",
        )

    async def save_payee_locations(
        self,
        locations: list[Row],
        change_batch_id: str | None = None,
    ) -> int:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_payee_locations(
            locations,
            change_batch_id=change_batch_id or "",
        )

    async def save_categories(
        self,
        category_groups: list[Row],
        categories: list[Row],
        change_batch_id: str | None = None,
    ) -> tuple[int, int]:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_categories(
            category_groups,
            categories,
            change_batch_id=change_batch_id or "",
        )

    async def save_month(
        self,
        month: Row,
        categories: list[Row],
        change_batch_id: str | None = None,
    ) -> tuple[int, int]:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_month(
            month,
            categories,
            change_batch_id=change_batch_id or "",
        )

    async def save_months(
        self,
        months: list[Row],
        change_batch_id: str | None = None,
    ) -> int:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_months(
            months,
            change_batch_id=change_batch_id or "",
        )

    async def save_scheduled_transactions(
        self,
        budget_id: str,
        transactions: list[Row],
        change_batch_id: str | None = None,
    ) -> tuple[int, int]:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().save_scheduled_transactions(
            budget_id,
            transactions,
            change_batch_id=change_batch_id or "",
        )

    async def record_changes(
        self,
        batch_id: str,
        budget_id: str,
        resource: str,
        rows: list[Row],
    ) -> int:
        """Compatibility forward to the change-log adapter."""
        return await self._changes().record_changes(
            batch_id,
            budget_id,
            resource,
            rows,
        )

    async def get_latest_changes(
        self,
        budget_id: str | None = None,
    ) -> list[Row]:
        """Compatibility forward to the change-log adapter."""
        return await self._changes().get_latest_changes(budget_id)

    async def get_server_knowledge(
        self,
        budget_id: str,
        resource: str,
    ) -> int | None:
        """Compatibility forward to the synchronization adapter."""
        return await self._sync().get_server_knowledge(
            budget_id,
            resource,
        )

    async def save_server_knowledge(
        self,
        budget_id: str,
        resource: str,
        server_knowledge: int | None,
    ) -> None:
        """Compatibility forward to the synchronization adapter."""
        await self._sync().save_server_knowledge(
            budget_id,
            resource,
            server_knowledge,
        )

    async def get_unnotified_transactions(self) -> list[Row]:
        """Compatibility forward to the notification adapter."""
        return await self._notifications().get_unnotified_transactions()

    async def mark_notified(self, transaction_id: str) -> None:
        """Compatibility forward to the notification adapter."""
        await self._notifications().mark_notified(transaction_id)
