"""Database adapter for transaction-notification persistence."""

from __future__ import annotations

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.models import TransactionNotification
from ynab_agent.db.sync import SqlSyncStore
from ynab_agent.services.sync import Row


class SqlTransactionNotificationStore:
    """Persist polled transactions and notification delivery markers."""

    def __init__(self, database: DatabaseManager) -> None:
        self._database = database
        self._sync = SqlSyncStore(database)

    async def save_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
    ) -> int:
        return await self._sync.save_transactions(
            plan_id,
            transactions,
            change_batch_id="",
        )

    async def get_unnotified_transactions(self) -> list[Row]:
        rows = await self._database.fetch_all(
            """
            SELECT t.*, a.name as account_name, p.name as payee_name
            FROM transactions t
            LEFT JOIN accounts a ON t.account_id = a.id
            LEFT JOIN payees p ON t.payee_id = p.id
            WHERE t.approved IS FALSE AND t.deleted IS FALSE
            AND t.id NOT IN (SELECT transaction_id FROM transaction_notifications)
            ORDER BY t.date DESC
            """
        )
        return [dict(row) for row in rows]

    async def mark_notified(self, transaction_id: str) -> None:
        async with self._database.session_factory() as session:
            await session.merge(
                TransactionNotification(transaction_id=transaction_id)
            )
            await session.commit()
