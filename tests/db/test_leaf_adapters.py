from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Never

import pytest

from ynab_agent.db.change_tracking import SqlChangeLog
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.notifications import SqlTransactionNotificationStore
from ynab_agent.db.sync import SqlSyncStore
from ynab_agent.services.change_tracking import ChangeLog
from ynab_agent.services.polling import TransactionPollStore
from ynab_agent.services.sync import Row, SyncStore


def _sync_port(store: SyncStore) -> SyncStore:
    return store


def _notification_port(store: TransactionPollStore) -> TransactionPollStore:
    return store


def _change_log_port(store: ChangeLog) -> ChangeLog:
    return store


async def _unexpected_compatibility_forward(
    *args: object,
    **kwargs: object,
) -> Never:
    raise AssertionError("focused adapter called a manager compatibility forward")


@pytest.mark.asyncio
async def test_sync_adapter_persists_bulk_resources_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = DatabaseManager(
        f"sqlite+aiosqlite:///{tmp_path / 'sync-adapter.db'}"
    )
    await database.initialize()
    for method_name in (
        "get_server_knowledge",
        "save_server_knowledge",
        "save_budgets",
        "save_accounts",
        "save_payees",
        "save_payee_locations",
        "save_categories",
        "save_months",
        "save_month",
        "save_transactions",
        "save_subtransactions",
        "save_scheduled_transactions",
    ):
        monkeypatch.setattr(
            database,
            method_name,
            _unexpected_compatibility_forward,
        )
    store = _sync_port(SqlSyncStore(database))
    try:
        await store.begin_sync_batch(
            "budget-1",
            "batch-1",
            started_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
        )
        assert await store.get_server_knowledge("budget-1", "accounts") is None
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            None,
            change_batch_id="batch-1",
        )
        assert await store.get_server_knowledge("budget-1", "accounts") is None
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            17,
            change_batch_id="batch-1",
        )
        assert await store.get_server_knowledge("budget-1", "accounts") == 17
        assert await database.fetch_all(
            """
            SELECT change_batch_id
            FROM sync_state
            WHERE budget_id = 'budget-1' AND resource = 'accounts'
            """
        ) == [{"change_batch_id": "batch-1"}]

        assert await store.save_budgets(
            [{"id": "budget-1", "name": "Main"}],
            change_batch_id="batch-1",
        ) == 1
        assert await store.save_accounts(
            "budget-1",
            [
                {
                    "id": "account-1",
                    "name": "Checking",
                    "type": "checking",
                    "on_budget": True,
                    "closed": False,
                    "balance": 100_000,
                    "deleted": False,
                }
            ],
            change_batch_id="batch-1",
        ) == 1
        assert await store.save_payees(
            [
                {
                    "id": "payee-1",
                    "budget_id": "budget-1",
                    "name": "Cafe",
                }
            ],
            change_batch_id="batch-1",
        ) == 1
        assert await store.save_payee_locations(
            [
                {
                    "id": "location-1",
                    "budget_id": "budget-1",
                    "payee_id": "payee-1",
                    "latitude": "39.0",
                    "longitude": "-86.0",
                }
            ],
            change_batch_id="batch-1",
        ) == 1
        assert await store.save_categories(
            [
                {
                    "id": "group-1",
                    "budget_id": "budget-1",
                    "name": "Living",
                }
            ],
            [
                {
                    "id": "category-1",
                    "budget_id": "budget-1",
                    "category_group_id": "group-1",
                    "name": "Dining",
                }
            ],
            change_batch_id="batch-1",
        ) == (1, 1)
        month: Row = {
            "id": "budget-1:2026-07-01",
            "budget_id": "budget-1",
            "month": "2026-07-01",
        }
        assert await store.save_months(
            [month],
            change_batch_id="batch-1",
        ) == 1
        assert await store.save_month(
            month,
            [
                {
                    "id": "budget-1:2026-07-01:category-1",
                    "budget_id": "budget-1",
                    "month": "2026-07-01",
                    "category_id": "category-1",
                    "category_group_id": "group-1",
                    "name": "Dining",
                }
            ],
            change_batch_id="batch-1",
        ) == (1, 1)
        assert await store.save_transactions(
            "budget-1",
            [
                {
                    "id": "transaction-1",
                    "account_id": "account-1",
                    "payee_id": "payee-1",
                    "category_id": "category-1",
                    "date": "2026-07-15",
                    "amount": -1_500,
                    "approved": False,
                }
            ],
            change_batch_id="batch-1",
        ) == 1
        assert await store.save_scheduled_transactions(
            "budget-1",
            [
                {
                    "id": "scheduled-1",
                    "account_id": "account-1",
                    "date_next": "2026-08-01",
                    "amount": -2_000,
                    "subtransactions": [
                        {
                            "id": "scheduled-split-1",
                            "amount": -2_000,
                            "category_id": "category-1",
                        }
                    ],
                }
            ],
            change_batch_id="batch-1",
        ) == (1, 1)
        await store.complete_sync_batch(
            "budget-1",
            "batch-1",
            completed_at=datetime(2026, 7, 31, 0, 1, tzinfo=timezone.utc),
        )
        assert await database.fetch_all(
            """
            SELECT state
            FROM calibration_sync_batches
            WHERE id = 'batch-1'
            """
        ) == [{"state": "completed"}]

        counts = await database.fetch_all(
            """
            SELECT
              (SELECT COUNT(*) FROM budgets) AS budgets,
              (SELECT COUNT(*) FROM accounts) AS accounts,
              (SELECT COUNT(*) FROM transactions) AS transactions,
              (SELECT COUNT(*) FROM scheduled_transactions) AS scheduled,
              (SELECT COUNT(*) FROM scheduled_subtransactions) AS scheduled_splits
            """
        )
        assert counts == [
            {
                "budgets": 1,
                "accounts": 1,
                "transactions": 1,
                "scheduled": 1,
                "scheduled_splits": 1,
            }
        ]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_change_and_notification_adapters_preserve_delivery_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = DatabaseManager(
        f"sqlite+aiosqlite:///{tmp_path / 'leaf-adapters.db'}"
    )
    await database.initialize()
    for method_name in (
        "record_changes",
        "get_latest_changes",
        "save_transactions",
        "get_unnotified_transactions",
        "mark_notified",
    ):
        monkeypatch.setattr(
            database,
            method_name,
            _unexpected_compatibility_forward,
        )
    changes = _change_log_port(SqlChangeLog(database))
    notifications = _notification_port(
        SqlTransactionNotificationStore(database)
    )
    try:
        await SqlSyncStore(database).save_accounts(
            "budget-1",
            [
                {
                    "id": "account-1",
                    "name": "Checking",
                    "on_budget": True,
                    "closed": False,
                    "deleted": False,
                }
            ],
            change_batch_id="",
        )
        assert await changes.record_changes(
            "batch-2",
            "budget-1",
            "accounts",
            [
                {
                    "id": "account-1",
                    "name": "Checking",
                    "deleted": False,
                }
            ],
        ) == 1
        latest = await changes.get_latest_changes("budget-1")
        assert [
            (row["batch_id"], row["entity_id"], row["action"])
            for row in latest
        ] == [("batch-2", "account-1", "updated")]

        assert await notifications.save_transactions(
            "budget-1",
            [
                {
                    "id": "transaction-1",
                    "account_id": "account-1",
                    "date": "2026-07-29",
                    "amount": -1_250,
                    "memo": "Lunch",
                    "approved": False,
                    "deleted": False,
                }
            ],
        ) == 1
        pending = await notifications.get_unnotified_transactions()
        assert [row["id"] for row in pending] == ["transaction-1"]
        assert pending[0]["account_name"] == "Checking"

        await notifications.mark_notified("transaction-1")
        assert await notifications.get_unnotified_transactions() == []
        await notifications.mark_notified("transaction-1")
        assert await notifications.get_unnotified_transactions() == []
    finally:
        await database.close()
