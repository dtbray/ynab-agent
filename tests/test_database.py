import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Text, create_engine, inspect

from ynab_agent.db.models import Base
from ynab_agent.db.manager import DatabaseManager


@pytest.mark.asyncio
async def test_database_manager_saves_and_queries_transactions(tmp_path):
    db = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'ynab.db'}")
    await db.initialize()

    count = await db.save_transactions(
        "budget-1",
        [
            {
                "id": "txn-1",
                "account_id": "account-1",
                "date": "2026-05-18",
                "amount": -12345,
                "memo": "Coffee",
                "cleared": "uncleared",
                "approved": False,
                "deleted": False,
            }
        ],
    )

    rows = await db.fetch_all("SELECT id, amount, approved FROM transactions")

    assert count == 1
    assert rows == [{"id": "txn-1", "amount": -12345, "approved": False}]

    await db.mark_notified("txn-1")
    rows = await db.fetch_all("SELECT transaction_id FROM transaction_notifications")
    assert rows == [{"transaction_id": "txn-1"}]

    assert await db.get_server_knowledge("budget-1", "transactions") is None
    await db.save_server_knowledge("budget-1", "transactions", 42)
    assert await db.get_server_knowledge("budget-1", "transactions") == 42
    await db.save_server_knowledge("budget-1", "transactions", 43)
    assert await db.get_server_knowledge("budget-1", "transactions") == 43

    await db.close()


@pytest.mark.asyncio
async def test_database_manager_preserves_long_import_payee_metadata(tmp_path):
    db = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'ynab.db'}")
    await db.initialize()
    imported_metadata = '{"raw": "' + ("x" * 1000) + '"}'

    await db.save_transactions(
        "budget-1",
        [
            {
                "id": "txn-long-import-payee",
                "import_payee_name": imported_metadata,
                "import_payee_name_original": imported_metadata,
                "deleted": False,
            }
        ],
    )

    rows = await db.fetch_all(
        """
        SELECT import_payee_name, import_payee_name_original
        FROM transactions
        WHERE id = :transaction_id
        """,
        {"transaction_id": "txn-long-import-payee"},
    )
    assert rows == [
        {
            "import_payee_name": imported_metadata,
            "import_payee_name_original": imported_metadata,
        }
    ]
    await db.close()


@pytest.mark.asyncio
async def test_database_manager_records_latest_sync_changes(tmp_path):
    db = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'ynab.db'}")
    await db.initialize()

    await db.save_accounts(
        "budget-1",
        [
            {
                "id": "account-1",
                "name": "Checking",
                "type": "checking",
                "on_budget": True,
                "closed": False,
                "balance": 1000,
                "deleted": False,
            }
        ],
        change_batch_id="batch-1",
    )
    await db.save_accounts(
        "budget-1",
        [
            {
                "id": "account-1",
                "name": "Checking",
                "type": "checking",
                "on_budget": True,
                "closed": False,
                "balance": 2000,
                "deleted": False,
            },
            {
                "id": "account-2",
                "name": "Old Savings",
                "type": "savings",
                "on_budget": True,
                "closed": True,
                "balance": 0,
                "deleted": True,
            },
        ],
        change_batch_id="batch-2",
    )

    rows = await db.get_latest_changes("budget-1")

    assert [row["batch_id"] for row in rows] == ["batch-2", "batch-2"]
    assert [(row["entity_id"], row["action"]) for row in rows] == [
        ("account-2", "deleted"),
        ("account-1", "updated"),
    ]

    await db.close()


def test_alembic_upgrade_creates_model_schema(tmp_path):
    db_path = tmp_path / "migrated.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        transaction_columns = {
            column["name"]: column["type"]
            for column in inspector.get_columns("transactions")
        }
    finally:
        engine.dispose()

    expected_tables = set(Base.metadata.tables) | {"alembic_version"}
    assert tables == expected_tables
    assert "sync_state" in tables
    assert isinstance(transaction_columns["import_payee_name"], Text)
    assert isinstance(transaction_columns["import_payee_name_original"], Text)
