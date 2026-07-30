import asyncio
from datetime import date

from typer.testing import CliRunner

from ynab_agent import cli
from ynab_agent.api.client import SyncResult
from ynab_agent.cli import app
from ynab_agent.db.manager import DatabaseManager


runner = CliRunner()


def _configure_empty_db(monkeypatch, tmp_path) -> str:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'sync.db'}"
    monkeypatch.setattr(cli.settings, "database_url", database_url)
    return database_url


def _fetch_all(database_url: str, sql: str) -> list[dict]:
    async def _fetch() -> list[dict]:
        db = DatabaseManager(database_url)
        try:
            return await db.fetch_all(sql)
        finally:
            await db.close()

    return asyncio.run(_fetch())


def test_resolve_plan_alias_uses_account_membership():
    plans = [
        {"id": "budget-1", "account_ids": ["account-a", "account-b"]},
        {"id": "budget-2", "account_ids": ["account-c"]},
    ]

    assert cli._resolve_plan_id(
        "last-used",
        plans,
        [{"id": "account-c"}],
    ) == "budget-2"


def test_sync_writes_ynab_payloads_and_checkpoints(monkeypatch, tmp_path):
    database_url = _configure_empty_db(monkeypatch, tmp_path)
    calls = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def get_plans(self, **kwargs):
            calls.append(("plans", None))
            return [{"id": "budget-1", "name": "Main"}]

        async def get_accounts(self, **kwargs):
            calls.append(("accounts", kwargs["last_knowledge_of_server"]))
            return SyncResult(
                [
                    {
                        "id": "account-1",
                        "name": "Checking",
                        "type": "checking",
                        "on_budget": True,
                        "closed": False,
                        "balance": 100000,
                        "cleared_balance": 100000,
                        "uncleared_balance": 0,
                        "deleted": False,
                    }
                ],
                10,
            )

        async def get_payees(self, **kwargs):
            calls.append(("payees", kwargs["last_knowledge_of_server"]))
            return SyncResult(
                [{"id": "payee-1", "budget_id": "budget-1", "name": "Market", "deleted": False}],
                11,
            )

        async def get_payee_locations(self, **kwargs):
            calls.append(("payee_locations", None))
            return [
                {
                    "id": "location-1",
                    "budget_id": "budget-1",
                    "payee_id": "payee-1",
                    "latitude": "40.0",
                    "longitude": "-75.0",
                    "deleted": False,
                }
            ]

        async def get_categories(self, **kwargs):
            calls.append(("categories", kwargs["last_knowledge_of_server"]))
            return SyncResult(
                (
                    [
                        {
                            "id": "group-1",
                            "budget_id": "budget-1",
                            "name": "Bills",
                            "hidden": False,
                            "deleted": False,
                        }
                    ],
                    [
                        {
                            "id": "category-1",
                            "budget_id": "budget-1",
                            "category_group_id": "group-1",
                            "name": "Groceries",
                            "hidden": False,
                            "deleted": False,
                        }
                    ],
                ),
                12,
            )

        async def get_months(self, **kwargs):
            calls.append(("months", kwargs["last_knowledge_of_server"]))
            return SyncResult(
                [
                    {
                        "id": "budget-1:2026-07-01",
                        "budget_id": "budget-1",
                        "month": "2026-07-01",
                        "income": 0,
                        "budgeted": 0,
                        "activity": 0,
                        "to_be_budgeted": 0,
                        "deleted": False,
                    }
                ],
                13,
            )

        async def get_month(self, *, plan_id, month):
            assert plan_id == "budget-1"
            assert month == date(2026, 7, 1)
            return (
                {
                    "id": "budget-1:2026-07-01",
                    "budget_id": "budget-1",
                    "month": "2026-07-01",
                    "income": 0,
                    "budgeted": 0,
                    "activity": 0,
                    "to_be_budgeted": 0,
                    "deleted": False,
                },
                [
                    {
                        "id": "budget-1:2026-07-01:category-1",
                        "budget_id": "budget-1",
                        "month": "2026-07-01",
                        "category_id": "category-1",
                        "budgeted": 250000,
                        "activity": -125000,
                        "balance": 125000,
                    }
                ],
            )

        async def get_transactions(self, **kwargs):
            calls.append(("transactions", kwargs["last_knowledge_of_server"], kwargs["since_date"]))
            return SyncResult(
                [
                    {
                        "id": "txn-1",
                        "account_id": "account-1",
                        "category_id": "category-1",
                        "payee_id": "payee-1",
                        "date": "2026-07-02",
                        "amount": -125000,
                        "cleared": "cleared",
                        "approved": True,
                        "deleted": False,
                        "subtransactions": [
                            {
                                "id": "subtxn-1",
                                "transaction_id": "txn-1",
                                "amount": -125000,
                                "category_id": "category-1",
                                "deleted": False,
                            }
                        ],
                    }
                ],
                14,
            )

        async def get_scheduled_transactions(self, **kwargs):
            calls.append(
                ("scheduled_transactions", kwargs["last_knowledge_of_server"])
            )
            return SyncResult(
                [
                    {
                        "id": "scheduled-1",
                        "account_id": "account-1",
                        "date_first": "2026-07-01",
                        "date_next": "2026-08-01",
                        "frequency": "monthly",
                        "amount": -125000,
                        "deleted": False,
                        "subtransactions": [
                            {
                                "id": "scheduled-subtxn-1",
                                "amount": -125000,
                                "category_id": "category-1",
                                "deleted": False,
                            }
                        ],
                    }
                ],
                15,
            )

    import ynab_agent.api.client

    monkeypatch.setattr(ynab_agent.api.client, "YnabClient", FakeClient)

    result = runner.invoke(
        app,
        ["sync", "--plan-id", "budget-1", "--month", "2026-07-01"],
    )

    assert result.exit_code == 0
    assert "Synced 1 plans, 1 accounts, 1 payees, 1 payee locations" in result.output
    assert calls == [
        ("plans", None),
        ("accounts", None),
        ("payees", None),
        ("payee_locations", None),
        ("categories", None),
        ("months", None),
        ("transactions", None, None),
        ("scheduled_transactions", None),
    ]
    assert _fetch_all(database_url, "SELECT name, balance FROM accounts") == [
        {"name": "Checking", "balance": 100000}
    ]
    assert _fetch_all(database_url, "SELECT amount FROM transactions") == [{"amount": -125000}]
    assert _fetch_all(
        database_url,
        "SELECT resource, server_knowledge FROM sync_state ORDER BY resource",
    ) == [
        {"resource": "accounts", "server_knowledge": 10},
        {"resource": "categories", "server_knowledge": 12},
        {"resource": "months", "server_knowledge": 13},
        {"resource": "payees", "server_knowledge": 11},
        {"resource": "scheduled_transactions", "server_knowledge": 15},
        {"resource": "transactions", "server_knowledge": 14},
    ]
    assert _fetch_all(database_url, "SELECT transaction_id FROM subtransactions") == [
        {"transaction_id": "txn-1"}
    ]
    assert _fetch_all(
        database_url,
        "SELECT scheduled_transaction_id FROM scheduled_subtransactions",
    ) == [{"scheduled_transaction_id": "scheduled-1"}]


def test_sync_uses_stored_checkpoints_and_can_skip_month_detail(monkeypatch, tmp_path):
    database_url = _configure_empty_db(monkeypatch, tmp_path)
    seen_checkpoints = {}

    async def _seed_checkpoint() -> None:
        db = DatabaseManager(database_url)
        await db.initialize()
        try:
            for resource, knowledge in {
                "accounts": 20,
                "payees": 21,
                "categories": 22,
                "months": 23,
                "transactions": 24,
                "scheduled_transactions": 25,
            }.items():
                await db.save_server_knowledge("budget-1", resource, knowledge)
        finally:
            await db.close()

    asyncio.run(_seed_checkpoint())

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def get_plans(self, **kwargs):
            return []

        async def get_accounts(self, **kwargs):
            seen_checkpoints["accounts"] = kwargs["last_knowledge_of_server"]
            return SyncResult([], 30)

        async def get_payees(self, **kwargs):
            seen_checkpoints["payees"] = kwargs["last_knowledge_of_server"]
            return SyncResult([], 31)

        async def get_payee_locations(self, **kwargs):
            return []

        async def get_categories(self, **kwargs):
            seen_checkpoints["categories"] = kwargs["last_knowledge_of_server"]
            return SyncResult(([], []), 32)

        async def get_months(self, **kwargs):
            seen_checkpoints["months"] = kwargs["last_knowledge_of_server"]
            return SyncResult([], 33)

        async def get_month(self, **kwargs):
            raise AssertionError("month detail should be skipped")

        async def get_transactions(self, **kwargs):
            seen_checkpoints["transactions"] = kwargs["last_knowledge_of_server"]
            return SyncResult([], 34)

        async def get_scheduled_transactions(self, **kwargs):
            seen_checkpoints["scheduled_transactions"] = kwargs[
                "last_knowledge_of_server"
            ]
            return SyncResult([], 35)

    import ynab_agent.api.client

    monkeypatch.setattr(ynab_agent.api.client, "YnabClient", FakeClient)

    result = runner.invoke(
        app,
        ["sync", "--plan-id", "budget-1", "--skip-month-detail"],
    )

    assert result.exit_code == 0
    assert seen_checkpoints == {
        "accounts": 20,
        "payees": 21,
        "categories": 22,
        "months": 23,
        "transactions": 24,
        "scheduled_transactions": 25,
    }


def test_full_sync_ignores_checkpoints_and_applies_since_date(monkeypatch, tmp_path):
    database_url = _configure_empty_db(monkeypatch, tmp_path)
    transaction_call = {}

    async def _seed_checkpoint() -> None:
        db = DatabaseManager(database_url)
        await db.initialize()
        try:
            await db.save_server_knowledge("budget-1", "transactions", 99)
        finally:
            await db.close()

    asyncio.run(_seed_checkpoint())

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def get_plans(self, **kwargs):
            return []

        async def get_accounts(self, **kwargs):
            assert kwargs["last_knowledge_of_server"] is None
            return SyncResult([], 30)

        async def get_payees(self, **kwargs):
            assert kwargs["last_knowledge_of_server"] is None
            return SyncResult([], 31)

        async def get_payee_locations(self, **kwargs):
            return []

        async def get_categories(self, **kwargs):
            assert kwargs["last_knowledge_of_server"] is None
            return SyncResult(([], []), 32)

        async def get_months(self, **kwargs):
            assert kwargs["last_knowledge_of_server"] is None
            return SyncResult([], 33)

        async def get_transactions(self, **kwargs):
            transaction_call.update(kwargs)
            return SyncResult([], 34)

        async def get_scheduled_transactions(self, **kwargs):
            assert kwargs["last_knowledge_of_server"] is None
            return SyncResult([], 35)

    import ynab_agent.api.client

    monkeypatch.setattr(ynab_agent.api.client, "YnabClient", FakeClient)

    result = runner.invoke(
        app,
        [
            "sync",
            "--plan-id",
            "budget-1",
            "--full",
            "--since",
            "2026-05-01",
            "--skip-month-detail",
        ],
    )

    assert result.exit_code == 0
    assert transaction_call["last_knowledge_of_server"] is None
    assert transaction_call["since_date"] == date(2026, 5, 1)
