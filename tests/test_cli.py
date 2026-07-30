import json

from typer.testing import CliRunner

from ynab_agent.cli import _dollars_to_milliunits, _reconciliation_status, app


runner = CliRunner()


def test_dollars_to_milliunits_uses_decimal_math():
    assert _dollars_to_milliunits("-4.25") == -4250
    assert _dollars_to_milliunits("1.005") == 1005


def test_transaction_create_defaults_to_dry_run():
    result = runner.invoke(
        app,
        [
            "transactions",
            "create",
            "--account-id",
            "7fc6f73c-098e-49bb-b215-a55d53434f92",
            "--amount",
            "-4.25",
            "--payee-name",
            "Coffee Shop",
            "--memo",
            "dry run",
            "--date",
            "2026-05-20",
            "--json",
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["dry_run"] is True
    assert output["transaction"]["amount"] == -4250
    assert output["transaction"]["payee_name"] == "Coffee Shop"


def test_transaction_create_requires_confirmation_for_execute():
    result = runner.invoke(
        app,
        [
            "transactions",
            "create",
            "--account-id",
            "7fc6f73c-098e-49bb-b215-a55d53434f92",
            "--amount",
            "-4.25",
            "--payee-name",
            "Coffee Shop",
            "--execute",
        ],
    )

    assert result.exit_code == 2


def test_reconcile_accounts_fetches_transactions_once(monkeypatch):
    calls = {"get_transactions": 0, "get_transactions_by_account": 0}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def get_accounts(self, plan_id):
            return [
                {
                    "id": "account-1",
                    "name": "Checking",
                    "type": "checking",
                    "balance": 100000,
                    "cleared_balance": 75000,
                    "uncleared_balance": 25000,
                    "direct_import_linked": True,
                    "direct_import_in_error": False,
                    "closed": False,
                    "deleted": False,
                }
            ]

        async def get_transactions(self, plan_id, since_date=None):
            calls["get_transactions"] += 1
            return [
                {
                    "id": "txn-1",
                    "account_id": "account-1",
                    "cleared": "cleared",
                    "deleted": False,
                }
            ]

        async def get_transactions_by_account(self, **kwargs):
            calls["get_transactions_by_account"] += 1
            return []

    import ynab_agent.api.client

    monkeypatch.setattr(ynab_agent.api.client, "YnabClient", FakeClient)

    result = runner.invoke(app, ["accounts", "reconcile", "--json"])

    assert result.exit_code == 0
    assert calls == {"get_transactions": 1, "get_transactions_by_account": 0}
    output = json.loads(result.stdout)
    assert output[0]["eligible_transactions"] == 1


def test_reconcile_accounts_bulk_reconciles_ready_accounts(monkeypatch):
    calls = {"reconcile_transactions": []}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def get_accounts(self, plan_id):
            return [
                {
                    "id": "account-1",
                    "name": "Checking",
                    "type": "checking",
                    "balance": 100000,
                    "cleared_balance": 75000,
                    "uncleared_balance": 25000,
                    "direct_import_linked": True,
                    "direct_import_in_error": False,
                    "closed": False,
                    "deleted": False,
                },
                {
                    "id": "account-2",
                    "name": "Credit Card",
                    "type": "creditCard",
                    "balance": -50000,
                    "cleared_balance": -50000,
                    "uncleared_balance": 0,
                    "direct_import_linked": True,
                    "direct_import_in_error": False,
                    "closed": False,
                    "deleted": False,
                },
            ]

        async def get_transactions(self, plan_id, since_date=None):
            return [
                {
                    "id": "txn-1",
                    "account_id": "account-1",
                    "cleared": "cleared",
                    "deleted": False,
                },
                {
                    "id": "txn-2",
                    "account_id": "account-2",
                    "cleared": "cleared",
                    "deleted": False,
                },
            ]

        async def reconcile_transactions(self, plan_id, transaction_ids):
            calls["reconcile_transactions"].append(transaction_ids)
            return {
                "transaction_ids": transaction_ids,
                "duplicate_import_ids": [],
                "server_knowledge": None,
                "transactions": [],
            }

    import ynab_agent.api.client

    monkeypatch.setattr(ynab_agent.api.client, "YnabClient", FakeClient)

    result = runner.invoke(app, ["accounts", "reconcile", "--execute", "--yes", "--json"])

    assert result.exit_code == 0
    assert calls["reconcile_transactions"] == [["txn-1", "txn-2"]]
    output = json.loads(result.stdout)
    assert output[0]["reconciled_transaction_ids"] == ["txn-1"]
    assert output[1]["reconciled_transaction_ids"] == ["txn-2"]


def test_reconciliation_status_marks_ready_account():
    account = {
        "id": "account-1",
        "name": "Checking",
        "type": "checking",
        "balance": 100000,
        "cleared_balance": 100000,
        "uncleared_balance": 0,
        "direct_import_linked": True,
        "direct_import_in_error": False,
        "closed": False,
        "deleted": False,
    }

    result = _reconciliation_status(
        account,
        [
            {"id": "txn-1", "cleared": "cleared", "deleted": False},
            {"id": "txn-2", "cleared": "reconciled", "deleted": False},
        ],
    )

    assert result["status"] == "ready"
    assert result["eligible_transaction_ids"] == ["txn-1"]
    assert result["eligible_transactions"] == 1


def test_reconciliation_status_keeps_uncleared_activity_reconcilable():
    account = {
        "id": "account-1",
        "name": "Checking",
        "type": "checking",
        "balance": 100000,
        "cleared_balance": 75000,
        "uncleared_balance": 25000,
        "direct_import_linked": True,
        "direct_import_in_error": False,
        "closed": False,
        "deleted": False,
    }

    result = _reconciliation_status(
        account,
        [{"id": "txn-1", "cleared": "uncleared", "deleted": False}],
    )

    assert result["status"] == "ready"
    assert result["uncleared_transactions"] == 1
    assert "uncleared activity 25.00" in result["reason"]


def test_reconciliation_status_surfaces_manual_and_import_error_accounts():
    manual_account = {
        "id": "account-1",
        "name": "Manual Checking",
        "type": "checking",
        "balance": 100000,
        "cleared_balance": 100000,
        "uncleared_balance": 0,
        "direct_import_linked": False,
        "direct_import_in_error": False,
        "closed": False,
        "deleted": False,
    }
    import_error_account = {
        **manual_account,
        "id": "account-2",
        "direct_import_linked": True,
        "direct_import_in_error": True,
    }

    assert _reconciliation_status(manual_account, [])["status"] == "manual_review"
    blocked = _reconciliation_status(import_error_account, [])
    assert blocked["status"] == "blocked"
    assert "direct import is in error" in blocked["reason"]
