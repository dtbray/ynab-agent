from datetime import date
from types import SimpleNamespace

import pytest

from ynab_agent.config import settings
from ynab_agent.api.client import SyncResult, YnabClient


@pytest.mark.asyncio
async def test_client_exposes_authenticated_sdk_apis(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        assert client.accounts is not None
        assert client.plans is not None
        assert client.transactions is not None


@pytest.mark.asyncio
async def test_get_transactions_returns_plain_dicts(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        transaction = SimpleNamespace(
            id="txn-1",
            account_id="account-1",
            category_id="category-1",
            payee_id="payee-1",
            date=None,
            amount=-12345,
            memo="Coffee",
            cleared="uncleared",
            approved=False,
            flag_color=None,
            flag_name=None,
            deleted=False,
            subtransactions=[
                SimpleNamespace(
                    id="subtxn-1",
                    transaction_id="txn-1",
                    amount=-12345,
                    memo="Split coffee",
                    payee_id="payee-1",
                    payee_name="Cafe",
                    category_id="category-1",
                    category_name="Dining Out",
                    transfer_account_id=None,
                    transfer_transaction_id=None,
                    deleted=False,
                )
            ],
        )
        response = SimpleNamespace(data=SimpleNamespace(transactions=[transaction], server_knowledge=42))

        def fake_get_transactions(**kwargs):
            assert kwargs["plan_id"] == "last-used"
            assert kwargs["type"] == "unapproved"
            assert kwargs["last_knowledge_of_server"] == 41
            return response

        monkeypatch.setattr(client.transactions, "get_transactions", fake_get_transactions)

        result = await client.get_transactions(
            transaction_type="unapproved",
            last_knowledge_of_server=41,
            include_server_knowledge=True,
        )

    assert isinstance(result, SyncResult)
    assert result.server_knowledge == 42
    assert result.data == [
        {
            "id": "txn-1",
            "account_id": "account-1",
            "category_id": "category-1",
            "payee_id": "payee-1",
            "transfer_account_id": None,
            "transfer_transaction_id": None,
                "matched_transaction_id": None,
                "import_id": None,
                "import_payee_name": None,
                "import_payee_name_original": None,
                "debt_transaction_type": None,
                "account_name": None,
                "payee_name": None,
                "category_name": None,
                "date": None,
            "amount": -12345,
            "memo": "Coffee",
            "cleared": "uncleared",
            "approved": False,
            "flag_color": None,
            "flag_name": None,
            "foreign_amount": None,
                "foreign_currency_code": None,
                "deleted": False,
                "subtransactions": [
                    {
                        "id": "subtxn-1",
                        "transaction_id": "txn-1",
                        "amount": -12345,
                        "memo": "Split coffee",
                        "payee_id": "payee-1",
                        "payee_name": "Cafe",
                        "category_id": "category-1",
                        "category_name": "Dining Out",
                        "transfer_account_id": None,
                        "transfer_transaction_id": None,
                        "deleted": False,
                    }
                ],
            }
        ]


@pytest.mark.asyncio
async def test_get_plans_preserves_currency_metadata(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        plan = SimpleNamespace(
            id="budget-1",
            name="Main",
            last_modified_on=None,
            first_month=date(2020, 1, 1),
            last_month=date(2026, 7, 1),
            date_format=SimpleNamespace(format="MM/DD/YYYY"),
            currency_format=SimpleNamespace(
                iso_code="USD",
                example_format="123,456.78",
                decimal_digits=2,
                decimal_separator=".",
                symbol_first=True,
                group_separator=",",
                currency_symbol="$",
                display_symbol=True,
            ),
            accounts=[SimpleNamespace(id="account-1")],
        )
        monkeypatch.setattr(
            client.plans,
            "get_plans",
            lambda **kwargs: SimpleNamespace(data=SimpleNamespace(plans=[plan])),
        )

        result = await client.get_plans()
        result_with_accounts = await client.get_plans(include_accounts=True)

    assert result == [
        {
            "id": "budget-1",
            "name": "Main",
            "last_modified_on": None,
            "first_month": "2020-01-01",
            "last_month": "2026-07-01",
            "date_format": "MM/DD/YYYY",
            "currency_format_iso_code": "USD",
            "currency_format_example": "123,456.78",
            "currency_decimal_digits": 2,
            "currency_decimal_separator": ".",
            "currency_symbol_first": True,
            "currency_group_separator": ",",
            "currency_symbol": "$",
            "currency_display_symbol": True,
        }
    ]
    assert result_with_accounts[0]["account_ids"] == ["account-1"]


@pytest.mark.asyncio
async def test_get_scheduled_transactions_preserves_split_detail(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        split = SimpleNamespace(
            id="scheduled-split-1",
            scheduled_transaction_id="scheduled-1",
            amount=-50000,
            memo="Rent portion",
            payee_id=None,
            payee_name=None,
            category_id="category-rent",
            category_name="Rent",
            transfer_account_id=None,
            deleted=False,
        )
        scheduled = SimpleNamespace(
            id="scheduled-1",
            date_first=date(2026, 1, 1),
            date_next=date(2026, 8, 1),
            frequency="monthly",
            amount=-50000,
            memo="Rent",
            flag_color=None,
            flag_name=None,
            account_id="account-1",
            payee_id="payee-1",
            category_id=None,
            transfer_account_id=None,
            account_name="Checking",
            payee_name="Landlord",
            category_name=None,
            deleted=False,
            subtransactions=[split],
        )
        monkeypatch.setattr(
            client.scheduled_transactions,
            "get_scheduled_transactions",
            lambda **kwargs: SimpleNamespace(
                data=SimpleNamespace(
                    scheduled_transactions=[scheduled],
                    server_knowledge=9,
                )
            ),
        )

        result = await client.get_scheduled_transactions(
            last_knowledge_of_server=8,
            include_server_knowledge=True,
        )

    assert isinstance(result, SyncResult)
    assert result.server_knowledge == 9
    assert result.data[0]["account_name"] == "Checking"
    assert result.data[0]["subtransactions"] == [
        {
            "id": "scheduled-split-1",
            "scheduled_transaction_id": "scheduled-1",
            "amount": -50000,
            "memo": "Rent portion",
            "payee_id": None,
            "payee_name": None,
            "category_id": "category-rent",
            "category_name": "Rent",
            "transfer_account_id": None,
            "deleted": False,
        }
    ]


def test_get_token_reports_missing_pat_config(monkeypatch):
    monkeypatch.setattr(settings, "ynab_auth_mode", "pat")
    monkeypatch.setattr(settings, "ynab_access_token", None)

    with pytest.raises(ValueError, match="YNAB_AUTH_MODE=pat requires YNAB_ACCESS_TOKEN"):
        YnabClient()._get_token()


def test_get_token_reports_unknown_auth_mode(monkeypatch):
    monkeypatch.setattr(settings, "ynab_auth_mode", "magic")

    with pytest.raises(ValueError, match="YNAB_AUTH_MODE must be one of"):
        YnabClient()._get_token()


@pytest.mark.asyncio
async def test_create_transaction_uses_sdk_wrapper(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        captured = {}

        def fake_create_transaction(plan_id, data):
            captured["plan_id"] = plan_id
            captured["data"] = data
            transaction = SimpleNamespace(
                id="txn-1",
                account_id="7fc6f73c-098e-49bb-b215-a55d53434f92",
                category_id=None,
                payee_id=None,
                date=None,
                amount=-4250,
                memo="API smoke test",
                cleared="uncleared",
                approved=False,
                flag_color=None,
                flag_name=None,
                deleted=False,
            )
            return SimpleNamespace(
                data=SimpleNamespace(
                    transaction_ids=["txn-1"],
                    duplicate_import_ids=[],
                    server_knowledge=123,
                    transaction=transaction,
                )
            )

        monkeypatch.setattr(client.transactions, "create_transaction", fake_create_transaction)

        result = await client.create_transaction(
            plan_id="last-used",
            account_id="7fc6f73c-098e-49bb-b215-a55d53434f92",
            transaction_date=date(2026, 5, 20),
            amount=-4250,
            payee_name="Coffee Shop",
            memo="API smoke test",
        )

    assert captured["plan_id"] == "last-used"
    assert captured["data"].transaction.amount == -4250
    assert captured["data"].transaction.payee_name == "Coffee Shop"
    assert result["transaction_ids"] == ["txn-1"]
    assert result["server_knowledge"] == 123


@pytest.mark.asyncio
async def test_get_transactions_by_account_returns_plain_dicts(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        transaction = SimpleNamespace(
            id="txn-1",
            account_id="account-1",
            category_id="category-1",
            payee_id="payee-1",
            date=None,
            amount=-12345,
            memo="Coffee",
            cleared="cleared",
            approved=True,
            flag_color=None,
            flag_name=None,
            deleted=False,
        )
        response = SimpleNamespace(data=SimpleNamespace(transactions=[transaction], server_knowledge=42))

        def fake_get_transactions_by_account(**kwargs):
            assert kwargs["plan_id"] == "last-used"
            assert kwargs["account_id"] == "account-1"
            assert kwargs["since_date"] == date(2026, 5, 1)
            return response

        monkeypatch.setattr(
            client.transactions,
            "get_transactions_by_account",
            fake_get_transactions_by_account,
        )

        result = await client.get_transactions_by_account(
            account_id="account-1",
            since_date=date(2026, 5, 1),
            include_server_knowledge=True,
        )

    assert isinstance(result, SyncResult)
    assert result.server_knowledge == 42
    assert result.data[0]["id"] == "txn-1"
    assert result.data[0]["cleared"] == "cleared"


@pytest.mark.asyncio
async def test_reconcile_transactions_uses_bulk_patch_wrapper(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        captured = {}

        def fake_update_transactions(plan_id, data):
            captured["plan_id"] = plan_id
            captured["data"] = data
            transaction = SimpleNamespace(
                id="txn-1",
                account_id="account-1",
                category_id=None,
                payee_id=None,
                date=None,
                amount=-4250,
                memo=None,
                cleared="reconciled",
                approved=True,
                flag_color=None,
                flag_name=None,
                deleted=False,
            )
            return SimpleNamespace(
                data=SimpleNamespace(
                    transaction_ids=["txn-1"],
                    duplicate_import_ids=[],
                    server_knowledge=124,
                    transactions=[transaction],
                )
            )

        monkeypatch.setattr(client.transactions, "update_transactions", fake_update_transactions)

        result = await client.reconcile_transactions(
            plan_id="last-used",
            transaction_ids=["txn-1"],
        )

    assert captured["plan_id"] == "last-used"
    assert captured["data"].transactions[0].id == "txn-1"
    assert captured["data"].transactions[0].cleared.value == "reconciled"
    assert result["transaction_ids"] == ["txn-1"]
    assert result["server_knowledge"] == 124
    assert result["transactions"][0]["cleared"] == "reconciled"


@pytest.mark.asyncio
async def test_reconcile_transactions_handles_empty_sdk_response(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        monkeypatch.setattr(client.transactions, "update_transactions", lambda **kwargs: None)

        result = await client.reconcile_transactions(
            plan_id="last-used",
            transaction_ids=["txn-1", "txn-2"],
        )

    assert result == {
        "transaction_ids": ["txn-1", "txn-2"],
        "duplicate_import_ids": [],
        "server_knowledge": None,
        "transactions": [],
    }


@pytest.mark.asyncio
async def test_update_month_category_budgeted_rejects_internal_category(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        async def fake_get_categories(plan_id):
            return (
                [
                    {
                        "id": "group-internal",
                        "budget_id": plan_id,
                        "name": "Internal Master Category",
                        "hidden": False,
                        "deleted": False,
                    }
                ],
                [
                    {
                        "id": "11111111-1111-4111-8111-111111111111",
                        "budget_id": plan_id,
                        "category_group_id": "group-internal",
                        "name": "Uncategorized",
                        "hidden": False,
                        "deleted": False,
                    }
                ],
            )

        monkeypatch.setattr(
            client,
            "get_categories",
            fake_get_categories,
        )

        with pytest.raises(ValueError, match="non-budgetable group"):
            await client.update_month_category_budgeted(
                plan_id="last-used",
                month=date(2026, 5, 1),
                category_id="11111111-1111-4111-8111-111111111111",
                budgeted=64540,
            )


@pytest.mark.asyncio
async def test_update_month_category_budgeted_rejects_credit_card_payment_category(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        async def fake_get_categories(plan_id):
            return (
                [
                    {
                        "id": "group-cards",
                        "budget_id": plan_id,
                        "name": "Credit Card Payments",
                        "hidden": False,
                        "deleted": False,
                    }
                ],
                [
                    {
                        "id": "22222222-2222-4222-8222-222222222222",
                        "budget_id": plan_id,
                        "category_group_id": "group-cards",
                        "name": "Alliant CC",
                        "hidden": False,
                        "deleted": False,
                    }
                ],
            )

        monkeypatch.setattr(
            client,
            "get_categories",
            fake_get_categories,
        )

        with pytest.raises(ValueError, match="non-budgetable group"):
            await client.update_month_category_budgeted(
                plan_id="last-used",
                month=date(2026, 5, 1),
                category_id="22222222-2222-4222-8222-222222222222",
                budgeted=64540,
            )


@pytest.mark.asyncio
async def test_update_month_category_budgeted_uses_sdk_for_budgetable_category(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        captured = {}
        category_id = "33333333-3333-4333-8333-333333333333"

        async def fake_get_categories(plan_id):
            return (
                [
                    {
                        "id": "group-monthly",
                        "budget_id": plan_id,
                        "name": "Monthly Bills",
                        "hidden": False,
                        "deleted": False,
                    }
                ],
                [
                    {
                        "id": category_id,
                        "budget_id": plan_id,
                        "category_group_id": "group-monthly",
                        "name": "Electric",
                        "hidden": False,
                        "deleted": False,
                    }
                ],
            )

        monkeypatch.setattr(
            client,
            "get_categories",
            fake_get_categories,
        )

        def fake_update_month_category(plan_id, month, category_id, data):
            captured["plan_id"] = plan_id
            captured["month"] = month
            captured["category_id"] = category_id
            captured["data"] = data
            return SimpleNamespace(
                data=SimpleNamespace(
                    category=SimpleNamespace(
                        id=category_id,
                        budgeted=120000,
                        activity=-100000,
                        balance=20000,
                    )
                )
            )

        monkeypatch.setattr(client.categories, "update_month_category", fake_update_month_category)

        result = await client.update_month_category_budgeted(
            plan_id="last-used",
            month=date(2026, 5, 1),
            category_id=category_id,
            budgeted=120000,
        )

    assert captured["plan_id"] == "last-used"
    assert str(captured["category_id"]) == category_id
    assert captured["data"].category.budgeted == 120000
    assert result["budgeted"] == 120000


@pytest.mark.asyncio
async def test_get_month_preserves_month_category_detail_metadata(monkeypatch):
    monkeypatch.setattr(YnabClient, "_get_token", lambda self: "token")

    async with YnabClient() as client:
        category = SimpleNamespace(
            id="category-food",
            category_group_id="group-bills",
            category_group_name="Bills",
            name="Groceries",
            hidden=False,
            deleted=False,
            budgeted=250000,
            activity=-125000,
            balance=125000,
            goal_type="NEED",
            goal_needs_whole_amount=False,
            goal_day=15,
            goal_cadence=1,
            goal_cadence_frequency=1,
            goal_creation_month=date(2026, 1, 1),
            goal_target=300000,
            goal_target_month=date(2026, 7, 1),
            goal_target_date=date(2026, 7, 15),
            goal_percentage_complete=50,
            goal_months_to_budget=1,
            goal_under_funded=50000,
            goal_overall_funded=225000,
            goal_overall_left=75000,
            goal_snoozed_at=None,
        )
        month_detail = SimpleNamespace(
            month=date(2026, 7, 1),
            note="",
            income=0,
            budgeted=250000,
            activity=-125000,
            to_be_budgeted=0,
            age_of_money=20,
            deleted=False,
            categories=[category],
        )

        def fake_get_plan_month(**kwargs):
            assert kwargs["plan_id"] == "last-used"
            assert kwargs["month"] == date(2026, 7, 1)
            return SimpleNamespace(data=SimpleNamespace(month=month_detail))

        monkeypatch.setattr(client.months, "get_plan_month", fake_get_plan_month)

        month_row, categories = await client.get_month(
            plan_id="last-used",
            month=date(2026, 7, 1),
        )

    assert month_row["id"] == "last-used:2026-07-01"
    assert categories == [
        {
            "id": "last-used:2026-07-01:category-food",
            "budget_id": "last-used",
            "month": "2026-07-01",
            "category_id": "category-food",
            "category_group_id": "group-bills",
            "category_group_name": "Bills",
            "name": "Groceries",
            "hidden": False,
            "internal": False,
            "deleted": False,
            "budgeted": 250000,
            "activity": -125000,
            "balance": 125000,
            "goal_type": "NEED",
            "goal_needs_whole_amount": False,
            "goal_day": 15,
            "goal_cadence": 1,
            "goal_cadence_frequency": 1,
            "goal_creation_month": "2026-01-01",
            "goal_target": 300000,
            "goal_target_month": "2026-07-01",
            "goal_target_date": "2026-07-15",
            "goal_percentage_complete": 50,
            "goal_months_to_budget": 1,
            "goal_under_funded": 50000,
            "goal_overall_funded": 225000,
            "goal_overall_left": 75000,
            "goal_snoozed_at": None,
        }
    ]
