from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Generic, Literal, TypeVar

import pytest

from ynab_agent.services.sync import Row, SyncRequest, SyncService


T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    data: T
    server_knowledge: int | None


class FakeGateway:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.plans: list[Row] = [{"id": "budget-1", "name": "Main"}]
        self.accounts = Page[list[Row]]([{"id": "account-1"}], 10)
        self.payees = Page[list[Row]]([{"id": "payee-1"}], 11)
        self.payee_locations: list[Row] = [{"id": "location-1"}]
        self.categories = Page[tuple[list[Row], list[Row]]](
            ([{"id": "group-1"}], [{"id": "category-1"}]),
            12,
        )
        self.months = Page[list[Row]]([{"id": "month-1"}], 13)
        self.month_detail: tuple[Row, list[Row]] = (
            {"id": "month-1"},
            [{"id": "month-category-1"}, {"id": "month-category-2"}],
        )
        self.transactions = Page[list[Row]]([{"id": "transaction-1"}], 14)
        self.scheduled = Page[list[Row]](
            [
                {
                    "id": "scheduled-1",
                    "subtransactions": [{"id": "scheduled-split-1"}],
                }
            ],
            15,
        )
        self.plan_calls: list[bool] = []
        self.account_calls: list[tuple[str, int | None]] = []
        self.payee_calls: list[tuple[str, int | None]] = []
        self.location_calls: list[str] = []
        self.category_calls: list[tuple[str, int | None]] = []
        self.month_calls: list[tuple[str, int | None]] = []
        self.month_detail_calls: list[tuple[str, date]] = []
        self.transaction_calls: list[tuple[str, date | None, int | None]] = []
        self.scheduled_calls: list[tuple[str, int | None]] = []

    async def get_plans(self, *, include_accounts: bool) -> list[Row]:
        self.events.append("gateway:plans")
        self.plan_calls.append(include_accounts)
        return self.plans

    async def get_accounts(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> Page[list[Row]]:
        assert include_server_knowledge is True
        self.events.append("gateway:accounts")
        self.account_calls.append((plan_id, last_knowledge_of_server))
        return self.accounts

    async def get_payees(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> Page[list[Row]]:
        assert include_server_knowledge is True
        self.events.append("gateway:payees")
        self.payee_calls.append((plan_id, last_knowledge_of_server))
        return self.payees

    async def get_payee_locations(self, *, plan_id: str) -> list[Row]:
        self.events.append("gateway:payee_locations")
        self.location_calls.append(plan_id)
        return self.payee_locations

    async def get_categories(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> Page[tuple[list[Row], list[Row]]]:
        assert include_server_knowledge is True
        self.events.append("gateway:categories")
        self.category_calls.append((plan_id, last_knowledge_of_server))
        return self.categories

    async def get_months(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> Page[list[Row]]:
        assert include_server_knowledge is True
        self.events.append("gateway:months")
        self.month_calls.append((plan_id, last_knowledge_of_server))
        return self.months

    async def get_month(
        self,
        *,
        plan_id: str,
        month: date,
    ) -> tuple[Row, list[Row]]:
        self.events.append("gateway:month_detail")
        self.month_detail_calls.append((plan_id, month))
        return self.month_detail

    async def get_transactions(
        self,
        *,
        plan_id: str,
        since_date: date | None,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> Page[list[Row]]:
        assert include_server_knowledge is True
        self.events.append("gateway:transactions")
        self.transaction_calls.append(
            (plan_id, since_date, last_knowledge_of_server)
        )
        return self.transactions

    async def get_scheduled_transactions(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> Page[list[Row]]:
        assert include_server_knowledge is True
        self.events.append("gateway:scheduled")
        self.scheduled_calls.append((plan_id, last_knowledge_of_server))
        return self.scheduled


class FakeStore:
    def __init__(
        self,
        events: list[str],
        *,
        checkpoints: dict[tuple[str, str], int] | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.events = events
        self.checkpoints = checkpoints or {}
        self.fail_on = fail_on
        self.saved_checkpoints: list[tuple[str, str, int | None]] = []

    def _saved(self, resource: str) -> None:
        self.events.append(f"store:save:{resource}")
        if self.fail_on == resource:
            raise RuntimeError(f"failed to save {resource}")

    async def get_server_knowledge(
        self,
        plan_id: str,
        resource: str,
    ) -> int | None:
        self.events.append(f"store:checkpoint:get:{resource}")
        return self.checkpoints.get((plan_id, resource))

    async def save_server_knowledge(
        self,
        plan_id: str,
        resource: str,
        server_knowledge: int | None,
    ) -> None:
        self.events.append(f"store:checkpoint:save:{resource}")
        self.saved_checkpoints.append((plan_id, resource, server_knowledge))

    async def save_budgets(
        self,
        budgets: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        assert change_batch_id == "batch-1"
        self._saved("budgets")
        return len(budgets)

    async def save_accounts(
        self,
        plan_id: str,
        accounts: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        assert plan_id
        assert change_batch_id == "batch-1"
        self._saved("accounts")
        return len(accounts)

    async def save_payees(
        self,
        payees: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        assert change_batch_id == "batch-1"
        self._saved("payees")
        return len(payees)

    async def save_payee_locations(
        self,
        payee_locations: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        assert change_batch_id == "batch-1"
        self._saved("payee_locations")
        return len(payee_locations)

    async def save_categories(
        self,
        category_groups: list[Row],
        categories: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]:
        assert change_batch_id == "batch-1"
        self._saved("categories")
        return len(category_groups), len(categories)

    async def save_months(
        self,
        months: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        assert change_batch_id == "batch-1"
        self._saved("months")
        return len(months)

    async def save_month(
        self,
        month: Row,
        categories: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]:
        assert month
        assert change_batch_id == "batch-1"
        self._saved("month_detail")
        return 1, len(categories)

    async def save_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        assert plan_id
        assert change_batch_id == "batch-1"
        self._saved("transactions")
        return len(transactions)

    async def save_scheduled_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]:
        assert plan_id
        assert change_batch_id == "batch-1"
        self._saved("scheduled")
        split_count = sum(
            len(splits)
            for transaction in transactions
            if isinstance((splits := transaction.get("subtransactions")), list)
        )
        return len(transactions), split_count


class FakeValuationCapture:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.observed_at: datetime | None = None

    async def capture_current_valuations(self, *, observed_at: datetime) -> object:
        self.events.append("service:capture_valuations")
        self.observed_at = observed_at
        return ()


def service(
    gateway: FakeGateway,
    store: FakeStore,
    *,
    valuation_capture: FakeValuationCapture | None = None,
) -> SyncService:
    return SyncService(
        gateway,
        store,
        batch_id_factory=lambda: "batch-1",
        valuation_capture=valuation_capture,
        clock=lambda: datetime(2026, 7, 29, 15, 0, tzinfo=timezone.utc),
    )


def request(**overrides: object) -> SyncRequest:
    values: dict[str, object] = {
        "plan_id": "budget-1",
        "since": date(2026, 5, 1),
        "month": date(2026, 7, 1),
    }
    values.update(overrides)
    return SyncRequest(
        plan_id=str(values["plan_id"]),
        since=values["since"] if isinstance(values["since"], date) else date.min,
        month=values["month"] if isinstance(values["month"], date) else date.min,
        full=values.get("full") is True,
        skip_month_detail=values.get("skip_month_detail") is True,
    )


@pytest.mark.asyncio
async def test_sync_preserves_fetch_persist_checkpoint_sequence() -> None:
    events: list[str] = []
    checkpoints = {
        ("budget-1", "accounts"): 1,
        ("budget-1", "payees"): 2,
        ("budget-1", "categories"): 3,
        ("budget-1", "months"): 4,
        ("budget-1", "transactions"): 5,
        ("budget-1", "scheduled_transactions"): 6,
    }
    gateway = FakeGateway(events)
    store = FakeStore(events, checkpoints=checkpoints)

    summary = await service(gateway, store).run(request())

    assert summary.change_batch_id == "batch-1"
    assert summary.requested_plan_id == "budget-1"
    assert summary.resolved_plan_id == "budget-1"
    assert summary.used_plan_alias is False
    assert summary.plan_count == 1
    assert summary.account_count == 1
    assert summary.payee_count == 1
    assert summary.payee_location_count == 1
    assert summary.category_group_count == 1
    assert summary.category_count == 1
    assert summary.month_count == 1
    assert summary.month_category_count == 2
    assert summary.transaction_count == 1
    assert summary.scheduled_transaction_count == 1
    assert summary.scheduled_subtransaction_count == 1
    assert gateway.account_calls == [("budget-1", 1)]
    assert gateway.payee_calls == [("budget-1", 2)]
    assert gateway.category_calls == [("budget-1", 3)]
    assert gateway.month_calls == [("budget-1", 4)]
    assert gateway.transaction_calls == [("budget-1", None, 5)]
    assert gateway.scheduled_calls == [("budget-1", 6)]
    assert events == [
        "gateway:plans",
        "store:save:budgets",
        "store:checkpoint:get:accounts",
        "gateway:accounts",
        "store:save:accounts",
        "store:checkpoint:save:accounts",
        "store:checkpoint:get:payees",
        "gateway:payees",
        "store:save:payees",
        "store:checkpoint:save:payees",
        "gateway:payee_locations",
        "store:save:payee_locations",
        "store:checkpoint:get:categories",
        "gateway:categories",
        "store:save:categories",
        "store:checkpoint:save:categories",
        "store:checkpoint:get:months",
        "gateway:months",
        "store:save:months",
        "store:checkpoint:save:months",
        "gateway:month_detail",
        "store:save:month_detail",
        "store:checkpoint:get:transactions",
        "gateway:transactions",
        "store:save:transactions",
        "store:checkpoint:save:transactions",
        "store:checkpoint:get:scheduled_transactions",
        "gateway:scheduled",
        "store:save:scheduled",
        "store:checkpoint:save:scheduled_transactions",
    ]


@pytest.mark.asyncio
async def test_full_sync_ignores_checkpoints_applies_since_and_skips_detail() -> None:
    events: list[str] = []
    gateway = FakeGateway(events)
    store = FakeStore(
        events,
        checkpoints={("budget-1", "transactions"): 99},
    )

    summary = await service(gateway, store).run(
        request(full=True, skip_month_detail=True)
    )

    assert summary.month_category_count == 0
    assert gateway.account_calls == [("budget-1", None)]
    assert gateway.transaction_calls == [
        ("budget-1", date(2026, 5, 1), None)
    ]
    assert gateway.month_detail_calls == []
    assert not any(event.startswith("store:checkpoint:get:") for event in events)


@pytest.mark.asyncio
async def test_alias_resolves_once_then_uses_concrete_plan_id() -> None:
    events: list[str] = []
    gateway = FakeGateway(events)
    gateway.plans = [
        {"id": "budget-1", "account_ids": ["account-other"]},
        {"id": "budget-2", "account_ids": ["account-1"]},
    ]
    store = FakeStore(events)

    summary = await service(gateway, store).run(request(plan_id="last-used"))

    assert summary.requested_plan_id == "last-used"
    assert summary.resolved_plan_id == "budget-2"
    assert summary.used_plan_alias is True
    assert gateway.plan_calls == [True]
    assert gateway.account_calls == [("last-used", None)]
    assert gateway.payee_calls == [("budget-2", None)]
    assert gateway.location_calls == ["budget-2"]
    assert store.saved_checkpoints[0] == ("budget-2", "accounts", 10)


@pytest.mark.asyncio
async def test_failed_persistence_does_not_advance_resource_checkpoint() -> None:
    events: list[str] = []
    gateway = FakeGateway(events)
    store = FakeStore(events, fail_on="payees")

    with pytest.raises(RuntimeError, match="failed to save payees"):
        await service(gateway, store).run(request())

    assert store.saved_checkpoints == [("budget-1", "accounts", 10)]
    assert gateway.category_calls == []
    assert "store:checkpoint:save:payees" not in events


@pytest.mark.asyncio
async def test_valuation_capture_runs_after_accounts_are_persisted_and_checkpointed() -> None:
    events: list[str] = []
    gateway = FakeGateway(events)
    store = FakeStore(events)
    capture = FakeValuationCapture(events)

    await service(gateway, store, valuation_capture=capture).run(request())

    assert events.index("store:save:accounts") < events.index(
        "store:checkpoint:save:accounts"
    )
    assert events.index("store:checkpoint:save:accounts") < events.index(
        "service:capture_valuations"
    )
    assert events.index("service:capture_valuations") < events.index("gateway:payees")
    assert capture.observed_at == datetime(
        2026,
        7,
        29,
        15,
        0,
        tzinfo=timezone.utc,
    )


@pytest.mark.asyncio
async def test_ambiguous_alias_stops_before_account_persistence() -> None:
    events: list[str] = []
    gateway = FakeGateway(events)
    gateway.plans = [
        {"id": "budget-1", "account_ids": ["account-other-1"]},
        {"id": "budget-2", "account_ids": ["account-other-2"]},
    ]
    store = FakeStore(events)

    with pytest.raises(RuntimeError, match="Could not resolve YNAB plan alias"):
        await service(gateway, store).run(request(plan_id="default"))

    assert "store:save:budgets" in events
    assert "store:save:accounts" not in events
    assert store.saved_checkpoints == []
