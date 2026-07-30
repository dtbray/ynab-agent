from __future__ import annotations

from datetime import date
from enum import StrEnum

import pytest

from ynab_agent.services.reconciliation import (
    AccountStatus,
    MutationState,
    ReconcileBatchResponse,
    ReconciliationApplyResult,
    ReconciliationRequest,
    ReconciliationService,
    Row,
)


class ClearedStatus(StrEnum):
    CLEARED = "cleared"
    UNCLEARED = "uncleared"
    RECONCILED = "reconciled"


class FakeReader:
    def __init__(
        self,
        accounts: list[Row],
        transactions: list[Row],
    ) -> None:
        self.accounts = accounts
        self.transactions = transactions
        self.account_calls: list[str] = []
        self.transaction_calls: list[tuple[str, date | None]] = []

    async def get_accounts(self, *, plan_id: str) -> list[Row]:
        self.account_calls.append(plan_id)
        return self.accounts

    async def get_transactions(
        self,
        *,
        plan_id: str,
        since_date: date | None,
    ) -> list[Row]:
        self.transaction_calls.append((plan_id, since_date))
        return self.transactions


class FakeWriter:
    def __init__(
        self,
        responses: list[ReconcileBatchResponse | Exception] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, list[str]]] = []

    async def reconcile_transactions(
        self,
        *,
        plan_id: str,
        transaction_ids: list[str],
    ) -> ReconcileBatchResponse:
        self.calls.append((plan_id, list(transaction_ids)))
        if not self.responses:
            return ReconcileBatchResponse(tuple(transaction_ids))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def account(
    account_id: str,
    *,
    name: str | None = None,
    closed: bool = False,
    deleted: bool = False,
    linked: bool = True,
    import_error: bool = False,
    balance: int = 100_000,
    cleared_balance: int = 100_000,
    uncleared_balance: int = 0,
) -> Row:
    return {
        "id": account_id,
        "name": name or account_id,
        "type": "checking",
        "closed": closed,
        "deleted": deleted,
        "direct_import_linked": linked,
        "direct_import_in_error": import_error,
        "balance": balance,
        "cleared_balance": cleared_balance,
        "uncleared_balance": uncleared_balance,
    }


def transaction(
    transaction_id: str,
    account_id: str,
    cleared: ClearedStatus = ClearedStatus.CLEARED,
    *,
    deleted: bool = False,
) -> Row:
    return {
        "id": transaction_id,
        "account_id": account_id,
        "cleared": cleared,
        "deleted": deleted,
    }


@pytest.mark.asyncio
async def test_plan_is_deterministic_and_fetches_each_collection_once() -> None:
    reader = FakeReader(
        [
            account("manual", linked=False),
            account(
                "ready",
                balance=100_000,
                cleared_balance=75_000,
                uncleared_balance=25_000,
            ),
            account("blocked", import_error=True),
            account("closed", closed=True),
        ],
        [
            transaction("ready-2", "ready"),
            transaction("manual-1", "manual"),
            transaction("ready-1", "ready"),
            transaction("ready-uncleared", "ready", ClearedStatus.UNCLEARED),
            transaction("ready-reconciled", "ready", ClearedStatus.RECONCILED),
            transaction("ready-deleted", "ready", deleted=True),
            transaction("blocked-1", "blocked"),
        ],
    )
    request = ReconciliationRequest(
        plan_id="budget-1",
        since=date(2026, 5, 1),
    )

    plan = await ReconciliationService(reader).plan(request)

    assert reader.account_calls == ["budget-1"]
    assert reader.transaction_calls == [("budget-1", date(2026, 5, 1))]
    assert [item.account_id for item in plan.accounts] == ["blocked", "manual", "ready"]
    assert [item.status for item in plan.accounts] == [
        AccountStatus.BLOCKED,
        AccountStatus.MANUAL_REVIEW,
        AccountStatus.READY,
    ]
    ready = plan.accounts[-1]
    assert ready.eligible_transaction_ids == ("ready-1", "ready-2")
    assert ready.uncleared_transaction_count == 1
    assert ready.reconciled_transaction_count == 1
    assert ready.reason == "uncleared activity 25.00"
    assert [mutation.transaction_id for mutation in plan.mutations] == [
        "ready-1",
        "ready-2",
    ]
    assert len({mutation.mutation_id for mutation in plan.mutations}) == 2

    reordered_reader = FakeReader(
        list(reversed(reader.accounts)),
        list(reversed(reader.transactions)),
    )
    reordered = await ReconciliationService(reordered_reader).plan(request)

    assert reordered == plan


@pytest.mark.asyncio
async def test_plan_selection_and_include_closed_match_cli_rules() -> None:
    reader = FakeReader(
        [
            account("ready"),
            account("closed", closed=True),
            account("deleted", deleted=True),
        ],
        [
            transaction("ready-1", "ready"),
            transaction("closed-1", "closed"),
            transaction("deleted-1", "deleted"),
        ],
    )
    service = ReconciliationService(reader)

    selected = await service.plan(
        ReconciliationRequest(
            plan_id="budget-1",
            account_ids=("closed",),
            include_closed=False,
        )
    )
    included = await service.plan(
        ReconciliationRequest(
            plan_id="budget-1",
            account_ids=("closed",),
            include_closed=True,
        )
    )

    assert selected.accounts == ()
    assert selected.mutations == ()
    assert len(included.accounts) == 1
    assert included.accounts[0].status is AccountStatus.SKIPPED
    assert included.accounts[0].reasons == ("account is closed or deleted",)
    assert included.mutations == ()


@pytest.mark.asyncio
async def test_apply_mutates_only_ready_accounts() -> None:
    reader = FakeReader(
        [
            account("ready"),
            account("manual", linked=False),
            account("blocked", import_error=True),
        ],
        [
            transaction("ready-1", "ready"),
            transaction("manual-1", "manual"),
            transaction("blocked-1", "blocked"),
        ],
    )
    writer = FakeWriter()
    service = ReconciliationService(reader, writer)
    plan = await service.plan(ReconciliationRequest(plan_id="budget-1"))

    result = await service.apply(plan)

    assert writer.calls == [("budget-1", ["ready-1"])]
    assert result.is_complete is True
    assert result.reconciled_transaction_ids("ready") == ("ready-1",)
    assert result.reconciled_transaction_ids("manual") == ()


@pytest.mark.asyncio
async def test_apply_preserves_api_batch_limit() -> None:
    reader = FakeReader(
        [account("ready")],
        [
            transaction(f"transaction-{index:03d}", "ready")
            for index in range(205)
        ],
    )
    writer = FakeWriter()
    service = ReconciliationService(reader, writer)
    plan = await service.plan(ReconciliationRequest(plan_id="budget-1"))

    result = await service.apply(plan)

    assert [len(transaction_ids) for _, transaction_ids in writer.calls] == [
        100,
        100,
        5,
    ]
    assert len(result.completed_mutation_ids) == 205
    assert result.is_complete is True


@pytest.mark.asyncio
async def test_partial_response_retry_sends_only_unconfirmed_items() -> None:
    reader = FakeReader(
        [account("ready")],
        [
            transaction("transaction-1", "ready"),
            transaction("transaction-2", "ready"),
            transaction("transaction-3", "ready"),
        ],
    )
    writer = FakeWriter(
        [
            ReconcileBatchResponse(
                ("transaction-1", "transaction-3", "unexpected")
            ),
            ReconcileBatchResponse(("transaction-2",)),
        ]
    )
    service = ReconciliationService(reader, writer)
    plan = await service.plan(ReconciliationRequest(plan_id="budget-1"))

    partial = await service.apply(plan)
    completed_before_retry = partial.completed_mutation_ids
    retried = await service.apply(plan, previous=partial)

    assert writer.calls == [
        (
            "budget-1",
            ["transaction-1", "transaction-2", "transaction-3"],
        ),
        ("budget-1", ["transaction-2"]),
    ]
    assert partial.pending_mutation_ids == (plan.mutations[1].mutation_id,)
    assert partial.unexpected_transaction_ids == ("unexpected",)
    assert set(completed_before_retry).issubset(retried.completed_mutation_ids)
    assert retried.unexpected_transaction_ids == ("unexpected",)
    assert retried.is_complete is True
    attempts = {
        result.transaction_id: result.attempts
        for result in retried.mutations
    }
    assert attempts == {
        "transaction-1": 1,
        "transaction-2": 2,
        "transaction-3": 1,
    }


@pytest.mark.asyncio
async def test_unobserved_api_failure_is_not_automatically_duplicated() -> None:
    reader = FakeReader(
        [account("ready")],
        [transaction("transaction-1", "ready")],
    )
    writer = FakeWriter([TimeoutError("response was not observed")])
    service = ReconciliationService(reader, writer)
    plan = await service.plan(ReconciliationRequest(plan_id="budget-1"))

    failed = await service.apply(plan)
    retried = await service.apply(plan, previous=failed)

    assert len(writer.calls) == 1
    assert failed.unknown_mutation_ids == (plan.mutations[0].mutation_id,)
    assert failed.mutations[0].state is MutationState.UNKNOWN
    assert failed.failures[0].error == (
        "TimeoutError: response was not observed"
    )
    assert retried == failed


@pytest.mark.asyncio
async def test_previous_result_must_match_the_plan() -> None:
    reader = FakeReader(
        [account("ready")],
        [transaction("transaction-1", "ready")],
    )
    writer = FakeWriter()
    service = ReconciliationService(reader, writer)
    plan = await service.plan(ReconciliationRequest(plan_id="budget-1"))
    previous = ReconciliationApplyResult(
        plan_identity="another-plan",
        mutations=(),
    )

    with pytest.raises(ValueError, match="different reconciliation plan"):
        await service.apply(plan, previous=previous)

    assert writer.calls == []
