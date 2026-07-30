"""Deterministic planning and resumable application of account reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from typing import Protocol


Row = dict[str, object]
MAX_RECONCILIATION_BATCH_SIZE = 100


class ReconciliationReader(Protocol):
    """Read operations required to build a reconciliation plan."""

    async def get_accounts(self, *, plan_id: str) -> list[Row]: ...

    async def get_transactions(
        self,
        *,
        plan_id: str,
        since_date: date | None,
    ) -> list[Row]: ...


class ReconciliationWriter(Protocol):
    """Mutation operation required to apply a reconciliation plan."""

    async def reconcile_transactions(
        self,
        *,
        plan_id: str,
        transaction_ids: list[str],
    ) -> ReconcileBatchResponse: ...


@dataclass(frozen=True)
class ReconcileBatchResponse:
    """Transaction IDs confirmed by one API mutation response."""

    transaction_ids: tuple[str, ...]
    server_knowledge: int | None = None


class AccountStatus(StrEnum):
    """Reconciliation readiness classification for an account."""

    READY = "ready"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    MANUAL_REVIEW = "manual_review"


class MutationState(StrEnum):
    """State of one planned transaction mutation."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReconciliationRequest:
    """Scope used to build a dry-run reconciliation plan."""

    plan_id: str
    account_ids: tuple[str, ...] = ()
    since: date | None = None
    include_closed: bool = False


@dataclass(frozen=True)
class AccountReconciliation:
    """Readiness and transaction counts for one selected account."""

    account_id: str
    name: str | None
    account_type: str | None
    status: AccountStatus
    balance: int
    cleared_balance: int
    uncleared_balance: int
    uncleared_delta: int
    eligible_transaction_ids: tuple[str, ...]
    uncleared_transaction_count: int
    reconciled_transaction_count: int
    last_reconciled_at: str | None
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        """Return the CLI-compatible joined reason text."""
        return "; ".join(self.reasons)


@dataclass(frozen=True)
class ReconciliationMutation:
    """One stable, independently reportable transaction mutation."""

    mutation_id: str
    account_id: str
    transaction_id: str


@dataclass(frozen=True)
class ReconciliationPlan:
    """Deterministic dry-run result that can later be applied."""

    identity: str
    request: ReconciliationRequest
    accounts: tuple[AccountReconciliation, ...]
    mutations: tuple[ReconciliationMutation, ...]


@dataclass(frozen=True)
class MutationResult:
    """Per-item outcome retained across partial retries."""

    mutation_id: str
    account_id: str
    transaction_id: str
    state: MutationState
    attempts: int = 0
    error: str | None = None


@dataclass(frozen=True)
class BatchFailure:
    """A batch whose API outcome could not be observed."""

    mutation_ids: tuple[str, ...]
    error: str


@dataclass(frozen=True)
class ReconciliationApplyResult:
    """Complete per-item state after one apply or retry attempt."""

    plan_identity: str
    mutations: tuple[MutationResult, ...]
    unexpected_transaction_ids: tuple[str, ...] = ()
    failures: tuple[BatchFailure, ...] = ()

    @property
    def completed_mutation_ids(self) -> tuple[str, ...]:
        """Return stable IDs confirmed as successfully applied."""
        return tuple(
            result.mutation_id
            for result in self.mutations
            if result.state is MutationState.SUCCEEDED
        )

    @property
    def pending_mutation_ids(self) -> tuple[str, ...]:
        """Return stable IDs safe to send on a subsequent retry."""
        return tuple(
            result.mutation_id
            for result in self.mutations
            if result.state is MutationState.PENDING
        )

    @property
    def unknown_mutation_ids(self) -> tuple[str, ...]:
        """Return stable IDs requiring a fresh read before safe retry."""
        return tuple(
            result.mutation_id
            for result in self.mutations
            if result.state is MutationState.UNKNOWN
        )

    @property
    def is_complete(self) -> bool:
        """Return whether every planned mutation was confirmed."""
        return all(
            result.state is MutationState.SUCCEEDED
            for result in self.mutations
        )

    def reconciled_transaction_ids(self, account_id: str) -> tuple[str, ...]:
        """Return confirmed transaction IDs for one account."""
        return tuple(
            result.transaction_id
            for result in self.mutations
            if result.account_id == account_id
            and result.state is MutationState.SUCCEEDED
        )


class ReconciliationService:
    """Build dry-run plans and safely apply their pending mutations."""

    def __init__(
        self,
        reader: ReconciliationReader,
        writer: ReconciliationWriter | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer

    async def plan(self, request: ReconciliationRequest) -> ReconciliationPlan:
        """Fetch accounts and transactions exactly once and build a stable plan."""
        accounts = await self._reader.get_accounts(plan_id=request.plan_id)
        transactions = await self._reader.get_transactions(
            plan_id=request.plan_id,
            since_date=request.since,
        )
        transactions_by_account: dict[str, list[Row]] = {}
        for transaction in transactions:
            account_id = transaction.get("account_id")
            if account_id is None:
                continue
            transactions_by_account.setdefault(str(account_id), []).append(transaction)

        selected_accounts = set(request.account_ids)
        account_plans: list[AccountReconciliation] = []
        mutations: list[ReconciliationMutation] = []
        seen_transaction_ids: set[str] = set()
        for account in accounts:
            raw_account_id = account.get("id")
            if raw_account_id is None:
                raise ValueError("account is missing an id")
            account_id = str(raw_account_id)
            if selected_accounts and account_id not in selected_accounts:
                continue
            if (
                not request.include_closed
                and (bool(account.get("closed")) or bool(account.get("deleted")))
            ):
                continue

            account_plan = classify_account(
                account,
                transactions_by_account.get(account_id, []),
            )
            account_plans.append(account_plan)
            if account_plan.status is not AccountStatus.READY:
                continue
            for transaction_id in account_plan.eligible_transaction_ids:
                if transaction_id in seen_transaction_ids:
                    continue
                seen_transaction_ids.add(transaction_id)
                mutations.append(
                    ReconciliationMutation(
                        mutation_id=_mutation_id(
                            request.plan_id,
                            account_id,
                            transaction_id,
                        ),
                        account_id=account_id,
                        transaction_id=transaction_id,
                    )
                )

        account_plans.sort(key=lambda account: account.account_id)
        mutations.sort(key=lambda mutation: (mutation.account_id, mutation.transaction_id))
        plan_accounts = tuple(account_plans)
        plan_mutations = tuple(mutations)
        return ReconciliationPlan(
            identity=_plan_identity(request, plan_accounts, plan_mutations),
            request=request,
            accounts=plan_accounts,
            mutations=plan_mutations,
        )

    async def apply(
        self,
        plan: ReconciliationPlan,
        *,
        previous: ReconciliationApplyResult | None = None,
    ) -> ReconciliationApplyResult:
        """Apply pending items, never resending mutations already confirmed."""
        if self._writer is None:
            raise RuntimeError("a reconciliation writer is required to apply a plan")
        results = _initial_results(plan, previous)
        result_by_id = {result.mutation_id: result for result in results}
        pending = [
            mutation
            for mutation in plan.mutations
            if result_by_id[mutation.mutation_id].state is MutationState.PENDING
        ]
        unexpected_transaction_ids: set[str] = set(
            previous.unexpected_transaction_ids if previous else ()
        )
        failures: list[BatchFailure] = list(previous.failures if previous else ())

        for batch in _chunked(pending, MAX_RECONCILIATION_BATCH_SIZE):
            transaction_ids = [mutation.transaction_id for mutation in batch]
            try:
                response = await self._writer.reconcile_transactions(
                    plan_id=plan.request.plan_id,
                    transaction_ids=transaction_ids,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                mutation_ids = tuple(mutation.mutation_id for mutation in batch)
                failures.append(BatchFailure(mutation_ids=mutation_ids, error=message))
                for mutation in batch:
                    current = result_by_id[mutation.mutation_id]
                    result_by_id[mutation.mutation_id] = replace(
                        current,
                        state=MutationState.UNKNOWN,
                        attempts=current.attempts + 1,
                        error=message,
                    )
                break

            requested_ids = set(transaction_ids)
            confirmed_ids = set(response.transaction_ids)
            unexpected_transaction_ids.update(confirmed_ids - requested_ids)
            for mutation in batch:
                current = result_by_id[mutation.mutation_id]
                succeeded = mutation.transaction_id in confirmed_ids
                result_by_id[mutation.mutation_id] = replace(
                    current,
                    state=(
                        MutationState.SUCCEEDED
                        if succeeded
                        else MutationState.PENDING
                    ),
                    attempts=current.attempts + 1,
                    error=None,
                )

        return ReconciliationApplyResult(
            plan_identity=plan.identity,
            mutations=tuple(
                result_by_id[mutation.mutation_id]
                for mutation in plan.mutations
            ),
            unexpected_transaction_ids=tuple(sorted(unexpected_transaction_ids)),
            failures=tuple(failures),
        )


def classify_account(
    account: Row,
    transactions: list[Row],
) -> AccountReconciliation:
    """Classify one account from an already-fetched transaction collection."""
    account_id = str(account["id"])
    balance = _as_milliunits(account.get("balance"))
    cleared_balance = _as_milliunits(account.get("cleared_balance"))
    uncleared_balance = _as_milliunits(account.get("uncleared_balance"))
    uncleared_delta = balance - cleared_balance
    reasons: list[str] = []
    status = AccountStatus.READY

    if bool(account.get("closed")) or bool(account.get("deleted")):
        status = AccountStatus.SKIPPED
        reasons.append("account is closed or deleted")
    if bool(account.get("direct_import_in_error")):
        status = AccountStatus.BLOCKED
        reasons.append("direct import is in error")
    if status is AccountStatus.READY and account.get("direct_import_linked") is False:
        status = AccountStatus.MANUAL_REVIEW
        reasons.append("not linked for direct import")
    if uncleared_balance != 0 or uncleared_delta != 0:
        reasons.append(f"uncleared activity {uncleared_delta / 1000:,.2f}")

    eligible_transaction_ids: list[str] = []
    uncleared_transaction_count = 0
    reconciled_transaction_count = 0
    for transaction in transactions:
        if bool(transaction.get("deleted")):
            continue
        cleared = _status_value(transaction.get("cleared"))
        if cleared == "cleared":
            transaction_id = transaction.get("id")
            if transaction_id is None:
                raise ValueError("eligible transaction is missing an id")
            eligible_transaction_ids.append(str(transaction_id))
        elif cleared == "uncleared":
            uncleared_transaction_count += 1
        elif cleared == "reconciled":
            reconciled_transaction_count += 1

    return AccountReconciliation(
        account_id=account_id,
        name=_optional_string(account.get("name")),
        account_type=_optional_string(account.get("type")),
        status=status,
        balance=balance,
        cleared_balance=cleared_balance,
        uncleared_balance=uncleared_balance,
        uncleared_delta=uncleared_delta,
        eligible_transaction_ids=tuple(sorted(eligible_transaction_ids)),
        uncleared_transaction_count=uncleared_transaction_count,
        reconciled_transaction_count=reconciled_transaction_count,
        last_reconciled_at=_optional_string(account.get("last_reconciled_at")),
        reasons=tuple(reasons),
    )


def _initial_results(
    plan: ReconciliationPlan,
    previous: ReconciliationApplyResult | None,
) -> tuple[MutationResult, ...]:
    if previous is None:
        return tuple(
            MutationResult(
                mutation_id=mutation.mutation_id,
                account_id=mutation.account_id,
                transaction_id=mutation.transaction_id,
                state=MutationState.PENDING,
            )
            for mutation in plan.mutations
        )
    if previous.plan_identity != plan.identity:
        raise ValueError("previous result belongs to a different reconciliation plan")
    previous_by_id = {result.mutation_id: result for result in previous.mutations}
    if set(previous_by_id) != {mutation.mutation_id for mutation in plan.mutations}:
        raise ValueError("previous result does not contain this plan's mutations")
    return tuple(previous_by_id[mutation.mutation_id] for mutation in plan.mutations)


def _chunked(
    mutations: list[ReconciliationMutation],
    size: int,
) -> list[list[ReconciliationMutation]]:
    return [
        mutations[index : index + size]
        for index in range(0, len(mutations), size)
    ]


def _plan_identity(
    request: ReconciliationRequest,
    accounts: tuple[AccountReconciliation, ...],
    mutations: tuple[ReconciliationMutation, ...],
) -> str:
    payload = {
        "plan_id": request.plan_id,
        "account_ids": sorted(set(request.account_ids)),
        "since": request.since.isoformat() if request.since else None,
        "include_closed": request.include_closed,
        "accounts": [
            {
                "account_id": account.account_id,
                "status": account.status.value,
                "eligible_transaction_ids": account.eligible_transaction_ids,
            }
            for account in accounts
        ],
        "mutation_ids": [mutation.mutation_id for mutation in mutations],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"reconciliation-plan:{sha256(encoded).hexdigest()}"


def _mutation_id(plan_id: str, account_id: str, transaction_id: str) -> str:
    encoded = "\0".join((plan_id, account_id, transaction_id)).encode()
    return f"reconciliation:{sha256(encoded).hexdigest()}"


def _as_milliunits(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float, Decimal, str)):
        return int(value)
    raise ValueError(f"invalid milliunit value: {value!r}")


def _status_value(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value or "")


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)
