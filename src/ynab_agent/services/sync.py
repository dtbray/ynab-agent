"""Typed orchestration for checkpointed YNAB cache synchronization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal, Protocol, TypeVar
from uuid import uuid4


Row = dict[str, object]
T_co = TypeVar("T_co", covariant=True)


class DeltaResult(Protocol[T_co]):
    """Read-only result returned by a delta-capable gateway endpoint."""

    @property
    def data(self) -> T_co:
        """Return the endpoint payload."""
        ...

    @property
    def server_knowledge(self) -> int | None:
        """Return the checkpoint associated with the payload."""
        ...


class SyncGateway(Protocol):
    """YNAB operations required by the synchronization workflow."""

    async def get_plans(self, *, include_accounts: bool) -> list[Row]: ...

    async def get_accounts(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> DeltaResult[list[Row]]: ...

    async def get_payees(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> DeltaResult[list[Row]]: ...

    async def get_payee_locations(self, *, plan_id: str) -> list[Row]: ...

    async def get_categories(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> DeltaResult[tuple[list[Row], list[Row]]]: ...

    async def get_months(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> DeltaResult[list[Row]]: ...

    async def get_month(self, *, plan_id: str, month: date) -> tuple[Row, list[Row]]: ...

    async def get_transactions(
        self,
        *,
        plan_id: str,
        since_date: date | None,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> DeltaResult[list[Row]]: ...

    async def get_scheduled_transactions(
        self,
        *,
        plan_id: str,
        last_knowledge_of_server: int | None,
        include_server_knowledge: Literal[True],
    ) -> DeltaResult[list[Row]]: ...


class SyncStore(Protocol):
    """Persistence operations required by the synchronization workflow."""

    async def get_server_knowledge(self, plan_id: str, resource: str) -> int | None: ...

    async def save_server_knowledge(
        self,
        plan_id: str,
        resource: str,
        server_knowledge: int | None,
    ) -> None: ...

    async def save_budgets(
        self,
        budgets: list[Row],
        *,
        change_batch_id: str,
    ) -> int: ...

    async def save_accounts(
        self,
        plan_id: str,
        accounts: list[Row],
        *,
        change_batch_id: str,
    ) -> int: ...

    async def save_payees(
        self,
        payees: list[Row],
        *,
        change_batch_id: str,
    ) -> int: ...

    async def save_payee_locations(
        self,
        payee_locations: list[Row],
        *,
        change_batch_id: str,
    ) -> int: ...

    async def save_categories(
        self,
        category_groups: list[Row],
        categories: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]: ...

    async def save_months(
        self,
        months: list[Row],
        *,
        change_batch_id: str,
    ) -> int: ...

    async def save_month(
        self,
        month: Row,
        categories: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]: ...

    async def save_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
        *,
        change_batch_id: str,
    ) -> int: ...

    async def save_scheduled_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]: ...


class ValuationCapture(Protocol):
    """Optional post-account-sync hook for dated wealth valuations."""

    async def capture_current_valuations(
        self,
        *,
        observed_at: datetime,
    ) -> object: ...


@dataclass(frozen=True)
class SyncRequest:
    """Inputs controlling one synchronization run."""

    plan_id: str
    since: date
    month: date
    full: bool = False
    skip_month_detail: bool = False


@dataclass(frozen=True)
class SyncSummary:
    """Counts and resolved identity produced by one synchronization run."""

    change_batch_id: str
    requested_plan_id: str
    resolved_plan_id: str
    used_plan_alias: bool
    plan_count: int
    account_count: int
    payee_count: int
    payee_location_count: int
    category_group_count: int
    category_count: int
    month_count: int
    month_category_count: int
    transaction_count: int
    scheduled_transaction_count: int
    scheduled_subtransaction_count: int


class SyncService:
    """Synchronize YNAB resources in checkpoint-safe dependency order."""

    def __init__(
        self,
        gateway: SyncGateway,
        store: SyncStore,
        *,
        batch_id_factory: Callable[[], str] | None = None,
        valuation_capture: ValuationCapture | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._batch_id_factory = batch_id_factory or (lambda: str(uuid4()))
        self._valuation_capture = valuation_capture
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def run(self, request: SyncRequest) -> SyncSummary:
        """Run one sync, checkpointing each resource only after it is persisted."""
        change_batch_id = self._batch_id_factory()
        uses_plan_alias = request.plan_id in {"last-used", "default"}

        plans = await self._gateway.get_plans(include_accounts=uses_plan_alias)
        plan_count = await self._store.save_budgets(
            plans,
            change_batch_id=change_batch_id,
        )

        account_result = await self._gateway.get_accounts(
            plan_id=request.plan_id,
            last_knowledge_of_server=(
                None
                if uses_plan_alias
                else await self._checkpoint(request, request.plan_id, "accounts")
            ),
            include_server_knowledge=True,
        )
        resolved_plan_id = (
            _resolve_plan_id(request.plan_id, plans, account_result.data)
            if uses_plan_alias
            else request.plan_id
        )
        account_count = await self._store.save_accounts(
            resolved_plan_id,
            account_result.data,
            change_batch_id=change_batch_id,
        )
        await self._store.save_server_knowledge(
            resolved_plan_id,
            "accounts",
            account_result.server_knowledge,
        )
        if self._valuation_capture is not None:
            await self._valuation_capture.capture_current_valuations(
                observed_at=self._clock(),
            )

        payee_result = await self._gateway.get_payees(
            plan_id=resolved_plan_id,
            last_knowledge_of_server=await self._checkpoint(
                request,
                resolved_plan_id,
                "payees",
            ),
            include_server_knowledge=True,
        )
        payee_count = await self._store.save_payees(
            payee_result.data,
            change_batch_id=change_batch_id,
        )
        await self._store.save_server_knowledge(
            resolved_plan_id,
            "payees",
            payee_result.server_knowledge,
        )

        payee_locations = await self._gateway.get_payee_locations(
            plan_id=resolved_plan_id,
        )
        payee_location_count = await self._store.save_payee_locations(
            payee_locations,
            change_batch_id=change_batch_id,
        )

        category_result = await self._gateway.get_categories(
            plan_id=resolved_plan_id,
            last_knowledge_of_server=await self._checkpoint(
                request,
                resolved_plan_id,
                "categories",
            ),
            include_server_knowledge=True,
        )
        category_groups, categories = category_result.data
        category_group_count, category_count = await self._store.save_categories(
            category_groups,
            categories,
            change_batch_id=change_batch_id,
        )
        await self._store.save_server_knowledge(
            resolved_plan_id,
            "categories",
            category_result.server_knowledge,
        )

        month_result = await self._gateway.get_months(
            plan_id=resolved_plan_id,
            last_knowledge_of_server=await self._checkpoint(
                request,
                resolved_plan_id,
                "months",
            ),
            include_server_knowledge=True,
        )
        month_count = await self._store.save_months(
            month_result.data,
            change_batch_id=change_batch_id,
        )
        await self._store.save_server_knowledge(
            resolved_plan_id,
            "months",
            month_result.server_knowledge,
        )

        month_category_count = 0
        if not request.skip_month_detail:
            budget_month, month_categories = await self._gateway.get_month(
                plan_id=resolved_plan_id,
                month=request.month,
            )
            _, month_category_count = await self._store.save_month(
                budget_month,
                month_categories,
                change_batch_id=change_batch_id,
            )

        transaction_result = await self._gateway.get_transactions(
            plan_id=resolved_plan_id,
            since_date=request.since if request.full else None,
            last_knowledge_of_server=await self._checkpoint(
                request,
                resolved_plan_id,
                "transactions",
            ),
            include_server_knowledge=True,
        )
        transaction_count = await self._store.save_transactions(
            resolved_plan_id,
            transaction_result.data,
            change_batch_id=change_batch_id,
        )
        await self._store.save_server_knowledge(
            resolved_plan_id,
            "transactions",
            transaction_result.server_knowledge,
        )

        scheduled_result = await self._gateway.get_scheduled_transactions(
            plan_id=resolved_plan_id,
            last_knowledge_of_server=await self._checkpoint(
                request,
                resolved_plan_id,
                "scheduled_transactions",
            ),
            include_server_knowledge=True,
        )
        scheduled_transaction_count, scheduled_subtransaction_count = (
            await self._store.save_scheduled_transactions(
                resolved_plan_id,
                scheduled_result.data,
                change_batch_id=change_batch_id,
            )
        )
        await self._store.save_server_knowledge(
            resolved_plan_id,
            "scheduled_transactions",
            scheduled_result.server_knowledge,
        )

        return SyncSummary(
            change_batch_id=change_batch_id,
            requested_plan_id=request.plan_id,
            resolved_plan_id=resolved_plan_id,
            used_plan_alias=uses_plan_alias,
            plan_count=plan_count,
            account_count=account_count,
            payee_count=payee_count,
            payee_location_count=payee_location_count,
            category_group_count=category_group_count,
            category_count=category_count,
            month_count=month_count,
            month_category_count=month_category_count,
            transaction_count=transaction_count,
            scheduled_transaction_count=scheduled_transaction_count,
            scheduled_subtransaction_count=scheduled_subtransaction_count,
        )

    async def _checkpoint(
        self,
        request: SyncRequest,
        plan_id: str,
        resource: str,
    ) -> int | None:
        if request.full:
            return None
        return await self._store.get_server_knowledge(plan_id, resource)


def _resolve_plan_id(alias: str, plans: list[Row], accounts: list[Row]) -> str:
    """Resolve a YNAB symbolic alias using account membership in plan summaries."""
    account_ids = {
        str(account_id)
        for account in accounts
        if (account_id := account.get("id")) is not None
    }
    matches = [
        str(plan_id)
        for plan in plans
        if (plan_id := plan.get("id")) is not None
        and account_ids.intersection(
            str(account_id)
            for account_id in _account_ids(plan.get("account_ids"))
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if len(plans) == 1 and plans[0].get("id") is not None:
        return str(plans[0]["id"])
    raise RuntimeError(
        f"Could not resolve YNAB plan alias {alias!r} to exactly one plan UUID; "
        "configure YNAB_PLAN_ID with the desired UUID"
    )


def _account_ids(value: object) -> list[object]:
    return value if isinstance(value, list) else []
