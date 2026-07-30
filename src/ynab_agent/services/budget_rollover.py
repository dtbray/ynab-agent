"""Deterministic planning and resumable application of budget rollover changes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from hashlib import sha256
import json
from typing import Protocol


NON_BUDGETABLE_CATEGORY_GROUPS = frozenset(
    {
        "Credit Card Payments",
        "Internal Master Category",
    }
)


class BudgetRolloverReader(Protocol):
    """Read operations required to build a rollover plan."""

    async def get_month_categories(
        self,
        *,
        plan_id: str,
        month: date,
    ) -> tuple[MonthCategory, ...]: ...


class BudgetRolloverWriter(Protocol):
    """Absolute category-budget mutation required to apply a rollover plan."""

    async def set_category_budget(
        self,
        *,
        plan_id: str,
        month: date,
        category_id: str,
        budgeted: int,
    ) -> CategoryUpdate: ...


@dataclass(frozen=True)
class MonthCategory:
    """One category's state in a budget month."""

    category_id: str
    group_name: str
    name: str
    budgeted: int
    balance: int
    hidden: bool = False
    deleted: bool = False


@dataclass(frozen=True)
class CategoryUpdate:
    """State returned after an absolute category-budget update."""

    budgeted: int
    balance: int | None = None


class RolloverPlanStatus(StrEnum):
    """Whether a category is eligible for paired rollover writes."""

    PLANNED = "planned"
    SKIPPED_NON_BUDGETABLE = "skipped_non_budgetable"


class StepState(StrEnum):
    """Progress of one absolute source or target write."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    SKIPPED = "skipped"


class CategoryExecutionStatus(StrEnum):
    """Derived execution state for one category pair."""

    PENDING = "pending"
    SOURCE_APPLIED = "source_applied"
    UPDATED = "updated"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    SKIPPED_NON_BUDGETABLE = "skipped_non_budgetable"


@dataclass(frozen=True)
class BudgetRolloverRequest:
    """Scope used to build a dry-run rollover plan."""

    plan_id: str
    source_month: date
    target_month: date | None = None


@dataclass(frozen=True)
class CategoryRollover:
    """Paired before/after values for one overspent category."""

    mutation_id: str
    category_id: str
    group_name: str
    category_name: str
    status: RolloverPlanStatus
    overspent: int
    source_budgeted_before: int
    source_budgeted_after: int
    target_budgeted_before: int
    target_budgeted_after: int


@dataclass(frozen=True)
class BudgetRolloverPlan:
    """Deterministic dry-run result that can later be applied."""

    identity: str
    plan_id: str
    source_month: date
    target_month: date
    categories: tuple[CategoryRollover, ...]

    @property
    def planned_categories(self) -> tuple[CategoryRollover, ...]:
        """Return categories eligible for paired writes."""
        return tuple(
            category
            for category in self.categories
            if category.status is RolloverPlanStatus.PLANNED
        )


@dataclass(frozen=True)
class CategoryExecutionResult:
    """Per-category progress retained across interrupted retries."""

    mutation_id: str
    category_id: str
    source_state: StepState
    target_state: StepState
    source_attempts: int = 0
    target_attempts: int = 0
    source_update: CategoryUpdate | None = None
    target_update: CategoryUpdate | None = None
    error: str | None = None

    @property
    def status(self) -> CategoryExecutionStatus:
        """Return the category-level status derived from its paired steps."""
        if self.source_state is StepState.SKIPPED:
            return CategoryExecutionStatus.SKIPPED_NON_BUDGETABLE
        if (
            self.source_state is StepState.CONFLICT
            or self.target_state is StepState.CONFLICT
        ):
            return CategoryExecutionStatus.CONFLICT
        if (
            self.source_state is StepState.UNKNOWN
            or self.target_state is StepState.UNKNOWN
        ):
            return CategoryExecutionStatus.UNKNOWN
        if (
            self.source_state is StepState.SUCCEEDED
            and self.target_state is StepState.SUCCEEDED
        ):
            return CategoryExecutionStatus.UPDATED
        if self.source_state is StepState.SUCCEEDED:
            return CategoryExecutionStatus.SOURCE_APPLIED
        return CategoryExecutionStatus.PENDING


@dataclass(frozen=True)
class BudgetRolloverApplyResult:
    """Complete per-category progress after one apply or retry attempt."""

    plan_identity: str
    categories: tuple[CategoryExecutionResult, ...]

    @property
    def is_complete(self) -> bool:
        """Return whether every category is updated or intentionally skipped."""
        return all(
            result.status
            in {
                CategoryExecutionStatus.UPDATED,
                CategoryExecutionStatus.SKIPPED_NON_BUDGETABLE,
            }
            for result in self.categories
        )

    @property
    def retry_mutation_ids(self) -> tuple[str, ...]:
        """Return categories with pending or safely repeatable unknown steps."""
        return tuple(
            result.mutation_id
            for result in self.categories
            if result.status
            in {
                CategoryExecutionStatus.PENDING,
                CategoryExecutionStatus.SOURCE_APPLIED,
                CategoryExecutionStatus.UNKNOWN,
            }
        )

    def result_for(self, mutation_id: str) -> CategoryExecutionResult:
        """Return one stable per-category result."""
        for result in self.categories:
            if result.mutation_id == mutation_id:
                return result
        raise KeyError(mutation_id)


class BudgetRolloverService:
    """Build dry-run plans and apply their absolute paired writes safely."""

    def __init__(
        self,
        reader: BudgetRolloverReader,
        writer: BudgetRolloverWriter | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer

    async def plan(self, request: BudgetRolloverRequest) -> BudgetRolloverPlan:
        """Read source and target months once and build a deterministic plan."""
        source_month = _first_of_month(request.source_month)
        target_month = _first_of_month(
            request.target_month or _next_month(source_month)
        )
        source_categories = await self._reader.get_month_categories(
            plan_id=request.plan_id,
            month=source_month,
        )
        target_categories = await self._reader.get_month_categories(
            plan_id=request.plan_id,
            month=target_month,
        )
        target_budgeted = {
            category.category_id: category.budgeted
            for category in target_categories
        }

        categories: list[CategoryRollover] = []
        seen_category_ids: set[str] = set()
        for source in source_categories:
            if (
                source.balance >= 0
                or source.hidden
                or source.deleted
                or source.category_id in seen_category_ids
            ):
                continue
            seen_category_ids.add(source.category_id)
            target_before = target_budgeted.get(source.category_id, 0)
            status = (
                RolloverPlanStatus.SKIPPED_NON_BUDGETABLE
                if source.group_name in NON_BUDGETABLE_CATEGORY_GROUPS
                else RolloverPlanStatus.PLANNED
            )
            categories.append(
                CategoryRollover(
                    mutation_id=_mutation_id(
                        request.plan_id,
                        source_month,
                        target_month,
                        source.category_id,
                    ),
                    category_id=source.category_id,
                    group_name=source.group_name,
                    category_name=source.name,
                    status=status,
                    overspent=-source.balance,
                    source_budgeted_before=source.budgeted,
                    source_budgeted_after=source.budgeted - source.balance,
                    target_budgeted_before=target_before,
                    target_budgeted_after=target_before + source.balance,
                )
            )

        categories.sort(
            key=lambda category: (
                -category.overspent,
                category.category_id,
            )
        )
        planned_categories = tuple(categories)
        return BudgetRolloverPlan(
            identity=_plan_identity(
                request.plan_id,
                source_month,
                target_month,
                planned_categories,
            ),
            plan_id=request.plan_id,
            source_month=source_month,
            target_month=target_month,
            categories=planned_categories,
        )

    async def apply(
        self,
        plan: BudgetRolloverPlan,
        *,
        previous: BudgetRolloverApplyResult | None = None,
        step_limit: int | None = None,
    ) -> BudgetRolloverApplyResult:
        """Apply paired writes, resuming at the first unfinished absolute step."""
        if self._writer is None:
            raise RuntimeError("a budget rollover writer is required to apply a plan")
        if step_limit is not None and step_limit < 1:
            raise ValueError("step_limit must be at least 1")

        results = _initial_results(plan, previous)
        if any(
            StepState.UNKNOWN in {result.source_state, result.target_state}
            for result in results
        ):
            results = await self._resolve_unknown_steps(plan, results)
        result_by_id = {result.mutation_id: result for result in results}
        completed_steps = 0

        for category in plan.categories:
            current = result_by_id[category.mutation_id]
            if category.status is RolloverPlanStatus.SKIPPED_NON_BUDGETABLE:
                continue
            if (
                current.source_state is StepState.CONFLICT
                or current.target_state is StepState.CONFLICT
            ):
                continue

            if current.source_state is not StepState.SUCCEEDED:
                try:
                    source_update = await self._writer.set_category_budget(
                        plan_id=plan.plan_id,
                        month=plan.source_month,
                        category_id=category.category_id,
                        budgeted=category.source_budgeted_after,
                    )
                except Exception as exc:
                    result_by_id[category.mutation_id] = replace(
                        current,
                        source_state=StepState.UNKNOWN,
                        source_attempts=current.source_attempts + 1,
                        error=_error_message(exc),
                    )
                    break
                current = replace(
                    current,
                    source_state=(
                        StepState.SUCCEEDED
                        if source_update.budgeted == category.source_budgeted_after
                        else StepState.CONFLICT
                    ),
                    source_attempts=current.source_attempts + 1,
                    source_update=source_update,
                    error=(
                        None
                        if source_update.budgeted == category.source_budgeted_after
                        else _conflict_message(
                            "source",
                            category.source_budgeted_after,
                            source_update.budgeted,
                        )
                    ),
                )
                result_by_id[category.mutation_id] = current
                completed_steps += 1
                if (
                    current.source_state is StepState.CONFLICT
                    or _limit_reached(step_limit, completed_steps)
                ):
                    break

            if current.target_state is not StepState.SUCCEEDED:
                try:
                    target_update = await self._writer.set_category_budget(
                        plan_id=plan.plan_id,
                        month=plan.target_month,
                        category_id=category.category_id,
                        budgeted=category.target_budgeted_after,
                    )
                except Exception as exc:
                    result_by_id[category.mutation_id] = replace(
                        current,
                        target_state=StepState.UNKNOWN,
                        target_attempts=current.target_attempts + 1,
                        error=_error_message(exc),
                    )
                    break
                current = replace(
                    current,
                    target_state=(
                        StepState.SUCCEEDED
                        if target_update.budgeted == category.target_budgeted_after
                        else StepState.CONFLICT
                    ),
                    target_attempts=current.target_attempts + 1,
                    target_update=target_update,
                    error=(
                        None
                        if target_update.budgeted == category.target_budgeted_after
                        else _conflict_message(
                            "target",
                            category.target_budgeted_after,
                            target_update.budgeted,
                        )
                    ),
                )
                result_by_id[category.mutation_id] = current
                completed_steps += 1
                if (
                    current.target_state is StepState.CONFLICT
                    or _limit_reached(step_limit, completed_steps)
                ):
                    break

        return BudgetRolloverApplyResult(
            plan_identity=plan.identity,
            categories=tuple(
                result_by_id[category.mutation_id]
                for category in plan.categories
            ),
        )

    async def _resolve_unknown_steps(
        self,
        plan: BudgetRolloverPlan,
        results: tuple[CategoryExecutionResult, ...],
    ) -> tuple[CategoryExecutionResult, ...]:
        """Refresh unknown writes before deciding whether repeating them is safe."""
        source_rows = await self._reader.get_month_categories(
            plan_id=plan.plan_id,
            month=plan.source_month,
        )
        target_rows = await self._reader.get_month_categories(
            plan_id=plan.plan_id,
            month=plan.target_month,
        )
        source_by_id = {row.category_id: row for row in source_rows}
        target_by_id = {row.category_id: row for row in target_rows}
        category_by_id = {
            category.mutation_id: category for category in plan.categories
        }
        refreshed: list[CategoryExecutionResult] = []
        for result in results:
            category = category_by_id[result.mutation_id]
            current = result
            if current.source_state is StepState.UNKNOWN:
                current = _resolve_unknown_step(
                    current,
                    step="source",
                    row=source_by_id.get(category.category_id),
                    before=category.source_budgeted_before,
                    after=category.source_budgeted_after,
                )
            if current.target_state is StepState.UNKNOWN:
                current = _resolve_unknown_step(
                    current,
                    step="target",
                    row=target_by_id.get(category.category_id),
                    before=category.target_budgeted_before,
                    after=category.target_budgeted_after,
                )
            refreshed.append(current)
        return tuple(refreshed)


def _initial_results(
    plan: BudgetRolloverPlan,
    previous: BudgetRolloverApplyResult | None,
) -> tuple[CategoryExecutionResult, ...]:
    if previous is None:
        return tuple(
            CategoryExecutionResult(
                mutation_id=category.mutation_id,
                category_id=category.category_id,
                source_state=(
                    StepState.SKIPPED
                    if category.status
                    is RolloverPlanStatus.SKIPPED_NON_BUDGETABLE
                    else StepState.PENDING
                ),
                target_state=(
                    StepState.SKIPPED
                    if category.status
                    is RolloverPlanStatus.SKIPPED_NON_BUDGETABLE
                    else StepState.PENDING
                ),
            )
            for category in plan.categories
        )
    if previous.plan_identity != plan.identity:
        raise ValueError("previous result belongs to a different rollover plan")
    previous_by_id = {
        result.mutation_id: result
        for result in previous.categories
    }
    if set(previous_by_id) != {
        category.mutation_id
        for category in plan.categories
    }:
        raise ValueError("previous result does not contain this plan's categories")
    return tuple(
        previous_by_id[category.mutation_id]
        for category in plan.categories
    )


def _plan_identity(
    plan_id: str,
    source_month: date,
    target_month: date,
    categories: tuple[CategoryRollover, ...],
) -> str:
    payload = {
        "plan_id": plan_id,
        "source_month": source_month.isoformat(),
        "target_month": target_month.isoformat(),
        "categories": [
            {
                "mutation_id": category.mutation_id,
                "status": category.status.value,
                "source_budgeted_before": category.source_budgeted_before,
                "source_budgeted_after": category.source_budgeted_after,
                "target_budgeted_before": category.target_budgeted_before,
                "target_budgeted_after": category.target_budgeted_after,
            }
            for category in categories
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"budget-rollover-plan:{sha256(encoded).hexdigest()}"


def _mutation_id(
    plan_id: str,
    source_month: date,
    target_month: date,
    category_id: str,
) -> str:
    encoded = "\0".join(
        (
            plan_id,
            source_month.isoformat(),
            target_month.isoformat(),
            category_id,
        )
    ).encode()
    return f"budget-rollover:{sha256(encoded).hexdigest()}"


def _first_of_month(value: date) -> date:
    return value.replace(day=1)


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _limit_reached(limit: int | None, completed_steps: int) -> bool:
    return limit is not None and completed_steps >= limit


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _conflict_message(step: str, expected: int, actual: int) -> str:
    return f"{step} update returned budgeted={actual}; expected {expected}"


def _resolve_unknown_step(
    result: CategoryExecutionResult,
    *,
    step: str,
    row: MonthCategory | None,
    before: int,
    after: int,
) -> CategoryExecutionResult:
    if row is None:
        state = StepState.CONFLICT
        update = None
        error = f"{step} category disappeared while resolving an unknown write"
    elif row.budgeted == after:
        state = StepState.SUCCEEDED
        update = CategoryUpdate(budgeted=row.budgeted, balance=row.balance)
        error = None
    elif row.budgeted == before:
        state = StepState.PENDING
        update = None
        error = None
    else:
        state = StepState.CONFLICT
        update = CategoryUpdate(budgeted=row.budgeted, balance=row.balance)
        error = (
            f"{step} category changed after an unobserved response: "
            f"expected prior {before} or applied {after}, found {row.budgeted}"
        )

    if step == "source":
        return replace(
            result,
            source_state=state,
            source_update=update,
            error=error,
        )
    return replace(
        result,
        target_state=state,
        target_update=update,
        error=error,
    )
