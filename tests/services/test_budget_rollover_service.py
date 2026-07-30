from __future__ import annotations

from datetime import date

import pytest

from ynab_agent.services.budget_rollover import (
    BudgetRolloverApplyResult,
    BudgetRolloverRequest,
    BudgetRolloverService,
    CategoryExecutionStatus,
    CategoryUpdate,
    MonthCategory,
    RolloverPlanStatus,
    StepState,
)


class FakeReader:
    def __init__(
        self,
        categories_by_month: dict[date, tuple[MonthCategory, ...]],
    ) -> None:
        self.categories_by_month = categories_by_month
        self.calls: list[tuple[str, date]] = []

    async def get_month_categories(
        self,
        *,
        plan_id: str,
        month: date,
    ) -> tuple[MonthCategory, ...]:
        self.calls.append((plan_id, month))
        return self.categories_by_month.get(month, ())


class FakeWriter:
    def __init__(
        self,
        responses: list[CategoryUpdate | Exception] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, date, str, int]] = []

    async def set_category_budget(
        self,
        *,
        plan_id: str,
        month: date,
        category_id: str,
        budgeted: int,
    ) -> CategoryUpdate:
        self.calls.append((plan_id, month, category_id, budgeted))
        if not self.responses:
            return CategoryUpdate(budgeted=budgeted, balance=budgeted)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def category(
    category_id: str,
    *,
    group_name: str = "Bills",
    name: str | None = None,
    budgeted: int = 0,
    balance: int = 0,
    hidden: bool = False,
    deleted: bool = False,
) -> MonthCategory:
    return MonthCategory(
        category_id=category_id,
        group_name=group_name,
        name=name or category_id,
        budgeted=budgeted,
        balance=balance,
        hidden=hidden,
        deleted=deleted,
    )


def reader() -> FakeReader:
    return FakeReader(
        {
            date(2026, 5, 1): (
                category(
                    "internal",
                    group_name="Internal Master Category",
                    budgeted=100_000,
                    balance=-25_000,
                ),
                category(
                    "groceries",
                    name="Groceries",
                    budgeted=200_000,
                    balance=-150_000,
                ),
                category(
                    "credit-card",
                    group_name="Credit Card Payments",
                    budgeted=75_000,
                    balance=-50_000,
                ),
                category("positive", budgeted=10_000, balance=5_000),
                category("hidden", balance=-10_000, hidden=True),
                category("deleted", balance=-10_000, deleted=True),
            ),
            date(2026, 6, 1): (
                category("groceries", budgeted=25_000),
                category("credit-card", budgeted=10_000),
            ),
        }
    )


def request() -> BudgetRolloverRequest:
    return BudgetRolloverRequest(
        plan_id="budget-1",
        source_month=date(2026, 5, 20),
    )


@pytest.mark.asyncio
async def test_plan_is_deterministic_with_paired_before_after_values() -> None:
    first_reader = reader()
    plan = await BudgetRolloverService(first_reader).plan(request())

    assert first_reader.calls == [
        ("budget-1", date(2026, 5, 1)),
        ("budget-1", date(2026, 6, 1)),
    ]
    assert plan.source_month == date(2026, 5, 1)
    assert plan.target_month == date(2026, 6, 1)
    assert [item.category_id for item in plan.categories] == [
        "groceries",
        "credit-card",
        "internal",
    ]
    groceries = plan.categories[0]
    assert groceries.status is RolloverPlanStatus.PLANNED
    assert groceries.overspent == 150_000
    assert groceries.source_budgeted_before == 200_000
    assert groceries.source_budgeted_after == 350_000
    assert groceries.target_budgeted_before == 25_000
    assert groceries.target_budgeted_after == -125_000
    assert plan.categories[1].status is RolloverPlanStatus.SKIPPED_NON_BUDGETABLE
    assert plan.categories[2].status is RolloverPlanStatus.SKIPPED_NON_BUDGETABLE
    assert len({item.mutation_id for item in plan.categories}) == 3

    reversed_reader = reader()
    reversed_reader.categories_by_month = {
        month: tuple(reversed(categories))
        for month, categories in reversed_reader.categories_by_month.items()
    }
    reordered = await BudgetRolloverService(reversed_reader).plan(request())

    assert reordered == plan


@pytest.mark.asyncio
async def test_apply_updates_only_budgetable_categories_in_paired_order() -> None:
    source = reader()
    writer = FakeWriter()
    service = BudgetRolloverService(source, writer)
    plan = await service.plan(request())

    result = await service.apply(plan)

    assert writer.calls == [
        (
            "budget-1",
            date(2026, 5, 1),
            "groceries",
            350_000,
        ),
        (
            "budget-1",
            date(2026, 6, 1),
            "groceries",
            -125_000,
        ),
    ]
    assert result.is_complete is True
    assert [item.status for item in result.categories] == [
        CategoryExecutionStatus.UPDATED,
        CategoryExecutionStatus.SKIPPED_NON_BUDGETABLE,
        CategoryExecutionStatus.SKIPPED_NON_BUDGETABLE,
    ]
    groceries = result.result_for(plan.categories[0].mutation_id)
    assert groceries.source_update == CategoryUpdate(350_000, 350_000)
    assert groceries.target_update == CategoryUpdate(-125_000, -125_000)


@pytest.mark.asyncio
async def test_interruption_after_source_resumes_with_target_only() -> None:
    source = reader()
    writer = FakeWriter()
    service = BudgetRolloverService(source, writer)
    plan = await service.plan(request())

    interrupted = await service.apply(plan, step_limit=1)
    resumed = await service.apply(plan, previous=interrupted)

    assert writer.calls == [
        ("budget-1", date(2026, 5, 1), "groceries", 350_000),
        ("budget-1", date(2026, 6, 1), "groceries", -125_000),
    ]
    partial = interrupted.result_for(plan.categories[0].mutation_id)
    assert partial.status is CategoryExecutionStatus.SOURCE_APPLIED
    assert partial.source_state is StepState.SUCCEEDED
    assert partial.target_state is StepState.PENDING
    assert resumed.is_complete is True
    completed = resumed.result_for(plan.categories[0].mutation_id)
    assert completed.source_attempts == 1
    assert completed.target_attempts == 1


@pytest.mark.asyncio
async def test_unknown_target_retry_does_not_repeat_confirmed_source() -> None:
    source = reader()
    writer = FakeWriter(
        [
            CategoryUpdate(350_000, 0),
            TimeoutError("target response was not observed"),
            CategoryUpdate(-125_000, 0),
        ]
    )
    service = BudgetRolloverService(source, writer)
    plan = await service.plan(request())

    interrupted = await service.apply(plan)
    resumed = await service.apply(plan, previous=interrupted)

    assert writer.calls == [
        ("budget-1", date(2026, 5, 1), "groceries", 350_000),
        ("budget-1", date(2026, 6, 1), "groceries", -125_000),
        ("budget-1", date(2026, 6, 1), "groceries", -125_000),
    ]
    partial = interrupted.result_for(plan.categories[0].mutation_id)
    assert partial.source_state is StepState.SUCCEEDED
    assert partial.target_state is StepState.UNKNOWN
    assert partial.error == "TimeoutError: target response was not observed"
    assert resumed.is_complete is True
    completed = resumed.result_for(plan.categories[0].mutation_id)
    assert completed.source_attempts == 1
    assert completed.target_attempts == 2
    assert completed.error is None


@pytest.mark.asyncio
async def test_unknown_source_retry_repeats_same_absolute_value_then_continues() -> None:
    source = reader()
    writer = FakeWriter(
        [
            TimeoutError("source response was not observed"),
            CategoryUpdate(350_000, 0),
            CategoryUpdate(-125_000, 0),
        ]
    )
    service = BudgetRolloverService(source, writer)
    plan = await service.plan(request())

    interrupted = await service.apply(plan)
    resumed = await service.apply(plan, previous=interrupted)

    assert writer.calls == [
        ("budget-1", date(2026, 5, 1), "groceries", 350_000),
        ("budget-1", date(2026, 5, 1), "groceries", 350_000),
        ("budget-1", date(2026, 6, 1), "groceries", -125_000),
    ]
    assert interrupted.retry_mutation_ids == (plan.categories[0].mutation_id,)
    assert resumed.is_complete is True
    completed = resumed.result_for(plan.categories[0].mutation_id)
    assert completed.source_attempts == 2
    assert completed.target_attempts == 1


@pytest.mark.asyncio
async def test_unknown_retry_refuses_to_overwrite_an_intervening_manual_edit() -> None:
    source = reader()
    writer = FakeWriter([TimeoutError("source response was not observed")])
    service = BudgetRolloverService(source, writer)
    plan = await service.plan(request())

    interrupted = await service.apply(plan)
    source.categories_by_month[date(2026, 5, 1)] = (
        category(
            "groceries",
            budgeted=300_000,
            balance=-150_000,
        ),
    )
    resumed = await service.apply(plan, previous=interrupted)

    assert writer.calls == [
        ("budget-1", date(2026, 5, 1), "groceries", 350_000),
    ]
    result = resumed.result_for(plan.categories[0].mutation_id)
    assert result.status is CategoryExecutionStatus.CONFLICT
    assert result.error == (
        "source category changed after an unobserved response: "
        "expected prior 200000 or applied 350000, found 300000"
    )


@pytest.mark.asyncio
async def test_conflicting_api_value_stops_before_target_and_requires_replan() -> None:
    source = reader()
    writer = FakeWriter([CategoryUpdate(349_000, 0)])
    service = BudgetRolloverService(source, writer)
    plan = await service.plan(request())

    conflicted = await service.apply(plan)
    repeated = await service.apply(plan, previous=conflicted)

    assert writer.calls == [
        ("budget-1", date(2026, 5, 1), "groceries", 350_000),
    ]
    result = conflicted.result_for(plan.categories[0].mutation_id)
    assert result.status is CategoryExecutionStatus.CONFLICT
    assert result.source_state is StepState.CONFLICT
    assert result.target_state is StepState.PENDING
    assert result.error == (
        "source update returned budgeted=349000; expected 350000"
    )
    assert repeated == conflicted
    assert conflicted.is_complete is False
    assert conflicted.retry_mutation_ids == ()


@pytest.mark.asyncio
async def test_previous_result_must_match_plan_and_step_limit_is_positive() -> None:
    source = reader()
    writer = FakeWriter()
    service = BudgetRolloverService(source, writer)
    plan = await service.plan(request())

    with pytest.raises(ValueError, match="step_limit must be at least 1"):
        await service.apply(plan, step_limit=0)

    with pytest.raises(ValueError, match="different rollover plan"):
        await service.apply(
            plan,
            previous=BudgetRolloverApplyResult(
                plan_identity="another-plan",
                categories=(),
            ),
        )

    assert writer.calls == []
