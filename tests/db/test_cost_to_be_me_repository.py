from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text

from ynab_agent.db.cost_to_be_me import SqlCostToBeMeRepository
from ynab_agent.db.manager import DatabaseManager


@pytest.mark.asyncio
async def test_cost_to_be_me_queries_are_plan_date_and_split_scoped(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(
        f"sqlite+aiosqlite:///{tmp_path / 'cost-to-be-me.db'}"
    )
    await database.initialize()
    try:
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO accounts (
                      id, budget_id, name, type, on_budget, closed, balance,
                      cleared_balance, uncleared_balance, deleted
                    )
                    VALUES
                      ('checking', 'plan-1', 'Checking', 'checking', 1, 0, 0, 0, 0, 0),
                      ('tracking', 'plan-1', 'Tracking', 'otherAsset', 0, 0, 0, 0, 0, 0),
                      ('other-checking', 'plan-2', 'Checking', 'checking', 1, 0,
                       0, 0, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO budget_months (
                      id, budget_id, month, income, deleted
                    )
                    VALUES
                      ('plan-1-2026-03', 'plan-1', '2026-03-01', 3000000, 0),
                      ('plan-1-2026-04', 'plan-1', '2026-04-01', 5000000, 0),
                      ('plan-1-deleted', 'plan-1', '2026-02-01', 7000000, 1),
                      ('plan-2-2026-04', 'plan-2', '2026-04-01', 9000000, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO category_groups (
                      id, budget_id, name, hidden, deleted
                    )
                    VALUES
                      ('bills', 'plan-1', 'Bills', 0, 0),
                      ('cards', 'plan-1', 'Credit Card Payments', 0, 0),
                      ('internal', 'plan-1', 'Internal Master Category', 0, 0),
                      ('hidden-group', 'plan-1', 'Hidden Group', 1, 0),
                      ('other-bills', 'plan-2', 'Bills', 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO categories (
                      id, budget_id, category_group_id, name, hidden, internal,
                      deleted
                    )
                    VALUES
                      ('food', 'plan-1', 'bills', 'Food', 0, 0, 0),
                      ('card', 'plan-1', 'cards', 'Visa', 0, 0, 0),
                      ('internal-cat', 'plan-1', 'internal', 'Ready', 0, 1, 0),
                      ('hidden-cat', 'plan-1', 'bills', 'Hidden', 1, 0, 0),
                      ('hidden-group-cat', 'plan-1', 'hidden-group', 'Secret', 0, 0, 0),
                      ('deleted-cat', 'plan-1', 'bills', 'Deleted', 0, 0, 1),
                      ('other-food', 'plan-2', 'other-bills', 'Food', 0, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO month_categories (
                      id, budget_id, month, category_id, hidden, internal,
                      budgeted, goal_under_funded, goal_target, deleted
                    )
                    VALUES
                      ('food-month', 'plan-1', '2026-05-01', 'food', 0, 0,
                       1200000, 50000, 25800000, 0),
                      ('card-month', 'plan-1', '2026-05-01', 'card', 0, 0,
                       100000, 0, 0, 0),
                      ('internal-month', 'plan-1', '2026-05-01', 'internal-cat', 0, 1,
                       100000, 0, 0, 0),
                      ('hidden-month', 'plan-1', '2026-05-01', 'hidden-cat', 0, 0,
                       100000, 0, 0, 0),
                      ('hidden-group-month', 'plan-1', '2026-05-01',
                       'hidden-group-cat', 0, 0, 100000, 0, 0, 0),
                      ('deleted-category-month', 'plan-1', '2026-05-01',
                       'deleted-cat', 0, 0, 100000, 0, 0, 0),
                      ('deleted-month', 'plan-1', '2026-05-01', 'food', 0, 0,
                       100000, 0, 0, 1),
                      ('other-month', 'plan-2', '2026-05-01', 'other-food', 0, 0,
                       9000000, 0, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO transactions (
                      id, budget_id, account_id, category_id, date, amount,
                      transfer_account_id, approved, deleted
                    )
                    VALUES
                      ('march-spend', 'plan-1', 'checking', 'food', '2026-03-01',
                       -500000, NULL, 1, 0),
                      ('april-spend', 'plan-1', 'checking', 'food', '2026-04-10',
                       -300000, NULL, 1, 0),
                      ('april-refund', 'plan-1', 'checking', 'food', '2026-04-11',
                       100000, NULL, 1, 0),
                      ('split-parent', 'plan-1', 'checking', 'food', '2026-04-12',
                       -1000000, NULL, 1, 0),
                      ('before-window', 'plan-1', 'checking', 'food', '2026-02-28',
                       -800000, NULL, 1, 0),
                      ('end-boundary', 'plan-1', 'checking', 'food', '2026-05-01',
                       -800000, NULL, 1, 0),
                      ('card-spend', 'plan-1', 'checking', 'card', '2026-04-15',
                       -900000, NULL, 1, 0),
                      ('internal-spend', 'plan-1', 'checking', 'internal-cat',
                       '2026-04-15', -900000, NULL, 1, 0),
                      ('transfer', 'plan-1', 'checking', 'food', '2026-04-16',
                       -900000, 'tracking', 1, 0),
                      ('tracking-spend', 'plan-1', 'tracking', 'food', '2026-04-17',
                       -900000, NULL, 1, 0),
                      ('deleted-spend', 'plan-1', 'checking', 'food', '2026-04-18',
                       -900000, NULL, 1, 1),
                      ('other-spend', 'plan-2', 'other-checking', 'other-food',
                       '2026-04-19', -900000, NULL, 1, 0),
                      ('income-mar', 'plan-1', 'checking', NULL, '2026-03-15',
                       3000000, NULL, 1, 0),
                      ('income-apr', 'plan-1', 'checking', NULL, '2026-04-15',
                       5000000, NULL, 1, 0),
                      ('income-end', 'plan-1', 'checking', NULL, '2026-05-01',
                       9000000, NULL, 1, 0),
                      ('categorized-income', 'plan-1', 'checking', 'card',
                       '2026-04-20', 9000000, NULL, 1, 0),
                      ('transfer-income', 'plan-1', 'checking', NULL, '2026-04-20',
                       9000000, 'tracking', 1, 0),
                      ('tracking-income', 'plan-1', 'tracking', NULL, '2026-04-20',
                       9000000, NULL, 1, 0),
                      ('deleted-income', 'plan-1', 'checking', NULL, '2026-04-20',
                       9000000, NULL, 1, 1),
                      ('other-income', 'plan-2', 'other-checking', NULL, '2026-04-20',
                       9000000, NULL, 1, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO subtransactions (
                      id, budget_id, transaction_id, amount, category_id,
                      transfer_account_id, deleted
                    )
                    VALUES
                      ('split-a', 'plan-1', 'split-parent', -60000, 'food', NULL, 0),
                      ('split-b', 'plan-1', 'split-parent', -40000, 'food', NULL, 0)
                    """
                )
            )
            await session.commit()

        repository = SqlCostToBeMeRepository(database)
        funding = await repository.list_funding_categories(
            month=date(2026, 5, 1),
            plan_id="plan-1",
        )
        spending = await repository.list_monthly_spending(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 5, 1),
            plan_id="plan-1",
        )
        income = await repository.list_monthly_ready_to_assign_income(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 5, 1),
            plan_id="plan-1",
        )
        boundary_spending = await repository.list_monthly_spending(
            start_date=date(2025, 12, 1),
            end_date=date(2026, 3, 1),
            plan_id="plan-1",
        )
        boundary_income = await repository.list_monthly_ready_to_assign_income(
            start_date=date(2025, 12, 1),
            end_date=date(2026, 3, 1),
            plan_id="plan-1",
        )

        assert len(funding) == 1
        assert funding[0].category_name == "Food"
        assert funding[0].assigned_milliunits == 1_200_000
        assert funding[0].still_underfunded_milliunits == 50_000
        assert funding[0].goal_period_target_milliunits == 25_800_000
        assert [(row.month, row.spending_milliunits) for row in spending] == [
            ("2026-01", 0),
            ("2026-02", 800_000),
            ("2026-03", 500_000),
            ("2026-04", 300_000),
        ]
        assert [(row.month, row.income_milliunits) for row in income] == [
            ("2026-01", 0),
            ("2026-02", 0),
            ("2026-03", 3_000_000),
            ("2026-04", 5_000_000),
        ]
        assert [
            (row.month, row.spending_milliunits)
            for row in boundary_spending
        ] == [
            ("2025-12", 0),
            ("2026-01", 0),
            ("2026-02", 800_000),
        ]
        assert [
            (row.month, row.income_milliunits)
            for row in boundary_income
        ] == [
            ("2025-12", 0),
            ("2026-01", 0),
            ("2026-02", 0),
        ]
    finally:
        await database.close()
