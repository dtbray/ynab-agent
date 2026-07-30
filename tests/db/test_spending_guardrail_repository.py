from __future__ import annotations

from datetime import date

from sqlalchemy import text
import pytest

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.spending_guardrails import SqlSpendingTierRepository
from ynab_agent.planning.spending_guardrails import SpendingTier
from ynab_agent.services.spending_guardrails import SpendingTierAssignment


@pytest.mark.asyncio
async def test_mappings_survive_rename_and_spending_is_split_aware(
    tmp_path,
) -> None:
    database = DatabaseManager(
        f"sqlite+aiosqlite:///{tmp_path / 'spending-guardrails.db'}"
    )
    await database.initialize()
    try:
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO category_groups (
                      id, budget_id, name, hidden, deleted
                    )
                    VALUES ('needs', 'budget-1', 'Needs', 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO categories (
                      id, budget_id, category_group_id, name,
                      hidden, internal, deleted
                    )
                    VALUES
                      ('food', 'budget-1', 'needs', 'Groceries', 0, 0, 0),
                      ('fun', 'budget-1', 'needs', 'Fun', 0, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO accounts (
                      id, budget_id, name, type, on_budget, closed,
                      balance, cleared_balance, uncleared_balance, deleted
                    )
                    VALUES
                      ('checking', 'budget-1', 'Checking', 'checking', 1, 0,
                       0, 0, 0, 0),
                      ('tracking', 'budget-1', 'Tracking', 'otherAsset', 0, 0,
                       0, 0, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO transactions (
                      id, budget_id, account_id, category_id,
                      transfer_account_id, date, amount, deleted
                    )
                    VALUES
                      ('food-outflow', 'budget-1', 'checking', 'food',
                       NULL, '2026-07-01', -100000, 0),
                      ('food-refund', 'budget-1', 'checking', 'food',
                       NULL, '2026-07-02', 10000, 0),
                      ('split-parent', 'budget-1', 'checking', NULL,
                       NULL, '2026-07-03', -100000, 0),
                      ('transfer', 'budget-1', 'checking', 'food',
                       'tracking', '2026-07-04', -50000, 0),
                      ('off-budget', 'budget-1', 'tracking', 'food',
                       NULL, '2026-07-05', -50000, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO subtransactions (
                      id, budget_id, transaction_id, category_id,
                      amount, deleted
                    )
                    VALUES
                      ('split-food', 'budget-1', 'split-parent', 'food',
                       -60000, 0),
                      ('split-fun', 'budget-1', 'split-parent', 'fun',
                       -40000, 0)
                    """
                )
            )
            await session.commit()

        repository = SqlSpendingTierRepository(database)
        food = await repository.upsert_mapping(
            SpendingTierAssignment(
                budget_id="budget-1",
                category_id="food",
                tier=SpendingTier.ESSENTIAL,
                essential_floor_milliunits=50_000,
            ),
            updated_at="2026-07-30T00:00:00+00:00",
        )
        fun = await repository.upsert_mapping(
            SpendingTierAssignment(
                budget_id="budget-1",
                category_id="fun",
                tier=SpendingTier.LIFESTYLE,
            ),
            updated_at="2026-07-30T00:00:00+00:00",
        )

        assert food is not None
        assert fun is not None
        observed = await repository.list_spending(
            budget_id="budget-1",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 8, 1),
        )
        assert {
            row.category_id: row.spending_milliunits for row in observed
        } == {"food": 150_000, "fun": 40_000}

        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE categories
                    SET name = 'Food at home'
                    WHERE id = 'food'
                    """
                )
            )
            await session.commit()
        mappings = await repository.list_mappings(budget_id="budget-1")

        renamed = next(
            row for row in mappings if row.category_id == "food"
        )
        assert renamed.category_name == "Food at home"
        assert renamed.tier is SpendingTier.ESSENTIAL
        assert renamed.essential_floor_milliunits == 50_000
    finally:
        await database.close()
