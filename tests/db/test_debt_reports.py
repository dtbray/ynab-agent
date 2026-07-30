from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text

from ynab_agent.db.debt_reports import SqlDebtReportRepository
from ynab_agent.db.manager import DatabaseManager


@pytest.mark.asyncio
async def test_account_queries_preserve_report_specific_types_and_plan_scope(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'debt-reports.db'}")
    await database.initialize()
    try:
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO category_groups (
                      id, budget_id, name, hidden, deleted
                    )
                    VALUES
                      ('payments', 'plan-1', 'Credit Card Payments', 0, 0),
                      ('other-payments', 'plan-2', 'Credit Card Payments', 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO categories (
                      id, budget_id, category_group_id, name, hidden, internal,
                      balance, deleted
                    )
                    VALUES
                      ('visa-payment', 'plan-1', 'payments', 'Visa', 0, 0, 100000, 0),
                      ('hidden-payment', 'plan-1', 'payments', 'Hidden Card', 1, 0,
                       500000, 0),
                      ('other-visa', 'plan-2', 'other-payments', 'Visa', 0, 0,
                       900000, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO accounts (
                      id, budget_id, name, type, on_budget, closed, balance,
                      cleared_balance, uncleared_balance, debt_minimum_payments, deleted
                    )
                    VALUES
                      ('visa', 'plan-1', 'Visa', 'creditCard', 1, 0, -500000,
                       -450000, -50000, '{"amount": 25000}', 0),
                      ('hidden', 'plan-1', 'Hidden Card', 'credit_card', 1, 0,
                       -200000, -200000, 0, NULL, 0),
                      ('mortgage', 'plan-1', 'Mortgage', 'mortgage', 0, 0, 0,
                       0, 0, NULL, 0),
                      ('medical', 'plan-1', 'Medical', 'medicalDebt', 0, 0, 0,
                       0, 0, NULL, 0),
                      ('negative', 'plan-1', 'Personal', 'checking', 1, 0, -100000,
                       -100000, 0, NULL, 0),
                      ('closed', 'plan-1', 'Closed', 'creditCard', 1, 1, -999000,
                       -999000, 0, NULL, 0),
                      ('deleted', 'plan-1', 'Deleted', 'creditCard', 1, 0, -999000,
                       -999000, 0, NULL, 1),
                      ('other-plan', 'plan-2', 'Visa', 'creditCard', 1, 0, -900000,
                       -900000, 0, NULL, 0)
                    """
                )
            )
            await session.commit()

        repository = SqlDebtReportRepository(database)
        cards = await repository.list_credit_cards(plan_id="plan-1")
        drag = await repository.list_debt_drag_accounts(plan_id="plan-1")
        plan = await repository.list_debt_plan_accounts(plan_id="plan-1")

        assert [row.account for row in cards] == ["Visa", "Hidden Card"]
        assert cards[0].payment_available_milliunits == 100_000
        assert cards[1].payment_available_milliunits == 0
        assert cards[0].minimum_payment_data == '{"amount": 25000}'
        assert {row.account for row in drag} == {
            "Visa",
            "Hidden Card",
            "Mortgage",
            "Personal",
        }
        assert {row.account for row in plan} == {
            "Visa",
            "Hidden Card",
            "Medical",
            "Personal",
        }
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_scheduled_query_is_date_state_and_plan_scoped(tmp_path: Path) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'scheduled-debt.db'}")
    await database.initialize()
    try:
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO category_groups (id, budget_id, name, hidden, deleted)
                    VALUES
                      ('debt', 'plan-1', 'Debt Payments', 0, 0),
                      ('bills', 'plan-1', 'Bills', 0, 0),
                      ('other-debt', 'plan-2', 'Debt Payments', 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO categories (
                      id, budget_id, category_group_id, name, hidden, internal, deleted
                    )
                    VALUES
                      ('loan', 'plan-1', 'debt', 'Student Loan', 0, 0, 0),
                      ('utility', 'plan-1', 'bills', 'Electricity', 0, 0, 0),
                      ('other-loan', 'plan-2', 'other-debt', 'Student Loan', 0, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO scheduled_transactions (
                      id, budget_id, category_id, date_next, frequency, amount, deleted
                    )
                    VALUES
                      ('included', 'plan-1', 'loan', '2026-06-01', 'monthly', -50000, 0),
                      ('future', 'plan-1', 'loan', '2026-07-01', 'monthly', -60000, 0),
                      ('positive', 'plan-1', 'loan', '2026-06-01', 'monthly', 10000, 0),
                      ('deleted', 'plan-1', 'loan', '2026-06-01', 'monthly', -70000, 1),
                      ('not-debt', 'plan-1', 'utility', '2026-06-01', 'monthly',
                       -80000, 0),
                      ('other-plan', 'plan-2', 'other-loan', '2026-06-01', 'monthly',
                       -90000, 0)
                    """
                )
            )
            await session.commit()

        rows = await SqlDebtReportRepository(
            database
        ).list_scheduled_debt_obligations(
            through=date(2026, 6, 1),
            plan_id="plan-1",
        )

        assert len(rows) == 1
        assert rows[0].category == "Student Loan"
        assert rows[0].amount_milliunits == -50_000
        assert rows[0].frequency == "monthly"
    finally:
        await database.close()
