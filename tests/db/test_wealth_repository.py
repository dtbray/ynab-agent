import asyncio
from datetime import date

from sqlalchemy import text

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.wealth import SqlWealthRepository


def test_cash_flow_query_is_account_and_date_scoped_and_retains_evidence(
    tmp_path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'wealth-repository.db'}"

    async def _exercise():
        database = DatabaseManager(database_url)
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
                          ('selected', 'budget-1', 'Selected', 'otherAsset', 0, 0,
                           100000, 100000, 0, 0),
                          ('other', 'budget-1', 'Other', 'otherAsset', 0, 0,
                           200000, 200000, 0, 0)
                        """
                    )
                )
                await session.execute(
                    text(
                        """
                        INSERT INTO transactions (
                          id, budget_id, account_id, transfer_account_id,
                          payee_name, memo, date, amount, cleared, approved, deleted
                        )
                        VALUES
                          ('included', 'budget-1', 'selected', 'checking',
                           'Transfer : Checking', 'payroll contribution',
                           '2026-01-02', 100000, 'cleared', 0, 0),
                          ('future', 'budget-1', 'selected', NULL,
                           'Future', NULL, '2026-02-01', 10000, 'cleared', 1, 0),
                          ('other-account', 'budget-1', 'other', NULL,
                           'Other', NULL, '2026-01-02', 10000, 'cleared', 1, 0),
                          ('deleted', 'budget-1', 'selected', NULL,
                           'Deleted', NULL, '2026-01-02', 10000, 'cleared', 1, 1)
                        """
                    )
                )
                await session.commit()

            return await SqlWealthRepository(database).get_cash_flow_candidates(
                ["selected"],
                through=date(2026, 1, 31),
            )
        finally:
            await database.close()

    rows = asyncio.run(_exercise())

    assert len(rows) == 1
    candidate = rows[0]
    assert candidate.transaction_id == "included"
    assert candidate.account_id == "selected"
    assert candidate.date == date(2026, 1, 2)
    assert candidate.transfer_account_id == "checking"
    assert candidate.payee_name == "Transfer : Checking"
    assert candidate.memo == "payroll contribution"
    assert candidate.approved is False
    assert candidate.cleared == "cleared"
