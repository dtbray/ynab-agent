from __future__ import annotations

from datetime import date, datetime
from typing import cast

import pytest

from ynab_agent.db.budget_activity_reports import (
    SqlBudgetActivityRepository,
)
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.obligation_reports import SqlObligationRepository
from ynab_agent.db.overview_reports import SqlOverviewReportRepository
from ynab_agent.db.spending_reports import SqlSpendingReportRepository
from ynab_agent.services.reports.budget_activity import (
    BudgetActivityRequest,
)
from ynab_agent.services.reports.obligations import ObligationRequest
from ynab_agent.services.reports.spending import (
    BurnRateGrouping,
    DateRangeRequest,
)


class RecordingDatabase:
    def __init__(self, *responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def fetch_all(
        self,
        sql: str,
        params: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append((sql, params or {}))
        return self.responses.pop(0)


def _as_manager(database: RecordingDatabase) -> DatabaseManager:
    return cast(DatabaseManager, cast(object, database))


@pytest.mark.asyncio
async def test_overview_queries_map_rows_and_scope_latest_batch() -> None:
    recorded_at = datetime(2026, 7, 1, 12, 0)
    database = RecordingDatabase(
        [
            {
                "bucket": "on_budget",
                "type": "checking",
                "accounts": 2,
                "balance": 4_000_000,
            }
        ],
        [{"batch_id": "batch-2"}],
        [
            {
                "batch_id": "batch-2",
                "budget_id": "plan-1",
                "resource": "accounts",
                "action": "updated",
                "entity_name": "Checking",
                "entity_id": "account-1",
                "recorded_at": recorded_at,
            }
        ],
        [{"check_name": "unapproved", "count": 3}],
    )
    repository = SqlOverviewReportRepository(_as_manager(database))

    net_worth = await repository.list_net_worth()
    changes = await repository.list_latest_changes("plan-1")
    hygiene = await repository.list_hygiene_checks(since=date(2026, 1, 1))

    assert net_worth[0].balance_milliunits == 4_000_000
    assert changes[0].recorded_at == recorded_at
    assert hygiene[0].count == 3
    assert database.calls[1][1] == {"plan_id": "plan-1"}
    assert database.calls[2][1] == {
        "batch_id": "batch-2",
        "plan_id": "plan-1",
    }


@pytest.mark.asyncio
async def test_budget_queries_map_all_three_category_views() -> None:
    activity = {
        "group_name": "Bills",
        "category": "Rent",
        "budgeted": 1_000_000,
        "activity": -1_000_000,
        "balance": 0,
    }
    database = RecordingDatabase(
        [{**activity, "overspent": 50_000}],
        [activity],
        [activity],
    )
    repository = SqlBudgetActivityRepository(_as_manager(database))
    request = BudgetActivityRequest(month=date(2026, 5, 1), limit=7)

    overspending = await repository.list_overspending(request)
    hidden = await repository.list_hidden_funds(request)
    month = await repository.list_month_activity(request)

    assert overspending[0].overspent_milliunits == 50_000
    assert hidden[0].category == "Rent"
    assert month[0].activity_milliunits == -1_000_000
    assert all(params == {"month": "2026-05-01", "limit": 7} for _, params in database.calls)
    assert all("c.budget_id = mc.budget_id" in sql for sql, _ in database.calls)


@pytest.mark.asyncio
async def test_spending_queries_use_portable_month_expression() -> None:
    database = RecordingDatabase(
        [
            {
                "month": "2026-05",
                "inflow": 100_000,
                "outflow": 50_000,
                "net": 50_000,
                "transactions": 2,
            }
        ],
        [
            {
                "bucket": "Bills",
                "outflow": 50_000,
                "transactions": 1,
                "active_months": 1,
            }
        ],
        [{"payee": "Market", "outflow": 50_000, "transactions": 1}],
    )
    repository = SqlSpendingReportRepository(_as_manager(database))

    await repository.list_cashflow(DateRangeRequest(since=date(2026, 1, 1)))
    await repository.list_burn_rate(
        DateRangeRequest(
            since=date(2026, 1, 1),
            through=date(2026, 5, 31),
            limit=5,
        ),
        grouping=BurnRateGrouping.GROUP,
    )
    await repository.list_top_spend(DateRangeRequest(since=date(2026, 1, 1), limit=5))

    sql = "\n".join(statement for statement, _ in database.calls)
    assert "strftime" not in sql.lower()
    assert "substr(t.date, 1, 7)" in sql
    assert "AND t.date <= :through" in database.calls[1][0]
    assert database.calls[1][1]["through"] == "2026-05-31"


@pytest.mark.asyncio
async def test_obligation_query_controls_inflow_filter_without_user_sql() -> None:
    row = {
        "due_date": "2026-05-25",
        "payee": "Landlord",
        "group_name": "Bills",
        "category": "Rent",
        "account": "Checking",
        "frequency": "monthly",
        "amount": -1_000_000,
        "memo": "June rent",
    }
    database = RecordingDatabase([row], [row])
    repository = SqlObligationRepository(_as_manager(database))

    expenses = await repository.list_obligations(
        ObligationRequest(
            through=date(2026, 5, 31),
            include_inflows=False,
            limit=30,
        )
    )
    await repository.list_obligations(
        ObligationRequest(
            through=date(2026, 5, 31),
            include_inflows=True,
            limit=30,
        )
    )

    assert expenses[0].amount_milliunits == -1_000_000
    assert "AND st.amount < 0" in database.calls[0][0]
    assert "AND st.amount < 0" not in database.calls[1][0]
