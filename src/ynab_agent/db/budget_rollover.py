"""Scoped cached-month adapter for budget rollover planning."""

from __future__ import annotations

from datetime import date

from ynab_agent.db._coercion import as_int
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.budget_rollover import MonthCategory


class SqlBudgetRolloverReader:
    """Read month category state without exposing SQL to command adapters."""

    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def get_month_categories(
        self,
        *,
        plan_id: str,
        month: date,
    ) -> tuple[MonthCategory, ...]:
        plan_filter = (
            ""
            if plan_id in {"last-used", "default"}
            else "AND mc.budget_id = :plan_id"
        )
        rows = await self._database.fetch_all(
            f"""
            SELECT
              mc.category_id,
              COALESCE(g.name, 'Unknown') AS group_name,
              COALESCE(c.name, mc.category_id) AS category_name,
              mc.budgeted,
              mc.balance,
              c.hidden,
              c.deleted
            FROM month_categories mc
            JOIN categories c
              ON c.id = mc.category_id
             AND c.budget_id = mc.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            WHERE mc.month = :month
              {plan_filter}
            ORDER BY mc.category_id
            """,
            {
                "month": month.isoformat(),
                "plan_id": plan_id,
            },
        )
        return tuple(
            MonthCategory(
                category_id=str(row["category_id"]),
                group_name=str(row["group_name"]),
                name=str(row["category_name"]),
                budgeted=as_int(row["budgeted"] or 0),
                balance=as_int(row["balance"] or 0),
                hidden=bool(row["hidden"]),
                deleted=bool(row["deleted"]),
            )
            for row in rows
        )
