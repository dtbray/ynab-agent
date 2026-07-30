"""SQL queries for cached category activity reports."""

from __future__ import annotations

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.reports.budget_activity import (
    BudgetActivityRequest,
    CategoryActivity,
    OverspendingCategory,
)


class SqlBudgetActivityRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def list_overspending(
        self,
        request: BudgetActivityRequest,
    ) -> tuple[OverspendingCategory, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT
              COALESCE(g.name, 'Unknown') AS group_name,
              c.name AS category,
              mc.budgeted,
              mc.activity,
              mc.balance,
              -mc.balance AS overspent
            FROM month_categories mc
            JOIN categories c
              ON c.id = mc.category_id
             AND c.budget_id = mc.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            WHERE mc.month = :month
              AND mc.balance < 0
              AND c.hidden IS FALSE
              AND c.deleted IS FALSE
            ORDER BY mc.balance ASC
            LIMIT :limit
            """,
            {"month": request.month.isoformat(), "limit": request.limit},
        )
        return tuple(
            OverspendingCategory(
                group_name=str(row["group_name"]),
                category=str(row["category"]),
                budgeted_milliunits=_optional_int(row["budgeted"]),
                activity_milliunits=_optional_int(row["activity"]),
                balance_milliunits=_optional_int(row["balance"]),
                overspent_milliunits=_optional_int(row["overspent"]),
            )
            for row in rows
        )

    async def list_hidden_funds(
        self,
        request: BudgetActivityRequest,
    ) -> tuple[CategoryActivity, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT
              COALESCE(g.name, 'Unknown') AS group_name,
              c.name AS category,
              mc.budgeted,
              mc.activity,
              mc.balance
            FROM month_categories mc
            JOIN categories c
              ON c.id = mc.category_id
             AND c.budget_id = mc.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            WHERE mc.month = :month
              AND c.hidden IS TRUE
              AND c.deleted IS FALSE
              AND (
                COALESCE(mc.budgeted, 0) != 0
                OR COALESCE(mc.balance, 0) != 0
              )
            ORDER BY
              ABS(COALESCE(mc.balance, 0)) DESC,
              ABS(COALESCE(mc.budgeted, 0)) DESC
            LIMIT :limit
            """,
            {"month": request.month.isoformat(), "limit": request.limit},
        )
        return tuple(_category_activity(row) for row in rows)

    async def list_month_activity(
        self,
        request: BudgetActivityRequest,
    ) -> tuple[CategoryActivity, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT
              COALESCE(g.name, 'Unknown') AS group_name,
              c.name AS category,
              mc.budgeted,
              mc.activity,
              mc.balance
            FROM month_categories mc
            JOIN categories c
              ON c.id = mc.category_id
             AND c.budget_id = mc.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            WHERE mc.month = :month
              AND c.hidden IS FALSE
              AND c.deleted IS FALSE
              AND mc.activity < 0
            ORDER BY mc.activity ASC
            LIMIT :limit
            """,
            {"month": request.month.isoformat(), "limit": request.limit},
        )
        return tuple(_category_activity(row) for row in rows)


def _category_activity(row: dict[str, object]) -> CategoryActivity:
    return CategoryActivity(
        group_name=str(row["group_name"]),
        category=str(row["category"]),
        budgeted_milliunits=_optional_int(row["budgeted"]),
        activity_milliunits=_optional_int(row["activity"]),
        balance_milliunits=_optional_int(row["balance"]),
    )


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None
