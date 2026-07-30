"""Scoped SQL adapter for the cost-to-be-me report."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from ynab_agent.db._coercion import as_int
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.reports.cost_to_be_me import (
    FundingCategory,
    MonthlyIncome,
    MonthlySpending,
)


class SqlCostToBeMeRepository:
    """Load bounded cost-to-be-me aggregates from the local YNAB cache."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def list_funding_categories(
        self,
        *,
        month: date,
        plan_id: str | None,
    ) -> Sequence[FundingCategory]:
        plan_filter = (
            "AND mc.budget_id = :plan_id"
            if plan_id is not None
            else ""
        )
        rows = await self.database.fetch_all(
            f"""
            SELECT
              COALESCE(g.name, 'Unknown') AS group_name,
              c.name AS category,
              COALESCE(mc.budgeted, 0) AS assigned,
              COALESCE(mc.goal_under_funded, 0) AS still_underfunded,
              COALESCE(mc.goal_target, 0) AS goal_period_target
            FROM month_categories mc
            JOIN categories c
              ON c.id = mc.category_id
             AND c.budget_id = mc.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            WHERE mc.month = :month
              {plan_filter}
              AND mc.deleted IS FALSE
              AND mc.hidden IS FALSE
              AND mc.internal IS FALSE
              AND c.deleted IS FALSE
              AND c.hidden IS FALSE
              AND c.internal IS FALSE
              AND COALESCE(g.hidden, FALSE) IS FALSE
              AND COALESCE(g.name, '') NOT IN (
                'Credit Card Payments',
                'Internal Master Category'
              )
            ORDER BY group_name, category
            """,
            {"month": month.isoformat(), "plan_id": plan_id},
        )
        return tuple(
            FundingCategory(
                group_name=str(row["group_name"]),
                category_name=str(row["category"]),
                assigned_milliunits=as_int(row["assigned"]),
                still_underfunded_milliunits=as_int(row["still_underfunded"]),
                goal_period_target_milliunits=as_int(row["goal_period_target"]),
            )
            for row in rows
        )

    async def list_monthly_spending(
        self,
        *,
        start_date: date,
        end_date: date,
        plan_id: str | None,
    ) -> Sequence[MonthlySpending]:
        plan_filter = (
            "AND t.budget_id = :plan_id"
            if plan_id is not None
            else ""
        )
        rows = await self.database.fetch_all(
            f"""
            WITH normalized AS (
              SELECT
                t.budget_id,
                t.date,
                t.account_id,
                t.category_id,
                t.transfer_account_id,
                t.amount
              FROM transactions t
              WHERE t.deleted IS FALSE
                {plan_filter}
                AND NOT EXISTS (
                  SELECT 1
                  FROM subtransactions st
                  WHERE st.transaction_id = t.id
                    AND st.budget_id = t.budget_id
                    AND st.deleted IS FALSE
                )
              UNION ALL
              SELECT
                t.budget_id,
                t.date,
                t.account_id,
                st.category_id,
                st.transfer_account_id,
                st.amount
              FROM transactions t
              JOIN subtransactions st
                ON st.transaction_id = t.id
               AND st.budget_id = t.budget_id
               AND st.deleted IS FALSE
              WHERE t.deleted IS FALSE
                {plan_filter}
            ),
            monthly AS (
              SELECT
                substr(n.date, 1, 7) AS month,
                -SUM(n.amount) AS spending
              FROM normalized n
              JOIN accounts a
                ON a.id = n.account_id
               AND a.budget_id = n.budget_id
              JOIN categories c
                ON c.id = n.category_id
               AND c.budget_id = n.budget_id
              LEFT JOIN category_groups g
                ON g.id = c.category_group_id
               AND g.budget_id = c.budget_id
              WHERE n.date >= :start_date
                AND n.date < :end_date
                AND n.transfer_account_id IS NULL
                AND a.on_budget IS TRUE
                AND COALESCE(g.name, '') NOT IN (
                  'Credit Card Payments',
                  'Internal Master Category'
                )
              GROUP BY substr(n.date, 1, 7)
            )
            SELECT month, spending
            FROM monthly
            WHERE spending > 0
            ORDER BY month
            """,
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "plan_id": plan_id,
            },
        )
        spending_by_month = {
            str(row["month"]): max(0, as_int(row["spending"]))
            for row in rows
        }
        return tuple(
            MonthlySpending(
                month=month,
                spending_milliunits=spending_by_month.get(month, 0),
            )
            for month in _month_labels(start_date, end_date)
        )

    async def list_monthly_ready_to_assign_income(
        self,
        *,
        start_date: date,
        end_date: date,
        plan_id: str | None,
    ) -> Sequence[MonthlyIncome]:
        plan_filter = (
            "AND bm.budget_id = :plan_id"
            if plan_id is not None
            else ""
        )
        rows = await self.database.fetch_all(
            f"""
            SELECT
              substr(bm.month, 1, 7) AS month,
              COALESCE(bm.income, 0) AS income
            FROM budget_months bm
            WHERE bm.deleted IS FALSE
              {plan_filter}
              AND bm.month >= :start_date
              AND bm.month < :end_date
            ORDER BY month
            """,
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "plan_id": plan_id,
            },
        )
        income_by_month = {
            str(row["month"]): max(0, as_int(row["income"]))
            for row in rows
        }
        return tuple(
            MonthlyIncome(
                month=month,
                income_milliunits=income_by_month.get(month, 0),
            )
            for month in _month_labels(start_date, end_date)
        )


def _month_labels(start: date, end: date) -> tuple[str, ...]:
    labels: list[str] = []
    current = start.replace(day=1)
    end_month = end.replace(day=1)
    while current < end_month:
        labels.append(f"{current:%Y-%m}")
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return tuple(labels)
