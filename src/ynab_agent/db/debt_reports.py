"""Bounded SQL read models for the debt report family."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from ynab_agent.db._coercion import as_int
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.debt_reports import (
    DebtAccountSnapshot,
    ScheduledDebtObligation,
)


_PAYMENT_CATEGORY_CTE = """
WITH payment_categories AS (
  SELECT
    c.budget_id,
    c.name,
    c.balance AS payment_available,
    c.goal_target,
    c.goal_under_funded,
    c.goal_overall_left
  FROM categories c
  LEFT JOIN category_groups g
    ON g.id = c.category_group_id
   AND g.budget_id = c.budget_id
  WHERE c.deleted IS FALSE
    AND c.hidden IS FALSE
    AND lower(COALESCE(g.name, '')) LIKE '%credit card%'
    AND (:plan_id IS NULL OR c.budget_id = :plan_id)
)
"""

_ACCOUNT_COLUMNS = """
  a.name AS account,
  a.type AS account_type,
  COALESCE(a.balance, 0) AS balance,
  COALESCE(a.cleared_balance, 0) AS cleared_balance,
  COALESCE(a.uncleared_balance, 0) AS uncleared_balance,
  COALESCE(pc.payment_available, 0) AS payment_available,
  COALESCE(pc.goal_target, 0) AS goal_target,
  COALESCE(pc.goal_under_funded, 0) AS goal_under_funded,
  COALESCE(pc.goal_overall_left, 0) AS goal_overall_left,
  a.debt_minimum_payments AS minimum_payment_data
"""

_ACCOUNT_JOIN = """
FROM accounts a
LEFT JOIN payment_categories pc
  ON pc.budget_id = a.budget_id
 AND lower(pc.name) = lower(a.name)
WHERE a.deleted IS FALSE
  AND a.closed IS FALSE
  AND (:plan_id IS NULL OR a.budget_id = :plan_id)
"""


class SqlDebtReportRepository:
    """SQL-backed debt reports with plan, state, and date bounds."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def list_credit_cards(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]:
        rows = await self.database.fetch_all(
            _PAYMENT_CATEGORY_CTE
            + """
SELECT
"""
            + _ACCOUNT_COLUMNS
            + _ACCOUNT_JOIN
            + """
  AND lower(COALESCE(a.type, '')) IN ('creditcard', 'credit_card')
ORDER BY CASE WHEN a.balance < 0 THEN -a.balance ELSE 0 END DESC, account
""",
            {"plan_id": plan_id},
        )
        return tuple(_account_snapshot(row) for row in rows)

    async def list_debt_drag_accounts(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]:
        rows = await self.database.fetch_all(
            _PAYMENT_CATEGORY_CTE
            + """
SELECT
"""
            + _ACCOUNT_COLUMNS
            + _ACCOUNT_JOIN
            + """
  AND (
    COALESCE(a.balance, 0) < 0
    OR lower(COALESCE(a.type, '')) IN (
      'creditcard',
      'credit_card',
      'lineofcredit',
      'otherliability',
      'mortgage'
    )
  )
ORDER BY CASE WHEN a.balance < 0 THEN -a.balance ELSE 0 END DESC, account
""",
            {"plan_id": plan_id},
        )
        return tuple(_account_snapshot(row) for row in rows)

    async def list_debt_plan_accounts(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]:
        rows = await self.database.fetch_all(
            _PAYMENT_CATEGORY_CTE
            + """
SELECT
"""
            + _ACCOUNT_COLUMNS
            + _ACCOUNT_JOIN
            + """
  AND (
    COALESCE(a.balance, 0) < 0
    OR lower(COALESCE(a.type, '')) IN (
      'creditcard',
      'credit_card',
      'lineofcredit',
      'otherliability',
      'medicaldebt'
    )
  )
ORDER BY CASE WHEN a.balance < 0 THEN -a.balance ELSE 0 END DESC, account
""",
            {"plan_id": plan_id},
        )
        return tuple(_account_snapshot(row) for row in rows)

    async def list_scheduled_debt_obligations(
        self,
        *,
        through: date,
        plan_id: str | None,
    ) -> Sequence[ScheduledDebtObligation]:
        rows = await self.database.fetch_all(
            """
            SELECT
              COALESCE(g.name, 'Unknown') AS group_name,
              COALESCE(c.name, 'Unknown') AS category,
              COALESCE(p.name, 'Unknown') AS payee,
              st.amount,
              COALESCE(st.frequency, '') AS frequency
            FROM scheduled_transactions st
            LEFT JOIN payees p
              ON p.id = st.payee_id
             AND p.budget_id = st.budget_id
            LEFT JOIN categories c
              ON c.id = st.category_id
             AND c.budget_id = st.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = st.budget_id
            WHERE st.deleted IS FALSE
              AND st.date_next IS NOT NULL
              AND st.date_next <= :through
              AND st.amount < 0
              AND (:plan_id IS NULL OR st.budget_id = :plan_id)
              AND (
                lower(COALESCE(g.name, '')) LIKE '%debt%'
                OR lower(COALESCE(g.name, '')) LIKE '%loan%'
                OR lower(COALESCE(g.name, '')) LIKE '%credit card%'
                OR lower(COALESCE(c.name, '')) LIKE '%debt%'
                OR lower(COALESCE(c.name, '')) LIKE '%loan%'
                OR lower(COALESCE(c.name, '')) LIKE '%student%'
                OR lower(COALESCE(c.name, '')) LIKE '%mortgage%'
                OR lower(COALESCE(c.name, '')) LIKE '%card%'
                OR lower(COALESCE(p.name, '')) LIKE '%loan%'
                OR lower(COALESCE(p.name, '')) LIKE '%student%'
              )
            ORDER BY st.amount ASC, st.id
            """,
            {"through": through.isoformat(), "plan_id": plan_id},
        )
        return tuple(
            ScheduledDebtObligation(
                group_name=str(row["group_name"]),
                category=str(row["category"]),
                payee=str(row["payee"]),
                amount_milliunits=as_int(row["amount"]),
                frequency=str(row["frequency"]),
            )
            for row in rows
        )


def _account_snapshot(row: dict[str, object]) -> DebtAccountSnapshot:
    return DebtAccountSnapshot(
        account=str(row["account"]),
        account_type=str(row["account_type"] or ""),
        balance_milliunits=as_int(row["balance"]),
        cleared_balance_milliunits=as_int(row["cleared_balance"]),
        uncleared_balance_milliunits=as_int(row["uncleared_balance"]),
        payment_available_milliunits=as_int(row["payment_available"]),
        goal_target_milliunits=as_int(row["goal_target"]),
        goal_under_funded_milliunits=as_int(row["goal_under_funded"]),
        goal_overall_left_milliunits=as_int(row["goal_overall_left"]),
        minimum_payment_data=(
            str(row["minimum_payment_data"])
            if row["minimum_payment_data"] is not None
            else None
        ),
    )
