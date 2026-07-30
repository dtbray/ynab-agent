"""SQL queries for cached scheduled obligations."""

from __future__ import annotations

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.reports.obligations import (
    ObligationRequest,
    ScheduledObligation,
)


class SqlObligationRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def list_obligations(
        self,
        request: ObligationRequest,
    ) -> tuple[ScheduledObligation, ...]:
        amount_filter = "" if request.include_inflows else "AND st.amount < 0"
        rows = await self.database.fetch_all(
            f"""
            SELECT
              st.date_next AS due_date,
              COALESCE(p.name, 'Unknown') AS payee,
              COALESCE(g.name, 'Unknown') AS group_name,
              c.name AS category,
              a.name AS account,
              st.frequency,
              st.amount,
              st.memo
            FROM scheduled_transactions st
            LEFT JOIN payees p
              ON p.id = st.payee_id
             AND p.budget_id = st.budget_id
            LEFT JOIN categories c
              ON c.id = st.category_id
             AND c.budget_id = st.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            LEFT JOIN accounts a
              ON a.id = st.account_id
             AND a.budget_id = st.budget_id
            WHERE st.deleted IS FALSE
              AND st.date_next IS NOT NULL
              AND st.date_next <= :through
              {amount_filter}
            ORDER BY st.date_next, st.amount
            LIMIT :limit
            """,
            {
                "through": request.through.isoformat(),
                "limit": request.limit,
            },
        )
        return tuple(
            ScheduledObligation(
                due_date=str(row["due_date"]),
                payee=str(row["payee"]),
                group_name=str(row["group_name"]),
                category=_optional_str(row["category"]),
                account=_optional_str(row["account"]),
                frequency=_optional_str(row["frequency"]),
                amount_milliunits=_optional_int(row["amount"]),
                memo=_optional_str(row["memo"]),
            )
            for row in rows
        )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None
