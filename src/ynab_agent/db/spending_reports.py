"""SQL queries for cached cashflow and spending reports."""

from __future__ import annotations

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.reports.spending import (
    BurnRateGrouping,
    CashflowMonth,
    DateRangeRequest,
    PayeeSpend,
    RawBurnRateBucket,
)


_CATEGORY_BUCKET_SQL = "COALESCE(g.name, 'Unknown') || ' / ' || COALESCE(c.name, 'Unknown')"
_GROUP_BUCKET_SQL = "COALESCE(g.name, 'Unknown')"


class SqlSpendingReportRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def list_cashflow(
        self,
        request: DateRangeRequest,
    ) -> tuple[CashflowMonth, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT
              substr(date, 1, 7) AS month,
              SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS inflow,
              SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END) AS outflow,
              SUM(amount) AS net,
              COUNT(*) AS transactions
            FROM transactions
            WHERE deleted IS FALSE
              AND date >= :since
              AND category_id IS NOT NULL
              AND transfer_account_id IS NULL
            GROUP BY substr(date, 1, 7)
            ORDER BY month
            """,
            {"since": request.since.isoformat()},
        )
        return tuple(
            CashflowMonth(
                month=str(row["month"]),
                inflow_milliunits=_optional_int(row["inflow"]),
                outflow_milliunits=_optional_int(row["outflow"]),
                net_milliunits=_optional_int(row["net"]),
                transaction_count=_required_int(row["transactions"]),
            )
            for row in rows
        )

    async def list_burn_rate(
        self,
        request: DateRangeRequest,
        *,
        grouping: BurnRateGrouping,
    ) -> tuple[RawBurnRateBucket, ...]:
        bucket_sql = (
            _GROUP_BUCKET_SQL if grouping is BurnRateGrouping.GROUP else _CATEGORY_BUCKET_SQL
        )
        through_filter = "AND t.date <= :through" if request.through is not None else ""
        rows = await self.database.fetch_all(
            f"""
            SELECT
              {bucket_sql} AS bucket,
              SUM(-t.amount) AS outflow,
              COUNT(*) AS transactions,
              COUNT(DISTINCT substr(t.date, 1, 7)) AS active_months
            FROM transactions t
            LEFT JOIN categories c
              ON c.id = t.category_id
             AND c.budget_id = t.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            WHERE t.deleted IS FALSE
              AND t.date >= :since
              {through_filter}
              AND t.amount < 0
              AND t.category_id IS NOT NULL
              AND t.transfer_account_id IS NULL
            GROUP BY {bucket_sql}
            ORDER BY SUM(-t.amount) DESC
            LIMIT :limit
            """,
            {
                "since": request.since.isoformat(),
                "through": (request.through.isoformat() if request.through is not None else None),
                "limit": request.limit,
            },
        )
        return tuple(
            RawBurnRateBucket(
                bucket=str(row["bucket"]),
                outflow_milliunits=_optional_int(row["outflow"]),
                transaction_count=_required_int(row["transactions"]),
                active_months=_required_int(row["active_months"]),
            )
            for row in rows
        )

    async def list_top_spend(
        self,
        request: DateRangeRequest,
    ) -> tuple[PayeeSpend, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT
              COALESCE(p.name, 'Unknown') AS payee,
              SUM(-t.amount) AS outflow,
              COUNT(*) AS transactions
            FROM transactions t
            LEFT JOIN payees p
              ON p.id = t.payee_id
             AND p.budget_id = t.budget_id
            WHERE t.deleted IS FALSE
              AND t.date >= :since
              AND t.amount < 0
              AND t.category_id IS NOT NULL
              AND t.transfer_account_id IS NULL
            GROUP BY COALESCE(p.name, 'Unknown')
            ORDER BY SUM(-t.amount) DESC
            LIMIT :limit
            """,
            {
                "since": request.since.isoformat(),
                "limit": request.limit,
            },
        )
        return tuple(
            PayeeSpend(
                payee=str(row["payee"]),
                outflow_milliunits=_optional_int(row["outflow"]),
                transaction_count=_required_int(row["transactions"]),
            )
            for row in rows
        )


def _optional_int(value: object) -> int | None:
    return _required_int(value) if value is not None else None


def _required_int(value: object) -> int:
    return int(str(value))
