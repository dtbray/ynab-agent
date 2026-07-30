"""SQL queries for cached overview and hygiene reports."""

from __future__ import annotations

from datetime import date, datetime

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.reports.overview import (
    CachedChange,
    HygieneCheck,
    NetWorthBucket,
)


class SqlOverviewReportRepository:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def list_net_worth(self) -> tuple[NetWorthBucket, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT
              CASE WHEN on_budget THEN 'on_budget' ELSE 'off_budget' END AS bucket,
              type,
              COUNT(*) AS accounts,
              SUM(balance) AS balance
            FROM accounts
            WHERE closed IS FALSE AND deleted IS FALSE
            GROUP BY
              CASE WHEN on_budget THEN 'on_budget' ELSE 'off_budget' END,
              type
            ORDER BY bucket, balance DESC
            """
        )
        return tuple(
            NetWorthBucket(
                bucket=str(row["bucket"]),
                account_type=str(row["type"]),
                account_count=_required_int(row["accounts"]),
                balance_milliunits=_optional_int(row["balance"]),
            )
            for row in rows
        )

    async def list_latest_changes(
        self,
        plan_id: str | None,
    ) -> tuple[CachedChange, ...]:
        params: dict[str, object] = {}
        plan_filter = ""
        if plan_id:
            params["plan_id"] = plan_id
            plan_filter = "WHERE budget_id = :plan_id"
        latest = await self.database.fetch_all(
            f"""
            SELECT batch_id
            FROM budget_changes
            {plan_filter}
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
            params,
        )
        if not latest:
            return ()

        params = {"batch_id": str(latest[0]["batch_id"])}
        plan_filter = ""
        if plan_id:
            params["plan_id"] = plan_id
            plan_filter = "AND budget_id = :plan_id"
        rows = await self.database.fetch_all(
            f"""
            SELECT
              batch_id,
              budget_id,
              resource,
              entity_id,
              action,
              entity_name,
              recorded_at
            FROM budget_changes
            WHERE batch_id = :batch_id
              {plan_filter}
            ORDER BY resource, action, entity_name, entity_id
            """,
            params,
        )
        return tuple(
            CachedChange(
                batch_id=str(row["batch_id"]),
                budget_id=str(row["budget_id"]),
                resource=str(row["resource"]),
                action=str(row["action"]),
                entity_name=(str(row["entity_name"]) if row["entity_name"] is not None else None),
                entity_id=str(row["entity_id"]),
                recorded_at=_recorded_at(row["recorded_at"]),
            )
            for row in rows
        )

    async def list_hygiene_checks(
        self,
        *,
        since: date,
    ) -> tuple[HygieneCheck, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT 'unapproved' AS check_name, COUNT(*) AS count
            FROM transactions
            WHERE deleted IS FALSE
              AND approved IS FALSE
              AND date >= :since
            UNION ALL
            SELECT 'uncategorized_nontransfer' AS check_name, COUNT(*) AS count
            FROM transactions
            WHERE deleted IS FALSE
              AND date >= :since
              AND category_id IS NULL
              AND transfer_account_id IS NULL
            UNION ALL
            SELECT 'direct_import_errors' AS check_name, COUNT(*) AS count
            FROM accounts
            WHERE deleted IS FALSE
              AND closed IS FALSE
              AND direct_import_in_error IS TRUE
            """,
            {"since": since.isoformat()},
        )
        return tuple(
            HygieneCheck(
                check_name=str(row["check_name"]),
                count=_required_int(row["count"]),
            )
            for row in rows
        )


def _optional_int(value: object) -> int | None:
    return _required_int(value) if value is not None else None


def _required_int(value: object) -> int:
    return int(str(value))


def _recorded_at(value: object) -> datetime | str:
    if isinstance(value, datetime):
        return value
    return str(value)
