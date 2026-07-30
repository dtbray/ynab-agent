"""Database adapter for cached synchronization change logs."""

from __future__ import annotations

from datetime import datetime
import json
from uuid import uuid4

from ynab_agent.db._upsert import bulk_upsert
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.models import BudgetChange
from ynab_agent.services.sync import Row


class SqlChangeLog:
    """Record and read synchronization changes without exposing the manager."""

    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def record_changes(
        self,
        batch_id: str,
        plan_id: str,
        resource: str,
        rows: list[Row],
    ) -> int:
        """Cache delta rows from a sync run as a queryable change log."""
        identified_rows = [row for row in rows if row.get("id") is not None]
        if not identified_rows:
            return 0

        existing_ids = await self._existing_change_ids(
            resource,
            [str(row["id"]) for row in identified_rows],
        )
        recorded_at = datetime.now()
        change_rows: list[Row] = [
            {
                "id": str(uuid4()),
                "batch_id": batch_id,
                "budget_id": plan_id,
                "resource": resource,
                "entity_id": str(row["id"]),
                "action": _change_action(row, existing_ids),
                "entity_name": _entity_name(row),
                "payload": json.dumps(row, default=str, sort_keys=True),
                "recorded_at": recorded_at,
            }
            for row in identified_rows
        ]
        await bulk_upsert(
            self._database.engine,
            self._database.session_factory,
            BudgetChange,
            change_rows,
        )
        return len(change_rows)

    async def get_latest_changes(
        self,
        plan_id: str | None = None,
    ) -> list[Row]:
        """Return cached changes from the most recent sync batch."""
        params: Row = {}
        plan_filter = ""
        if plan_id:
            params["plan_id"] = plan_id
            plan_filter = "WHERE budget_id = :plan_id"

        latest = await self._database.fetch_all(
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
            return []

        params = {"batch_id": latest[0]["batch_id"]}
        plan_filter = ""
        if plan_id:
            params["plan_id"] = plan_id
            plan_filter = "AND budget_id = :plan_id"
        rows = await self._database.fetch_all(
            f"""
            SELECT batch_id, budget_id, resource, entity_id, action, entity_name, recorded_at
            FROM budget_changes
            WHERE batch_id = :batch_id
              {plan_filter}
            ORDER BY resource, action, entity_name, entity_id
            """,
            params,
        )
        return [dict(row) for row in rows]

    async def _existing_change_ids(
        self,
        resource: str,
        entity_ids: list[str],
    ) -> set[str]:
        table_name = _change_resource_table(resource)
        if not table_name or not entity_ids:
            return set()

        placeholders = ", ".join(
            f":id_{index}" for index in range(len(entity_ids))
        )
        params = {
            f"id_{index}": entity_id
            for index, entity_id in enumerate(entity_ids)
        }
        rows = await self._database.fetch_all(
            f"SELECT id FROM {table_name} WHERE id IN ({placeholders})",
            params,
        )
        return {str(row["id"]) for row in rows}


def _entity_name(row: Row) -> str | None:
    name = row.get("name") or row.get("memo") or row.get("month")
    return str(name) if name is not None else None


def _change_action(row: Row, existing_ids: set[str]) -> str:
    if row.get("deleted") is True:
        return "deleted"
    return "updated" if str(row["id"]) in existing_ids else "created"


def _change_resource_table(resource: str) -> str | None:
    return {
        "accounts": "accounts",
        "budgets": "budgets",
        "categories": "categories",
        "category_groups": "category_groups",
        "month_categories": "month_categories",
        "month_detail": "budget_months",
        "months": "budget_months",
        "payee_locations": "payee_locations",
        "payees": "payees",
        "scheduled_subtransactions": "scheduled_subtransactions",
        "scheduled_transactions": "scheduled_transactions",
        "subtransactions": "subtransactions",
        "transactions": "transactions",
    }.get(resource)
