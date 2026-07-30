"""SQL adapter for stable spending-tier mappings and YNAB outflows."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from ynab_agent.db._coercion import as_int
from ynab_agent.db._upsert import bulk_upsert
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.models import SpendingTierMappingRecord
from ynab_agent.planning.spending_guardrails import SpendingTier
from ynab_agent.services.spending_guardrails import (
    CategorySpendingAggregate,
    SpendingTierAssignment,
    SpendingTierMapping,
)


class SqlSpendingTierRepository:
    """Persist identity mappings outside sync-owned category rows."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def upsert_mapping(
        self,
        assignment: SpendingTierAssignment,
        *,
        updated_at: str,
    ) -> SpendingTierMapping | None:
        category = await self.database.fetch_all(
            """
            SELECT id
            FROM categories
            WHERE budget_id = :budget_id
              AND id = :category_id
            """,
            {
                "budget_id": assignment.budget_id,
                "category_id": assignment.category_id,
            },
        )
        if not category:
            return None
        await bulk_upsert(
            self.database.engine,
            self.database.session_factory,
            SpendingTierMappingRecord,
            [
                {
                    "budget_id": assignment.budget_id,
                    "category_id": assignment.category_id,
                    "tier": assignment.tier.value,
                    "essential_floor_milliunits": (
                        assignment.essential_floor_milliunits
                    ),
                    "note": assignment.note,
                    "updated_at": updated_at,
                }
            ],
        )
        rows = await self.list_mappings(budget_id=assignment.budget_id)
        return next(
            (
                row
                for row in rows
                if row.category_id == assignment.category_id
            ),
            None,
        )

    async def list_mappings(
        self,
        *,
        budget_id: str,
    ) -> Sequence[SpendingTierMapping]:
        rows = await self.database.fetch_all(
            """
            SELECT
              m.budget_id,
              m.category_id,
              m.tier,
              m.essential_floor_milliunits,
              m.note,
              m.updated_at,
              c.name AS category_name,
              g.name AS category_group_name,
              CASE
                WHEN c.id IS NULL THEN TRUE
                ELSE COALESCE(c.hidden, FALSE)
              END AS category_hidden,
              CASE
                WHEN c.id IS NULL THEN TRUE
                ELSE COALESCE(c.deleted, FALSE)
              END AS category_deleted
            FROM spending_tier_mappings m
            LEFT JOIN categories c
              ON c.id = m.category_id
             AND c.budget_id = m.budget_id
            LEFT JOIN category_groups g
              ON g.id = c.category_group_id
             AND g.budget_id = c.budget_id
            WHERE m.budget_id = :budget_id
            ORDER BY m.tier, COALESCE(g.name, ''), COALESCE(c.name, ''), m.category_id
            """,
            {"budget_id": budget_id},
        )
        return tuple(
            SpendingTierMapping(
                budget_id=str(row["budget_id"]),
                category_id=str(row["category_id"]),
                tier=SpendingTier(str(row["tier"])),
                essential_floor_milliunits=(
                    as_int(row["essential_floor_milliunits"])
                    if row["essential_floor_milliunits"] is not None
                    else None
                ),
                note=str(row["note"]) if row["note"] is not None else None,
                category_name=(
                    str(row["category_name"])
                    if row["category_name"] is not None
                    else None
                ),
                category_group_name=(
                    str(row["category_group_name"])
                    if row["category_group_name"] is not None
                    else None
                ),
                category_hidden=bool(row["category_hidden"]),
                category_deleted=bool(row["category_deleted"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        )

    async def list_spending(
        self,
        *,
        budget_id: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[CategorySpendingAggregate]:
        rows = await self.database.fetch_all(
            """
            WITH normalized AS (
              SELECT
                t.budget_id,
                t.date,
                t.account_id,
                t.category_id,
                t.transfer_account_id,
                t.amount
              FROM transactions t
              WHERE t.budget_id = :budget_id
                AND t.deleted IS FALSE
                AND NOT EXISTS (
                  SELECT 1
                  FROM subtransactions child
                  WHERE child.transaction_id = t.id
                    AND child.budget_id = t.budget_id
                    AND child.deleted IS FALSE
                )

              UNION ALL

              SELECT
                child.budget_id,
                t.date,
                t.account_id,
                child.category_id,
                child.transfer_account_id,
                child.amount
              FROM transactions t
              JOIN subtransactions child
                ON child.transaction_id = t.id
               AND child.budget_id = t.budget_id
              WHERE t.budget_id = :budget_id
                AND t.deleted IS FALSE
                AND child.deleted IS FALSE
            ),
            observed AS (
              SELECT
                n.budget_id,
                n.category_id,
                -SUM(COALESCE(n.amount, 0)) AS spending_milliunits
              FROM normalized n
              JOIN accounts a
                ON a.id = n.account_id
               AND a.budget_id = n.budget_id
              WHERE n.date >= :start_date
                AND n.date < :end_date
                AND n.category_id IS NOT NULL
                AND n.transfer_account_id IS NULL
                AND a.on_budget IS TRUE
              GROUP BY n.budget_id, n.category_id
            )
            SELECT
              mapping.category_id,
              mapping.tier,
              COALESCE(observed.spending_milliunits, 0) AS spending_milliunits
            FROM spending_tier_mappings mapping
            LEFT JOIN observed
              ON observed.category_id = mapping.category_id
             AND observed.budget_id = mapping.budget_id
            WHERE mapping.budget_id = :budget_id
            ORDER BY mapping.category_id
            """,
            {
                "budget_id": budget_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
        return tuple(
            CategorySpendingAggregate(
                category_id=str(row["category_id"]),
                tier=SpendingTier(str(row["tier"])),
                spending_milliunits=max(
                    0,
                    as_int(row["spending_milliunits"]),
                ),
            )
            for row in rows
        )
