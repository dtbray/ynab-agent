"""SQL adapter for immutable continuous-calibration state."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from typing import Protocol, cast
from uuid import uuid5

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.scenario_comparison import SqlScenarioRevisionRepository
from ynab_agent.planning.models import LIQUID_PORTFOLIO_ROLES, WealthScenario
from ynab_agent.services.calibration import (
    CALIBRATION_NAMESPACE,
    AlertState,
    CalibrationAlert,
    CalibrationCaptureFailure,
    CalibrationProfile,
    CalibrationProfileState,
    CalibrationRun,
    CalibrationRunState,
    CalibrationSourceAccount,
    CalibrationSourceState,
)
from ynab_agent.services.calibration_models import (
    CalibrationObservation,
    CalibrationSnapshot,
    CalibrationSnapshotManifest,
    DriftEvent,
    DriftKind,
    SourceWatermark,
    canonical_content_sha256,
)
from ynab_agent.services.scenario_comparison import ScenarioRevision
from ynab_agent.services.calibration_attribution import AttributionReport


class _RowCountResult(Protocol):
    rowcount: int


class SqlCalibrationRepository:
    """Append-only repository with atomic idempotency and tamper checks."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database
        self._scenarios = SqlScenarioRevisionRepository(database)

    async def create_profile(self, profile: CalibrationProfile) -> None:
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_profiles (
                          id, scenario_revision_id, budget_id,
                          profile_json, profile_sha256, created_at
                        ) VALUES (
                          :id, :scenario_revision_id, :budget_id,
                          :profile_json, :profile_sha256, :created_at
                        )
                        """
                    ),
                    {
                        "id": profile.id,
                        "scenario_revision_id": profile.scenario_revision_id,
                        "budget_id": profile.budget_id,
                        "profile_json": profile.model_dump_json(),
                        "profile_sha256": profile.profile_sha256,
                        "created_at": profile.created_at.isoformat(),
                    },
                )
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_profile_heads (profile_id, snapshot_id)
                        VALUES (:profile_id, NULL)
                        """
                    ),
                    {"profile_id": profile.id},
                )
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_profile_events (
                          id, profile_id, state, created_at
                        ) VALUES (
                          :id, :profile_id, :state, :created_at
                        )
                        """
                    ),
                    {
                        "id": str(
                            uuid5(
                                CALIBRATION_NAMESPACE,
                                f"profile-event:{profile.id}:active",
                            )
                        ),
                        "profile_id": profile.id,
                        "state": CalibrationProfileState.ACTIVE.value,
                        "created_at": profile.created_at.isoformat(),
                    },
                )
                await session.commit()
        except IntegrityError as error:
            raise ValueError("calibration profile identity already exists") from error

    async def get_profile(self, profile_id: str) -> CalibrationProfile | None:
        rows = await self.database.fetch_all(
            """
            SELECT profile_json, profile_sha256
            FROM calibration_profiles
            WHERE id = :profile_id
            """,
            {"profile_id": profile_id},
        )
        if not rows:
            return None
        profile = CalibrationProfile.model_validate_json(str(rows[0]["profile_json"]))
        if profile.profile_sha256 != rows[0]["profile_sha256"]:
            raise ValueError("persisted calibration profile hash mismatch")
        return profile

    async def list_profiles(
        self,
        budget_id: str | None = None,
    ) -> tuple[CalibrationProfile, ...]:
        where = ""
        params: dict[str, object] = {}
        if budget_id is not None:
            where = "WHERE budget_id = :budget_id"
            params["budget_id"] = budget_id
        rows = await self.database.fetch_all(
            f"""
            SELECT id
            FROM calibration_profiles
            {where}
            ORDER BY created_at, id
            """,
            params,
        )
        profiles = [
            profile
            for row in rows
            if (profile := await self.get_profile(str(row["id"]))) is not None
        ]
        return tuple(profiles)

    async def list_active_profiles(
        self,
        budget_id: str,
    ) -> tuple[CalibrationProfile, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT p.id
            FROM calibration_profiles p
            JOIN calibration_profile_events e ON e.id = (
              SELECT e2.id
              FROM calibration_profile_events e2
              WHERE e2.profile_id = p.id
              ORDER BY e2.created_at DESC, e2.id DESC
              LIMIT 1
            )
            WHERE p.budget_id = :budget_id
              AND e.state = :active
            ORDER BY p.created_at, p.id
            """,
            {
                "budget_id": budget_id,
                "active": CalibrationProfileState.ACTIVE.value,
            },
        )
        profiles: list[CalibrationProfile] = []
        for row in rows:
            profile = await self.get_profile(str(row["id"]))
            if profile is not None:
                profiles.append(profile)
        return tuple(profiles)

    async def set_profile_state(
        self,
        profile_id: str,
        state: CalibrationProfileState,
        *,
        at: datetime,
    ) -> bool:
        if await self.get_profile(profile_id) is None:
            return False
        event_id = str(
            uuid5(
                CALIBRATION_NAMESPACE,
                f"profile-event:{profile_id}:{state.value}:{at.isoformat()}",
            )
        )
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_profile_events (
                          id, profile_id, state, created_at
                        ) VALUES (
                          :id, :profile_id, :state, :created_at
                        )
                        """
                    ),
                    {
                        "id": event_id,
                        "profile_id": profile_id,
                        "state": state.value,
                        "created_at": at.isoformat(),
                    },
                )
                await session.commit()
        except IntegrityError:
            pass
        return True

    async def get_profile_state(
        self,
        profile_id: str,
    ) -> CalibrationProfileState | None:
        rows = await self.database.fetch_all(
            """
            SELECT state
            FROM calibration_profile_events
            WHERE profile_id = :profile_id
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            {"profile_id": profile_id},
        )
        return CalibrationProfileState(str(rows[0]["state"])) if rows else None

    async def begin_sync_batch(
        self,
        budget_id: str,
        source_sync_batch_id: str,
        *,
        started_at: datetime,
    ) -> None:
        """Claim the budget cache generation and supersede an abandoned writer."""
        async with self.database.session_factory() as session:
            try:
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_sync_batches (
                          id, budget_id, state, started_at
                        ) VALUES (
                          :id, :budget_id, 'started', :started_at
                        )
                        """
                    ),
                    {
                        "id": source_sync_batch_id,
                        "budget_id": budget_id,
                        "started_at": started_at.isoformat(),
                    },
                )
            except IntegrityError:
                await session.rollback()
                existing = await self.database.fetch_all(
                    """
                    SELECT budget_id, state, started_at
                    FROM calibration_sync_batches
                    WHERE id = :id
                    """,
                    {"id": source_sync_batch_id},
                )
                if not existing or (
                    str(existing[0]["budget_id"]),
                    str(existing[0]["state"]),
                    _as_datetime(existing[0]["started_at"]),
                ) != (budget_id, "started", started_at):
                    raise ValueError("sync batch lifecycle identity conflict") from None
                head = await self.database.fetch_all(
                    """
                    SELECT batch_id
                    FROM calibration_sync_batch_heads
                    WHERE budget_id = :budget_id
                    """,
                    {"budget_id": budget_id},
                )
                if head and str(head[0]["batch_id"]) == source_sync_batch_id:
                    return
                raise ValueError("sync batch is no longer the active writer") from None

            await session.execute(
                text(
                    """
                    INSERT INTO calibration_sync_batch_heads (budget_id, batch_id)
                    VALUES (:budget_id, NULL)
                    ON CONFLICT (budget_id) DO NOTHING
                    """
                ),
                {"budget_id": budget_id},
            )
            lock_result = await session.execute(
                text(
                    """
                    UPDATE calibration_sync_batch_heads
                    SET batch_id = batch_id
                    WHERE budget_id = :budget_id
                    """
                ),
                {"budget_id": budget_id},
            )
            if cast(_RowCountResult, lock_result).rowcount != 1:
                await session.rollback()
                raise RuntimeError("could not lock sync batch head")
            head_result = await session.execute(
                text(
                    """
                    SELECT batch_id
                    FROM calibration_sync_batch_heads
                    WHERE budget_id = :budget_id
                    """
                ),
                {"budget_id": budget_id},
            )
            predecessor = head_result.mappings().one()["batch_id"]
            predecessor_id = (
                str(predecessor)
                if predecessor is not None and str(predecessor) != source_sync_batch_id
                else None
            )
            if predecessor_id is not None:
                await session.execute(
                    text(
                        """
                        UPDATE calibration_sync_batches
                        SET state = CASE
                              WHEN state = 'started' THEN 'superseded'
                              ELSE state
                            END,
                            superseded_by = :source_sync_batch_id
                        WHERE id = :predecessor_id
                          AND budget_id = :budget_id
                        """
                    ),
                    {
                        "source_sync_batch_id": source_sync_batch_id,
                        "predecessor_id": predecessor_id,
                        "budget_id": budget_id,
                    },
                )
            await session.execute(
                text(
                    """
                    UPDATE calibration_sync_batches
                    SET predecessor_id = :predecessor_id
                    WHERE id = :source_sync_batch_id
                    """
                ),
                {
                    "predecessor_id": predecessor_id,
                    "source_sync_batch_id": source_sync_batch_id,
                },
            )
            await session.execute(
                text(
                    """
                    UPDATE calibration_sync_batch_heads
                    SET batch_id = :source_sync_batch_id
                    WHERE budget_id = :budget_id
                    """
                ),
                {
                    "source_sync_batch_id": source_sync_batch_id,
                    "budget_id": budget_id,
                },
            )
            await session.commit()

    async def record_completed_sync_batch(
        self,
        budget_id: str,
        source_sync_batch_id: str,
        *,
        completed_at: datetime,
    ) -> None:
        existing = await self._completed_sync_batch(source_sync_batch_id)
        if existing is not None:
            if existing[0] != budget_id:
                raise ValueError("completed sync batch identity conflict")
            return
        async with self.database.session_factory() as session:
            lock_result = await session.execute(
                text(
                    """
                    UPDATE calibration_sync_batch_heads
                    SET batch_id = batch_id
                    WHERE budget_id = :budget_id
                      AND batch_id = :source_sync_batch_id
                    """
                ),
                {
                    "budget_id": budget_id,
                    "source_sync_batch_id": source_sync_batch_id,
                },
            )
            if cast(_RowCountResult, lock_result).rowcount != 1:
                await session.rollback()
                raise ValueError("sync batch is no longer the active writer")
            lifecycle_result = await session.execute(
                text(
                    """
                    SELECT state, unowned_source_dirty
                    FROM calibration_sync_batches
                    WHERE id = :source_sync_batch_id
                      AND budget_id = :budget_id
                    """
                ),
                {
                    "source_sync_batch_id": source_sync_batch_id,
                    "budget_id": budget_id,
                },
            )
            lifecycle_row = lifecycle_result.mappings().one_or_none()
            if lifecycle_row is None:
                await session.rollback()
                raise ValueError("sync batch lifecycle cannot be completed")
            if str(lifecycle_row["state"]) == "completed":
                await session.rollback()
                return
            if str(lifecycle_row["state"]) != "started":
                await session.rollback()
                raise ValueError("sync batch lifecycle cannot be completed")
            unowned_source_dirty = bool(lifecycle_row["unowned_source_dirty"])
            checkpoint_result = await session.execute(
                text(
                    """
                    SELECT budget_id, resource, server_knowledge, synced_at,
                           change_batch_id
                    FROM sync_state
                    WHERE budget_id = :budget_id
                    ORDER BY resource
                    """
                ),
                {"budget_id": budget_id},
            )
            rows = list(checkpoint_result.mappings())
            if not rows:
                await session.rollback()
                raise ValueError("completed sync batch has no durable watermarks")
            if any(
                str(row["change_batch_id"] or "") != source_sync_batch_id
                for row in rows
            ):
                await session.rollback()
                raise ValueError(
                    "completed sync batch does not own every durable watermark"
                )
            if unowned_source_dirty:
                await session.rollback()
                raise ValueError(
                    "sync batch cannot complete after an unowned source write"
                )
            (
                unfinished_lineage,
                lineage_has_completed,
                lineage_has_unowned_write,
            ) = (
                await self._sync_batch_lineage(
                    session,
                    budget_id,
                    source_sync_batch_id,
                )
            )
            source_owners_sql = """
                SELECT change_batch_id AS owner_batch_id
                FROM accounts
                WHERE budget_id = :budget_id
                UNION ALL
                SELECT change_batch_id AS owner_batch_id
                FROM transactions
                WHERE budget_id = :budget_id
                UNION ALL
                SELECT change_batch_id AS owner_batch_id
                FROM categories
                WHERE budget_id = :budget_id
            """
            unowned_result = await session.execute(
                text(
                    f"""
                    SELECT 1
                    FROM ({source_owners_sql}) source_owners
                    WHERE owner_batch_id IS NULL
                    LIMIT 1
                    """
                ),
                {"budget_id": budget_id},
            )
            has_unowned_source = unowned_result.first() is not None
            if has_unowned_source and (
                lineage_has_completed or lineage_has_unowned_write
            ):
                await session.rollback()
                raise ValueError(
                    "sync batch cannot adopt unowned evidence after lifecycle bootstrap"
                )
            validation_params: dict[str, object] = {"budget_id": budget_id}
            unfinished_placeholders = []
            for index, batch_id in enumerate(sorted(unfinished_lineage)):
                key = f"unfinished_batch_{index}"
                validation_params[key] = batch_id
                unfinished_placeholders.append(f":{key}")
            invalid_owner_result = await session.execute(
                text(
                    f"""
                    SELECT owner_batch_id
                    FROM ({source_owners_sql}) source_owners
                    WHERE owner_batch_id IS NOT NULL
                      AND owner_batch_id NOT IN (
                        {', '.join(unfinished_placeholders)}
                      )
                      AND NOT EXISTS (
                        SELECT 1
                        FROM calibration_sync_batches b
                        WHERE b.id = source_owners.owner_batch_id
                          AND b.budget_id = :budget_id
                          AND b.state = 'completed'
                      )
                    LIMIT 1
                    """
                ),
                validation_params,
            )
            if invalid_owner_result.first() is not None:
                await session.rollback()
                raise ValueError(
                    "sync batch cannot adopt evidence outside its predecessor lineage"
                )
            adoption_params: dict[str, object] = {
                "source_sync_batch_id": source_sync_batch_id,
                "budget_id": budget_id,
            }
            adoption_placeholders = []
            for index, batch_id in enumerate(sorted(unfinished_lineage)):
                key = f"adoption_batch_{index}"
                adoption_params[key] = batch_id
                adoption_placeholders.append(f":{key}")
            adoption_predicates = [
                f"change_batch_id IN ({', '.join(adoption_placeholders)})"
            ]
            if has_unowned_source:
                adoption_predicates.append("change_batch_id IS NULL")
            adoption_where = " OR ".join(adoption_predicates)
            for table_name in ("accounts", "transactions", "categories"):
                await session.execute(
                    text(
                        f"""
                        UPDATE {table_name}
                        SET change_batch_id = :source_sync_batch_id
                        WHERE budget_id = :budget_id
                          AND ({adoption_where})
                        """
                    ),
                    adoption_params,
                )
            watermarks = tuple(
                SourceWatermark(
                    budget_id=str(row["budget_id"]),
                    resource=str(row["resource"]),
                    server_knowledge=int(str(row["server_knowledge"])),
                    synced_at=min(_as_datetime(row["synced_at"]), completed_at),
                    change_batch_id=source_sync_batch_id,
                )
                for row in rows
            )
            watermarks_json = json.dumps(
                [row.model_dump(mode="json") for row in watermarks],
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            update_result = await session.execute(
                text(
                    """
                    UPDATE calibration_sync_batches
                    SET state = 'completed',
                        completed_at = :completed_at,
                        watermarks_json = :watermarks_json,
                        watermarks_sha256 = :watermarks_sha256
                    WHERE id = :source_sync_batch_id
                      AND budget_id = :budget_id
                      AND state = 'started'
                    """
                ),
                {
                    "source_sync_batch_id": source_sync_batch_id,
                    "budget_id": budget_id,
                    "completed_at": completed_at.isoformat(),
                    "watermarks_json": watermarks_json,
                    "watermarks_sha256": canonical_content_sha256(watermarks),
                },
            )
            if cast(_RowCountResult, update_result).rowcount != 1:
                await session.rollback()
                raise ValueError("sync batch lifecycle cannot be completed")
            await session.commit()

    async def _sync_batch_lineage(
        self,
        session: AsyncSession,
        budget_id: str,
        source_sync_batch_id: str,
    ) -> tuple[set[str], bool, bool]:
        lineage: set[str] = set()
        unfinished_lineage: set[str] = set()
        has_completed = False
        has_unowned_write = False
        current: str | None = source_sync_batch_id
        while current is not None:
            if current in lineage:
                raise ValueError("sync batch predecessor lineage contains a cycle")
            lineage.add(current)
            result = await session.execute(
                text(
                    """
                    SELECT budget_id, predecessor_id, state,
                           unowned_source_dirty
                    FROM calibration_sync_batches
                    WHERE id = :batch_id
                    """
                ),
                {"batch_id": current},
            )
            row = result.mappings().one_or_none()
            if row is None or str(row["budget_id"]) != budget_id:
                raise ValueError("sync batch predecessor lineage is invalid")
            row_completed = str(row["state"]) == "completed"
            has_completed = has_completed or row_completed
            if not row_completed:
                unfinished_lineage.add(current)
            has_unowned_write = has_unowned_write or bool(
                row["unowned_source_dirty"]
            )
            if row_completed:
                break
            current = (
                str(row["predecessor_id"])
                if row["predecessor_id"] is not None
                else None
            )
        return (
            unfinished_lineage,
            has_completed,
            has_unowned_write,
        )

    async def get_scenario_revision(
        self,
        revision_id: str,
    ) -> ScenarioRevision | None:
        return await self._scenarios.get_revision(revision_id)

    async def load_source_state(
        self,
        profile: CalibrationProfile,
        scenario: WealthScenario,
        *,
        source_sync_batch_id: str,
        as_of: datetime,
    ) -> CalibrationSourceState:
        completed = await self._completed_sync_batch(source_sync_batch_id)
        if completed is None or completed[0] != profile.budget_id:
            raise ValueError("calibration requires a recorded completed sync batch")
        _, completed_at, watermarks = completed
        await self._assert_cache_has_no_incomplete_batch(profile.budget_id)
        await self._assert_cache_has_no_newer_completed_batch(
            profile.budget_id,
            source_sync_batch_id,
            completed_at,
        )
        if as_of < completed_at:
            raise ValueError("calibration capture cannot predate sync completion")
        as_of = completed_at
        await self._assert_current_checkpoint(
            profile.budget_id,
            completed_at,
            watermarks,
        )
        liquid_ids = tuple(
            sorted(
                account.id
                for account in scenario.accounts
                if account.role in LIQUID_PORTFOLIO_ROLES
            )
        )
        scenario_ids = tuple(sorted(account.id for account in scenario.accounts))
        account_ids = scenario_ids or liquid_ids
        account_rows = await self._selected_accounts(profile.budget_id, account_ids)
        if account_ids and {str(row["id"]) for row in account_rows} != set(account_ids):
            raise ValueError("calibration source is missing selected scenario accounts")
        latest_activity = await self._latest_activity(profile.budget_id, account_ids)
        valuation_rows = await self._valuation_history(account_ids)
        accounts = tuple(
            self._source_account(
                row,
                latest_activity=latest_activity.get(str(row["id"])),
                valuation_rows=valuation_rows.get(str(row["id"]), ()),
            )
            for row in account_rows
        )
        start = _lookback_start(
            as_of.date(),
            profile.policy.lookback_months,
        )
        spending_rows = await self.database.fetch_all(
            """
            SELECT t.id, t.account_id, t.date, t.amount,
                   t.change_batch_id, c.change_batch_id AS category_change_batch_id
            FROM transactions t
            JOIN accounts a ON a.id = t.account_id
            LEFT JOIN categories c ON c.id = t.category_id
            WHERE t.budget_id = :budget_id
              AND t.date >= :start_date
              AND t.date <= :end_date
              AND t.amount < 0
              AND t.transfer_account_id IS NULL
              AND t.deleted IS FALSE
              AND a.on_budget IS TRUE
              AND (c.internal IS NULL OR c.internal IS FALSE)
            ORDER BY t.id
            """,
            {
                "budget_id": profile.budget_id,
                "start_date": start.isoformat(),
                "end_date": as_of.date().isoformat(),
            },
        )
        contribution_rows = await self._contribution_rows(
            profile.budget_id,
            liquid_ids,
            start,
            as_of.date(),
        )
        covered_days = max(1, (as_of.date() - start).days + 1)
        annualizer = 365.2425 / covered_days
        annual_spending = round(
            -sum(int(str(row["amount"])) for row in spending_rows) / 1_000 * annualizer,
            2,
        )
        annual_contribution = round(
            sum(int(str(row["amount"])) for row in contribution_rows) / 1_000 * annualizer,
            2,
        )
        await self._assert_cache_has_no_incomplete_batch(profile.budget_id)
        await self._assert_cache_has_no_newer_completed_batch(
            profile.budget_id,
            source_sync_batch_id,
            completed_at,
        )
        await self._assert_current_checkpoint(
            profile.budget_id,
            completed_at,
            watermarks,
        )
        return CalibrationSourceState(
            budget_id=profile.budget_id,
            source_sync_batch_id=source_sync_batch_id,
            observed_at=as_of,
            annual_spending=annual_spending,
            annual_contribution=annual_contribution,
            spending_sha256=canonical_content_sha256(
                {
                    "query": "annual-spending-v1",
                    "start": start,
                    "end": as_of.date(),
                    "rows": spending_rows,
                }
            ),
            contribution_sha256=canonical_content_sha256(
                {
                    "query": "annual-contribution-v1",
                    "account_ids": liquid_ids,
                    "start": start,
                    "end": as_of.date(),
                    "rows": contribution_rows,
                }
            ),
            accounts=accounts,
            watermarks=watermarks,
        )

    async def _assert_current_checkpoint(
        self,
        budget_id: str,
        completed_at: datetime,
        watermarks: tuple[SourceWatermark, ...],
    ) -> None:
        current_watermarks = await self.database.fetch_all(
            """
            SELECT resource, server_knowledge, synced_at, change_batch_id
            FROM sync_state
            WHERE budget_id = :budget_id
            ORDER BY resource
            """,
            {"budget_id": budget_id},
        )
        expected_checkpoint = tuple(
            (
                row.resource,
                row.server_knowledge,
                row.synced_at,
                row.change_batch_id,
            )
            for row in watermarks
        )
        current_checkpoint = tuple(
            (
                str(row["resource"]),
                int(str(row["server_knowledge"])),
                min(_as_datetime(row["synced_at"]), completed_at),
                str(row["change_batch_id"] or ""),
            )
            for row in current_watermarks
        )
        if current_checkpoint != expected_checkpoint:
            raise ValueError(
                "completed sync batch no longer matches the current cache checkpoint"
            )

    async def _assert_cache_has_no_incomplete_batch(
        self,
        budget_id: str,
    ) -> None:
        in_progress = await self.database.fetch_all(
            """
            SELECT change_batch_id
            FROM accounts
            WHERE budget_id = :budget_id
              AND NOT EXISTS (
                SELECT 1
                FROM calibration_sync_batches b
                WHERE b.id = accounts.change_batch_id
                  AND b.budget_id = accounts.budget_id
                  AND b.state = 'completed'
              )
            UNION
            SELECT change_batch_id
            FROM transactions
            WHERE budget_id = :budget_id
              AND NOT EXISTS (
                SELECT 1
                FROM calibration_sync_batches b
                WHERE b.id = transactions.change_batch_id
                  AND b.budget_id = transactions.budget_id
                  AND b.state = 'completed'
              )
            UNION
            SELECT change_batch_id
            FROM categories
            WHERE budget_id = :budget_id
              AND NOT EXISTS (
                SELECT 1
                FROM calibration_sync_batches b
                WHERE b.id = categories.change_batch_id
                  AND b.budget_id = categories.budget_id
                  AND b.state = 'completed'
              )
            LIMIT 1
            """,
            {"budget_id": budget_id},
        )
        if in_progress:
            raise ValueError("calibration cache contains an incomplete sync batch")

    async def _assert_cache_has_no_newer_completed_batch(
        self,
        budget_id: str,
        source_sync_batch_id: str,
        completed_at: datetime,
    ) -> None:
        rows = await self.database.fetch_all(
            """
            SELECT DISTINCT b.id, b.completed_at
            FROM calibration_sync_batches b
            JOIN (
              SELECT change_batch_id
              FROM accounts
              WHERE budget_id = :budget_id
              UNION
              SELECT change_batch_id
              FROM transactions
              WHERE budget_id = :budget_id
              UNION
              SELECT change_batch_id
              FROM categories
              WHERE budget_id = :budget_id
            ) source_batches ON source_batches.change_batch_id = b.id
            WHERE b.budget_id = :budget_id
              AND b.state = 'completed'
              AND b.id != :source_sync_batch_id
            """,
            {
                "budget_id": budget_id,
                "source_sync_batch_id": source_sync_batch_id,
            },
        )
        if any(
            _as_datetime(row["completed_at"]) >= completed_at
            for row in rows
        ):
            raise ValueError(
                "calibration cache contains a newer completed sync batch"
            )

    async def _completed_sync_batch(
        self,
        source_sync_batch_id: str,
    ) -> tuple[str, datetime, tuple[SourceWatermark, ...]] | None:
        rows = await self.database.fetch_all(
            """
            SELECT budget_id, completed_at, watermarks_json, watermarks_sha256
            FROM calibration_sync_batches
            WHERE id = :batch_id
              AND state = 'completed'
            """,
            {"batch_id": source_sync_batch_id},
        )
        if not rows:
            return None
        row = rows[0]
        if (
            row["completed_at"] is None
            or row["watermarks_json"] is None
            or row["watermarks_sha256"] is None
        ):
            raise ValueError("completed sync batch proof is incomplete")
        decoded = json.loads(str(row["watermarks_json"]))
        watermarks = tuple(SourceWatermark.model_validate(item) for item in decoded)
        if canonical_content_sha256(watermarks) != str(row["watermarks_sha256"]):
            raise ValueError("completed sync batch watermark hash mismatch")
        if any(watermark.change_batch_id != source_sync_batch_id for watermark in watermarks):
            raise ValueError("completed sync batch watermark identity mismatch")
        return str(row["budget_id"]), _as_datetime(row["completed_at"]), watermarks

    async def get_snapshot_for_batch(
        self,
        profile_id: str,
        source_sync_batch_id: str,
    ) -> CalibrationSnapshot | None:
        rows = await self.database.fetch_all(
            """
            SELECT id
            FROM calibration_snapshots
            WHERE profile_id = :profile_id
              AND source_sync_batch_id = :source_sync_batch_id
            """,
            {
                "profile_id": profile_id,
                "source_sync_batch_id": source_sync_batch_id,
            },
        )
        return await self._get_snapshot(str(rows[0]["id"])) if rows else None

    async def get_latest_snapshot(
        self,
        profile_id: str,
    ) -> CalibrationSnapshot | None:
        rows = await self.database.fetch_all(
            """
            SELECT s.id
            FROM calibration_profile_heads h
            JOIN calibration_snapshots s ON s.id = h.snapshot_id
            WHERE h.profile_id = :profile_id
            """,
            {"profile_id": profile_id},
        )
        return await self._get_snapshot(str(rows[0]["id"])) if rows else None

    async def _get_snapshot(self, snapshot_id: str) -> CalibrationSnapshot | None:
        rows = await self.database.fetch_all(
            """
            SELECT id, profile_id, previous_snapshot_id, source_sync_batch_id,
                   manifest_sha256, manifest_json, created_at
            FROM calibration_snapshots
            WHERE id = :snapshot_id
            """,
            {"snapshot_id": snapshot_id},
        )
        if not rows:
            return None
        row = rows[0]
        snapshot = CalibrationSnapshot(
            id=str(row["id"]),
            manifest=CalibrationSnapshotManifest.model_validate_json(str(row["manifest_json"])),
            manifest_sha256=str(row["manifest_sha256"]),
            created_at=_as_datetime(row["created_at"]),
        )
        if (
            str(row["profile_id"]) != snapshot.manifest.profile_id
            or (
                str(row["previous_snapshot_id"])
                if row["previous_snapshot_id"] is not None
                else None
            )
            != snapshot.manifest.previous_snapshot_id
            or str(row["source_sync_batch_id"])
            != snapshot.manifest.source_sync_batch_id
        ):
            raise ValueError("persisted calibration snapshot columns do not match manifest")
        observation_rows = await self.database.fetch_all(
            """
            SELECT id, kind, subject_id, content_sha256,
                   observation_json, observed_at
            FROM calibration_observations
            WHERE snapshot_id = :snapshot_id
            ORDER BY id, content_sha256
            """,
            {"snapshot_id": snapshot_id},
        )
        persisted_observations = tuple(
            CalibrationObservation.model_validate_json(str(child["observation_json"]))
            for child in observation_rows
        )
        if persisted_observations != tuple(
            sorted(snapshot.manifest.observations, key=lambda item: (item.id, item.content_sha256))
        ):
            raise ValueError("persisted calibration observations do not match manifest")
        for child, observation in zip(
            observation_rows,
            persisted_observations,
            strict=True,
        ):
            if (
                str(child["id"]) != observation.id
                or str(child["kind"]) != observation.kind.value
                or str(child["subject_id"]) != observation.subject_id
                or str(child["content_sha256"]) != observation.content_sha256
                or _as_datetime(child["observed_at"]) != observation.observed_at
            ):
                raise ValueError("persisted calibration observation columns are invalid")
        event_rows = await self.database.fetch_all(
            """
            SELECT content_sha256, kind, subject_id, material, event_json
            FROM calibration_drift_events
            WHERE snapshot_id = :snapshot_id
            ORDER BY content_sha256
            """,
            {"snapshot_id": snapshot_id},
        )
        persisted_events = tuple(
            DriftEvent.model_validate_json(str(child["event_json"]))
            for child in event_rows
        )
        if persisted_events != tuple(
            sorted(snapshot.manifest.drift_events, key=lambda item: item.content_sha256)
        ):
            raise ValueError("persisted calibration drift events do not match manifest")
        for child, event in zip(event_rows, persisted_events, strict=True):
            if (
                str(child["content_sha256"]) != event.content_sha256
                or str(child["kind"]) != event.kind.value
                or str(child["subject_id"]) != event.subject_id
                or bool(child["material"]) is not event.material
            ):
                raise ValueError("persisted calibration drift columns are invalid")
        return snapshot

    async def get_snapshot(self, snapshot_id: str) -> CalibrationSnapshot | None:
        """Return and verify one immutable snapshot by identity."""
        return await self._get_snapshot(snapshot_id)

    async def prepare_run_recovery(self) -> None:
        """Return interrupted work to the durable accepted queue."""
        async with self.database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE calibration_runs
                    SET state = :accepted,
                        started_at = NULL
                    WHERE state = :running
                    """
                ),
                {
                    "accepted": CalibrationRunState.ACCEPTED.value,
                    "running": CalibrationRunState.RUNNING.value,
                },
            )
            await session.commit()

    async def create_snapshot(self, snapshot: CalibrationSnapshot) -> None:
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_snapshots (
                          id, profile_id, previous_snapshot_id,
                          source_sync_batch_id, manifest_sha256,
                          manifest_json, created_at
                        ) VALUES (
                          :id, :profile_id, :previous_snapshot_id,
                          :source_sync_batch_id, :manifest_sha256,
                          :manifest_json, :created_at
                        )
                        """
                    ),
                    {
                        "id": snapshot.id,
                        "profile_id": snapshot.manifest.profile_id,
                        "previous_snapshot_id": (snapshot.manifest.previous_snapshot_id),
                        "source_sync_batch_id": (snapshot.manifest.source_sync_batch_id),
                        "manifest_sha256": snapshot.manifest_sha256,
                        "manifest_json": snapshot.manifest.model_dump_json(),
                        "created_at": snapshot.created_at.isoformat(),
                    },
                )
                head_result = cast(
                    _RowCountResult,
                    await session.execute(
                        text(
                            """
                            UPDATE calibration_profile_heads
                            SET snapshot_id = :snapshot_id
                            WHERE profile_id = :profile_id
                              AND (
                                snapshot_id = :previous_snapshot_id
                                OR (
                                  snapshot_id IS NULL
                                  AND :previous_snapshot_id IS NULL
                                )
                              )
                            """
                        ),
                        {
                            "snapshot_id": snapshot.id,
                            "profile_id": snapshot.manifest.profile_id,
                            "previous_snapshot_id": snapshot.manifest.previous_snapshot_id,
                        },
                    ),
                )
                if head_result.rowcount != 1:
                    await session.rollback()
                    existing = await self.get_snapshot_for_batch(
                        snapshot.manifest.profile_id,
                        snapshot.manifest.source_sync_batch_id,
                    )
                    if existing == snapshot:
                        return
                    raise ValueError(
                        "calibration snapshot chain advanced during capture; retry the batch"
                    )
                for observation in snapshot.manifest.observations:
                    await session.execute(
                        text(
                            """
                            INSERT INTO calibration_observations (
                              id, snapshot_id, kind, subject_id,
                              content_sha256, observation_json, observed_at
                            ) VALUES (
                              :id, :snapshot_id, :kind, :subject_id,
                              :content_sha256, :observation_json, :observed_at
                            )
                            """
                        ),
                        {
                            "id": observation.id,
                            "snapshot_id": snapshot.id,
                            "kind": observation.kind.value,
                            "subject_id": observation.subject_id,
                            "content_sha256": observation.content_sha256,
                            "observation_json": observation.model_dump_json(),
                            "observed_at": observation.observed_at.isoformat(),
                        },
                    )
                for event in snapshot.manifest.drift_events:
                    await session.execute(
                        text(
                            """
                            INSERT INTO calibration_drift_events (
                              snapshot_id, content_sha256, kind,
                              subject_id, material, event_json
                            ) VALUES (
                              :snapshot_id, :content_sha256, :kind,
                              :subject_id, :material, :event_json
                            )
                            """
                        ),
                        {
                            "snapshot_id": snapshot.id,
                            "content_sha256": event.content_sha256,
                            "kind": event.kind.value,
                            "subject_id": event.subject_id,
                            "material": event.material,
                            "event_json": event.model_dump_json(),
                        },
                    )
                await session.commit()
        except IntegrityError as error:
            existing = await self.get_snapshot_for_batch(
                snapshot.manifest.profile_id,
                snapshot.manifest.source_sync_batch_id,
            )
            if existing != snapshot:
                raise ValueError("calibration snapshot identity conflict") from error

    async def create_run(self, run: CalibrationRun) -> CalibrationRun:
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_runs (
                          id, snapshot_id, state, created_at
                        ) VALUES (
                          :id, :snapshot_id, :state, :created_at
                        )
                        """
                    ),
                    {
                        "id": run.id,
                        "snapshot_id": run.snapshot_id,
                        "state": run.state.value,
                        "created_at": run.created_at.isoformat(),
                    },
                )
                await session.commit()
        except IntegrityError:
            existing = await self.get_run(run.id)
            if existing is None:
                raise
            return existing
        created = await self.get_run(run.id)
        if created is None:
            raise RuntimeError("created calibration run could not be read")
        return created

    async def get_run(self, run_id: str) -> CalibrationRun | None:
        rows = await self.database.fetch_all(
            """
            SELECT id, snapshot_id, state, result_sha256, result_json,
                   error_code, created_at, started_at, completed_at
            FROM calibration_runs
            WHERE id = :run_id
            """,
            {"run_id": run_id},
        )
        return _run_from_row(rows[0]) if rows else None

    async def list_runs(
        self,
        *,
        state: CalibrationRunState | None = None,
        limit: int = 100,
    ) -> tuple[CalibrationRun, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("calibration run limit must be between 1 and 1000")
        where = ""
        params: dict[str, object] = {"limit": limit}
        if state is not None:
            where = "WHERE state = :state"
            params["state"] = state.value
        rows = await self.database.fetch_all(
            f"""
            SELECT id, snapshot_id, state, result_sha256, result_json,
                   error_code, created_at, started_at, completed_at
            FROM calibration_runs
            {where}
            ORDER BY created_at, id
            LIMIT :limit
            """,
            params,
        )
        return tuple(_run_from_row(row) for row in rows)

    async def list_profile_runs(
        self,
        profile_id: str,
        *,
        limit: int = 100,
    ) -> tuple[CalibrationRun, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("calibration run limit must be between 1 and 1000")
        rows = await self.database.fetch_all(
            """
            SELECT r.id, r.snapshot_id, r.state, r.result_sha256, r.result_json,
                   r.error_code, r.created_at, r.started_at, r.completed_at
            FROM calibration_runs r
            JOIN calibration_snapshots s ON s.id = r.snapshot_id
            WHERE s.profile_id = :profile_id
            ORDER BY r.created_at DESC, r.id DESC
            LIMIT :limit
            """,
            {"profile_id": profile_id, "limit": limit},
        )
        return tuple(_run_from_row(row) for row in rows)

    async def record_capture_failure(
        self,
        failure: CalibrationCaptureFailure,
    ) -> None:
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_capture_failures (
                          id, profile_id, source_sync_batch_id,
                          error_code, created_at
                        ) VALUES (
                          :id, :profile_id, :source_sync_batch_id,
                          :error_code, :created_at
                        )
                        """
                    ),
                    failure.model_dump(mode="json"),
                )
                await session.commit()
        except IntegrityError:
            return

    async def list_capture_failures(
        self,
        profile_id: str,
        *,
        limit: int = 100,
    ) -> tuple[CalibrationCaptureFailure, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("calibration failure limit must be between 1 and 1000")
        rows = await self.database.fetch_all(
            """
            SELECT id, profile_id, source_sync_batch_id, error_code, created_at
            FROM calibration_capture_failures
            WHERE profile_id = :profile_id
            ORDER BY created_at DESC, id DESC
            LIMIT :limit
            """,
            {"profile_id": profile_id, "limit": limit},
        )
        return tuple(
            CalibrationCaptureFailure(
                id=str(row["id"]),
                profile_id=str(row["profile_id"]),
                source_sync_batch_id=str(row["source_sync_batch_id"]),
                error_code=str(row["error_code"]),
                created_at=_as_datetime(row["created_at"]),
            )
            for row in rows
        )

    async def mark_run_running(
        self,
        run_id: str,
        at: datetime,
    ) -> CalibrationRun | None:
        changed = await self._transition_run(
            run_id,
            expected=CalibrationRunState.ACCEPTED,
            state=CalibrationRunState.RUNNING,
            values={"started_at": at.isoformat()},
        )
        return await self.get_run(run_id) if changed else None

    async def mark_run_succeeded(
        self,
        run_id: str,
        *,
        result: dict[str, object],
        result_sha256: str,
        completed_at: datetime,
    ) -> CalibrationRun | None:
        if canonical_content_sha256(result) != result_sha256:
            raise ValueError("calibration result hash mismatch")
        await self._transition_run(
            run_id,
            expected=CalibrationRunState.RUNNING,
            state=CalibrationRunState.SUCCEEDED,
            values={
                "result_json": json.dumps(
                    result,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "result_sha256": result_sha256,
                "completed_at": completed_at.isoformat(),
            },
        )
        return await self.get_run(run_id)

    async def mark_run_failed(
        self,
        run_id: str,
        *,
        error_code: str,
        completed_at: datetime,
    ) -> CalibrationRun | None:
        await self._transition_run(
            run_id,
            expected=CalibrationRunState.RUNNING,
            state=CalibrationRunState.FAILED,
            values={
                "error_code": error_code[:64],
                "completed_at": completed_at.isoformat(),
            },
        )
        return await self.get_run(run_id)

    async def store_attribution(
        self,
        run_id: str,
        report: AttributionReport,
        *,
        created_at: datetime,
    ) -> None:
        report_id = str(
            uuid5(
                CALIBRATION_NAMESPACE,
                f"attribution:{run_id}:{report.content_sha256}",
            )
        )
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO calibration_attribution_reports (
                          id, run_id, content_sha256, report_json, created_at
                        ) VALUES (
                          :id, :run_id, :content_sha256, :report_json, :created_at
                        )
                        """
                    ),
                    {
                        "id": report_id,
                        "run_id": run_id,
                        "content_sha256": report.content_sha256,
                        "report_json": report.model_dump_json(),
                        "created_at": created_at.isoformat(),
                    },
                )
                await session.commit()
        except IntegrityError:
            rows = await self.database.fetch_all(
                """
                SELECT content_sha256, report_json
                FROM calibration_attribution_reports
                WHERE run_id = :run_id
                """,
                {"run_id": run_id},
            )
            if (
                not rows
                or rows[0]["content_sha256"] != report.content_sha256
                or AttributionReport.model_validate_json(str(rows[0]["report_json"])) != report
            ):
                raise ValueError("calibration attribution identity conflict")

    async def _transition_run(
        self,
        run_id: str,
        *,
        expected: CalibrationRunState,
        state: CalibrationRunState,
        values: dict[str, object],
    ) -> bool:
        assignments = ["state = :state", *[f"{key} = :{key}" for key in values]]
        async with self.database.session_factory() as session:
            result = await session.execute(
                text(
                    f"""
                    UPDATE calibration_runs
                    SET {", ".join(assignments)}
                    WHERE id = :run_id
                      AND state = :expected
                    """
                ),
                {
                    "run_id": run_id,
                    "expected": expected.value,
                    "state": state.value,
                    **values,
                },
            )
            await session.commit()
        return cast(_RowCountResult, result).rowcount == 1

    async def reconcile_alerts(
        self,
        profile_id: str,
        snapshot: CalibrationSnapshot,
        *,
        at: datetime,
    ) -> tuple[CalibrationAlert, ...]:
        head = await self.database.fetch_all(
            """
            SELECT snapshot_id
            FROM calibration_profile_heads
            WHERE profile_id = :profile_id
            """,
            {"profile_id": profile_id},
        )
        if (
            not head
            or head[0]["snapshot_id"] is None
            or str(head[0]["snapshot_id"]) != snapshot.id
        ):
            return await self.list_alerts(profile_id)
        material = {
            (event.kind.value, event.subject_id): event
            for event in snapshot.manifest.drift_events
            if event.material
        }
        current = {
            (alert.drift_kind.value, alert.subject_id): alert
            for alert in await self.list_alerts(profile_id)
        }
        for identity, event in material.items():
            alert = current.get(identity)
            if alert is None:
                fingerprint = hashlib.sha256(
                    f"{profile_id}:{identity[0]}:{identity[1]}".encode()
                ).hexdigest()
                alert_id = str(
                    uuid5(
                        CALIBRATION_NAMESPACE,
                        f"alert:{fingerprint}",
                    )
                )
                try:
                    async with self.database.session_factory() as session:
                        await session.execute(
                            text(
                                """
                                INSERT INTO calibration_alerts (
                                  id, profile_id, drift_kind, subject_id,
                                  fingerprint_sha256, created_at
                                ) VALUES (
                                  :id, :profile_id, :drift_kind, :subject_id,
                                  :fingerprint_sha256, :created_at
                                )
                                """
                            ),
                            {
                                "id": alert_id,
                                "profile_id": profile_id,
                                "drift_kind": event.kind.value,
                                "subject_id": event.subject_id,
                                "fingerprint_sha256": fingerprint,
                                "created_at": at.isoformat(),
                            },
                        )
                        await session.execute(
                            text(
                                """
                                INSERT INTO calibration_alert_heads (
                                  alert_id, event_id
                                ) VALUES (
                                  :alert_id, NULL
                                )
                                """
                            ),
                            {"alert_id": alert_id},
                        )
                        await session.commit()
                except IntegrityError:
                    pass
                await self.transition_alert(
                    alert_id,
                    profile_id,
                    snapshot.id,
                    AlertState.OPEN,
                    at=at,
                )
            elif alert.state is AlertState.RESOLVED:
                await self.transition_alert(
                    alert.id,
                    profile_id,
                    snapshot.id,
                    AlertState.OPEN,
                    at=at,
                )
        for identity, alert in current.items():
            if identity not in material and alert.state is not AlertState.RESOLVED:
                await self.transition_alert(
                    alert.id,
                    profile_id,
                    snapshot.id,
                    AlertState.RESOLVED,
                    at=at,
                )
        return await self.list_alerts(profile_id)

    async def list_alerts(
        self,
        profile_id: str,
    ) -> tuple[CalibrationAlert, ...]:
        rows = await self.database.fetch_all(
            """
            SELECT a.id, a.profile_id, a.drift_kind, a.subject_id,
                   a.created_at, e.state, e.created_at AS state_changed_at
            FROM calibration_alerts a
            JOIN calibration_alert_heads h ON h.alert_id = a.id
            JOIN calibration_alert_events e ON e.id = h.event_id
            WHERE a.profile_id = :profile_id
            ORDER BY a.created_at, a.id
            """,
            {"profile_id": profile_id},
        )
        return tuple(
            CalibrationAlert(
                id=str(row["id"]),
                profile_id=str(row["profile_id"]),
                drift_kind=DriftKind(str(row["drift_kind"])),
                subject_id=str(row["subject_id"]),
                state=AlertState(str(row["state"])),
                created_at=_as_datetime(row["created_at"]),
                state_changed_at=_as_datetime(row["state_changed_at"]),
            )
            for row in rows
        )

    async def transition_alert(
        self,
        alert_id: str,
        profile_id: str,
        snapshot_id: str,
        state: AlertState,
        *,
        at: datetime,
    ) -> CalibrationAlert | None:
        event_id = str(
            uuid5(
                CALIBRATION_NAMESPACE,
                f"alert-event:{alert_id}:{snapshot_id}:{state.value}",
            )
        )
        for _ in range(3):
            ownership = await self.database.fetch_all(
                """
                SELECT ah.event_id, e.state
                FROM calibration_alerts a
                JOIN calibration_snapshots s ON s.id = :snapshot_id
                JOIN calibration_profile_heads ph
                  ON ph.profile_id = a.profile_id
                 AND ph.snapshot_id = s.id
                JOIN calibration_alert_heads ah ON ah.alert_id = a.id
                LEFT JOIN calibration_alert_events e ON e.id = ah.event_id
                WHERE a.id = :alert_id
                  AND a.profile_id = :profile_id
                  AND s.profile_id = a.profile_id
                """,
                {
                    "alert_id": alert_id,
                    "profile_id": profile_id,
                    "snapshot_id": snapshot_id,
                },
            )
            if not ownership:
                return None
            expected_event_id = (
                str(ownership[0]["event_id"])
                if ownership[0]["event_id"] is not None
                else None
            )
            current_state = (
                AlertState(str(ownership[0]["state"]))
                if ownership[0]["state"] is not None
                else None
            )
            if current_state is None:
                if state is not AlertState.OPEN:
                    raise ValueError(
                        "calibration alert must open before later transitions"
                    )
            elif current_state is state:
                alerts = await self.list_alerts(profile_id)
                return next(
                    (alert for alert in alerts if alert.id == alert_id),
                    None,
                )
            elif (
                current_state,
                state,
            ) not in {
                (AlertState.OPEN, AlertState.ACKNOWLEDGED),
                (AlertState.OPEN, AlertState.RESOLVED),
                (AlertState.ACKNOWLEDGED, AlertState.RESOLVED),
                (AlertState.RESOLVED, AlertState.OPEN),
            }:
                raise ValueError("illegal calibration alert lifecycle transition")
            try:
                async with self.database.session_factory() as session:
                    await session.execute(
                        text(
                            """
                            INSERT INTO calibration_alert_events (
                              id, alert_id, snapshot_id, state, created_at
                            ) VALUES (
                              :id, :alert_id, :snapshot_id, :state, :created_at
                            )
                            """
                        ),
                        {
                            "id": event_id,
                            "alert_id": alert_id,
                            "snapshot_id": snapshot_id,
                            "state": state.value,
                            "created_at": at.isoformat(),
                        },
                    )
                    head_result = cast(
                        _RowCountResult,
                        await session.execute(
                            text(
                                """
                                UPDATE calibration_alert_heads
                                SET event_id = :event_id
                                WHERE alert_id = :alert_id
                                  AND (
                                    event_id = :expected_event_id
                                    OR (
                                      event_id IS NULL
                                      AND :expected_event_id IS NULL
                                    )
                                  )
                                """
                            ),
                            {
                                "event_id": event_id,
                                "alert_id": alert_id,
                                "expected_event_id": expected_event_id,
                            },
                        ),
                    )
                    if head_result.rowcount != 1:
                        await session.rollback()
                        continue
                    await session.commit()
            except IntegrityError:
                continue
            alerts = await self.list_alerts(profile_id)
            return next((alert for alert in alerts if alert.id == alert_id), None)
        raise RuntimeError("calibration alert transition lost concurrent CAS")

    async def _selected_accounts(
        self,
        budget_id: str,
        account_ids: tuple[str, ...],
    ) -> list[dict[str, object]]:
        if not account_ids:
            return []
        placeholders = ", ".join(f":account_{index}" for index in range(len(account_ids)))
        return await self.database.fetch_all(
            f"""
            SELECT id, type, balance, last_reconciled_at,
                   debt_original_balance, change_batch_id
            FROM accounts
            WHERE budget_id = :budget_id
              AND id IN ({placeholders})
              AND deleted IS FALSE
            ORDER BY id
            """,
            {
                "budget_id": budget_id,
                **{f"account_{index}": account_id for index, account_id in enumerate(account_ids)},
            },
        )

    async def _latest_activity(
        self,
        budget_id: str,
        account_ids: tuple[str, ...],
    ) -> dict[str, datetime]:
        if not account_ids:
            return {}
        placeholders = ", ".join(f":account_{index}" for index in range(len(account_ids)))
        rows = await self.database.fetch_all(
            f"""
            SELECT account_id, MAX(date) AS latest_date
            FROM transactions
            WHERE budget_id = :budget_id
              AND account_id IN ({placeholders})
              AND deleted IS FALSE
            GROUP BY account_id
            """,
            {
                "budget_id": budget_id,
                **{f"account_{index}": account_id for index, account_id in enumerate(account_ids)},
            },
        )
        return {
            str(row["account_id"]): datetime.combine(
                date.fromisoformat(str(row["latest_date"])),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            for row in rows
            if row["latest_date"] is not None
        }

    async def _valuation_history(
        self,
        account_ids: tuple[str, ...],
    ) -> dict[str, tuple[dict[str, object], ...]]:
        if not account_ids:
            return {}
        placeholders = ", ".join(f":account_{index}" for index in range(len(account_ids)))
        rows = await self.database.fetch_all(
            f"""
            SELECT account_id, valuation_date, balance_milliunits,
                   observed_at, reviewed
            FROM account_valuation_snapshots
            WHERE account_id IN ({placeholders})
            ORDER BY account_id, valuation_date DESC, observed_at DESC
            """,
            {f"account_{index}": account_id for index, account_id in enumerate(account_ids)},
        )
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(str(row["account_id"]), []).append(row)
        return {key: tuple(value) for key, value in grouped.items()}

    async def _contribution_rows(
        self,
        budget_id: str,
        account_ids: tuple[str, ...],
        start: date,
        end: date,
    ) -> list[dict[str, object]]:
        if not account_ids:
            return []
        placeholders = ", ".join(f":account_{index}" for index in range(len(account_ids)))
        return await self.database.fetch_all(
            f"""
            SELECT id, account_id, date, amount, change_batch_id
            FROM transactions
            WHERE budget_id = :budget_id
              AND account_id IN ({placeholders})
              AND date >= :start_date
              AND date <= :end_date
              AND amount > 0
              AND transfer_account_id IS NOT NULL
              AND transfer_account_id NOT IN ({placeholders})
              AND deleted IS FALSE
            ORDER BY id
            """,
            {
                "budget_id": budget_id,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                **{f"account_{index}": account_id for index, account_id in enumerate(account_ids)},
            },
        )

    def _source_account(
        self,
        row: dict[str, object],
        *,
        latest_activity: datetime | None,
        valuation_rows: tuple[dict[str, object], ...],
    ) -> CalibrationSourceAccount:
        account_id = str(row["id"])
        balance_milliunits = int(str(row["balance"] or 0))
        latest_reviewed = next(
            (
                _as_datetime(item["observed_at"])
                for item in valuation_rows
                if bool(item["reviewed"])
            ),
            None,
        )
        unchanged_since: datetime | None = None
        for item in valuation_rows:
            if int(str(item["balance_milliunits"])) != balance_milliunits:
                break
            unchanged_since = datetime.combine(
                date.fromisoformat(str(item["valuation_date"])),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
        account_type = str(row["type"] or "").casefold()
        is_debt = row["debt_original_balance"] is not None or account_type in {
            "creditcard",
            "mortgage",
            "lineofcredit",
        }
        material = {
            "account_id": account_id,
            "balance_milliunits": balance_milliunits,
            "latest_activity_at": latest_activity,
            "last_reconciled_at": row["last_reconciled_at"],
            "latest_reviewed_at": latest_reviewed,
            "unchanged_since": unchanged_since,
            "change_batch_id": row["change_batch_id"],
        }
        return CalibrationSourceAccount(
            account_id=account_id,
            balance=round(balance_milliunits / 1_000, 2),
            debt_balance=(round(max(0, -balance_milliunits) / 1_000, 2) if is_debt else None),
            latest_activity_at=latest_activity,
            last_reconciled_at=(
                _as_datetime(row["last_reconciled_at"])
                if row["last_reconciled_at"] is not None
                else None
            ),
            latest_reviewed_at=latest_reviewed,
            unchanged_since=unchanged_since,
            content_sha256=canonical_content_sha256(material),
        )


def _run_from_row(row: dict[str, object]) -> CalibrationRun:
    result = (
        cast(dict[str, object], json.loads(str(row["result_json"])))
        if row["result_json"] is not None
        else None
    )
    if result is not None and canonical_content_sha256(result) != row["result_sha256"]:
        raise ValueError("persisted calibration result hash mismatch")
    return CalibrationRun(
        id=str(row["id"]),
        snapshot_id=str(row["snapshot_id"]),
        state=CalibrationRunState(str(row["state"])),
        result_sha256=(str(row["result_sha256"]) if row["result_sha256"] is not None else None),
        result=result,
        error_code=(str(row["error_code"]) if row["error_code"] is not None else None),
        created_at=_as_datetime(row["created_at"]),
        started_at=(_as_datetime(row["started_at"]) if row["started_at"] is not None else None),
        completed_at=(
            _as_datetime(row["completed_at"]) if row["completed_at"] is not None else None
        ),
    )


def _lookback_start(end: date, months: int) -> date:
    year = end.year
    month = end.month - months + 1
    while month <= 0:
        year -= 1
        month += 12
    return date(year, month, 1)


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        resolved = value
    else:
        text_value = str(value).replace("Z", "+00:00")
        resolved = datetime.fromisoformat(text_value)
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)
