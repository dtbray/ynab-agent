from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4

import pytest
from sqlalchemy import text

from ynab_agent.db.calibration import SqlCalibrationRepository
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.notifications import SqlTransactionNotificationStore
from ynab_agent.db.scenario_comparison import SqlScenarioRevisionRepository
from ynab_agent.db.sync import SqlSyncStore
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.planning.historical import (
    HistoricalGapPolicy,
    HistoricalOrderPolicy,
    HistoricalSeries,
)
from ynab_agent.services.calibration import (
    AlertState,
    CalibrationCaptureFailure,
    CalibrationProfile,
    CalibrationProfileState,
    CalibrationRunState,
    CalibrationService,
    ReviewedAllocation,
    build_calibration_snapshot,
    default_calibration_policy,
)
from ynab_agent.services.calibration_models import (
    DriftKind,
    FrozenFloatMap,
    canonical_content_sha256,
)
from ynab_agent.services.calibration_execution import run_calibration_snapshot
from ynab_agent.services.planner_jobs import (
    HistoricalDatasetSnapshot,
    PlannerExecutionPolicy,
)
from ynab_agent.services.scenario_comparison import ScenarioComparisonService
from ynab_agent.services.wealth import ResolvedStartingPortfolio
from ynab_agent.workers.calibration import CalibrationWorker


NOW = datetime(2090, 7, 31, 12, tzinfo=timezone.utc)


class ExplicitPortfolioResolver:
    async def resolve_starting_portfolio_with_provenance(
        self,
        scenario: WealthScenario,
    ) -> ResolvedStartingPortfolio:
        assert scenario.starting_portfolio is not None
        return ResolvedStartingPortfolio(
            value=scenario.starting_portfolio,
            provenance=ValuationProvenance(source="calibration_fixture"),
        )


def _runner(payload_json: str) -> str:
    return run_calibration_snapshot(payload_json)


async def _seed_source(database: DatabaseManager) -> None:
    async with database.session_factory() as session:
        await session.execute(
            text(
                """
            INSERT INTO budgets (id, name)
            VALUES ('budget-1', 'Private budget')
            """
            )
        )
        await session.execute(
            text(
                """
            INSERT INTO accounts (
              id, budget_id, name, type, on_budget, balance, deleted
            ) VALUES (
              'brokerage', 'budget-1', 'Brokerage', 'otherAsset',
              FALSE, 300000000, FALSE
            )
            """
            )
        )
        await session.execute(
            text(
                """
                INSERT INTO accounts (
                  id, budget_id, name, type, on_budget, balance,
                  debt_original_balance, deleted
                ) VALUES (
                  'mortgage', 'budget-1', 'Mortgage', 'mortgage', FALSE,
                  -100000000, 100000000, FALSE
                )
                """
            )
        )
        await session.execute(
            text(
                """
            INSERT INTO accounts (
              id, budget_id, name, type, on_budget, balance, deleted
            ) VALUES (
              'checking', 'budget-1', 'Checking', 'checking', TRUE, 100000, FALSE
            )
            """
            )
        )
        await session.execute(
            text(
                """
            INSERT INTO categories (
              id, budget_id, name, hidden, internal, deleted
            ) VALUES (
              'groceries', 'budget-1', 'Groceries', FALSE, FALSE, FALSE
            )
            """
            )
        )
        await session.execute(
            text(
                """
            INSERT INTO transactions (
              id, budget_id, account_id, category_id,
              date, amount, deleted
            ) VALUES (
              'txn-1', 'budget-1', 'checking', 'groceries',
              '2090-07-01', -60000000, FALSE
            )
            """
            )
        )
        await session.execute(
            text(
                """
            INSERT INTO sync_state (
              budget_id, resource, server_knowledge, synced_at, change_batch_id
            ) VALUES (
              'budget-1', 'accounts', 42, '2090-07-31T11:59:00+00:00',
              'batch-42'
            )
            """
            )
        )
        await session.commit()


async def _tag_sync_batch(
    database: DatabaseManager,
    repository: SqlCalibrationRepository,
    batch_id: str,
    *,
    started_at: datetime = NOW - timedelta(minutes=1),
) -> None:
    await repository.begin_sync_batch(
        "budget-1",
        batch_id,
        started_at=started_at,
    )
    async with database.session_factory() as session:
        await session.execute(
            text(
                """
                UPDATE sync_state
                SET change_batch_id = :batch_id
                WHERE budget_id = 'budget-1'
                """
            ),
            {"batch_id": batch_id},
        )
        await session.commit()


@pytest.mark.asyncio
async def test_later_partial_sync_supersedes_crashed_completed_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'sync-recovery.db'}")
    try:
        await database.initialize()
        await _seed_source(database)
        repository = SqlCalibrationRepository(database)
        store = SqlSyncStore(database)
        crash_batch = "crashed-after-final-checkpoint"
        recovery_batch = "later-partial-delta"
        await repository.begin_sync_batch(
            "budget-1",
            crash_batch,
            started_at=NOW - timedelta(minutes=2),
        )
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE sync_state
                    SET server_knowledge = 43,
                        change_batch_id = :batch_id
                    WHERE budget_id = 'budget-1'
                    """
                ),
                {"batch_id": crash_batch},
            )
            await session.commit()

        await repository.begin_sync_batch(
            "budget-1",
            recovery_batch,
            started_at=NOW - timedelta(minutes=1),
        )
        with pytest.raises(RuntimeError, match="superseded"):
            await store.save_server_knowledge(
                "budget-1",
                "accounts",
                44,
                change_batch_id=crash_batch,
            )
        # The later delta contains no account/category/transaction rows. Its
        # fresh final checkpoint confirms the inherited cache generation.
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            44,
            change_batch_id=recovery_batch,
        )
        await repository.record_completed_sync_batch(
            "budget-1",
            recovery_batch,
            completed_at=NOW,
        )
        proof_before = await database.fetch_all(
            """
            SELECT completed_at, watermarks_json, watermarks_sha256
            FROM calibration_sync_batches
            WHERE id = :batch_id
            """,
            {"batch_id": recovery_batch},
        )
        with pytest.raises(RuntimeError, match="superseded"):
            await store.save_accounts(
                "budget-1",
                [
                    {
                        "id": "checking",
                        "name": "Late mutation",
                        "balance": 999_999_000,
                        "deleted": False,
                    }
                ],
                change_batch_id=recovery_batch,
            )
        with pytest.raises(RuntimeError, match="superseded"):
            await store.save_server_knowledge(
                "budget-1",
                "accounts",
                99,
                change_batch_id=recovery_batch,
            )
        assert await database.fetch_all(
            """
            SELECT name, balance
            FROM accounts
            WHERE id = 'checking'
            """
        ) == [{"name": "Checking", "balance": 100_000}]
        assert await database.fetch_all(
            """
            SELECT server_knowledge
            FROM sync_state
            WHERE budget_id = 'budget-1' AND resource = 'accounts'
            """
        ) == [{"server_knowledge": 44}]
        assert await database.fetch_all(
            """
            SELECT completed_at, watermarks_json, watermarks_sha256
            FROM calibration_sync_batches
            WHERE id = :batch_id
            """,
            {"batch_id": recovery_batch},
        ) == proof_before

        profile = CalibrationProfile.create(
            id=str(uuid4()),
            scenario_revision_id=str(uuid4()),
            budget_id="budget-1",
            policy=default_calibration_policy(),
            created_at=NOW - timedelta(days=1),
        )
        source = await repository.load_source_state(
            profile,
            WealthScenario(
                name="Recovery fixture",
                current_age=30,
                retirement_age=35,
                end_age=40,
                accounts=[{"id": "checking", "role": "retirement"}],
                starting_portfolio=100,
                annual_contribution=0,
                annual_spending=10,
                trials=100,
            ),
            source_sync_batch_id=recovery_batch,
            as_of=NOW,
        )
        assert source.source_sync_batch_id == recovery_batch
        lifecycle = await database.fetch_all(
            """
            SELECT id, state, superseded_by
            FROM calibration_sync_batches
            ORDER BY started_at
            """
        )
        assert lifecycle == [
            {
                "id": crash_batch,
                "state": "superseded",
                "superseded_by": recovery_batch,
            },
            {
                "id": recovery_batch,
                "state": "completed",
                "superseded_by": None,
            },
        ]
        for table_name in ("accounts", "transactions", "categories"):
            assert await database.fetch_all(
                f"""
                SELECT DISTINCT change_batch_id
                FROM {table_name}
                WHERE budget_id = 'budget-1'
                """
            ) == [{"change_batch_id": recovery_batch}]
        old_evidence_batch = "reject-pre-completion-evidence"
        await repository.begin_sync_batch(
            "budget-1",
            old_evidence_batch,
            started_at=NOW + timedelta(microseconds=1),
        )
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE accounts
                    SET change_batch_id = :old_batch_id
                    WHERE id = 'checking'
                    """
                ),
                {"old_batch_id": crash_batch},
            )
            await session.commit()
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            45,
            change_batch_id=old_evidence_batch,
        )
        with pytest.raises(ValueError, match="outside its predecessor lineage"):
            await repository.record_completed_sync_batch(
                "budget-1",
                old_evidence_batch,
                completed_at=NOW + timedelta(microseconds=2),
            )
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE accounts
                    SET change_batch_id = :recovery_batch
                    WHERE id = 'checking'
                    """
                ),
                {"recovery_batch": recovery_batch},
            )
            await session.commit()
        concurrent_completion_batch = "overlapping-completion"
        await repository.begin_sync_batch(
            "budget-1",
            concurrent_completion_batch,
            started_at=NOW + timedelta(seconds=1),
        )
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            46,
            change_batch_id=concurrent_completion_batch,
        )
        original_completed = repository._completed_sync_batch
        missing_reads = 0
        both_prechecked = asyncio.Event()

        async def force_overlapping_precheck(batch_id: str):
            nonlocal missing_reads
            if batch_id != concurrent_completion_batch:
                return await original_completed(batch_id)
            missing_reads += 1
            if missing_reads == 2:
                both_prechecked.set()
            await both_prechecked.wait()
            return None

        monkeypatch.setattr(
            repository,
            "_completed_sync_batch",
            force_overlapping_precheck,
        )
        await asyncio.gather(
            repository.record_completed_sync_batch(
                "budget-1",
                concurrent_completion_batch,
                completed_at=NOW + timedelta(seconds=2),
            ),
            repository.record_completed_sync_batch(
                "budget-1",
                concurrent_completion_batch,
                completed_at=NOW + timedelta(seconds=2),
            ),
        )
        monkeypatch.setattr(
            repository,
            "_completed_sync_batch",
            original_completed,
        )
        for table_name in ("accounts", "transactions", "categories"):
            assert await database.fetch_all(
                f"""
                SELECT DISTINCT change_batch_id
                FROM {table_name}
                WHERE budget_id = 'budget-1'
                """
            ) == [{"change_batch_id": recovery_batch}]
        third_successful_batch = "third-successful-empty-delta"
        await repository.begin_sync_batch(
            "budget-1",
            third_successful_batch,
            started_at=NOW + timedelta(seconds=3),
        )
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            47,
            change_batch_id=third_successful_batch,
        )
        await repository.record_completed_sync_batch(
            "budget-1",
            third_successful_batch,
            completed_at=NOW + timedelta(seconds=4),
        )
        for table_name in ("accounts", "transactions", "categories"):
            assert await database.fetch_all(
                f"""
                SELECT DISTINCT change_batch_id
                FROM {table_name}
                WHERE budget_id = 'budget-1'
                """
            ) == [{"change_batch_id": recovery_batch}]
        await SqlTransactionNotificationStore(database).save_transactions(
            "budget-1",
            [
                {
                    "id": "txn-1",
                    "account_id": "checking",
                    "category_id": "groceries",
                    "date": "2090-07-01",
                    "amount": -61_000_000,
                    "deleted": False,
                },
                {
                    "id": "txn-notification-only",
                    "account_id": "checking",
                    "category_id": "groceries",
                    "date": "2090-07-02",
                    "amount": -1_000,
                    "deleted": False,
                },
            ],
        )
        with pytest.raises(ValueError, match="incomplete sync batch"):
            await repository.load_source_state(
                profile,
                WealthScenario(
                    name="Dirty notification fixture",
                    current_age=30,
                    retirement_age=35,
                    end_age=40,
                    accounts=[{"id": "checking", "role": "retirement"}],
                    starting_portfolio=100,
                    annual_contribution=0,
                    annual_spending=10,
                    trials=100,
                ),
                source_sync_batch_id=recovery_batch,
                as_of=NOW,
            )
        poll_recovery_batch = "partial-after-unowned-notification"
        await repository.begin_sync_batch(
            "budget-1",
            poll_recovery_batch,
            started_at=NOW + timedelta(minutes=1),
        )
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            48,
            change_batch_id=poll_recovery_batch,
        )
        with pytest.raises(ValueError, match="cannot adopt unowned evidence"):
            await repository.record_completed_sync_batch(
                "budget-1",
                poll_recovery_batch,
                completed_at=NOW + timedelta(minutes=2),
            )
        dirty_active_batch = "unowned-write-during-active-sync"
        await repository.begin_sync_batch(
            "budget-1",
            dirty_active_batch,
            started_at=NOW + timedelta(minutes=3),
        )
        async with database.session_factory() as session:
            for table_name in ("accounts", "transactions", "categories"):
                await session.execute(
                    text(
                        f"""
                        UPDATE {table_name}
                        SET change_batch_id = :batch_id
                        WHERE budget_id = 'budget-1'
                        """
                    ),
                    {"batch_id": dirty_active_batch},
                )
            await session.commit()
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            49,
            change_batch_id=dirty_active_batch,
        )
        await SqlTransactionNotificationStore(database).save_transactions(
            "budget-1",
            [
                {
                    "id": "txn-during-active",
                    "account_id": "checking",
                    "category_id": "groceries",
                    "date": "2090-07-03",
                    "amount": -2_000,
                    "deleted": False,
                }
            ],
        )
        with pytest.raises(ValueError, match="unowned source write"):
            await repository.record_completed_sync_batch(
                "budget-1",
                dirty_active_batch,
                completed_at=NOW + timedelta(minutes=4),
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_bootstrap_recovery_rejects_unowned_write_in_crashed_lineage(
    tmp_path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'dirty-bootstrap.db'}")
    try:
        await database.initialize()
        await _seed_source(database)
        repository = SqlCalibrationRepository(database)
        store = SqlSyncStore(database)
        await repository.begin_sync_batch(
            "budget-1",
            "dirty-bootstrap-crash",
            started_at=NOW - timedelta(minutes=2),
        )
        await SqlTransactionNotificationStore(database).save_transactions(
            "budget-1",
            [
                {
                    "id": "txn-dirty-bootstrap",
                    "account_id": "checking",
                    "category_id": "groceries",
                    "date": "2090-07-02",
                    "amount": -1_000,
                    "deleted": False,
                }
            ],
        )
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            43,
            change_batch_id="dirty-bootstrap-crash",
        )
        await repository.begin_sync_batch(
            "budget-1",
            "dirty-bootstrap-recovery",
            started_at=NOW - timedelta(minutes=1),
        )
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            44,
            change_batch_id="dirty-bootstrap-recovery",
        )
        with pytest.raises(ValueError, match="lifecycle bootstrap"):
            await repository.record_completed_sync_batch(
                "budget-1",
                "dirty-bootstrap-recovery",
                completed_at=NOW,
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_completion_validates_many_completed_source_owners_in_bulk(
    tmp_path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'owner-scale.db'}")
    try:
        await database.initialize()
        await _seed_source(database)
        repository = SqlCalibrationRepository(database)
        store = SqlSyncStore(database)
        await repository.begin_sync_batch(
            "budget-1",
            "owner-scale-bootstrap",
            started_at=NOW - timedelta(minutes=2),
        )
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            43,
            change_batch_id="owner-scale-bootstrap",
        )
        await repository.record_completed_sync_batch(
            "budget-1",
            "owner-scale-bootstrap",
            completed_at=NOW - timedelta(minutes=1),
        )
        owner_rows = [
            {
                "id": f"owner-{index:04d}",
                "budget_id": "budget-1",
                "started_at": (NOW - timedelta(seconds=2)).isoformat(),
                "completed_at": (NOW - timedelta(seconds=1)).isoformat(),
            }
            for index in range(1_005)
        ]
        transaction_rows = [
            {
                "id": f"scale-txn-{index:04d}",
                "owner_id": f"owner-{index:04d}",
            }
            for index in range(1_005)
        ]
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO calibration_sync_batches (
                      id, budget_id, state, started_at, completed_at
                    ) VALUES (
                      :id, :budget_id, 'completed', :started_at, :completed_at
                    )
                    """
                ),
                owner_rows,
            )
            await session.execute(
                text(
                    """
                    INSERT INTO transactions (
                      id, budget_id, account_id, date, amount,
                      deleted, change_batch_id
                    ) VALUES (
                      :id, 'budget-1', 'checking', '2090-07-01', -1,
                      FALSE, :owner_id
                    )
                    """
                ),
                transaction_rows,
            )
            await session.commit()
        await repository.begin_sync_batch(
            "budget-1",
            "owner-scale-current",
            started_at=NOW,
        )
        await store.save_server_knowledge(
            "budget-1",
            "accounts",
            44,
            change_batch_id="owner-scale-current",
        )
        await repository.record_completed_sync_batch(
            "budget-1",
            "owner-scale-current",
            completed_at=NOW + timedelta(seconds=1),
        )
        assert await database.fetch_all(
            """
            SELECT COUNT(DISTINCT change_batch_id) AS owner_count
            FROM transactions
            WHERE id LIKE 'scale-txn-%'
            """
        ) == [{"owner_count": 1_005}]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_capture_is_idempotent_replayable_and_worker_restart_safe(
    tmp_path,
    monkeypatch,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'calibration.db'}")
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        await database.initialize()
        await _seed_source(database)
        scenario_repository = SqlScenarioRevisionRepository(database)
        scenario_service = ScenarioComparisonService(
            repository=scenario_repository,
            portfolio_resolver=ExplicitPortfolioResolver(),
            execution_policy=PlannerExecutionPolicy(
                maximum_working_bytes=16 * 1024 * 1024,
                in_memory_path_bytes=8 * 1024 * 1024,
                maximum_temporary_bytes=32 * 1024 * 1024,
                batch_size=100,
            ),
        )
        revision = await scenario_service.save_revision(
            WealthScenario(
                name="Living plan",
                current_age=30,
                retirement_age=65,
                end_age=90,
                accounts=[
                    {"id": "checking", "role": "retirement"},
                    {"id": "mortgage", "role": "debt"},
                ],
                starting_portfolio=250_000,
                tax_buckets=[
                    {
                        "tax_treatment": "tax_deferred",
                        "account_id": "checking",
                        "starting_balance": 250_000,
                        "contribution_fraction": 1,
                    }
                ],
                tax_assumptions={
                    "ordinary_income_tax_rate": 0,
                    "long_term_capital_gains_tax_rate": 0,
                    "withdrawal_order": ["tax_deferred"],
                    "retirement_surplus_destination": "tax_deferred",
                    "apply_required_minimum_distributions": False,
                },
                portfolio_allocation={
                    "market": {
                        "us_equity": {
                            "expected_return": 0.08,
                            "volatility": 0.18,
                        },
                        "international_equity": {
                            "expected_return": 0.07,
                            "volatility": 0.20,
                        },
                        "bonds": {
                            "expected_return": 0.04,
                            "volatility": 0.07,
                        },
                        "cash": {
                            "expected_return": 0.025,
                            "volatility": 0.01,
                        },
                        "correlation": {
                            "values": [
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ]
                        },
                    },
                    "accounts": [
                        {
                            "account_id": "checking",
                            "portfolio_weight": 1,
                            "target": {
                                "us_equity": 0.7,
                                "international_equity": 0.15,
                                "bonds": 0.1,
                                "cash": 0.05,
                            },
                        }
                    ],
                },
                annual_contribution=7_500,
                annual_spending=48_000,
                trials=100,
            )
        )
        repository = SqlCalibrationRepository(database)
        service = CalibrationService(repository, clock=lambda: NOW)
        profile = await service.create_profile(
            scenario_revision_id=revision.id,
            budget_id="budget-1",
            reviewed_allocations=(
                ReviewedAllocation(
                    account_id="checking",
                    weights=FrozenFloatMap(
                        {
                            "us_equity": 0.6,
                            "international_equity": 0.15,
                            "bonds": 0.2,
                            "cash": 0.05,
                        }
                    ),
                    reviewed_at=NOW.replace(day=30),
                    source_sha256=canonical_content_sha256(
                        {
                            "account_id": "checking",
                            "weights": [0.6, 0.15, 0.2, 0.05],
                            "reviewed_at": NOW.replace(day=30),
                        }
                    ),
                ),
            ),
        )
        with pytest.raises(ValueError, match="recorded completed sync batch"):
            await service.capture_after_sync(
                profile.id,
                source_sync_batch_id="unrecorded-batch",
            )
        await repository.begin_sync_batch(
            "budget-1",
            "batch-42",
            started_at=NOW - timedelta(minutes=1),
        )
        await repository.record_completed_sync_batch(
            "budget-1",
            "batch-42",
            completed_at=NOW,
        )
        await repository.record_completed_sync_batch(
            "budget-1",
            "batch-42",
            completed_at=NOW + timedelta(seconds=1),
        )

        snapshot, run, created = await service.capture_after_sync(
            profile.id,
            source_sync_batch_id="batch-42",
        )
        duplicate_snapshot, duplicate_run, duplicate_created = await service.capture_after_sync(
            profile.id,
            source_sync_batch_id="batch-42",
        )
        assert created is True
        assert duplicate_created is False
        assert duplicate_snapshot == snapshot
        assert duplicate_run == run
        assert snapshot.manifest.source_watermarks[0].server_knowledge == 42
        assert all(
            link.source_id
            for observation in snapshot.manifest.observations
            for link in observation.source_links
        )
        assert any(event.material for event in snapshot.manifest.drift_events)
        assert {event.kind for event in snapshot.manifest.drift_events} >= {
            DriftKind.BALANCE,
            DriftKind.ALLOCATION,
            DriftKind.STALENESS,
        }
        contribution = next(
            event
            for event in snapshot.manifest.drift_events
            if event.kind is DriftKind.CONTRIBUTION
        )
        assert contribution.observed_value == 0
        first_execution = run_calibration_snapshot(snapshot.model_dump_json())
        assert run_calibration_snapshot(snapshot.model_dump_json()) == first_execution
        execution_payload = json.loads(first_execution)
        assert execution_payload["calibrated_scenario"]["annual_contribution"] == 7_500
        assert execution_payload["attribution"][
            "calibration_snapshot_manifest"
        ] == snapshot.manifest.model_dump(mode="json")
        report = execution_payload["attribution"]
        assert len(report["contributions"]) <= 5
        assert {
            drift_hash
            for contribution_row in report["contributions"]
            for drift_hash in contribution_row["drift_event_sha256"]
        } == {event.content_sha256 for event in snapshot.manifest.drift_events}
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE accounts
                    SET change_batch_id = 'in-progress-batch'
                    WHERE id = 'checking'
                    """
                )
            )
            await session.commit()
        with pytest.raises(ValueError, match="incomplete sync batch"):
            await repository.load_source_state(
                profile,
                revision.manifest.scenario,
                source_sync_batch_id="batch-42",
                as_of=NOW,
            )
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE accounts
                    SET change_batch_id = 'batch-42'
                    WHERE id = 'checking'
                    """
                )
            )
            await session.commit()
        original_contribution_rows = repository._contribution_rows

        async def complete_competing_sync(*args, **kwargs):
            rows = await original_contribution_rows(*args, **kwargs)
            competing_at = NOW + timedelta(minutes=1)
            await repository.begin_sync_batch(
                "budget-1",
                "overlapping-category-batch",
                started_at=competing_at - timedelta(seconds=1),
            )
            async with database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        UPDATE categories
                        SET change_batch_id = 'overlapping-category-batch'
                        WHERE id = 'groceries'
                        """
                    )
                )
                await session.execute(
                    text(
                        """
                        UPDATE sync_state
                        SET server_knowledge = 43,
                            synced_at = :synced_at,
                            change_batch_id = 'overlapping-category-batch'
                        WHERE budget_id = 'budget-1'
                          AND resource = 'accounts'
                        """
                    ),
                    {"synced_at": competing_at.isoformat()},
                )
                await session.commit()
            await repository.record_completed_sync_batch(
                "budget-1",
                "overlapping-category-batch",
                completed_at=competing_at,
            )
            return rows

        monkeypatch.setattr(
            repository,
            "_contribution_rows",
            complete_competing_sync,
        )
        with pytest.raises(
            ValueError,
            match=(
                "does not own every durable watermark"
                "|newer completed sync batch"
                "|current cache checkpoint"
            ),
        ):
            await repository.load_source_state(
                profile,
                revision.manifest.scenario,
                source_sync_batch_id="batch-42",
                as_of=NOW,
            )
        monkeypatch.setattr(
            repository,
            "_contribution_rows",
            original_contribution_rows,
        )
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE categories
                    SET change_batch_id = 'overlapping-category-batch'
                    WHERE id = 'groceries'
                    """
                )
            )
            await session.execute(
                text(
                    """
                    UPDATE sync_state
                    SET server_knowledge = 43,
                        synced_at = '2090-07-31T12:01:00+00:00',
                        change_batch_id = 'overlapping-category-batch'
                    WHERE budget_id = 'budget-1'
                      AND resource = 'accounts'
                    """
                )
            )
            await session.commit()
        masked_a_at = NOW + timedelta(minutes=2)
        masked_b_at = NOW + timedelta(minutes=3)
        await repository.begin_sync_batch(
            "budget-1",
            "masked-batch-a",
            started_at=masked_a_at - timedelta(seconds=1),
        )
        await repository.begin_sync_batch(
            "budget-1",
            "masked-batch-b",
            started_at=masked_b_at - timedelta(seconds=1),
        )
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE accounts
                    SET change_batch_id = 'masked-batch-b'
                    WHERE id = 'checking'
                    """
                )
            )
            await session.execute(
                text(
                    """
                    UPDATE sync_state
                    SET server_knowledge = 44,
                        synced_at = :synced_at,
                        change_batch_id = 'masked-batch-b'
                    WHERE budget_id = 'budget-1'
                      AND resource = 'accounts'
                    """
                ),
                {"synced_at": masked_b_at.isoformat()},
            )
            await session.commit()
        with pytest.raises(
            ValueError,
            match="no longer the active writer|does not own every durable watermark",
        ):
            await repository.record_completed_sync_batch(
                "budget-1",
                "masked-batch-a",
                completed_at=masked_a_at,
            )
        await repository.record_completed_sync_batch(
            "budget-1",
            "masked-batch-b",
            completed_at=masked_b_at,
        )
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE accounts
                    SET change_batch_id = 'masked-batch-b'
                    WHERE id = 'checking'
                    """
                )
            )
            await session.execute(
                text(
                    """
                    UPDATE sync_state
                    SET server_knowledge = 44,
                        synced_at = '2090-07-31T12:03:00+00:00',
                        change_batch_id = 'masked-batch-b'
                    WHERE budget_id = 'budget-1'
                      AND resource = 'accounts'
                    """
                )
            )
            await session.commit()

        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE accounts
                    SET balance = 0
                    WHERE id = 'mortgage'
                    """
                )
            )
            await session.commit()
        next_service = CalibrationService(
            repository,
            clock=lambda: NOW.replace(day=31) + timedelta(days=1),
        )
        await _tag_sync_batch(database, repository, "batch-43")
        await repository.record_completed_sync_batch(
            "budget-1",
            "batch-43",
            completed_at=NOW + timedelta(days=1),
        )
        second_snapshot, _, second_created = await next_service.capture_after_sync(
            profile.id,
            source_sync_batch_id="batch-43",
        )
        assert second_created is True
        payoff = next(
            event
            for event in second_snapshot.manifest.drift_events
            if event.kind is DriftKind.DEBT_PAYOFF
        )
        assert payoff.material is True
        assert payoff.baseline_value == 100_000
        assert payoff.observed_value == 0

        alerts = await repository.list_alerts(profile.id)
        assert len(alerts) >= 2
        assert "60000" not in json.dumps([alert.model_dump(mode="json") for alert in alerts])
        competing_results = await asyncio.gather(
            repository.transition_alert(
                alerts[-1].id,
                profile.id,
                second_snapshot.id,
                AlertState.ACKNOWLEDGED,
                at=second_snapshot.created_at + timedelta(seconds=2),
            ),
            repository.transition_alert(
                alerts[-1].id,
                profile.id,
                second_snapshot.id,
                AlertState.RESOLVED,
                at=second_snapshot.created_at + timedelta(seconds=1),
            ),
            return_exceptions=True,
        )
        assert any(not isinstance(result, Exception) for result in competing_results)
        concurrently_transitioned = next(
            alert
            for alert in await repository.list_alerts(profile.id)
            if alert.id == alerts[-1].id
        )
        assert concurrently_transitioned.state is AlertState.RESOLVED
        rejected = await repository.transition_alert(
            alerts[0].id,
            "another-profile",
            second_snapshot.id,
            AlertState.ACKNOWLEDGED,
            at=second_snapshot.created_at + timedelta(seconds=1),
        )
        assert rejected is None
        unchanged = await repository.list_alerts(profile.id)
        assert unchanged[0].state is alerts[0].state
        acknowledged = await repository.transition_alert(
            alerts[0].id,
            profile.id,
            second_snapshot.id,
            AlertState.ACKNOWLEDGED,
            at=second_snapshot.created_at + timedelta(seconds=1),
        )
        assert acknowledged is not None
        assert acknowledged.state is AlertState.ACKNOWLEDGED
        idempotent_ack = await repository.transition_alert(
            alerts[0].id,
            profile.id,
            second_snapshot.id,
            AlertState.ACKNOWLEDGED,
            at=second_snapshot.created_at + timedelta(seconds=2),
        )
        assert idempotent_ack == acknowledged
        resolved = await repository.transition_alert(
            alerts[0].id,
            profile.id,
            second_snapshot.id,
            AlertState.RESOLVED,
            at=second_snapshot.created_at + timedelta(seconds=3),
        )
        assert resolved is not None
        with pytest.raises(ValueError, match="illegal"):
            await repository.transition_alert(
                alerts[0].id,
                profile.id,
                second_snapshot.id,
                AlertState.ACKNOWLEDGED,
                at=second_snapshot.created_at + timedelta(seconds=4),
            )

        worker = CalibrationWorker(
            repository,
            executor=executor,
            runner=_runner,
        )
        assert await worker.run_once() == 2
        completed = await repository.get_run(run.id)
        assert completed is not None
        assert completed.state is CalibrationRunState.SUCCEEDED
        assert completed.result is not None
        assert completed.result["snapshot_id"] == snapshot.id
        assert {
            item.snapshot_id
            for item in await repository.list_profile_runs(profile.id)
        } == {snapshot.id, second_snapshot.id}

        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE calibration_runs
                    SET state = 'running', started_at = :started_at
                    WHERE id = :run_id
                    """
                ),
                {
                    "run_id": run.id,
                    "started_at": NOW.isoformat(),
                },
            )
            await session.commit()
        await repository.prepare_run_recovery()
        recovered = await repository.get_run(run.id)
        assert recovered is not None
        assert recovered.state is CalibrationRunState.ACCEPTED
        assert recovered.started_at is None

        reloaded = await repository.get_snapshot(snapshot.id)
        assert reloaded == snapshot
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE calibration_observations
                    SET subject_id = 'tampered'
                    WHERE snapshot_id = :snapshot_id
                      AND id = (
                        SELECT id
                        FROM calibration_observations
                        WHERE snapshot_id = :snapshot_id
                        LIMIT 1
                      )
                    """
                ),
                {"snapshot_id": second_snapshot.id},
            )
            await session.commit()
        with pytest.raises(ValueError, match="observation columns"):
            await repository.get_snapshot(second_snapshot.id)
        corrupted = json.loads(snapshot.manifest.model_dump_json())
        corrupted["resolved_scenario"]["annual_spending"] = 99_000
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE calibration_snapshots
                    SET manifest_json = :manifest_json
                    WHERE id = :snapshot_id
                    """
                ),
                {
                    "snapshot_id": snapshot.id,
                    "manifest_json": json.dumps(corrupted),
                },
            )
            await session.commit()
        with pytest.raises(ValueError):
            await repository.get_snapshot(snapshot.id)

        failure = CalibrationCaptureFailure(
            id="00000000-0000-4000-8000-000000000099",
            profile_id=profile.id,
            source_sync_batch_id="failed-batch",
            error_code="invalid_source",
            created_at=NOW + timedelta(days=2),
        )
        await repository.record_capture_failure(failure)
        assert await repository.list_capture_failures(profile.id) == (failure,)

        assert await service.disable_profile(profile.id) is True
        assert await repository.get_profile_state(profile.id) is CalibrationProfileState.DISABLED
        assert await repository.list_active_profiles("budget-1") == ()
    finally:
        executor.shutdown(wait=True)
        await database.close()


@pytest.mark.asyncio
async def test_non_tax_snapshot_reweights_accounts_and_replays_historical_data(
    tmp_path,
) -> None:
    database = DatabaseManager(f"sqlite+aiosqlite:///{tmp_path / 'calibration-nontax.db'}")
    try:
        await database.initialize()
        await _seed_source(database)
        scenario_service = ScenarioComparisonService(
            repository=SqlScenarioRevisionRepository(database),
            portfolio_resolver=ExplicitPortfolioResolver(),
            execution_policy=PlannerExecutionPolicy(
                maximum_working_bytes=16 * 1024 * 1024,
                in_memory_path_bytes=8 * 1024 * 1024,
                maximum_temporary_bytes=32 * 1024 * 1024,
                batch_size=100,
            ),
        )
        market = {
            "us_equity": {"expected_return": 0.08, "volatility": 0.18},
            "international_equity": {
                "expected_return": 0.07,
                "volatility": 0.20,
            },
            "bonds": {"expected_return": 0.04, "volatility": 0.07},
            "cash": {"expected_return": 0.025, "volatility": 0.01},
            "correlation": {
                "values": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            },
        }
        target = FrozenFloatMap(
            {
                "us_equity": 0.7,
                "international_equity": 0.15,
                "bonds": 0.1,
                "cash": 0.05,
            }
        )
        allocation_revision = await scenario_service.save_revision(
            WealthScenario(
                name="Non-tax allocation",
                current_age=30,
                retirement_age=35,
                end_age=40,
                accounts=[
                    {"id": "checking", "role": "retirement"},
                    {"id": "brokerage", "role": "taxable"},
                ],
                starting_portfolio=250_000,
                portfolio_allocation={
                    "market": market,
                    "accounts": [
                        {
                            "account_id": "checking",
                            "portfolio_weight": 0.5,
                            "target": target,
                        },
                        {
                            "account_id": "brokerage",
                            "portfolio_weight": 0.5,
                            "target": target,
                        },
                    ],
                },
                annual_contribution=7_500,
                annual_spending=48_000,
                trials=100,
            )
        )
        repository = SqlCalibrationRepository(database)
        service = CalibrationService(repository, clock=lambda: NOW)
        reviewed = tuple(
            ReviewedAllocation(
                account_id=account_id,
                weights=target,
                reviewed_at=NOW - timedelta(days=1),
                source_sha256=canonical_content_sha256(
                    {"account_id": account_id, "target": target}
                ),
            )
            for account_id in ("checking", "brokerage")
        )
        profile = await service.create_profile(
            scenario_revision_id=allocation_revision.id,
            budget_id="budget-1",
            reviewed_allocations=reviewed,
        )
        await _tag_sync_batch(database, repository, "allocation-batch")
        await repository.record_completed_sync_batch(
            "budget-1",
            "allocation-batch",
            completed_at=NOW,
        )
        snapshot, _, _ = await service.capture_after_sync(
            profile.id,
            source_sync_batch_id="allocation-batch",
        )
        portfolio_balance = next(
            event
            for event in snapshot.manifest.drift_events
            if event.kind is DriftKind.BALANCE and event.subject_id == "portfolio"
        )
        assert portfolio_balance.details["account_balances"] == FrozenFloatMap(
            {"checking": 100, "brokerage": 300_000}
        )
        execution = json.loads(run_calibration_snapshot(snapshot.model_dump_json()))
        accounts = execution["calibrated_scenario"]["portfolio_allocation"]["accounts"]
        weights = {row["account_id"]: row["portfolio_weight"] for row in accounts}
        assert weights["checking"] == pytest.approx(100 / 300_100)
        assert weights["brokerage"] == pytest.approx(300_000 / 300_100)

        historical_series = HistoricalSeries(
            years=tuple(range(2000, 2010)),
            nominal_returns=(0.05,) * 10,
            inflation_rates=(0.02,) * 10,
            source="test",
            sha256="1" * 64,
            order_policy=HistoricalOrderPolicy.REQUIRE_ASCENDING,
            gap_policy=HistoricalGapPolicy.REJECT,
        )
        historical_revision = await scenario_service.save_revision(
            WealthScenario(
                name="Historical living plan",
                current_age=30,
                retirement_age=35,
                end_age=40,
                accounts=[{"id": "checking", "role": "retirement"}],
                starting_portfolio=100,
                annual_contribution=0,
                annual_spending=10,
                return_model="historical_bootstrap",
                historical_block_size=2,
                trials=100,
            ),
            historical_dataset=HistoricalDatasetSnapshot(
                dataset_id="test-history",
                years=historical_series.years,
                nominal_returns=historical_series.nominal_returns,
                inflation_rates=historical_series.inflation_rates,
                content_sha256="1" * 64,
                observations_sha256=historical_series.observations_sha256,
                order_policy=HistoricalOrderPolicy.REQUIRE_ASCENDING,
                gap_policy=HistoricalGapPolicy.REJECT,
            ),
        )
        historical_profile = await service.create_profile(
            scenario_revision_id=historical_revision.id,
            budget_id="budget-1",
        )
        await _tag_sync_batch(database, repository, "historical-batch")
        await repository.record_completed_sync_batch(
            "budget-1",
            "historical-batch",
            completed_at=NOW + timedelta(seconds=1),
        )
        historical_service = CalibrationService(
            repository,
            clock=lambda: NOW + timedelta(seconds=1),
        )
        historical_snapshot, _, _ = await historical_service.capture_after_sync(
            historical_profile.id,
            source_sync_batch_id="historical-batch",
        )
        assert historical_snapshot.manifest.historical_dataset is not None
        historical_result = json.loads(
            run_calibration_snapshot(historical_snapshot.model_dump_json())
        )
        assert historical_result["snapshot_id"] == historical_snapshot.id

        broken_revision = await scenario_service.save_revision(
            WealthScenario(
                name="Missing source account",
                current_age=30,
                retirement_age=35,
                end_age=40,
                accounts=[{"id": "missing-account", "role": "retirement"}],
                starting_portfolio=1,
                annual_contribution=0,
                annual_spending=1,
                trials=100,
            )
        )
        broken_profile = await service.create_profile(
            scenario_revision_id=broken_revision.id,
            budget_id="budget-1",
        )
        hook_service = CalibrationService(
            repository,
            clock=lambda: NOW + timedelta(seconds=2),
        )
        await _tag_sync_batch(database, repository, "isolated-failure-batch")
        await repository.record_completed_sync_batch(
            "budget-1",
            "isolated-failure-batch",
            completed_at=NOW + timedelta(seconds=2),
        )
        await hook_service.after_sync(
            budget_id="budget-1",
            source_sync_batch_id="isolated-failure-batch",
        )
        isolated_snapshot = await repository.get_snapshot_for_batch(
            profile.id,
            "isolated-failure-batch",
        )
        assert isolated_snapshot is not None
        failures = await repository.list_capture_failures(broken_profile.id)
        assert len(failures) == 1
        assert failures[0].error_code == "invalid_source"
        assert await hook_service.disable_profile(broken_profile.id) is True
        assert broken_profile not in await repository.list_active_profiles("budget-1")

        concurrent_at = NOW + timedelta(seconds=3)
        sources = {}
        for batch_id in ("concurrent-a", "concurrent-b"):
            await _tag_sync_batch(database, repository, batch_id)
            await repository.record_completed_sync_batch(
                "budget-1",
                batch_id,
                completed_at=concurrent_at,
            )
            sources[batch_id] = await repository.load_source_state(
                profile,
                allocation_revision.manifest.scenario,
                source_sync_batch_id=batch_id,
                as_of=concurrent_at,
            )
        previous = await repository.get_latest_snapshot(profile.id)
        assert previous is not None
        for alert in await repository.list_alerts(profile.id):
            if alert.state is not AlertState.RESOLVED:
                await repository.transition_alert(
                    alert.id,
                    profile.id,
                    previous.id,
                    AlertState.RESOLVED,
                    at=concurrent_at - timedelta(microseconds=1),
                )
        candidates = {
            batch_id: build_calibration_snapshot(
                profile=profile,
                revision=allocation_revision,
                source=source,
                previous=previous,
                created_at=concurrent_at,
            )
            for batch_id, source in sources.items()
        }
        outcomes = await asyncio.gather(
            *(repository.create_snapshot(candidate) for candidate in candidates.values()),
            return_exceptions=True,
        )
        assert sum(outcome is None for outcome in outcomes) == 1
        assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
        persisted = {
            batch_id: await repository.get_snapshot_for_batch(profile.id, batch_id)
            for batch_id in candidates
        }
        assert sum(value is not None for value in persisted.values()) == 1
        winner = next(value for value in persisted.values() if value is not None)
        assert winner is not None
        assert winner.manifest.previous_snapshot_id == previous.id
        crash_events_before = await database.fetch_all(
            """
            SELECT id
            FROM calibration_alert_events
            WHERE snapshot_id = :snapshot_id
            """,
            {"snapshot_id": winner.id},
        )
        assert crash_events_before == []
        losing_batch = next(
            batch_id for batch_id, value in persisted.items() if value is None
        )
        retry_service = CalibrationService(
            repository,
            clock=lambda: concurrent_at + timedelta(microseconds=1),
        )
        replayed_after_crash, _, created_after_crash = (
            await retry_service.capture_after_sync(
                profile.id,
                source_sync_batch_id=winner.manifest.source_sync_batch_id,
            )
        )
        assert replayed_after_crash == winner
        assert created_after_crash is False
        assert await database.fetch_all(
            """
            SELECT id
            FROM calibration_alert_events
            WHERE snapshot_id = :snapshot_id
            """,
            {"snapshot_id": winner.id},
        )
        retried, _, _ = await retry_service.capture_after_sync(
            profile.id,
            source_sync_batch_id=losing_batch,
        )
        assert retried.manifest.previous_snapshot_id == winner.id
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE calibration_snapshots
                    SET created_at = '2200-01-01T00:00:00+00:00'
                    WHERE id = :snapshot_id
                    """
                ),
                {"snapshot_id": snapshot.id},
            )
            await session.commit()
        assert await repository.get_latest_snapshot(profile.id) == retried
        head_alerts = await repository.list_alerts(profile.id)
        await repository.reconcile_alerts(
            profile.id,
            winner,
            at=concurrent_at + timedelta(microseconds=2),
        )
        assert await repository.list_alerts(profile.id) == head_alerts
        stale_retry, _, stale_created = await retry_service.capture_after_sync(
            profile.id,
            source_sync_batch_id=winner.manifest.source_sync_batch_id,
        )
        assert stale_retry == winner
        assert stale_created is False
        assert await repository.list_alerts(profile.id) == head_alerts

        unrelated_snapshots = [
            {
                "snapshot_id": f"10000000-0000-4000-8000-{index:012d}",
                "profile_id": historical_profile.id,
                "batch_id": f"unrelated-{index}",
                "manifest_sha256": f"{index:064x}",
                "created_at": f"2000-01-01T00:00:{index % 60:02d}+00:00",
            }
            for index in range(101)
        ]
        unrelated_runs = [
            {
                "run_id": f"20000000-0000-4000-8000-{index:012d}",
                "snapshot_id": row["snapshot_id"],
                "created_at": row["created_at"],
            }
            for index, row in enumerate(unrelated_snapshots)
        ]
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO calibration_snapshots (
                      id, profile_id, previous_snapshot_id,
                      source_sync_batch_id, manifest_sha256,
                      manifest_json, created_at
                    ) VALUES (
                      :snapshot_id, :profile_id, NULL,
                      :batch_id, :manifest_sha256, '{}', :created_at
                    )
                    """
                ),
                unrelated_snapshots,
            )
            await session.execute(
                text(
                    """
                    INSERT INTO calibration_runs (
                      id, snapshot_id, state, created_at
                    ) VALUES (
                      :run_id, :snapshot_id, 'accepted', :created_at
                    )
                    """
                ),
                unrelated_runs,
            )
            await session.commit()
        scoped_runs = await repository.list_profile_runs(profile.id, limit=100)
        assert {row.snapshot_id for row in scoped_runs} == {
            snapshot.id,
            isolated_snapshot.id,
            winner.id,
            retried.id,
        }
    finally:
        await database.close()
