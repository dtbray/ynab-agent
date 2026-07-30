import asyncio
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import text

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.wealth import SqlWealthRepository
from ynab_agent.services.valuation_snapshots import (
    AccountValuationSnapshot,
    ValuationSnapshotConflictError,
    ValuationSnapshotSource,
)


async def _create_snapshot_table(database: DatabaseManager) -> None:
    async with database.session_factory() as session:
        await session.execute(
            text(
                """
                CREATE TABLE account_valuation_snapshots (
                  account_id VARCHAR(64) NOT NULL,
                  valuation_date VARCHAR(10) NOT NULL,
                  balance_milliunits BIGINT NOT NULL,
                  source VARCHAR(32) NOT NULL,
                  observed_at VARCHAR(64) NOT NULL,
                  reviewed BOOLEAN NOT NULL,
                  PRIMARY KEY (account_id, valuation_date, source)
                )
                """
            )
        )
        await session.commit()


def _snapshot(
    *,
    account_id: str = "brokerage",
    valuation_date: date = date(2026, 7, 29),
    balance_milliunits: int,
    observed_at: datetime,
    source: ValuationSnapshotSource = ValuationSnapshotSource.YNAB_SYNC,
    reviewed: bool = False,
) -> AccountValuationSnapshot:
    return AccountValuationSnapshot(
        account_id=account_id,
        valuation_date=valuation_date,
        balance_milliunits=balance_milliunits,
        source=source,
        observed_at=observed_at,
        reviewed=reviewed,
    )


def test_snapshot_upsert_is_unique_idempotent_and_newest_observation_wins(
    tmp_path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'snapshot-upsert.db'}"

    async def _exercise():
        database = DatabaseManager(database_url)
        try:
            await _create_snapshot_table(database)
            repository = SqlWealthRepository(database)
            noon = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
            newer = datetime(2026, 7, 29, 13, 0, tzinfo=timezone.utc)
            older = datetime(2026, 7, 29, 11, 0, tzinfo=timezone.utc)

            assert await repository.upsert_valuation_snapshots(
                [_snapshot(balance_milliunits=100_000, observed_at=noon)]
            ) == 1
            assert await repository.upsert_valuation_snapshots(
                [_snapshot(balance_milliunits=120_000, observed_at=newer)]
            ) == 1
            assert await repository.upsert_valuation_snapshots(
                [_snapshot(balance_milliunits=90_000, observed_at=older)]
            ) == 0
            assert await repository.upsert_valuation_snapshots(
                [_snapshot(balance_milliunits=120_000, observed_at=newer)]
            ) == 0

            rows = await database.fetch_all(
                """
                SELECT balance_milliunits, observed_at, reviewed
                FROM account_valuation_snapshots
                """
            )
            return rows, newer
        finally:
            await database.close()

    rows, newer = asyncio.run(_exercise())
    assert rows == [
        {
            "balance_milliunits": 120_000,
            "observed_at": newer.isoformat(),
            "reviewed": False,
        }
    ]


def test_same_identity_and_timestamp_with_different_balance_is_a_conflict(
    tmp_path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'snapshot-conflict.db'}"

    async def _exercise() -> None:
        database = DatabaseManager(database_url)
        try:
            await _create_snapshot_table(database)
            repository = SqlWealthRepository(database)
            observed_at = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
            await repository.upsert_valuation_snapshots(
                [_snapshot(balance_milliunits=100_000, observed_at=observed_at)]
            )
            with pytest.raises(ValuationSnapshotConflictError, match="conflicting"):
                await repository.upsert_valuation_snapshots(
                    [_snapshot(balance_milliunits=110_000, observed_at=observed_at)]
                )
        finally:
            await database.close()

    asyncio.run(_exercise())


def test_reviewed_lookup_is_exact_and_deterministically_selects_latest_source(
    tmp_path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'snapshot-lookup.db'}"
    target_date = date(2025, 12, 31)

    async def _exercise():
        database = DatabaseManager(database_url)
        try:
            await _create_snapshot_table(database)
            repository = SqlWealthRepository(database)
            await repository.upsert_valuation_snapshots(
                [
                    _snapshot(
                        valuation_date=target_date,
                        balance_milliunits=100_000,
                        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    ),
                    _snapshot(
                        valuation_date=target_date,
                        balance_milliunits=110_000,
                        observed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                        source=ValuationSnapshotSource.USER_REVIEWED,
                        reviewed=True,
                    ),
                    _snapshot(
                        valuation_date=target_date,
                        balance_milliunits=115_000,
                        observed_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
                        source=ValuationSnapshotSource.IMPORTED_REVIEWED,
                        reviewed=True,
                    ),
                    _snapshot(
                        account_id="ira",
                        valuation_date=target_date,
                        balance_milliunits=250_000,
                        observed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                        source=ValuationSnapshotSource.USER_REVIEWED,
                        reviewed=True,
                    ),
                    _snapshot(
                        account_id="ira",
                        valuation_date=date(2025, 12, 30),
                        balance_milliunits=249_000,
                        observed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                        source=ValuationSnapshotSource.IMPORTED_REVIEWED,
                        reviewed=True,
                    ),
                ]
            )
            rows = await repository.get_reviewed_valuation_snapshots(
                ["brokerage", "ira"],
                valuation_date=target_date,
            )
            return rows
        finally:
            await database.close()

    rows = asyncio.run(_exercise())

    assert [(row.account_id, row.balance_milliunits, row.source) for row in rows] == [
        ("brokerage", 115_000, ValuationSnapshotSource.IMPORTED_REVIEWED),
        ("ira", 250_000, ValuationSnapshotSource.USER_REVIEWED),
    ]
    assert all(row.valuation_date == target_date and row.reviewed for row in rows)
