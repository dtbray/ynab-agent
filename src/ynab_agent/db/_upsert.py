"""Dialect-aware bulk upsert used by focused cache adapters."""

from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ynab_agent.db.models import Base
from ynab_agent.services.sync import Row


async def bulk_upsert(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    model: type[Base],
    rows: list[Row],
    *,
    active_sync_batch: tuple[str, str] | None = None,
    unbatched_source_budget_id: str | None = None,
) -> None:
    """Upsert rows in bounded batches on SQLite and PostgreSQL."""
    if not rows:
        return

    if len(rows) > 1:
        maximum_rows = max(1, 30_000 // len(rows[0]))
        if len(rows) > maximum_rows:
            for index in range(0, len(rows), maximum_rows):
                await bulk_upsert(
                    engine,
                    session_factory,
                    model,
                    rows[index : index + maximum_rows],
                    active_sync_batch=active_sync_batch,
                    unbatched_source_budget_id=unbatched_source_budget_id,
                )
            return

    table: Any = model.__table__
    dialect = engine.dialect.name
    statement: Any
    if dialect == "postgresql":
        statement = postgresql_insert(table).values(rows)
    elif dialect == "sqlite":
        statement = sqlite_insert(table).values(rows)
    else:
        async with session_factory() as session:
            await _guard_sync_write(
                session,
                active_sync_batch,
                unbatched_source_budget_id,
            )
            for row in rows:
                await session.merge(model(**row))
            await session.commit()
        return

    primary_keys = [column.name for column in table.primary_key.columns]
    update_columns = {
        column.name: getattr(statement.excluded, column.name)
        for column in table.columns
        if column.name not in primary_keys
    }
    statement = statement.on_conflict_do_update(
        index_elements=primary_keys,
        set_=update_columns,
    )
    async with session_factory() as session:
        await _guard_sync_write(
            session,
            active_sync_batch,
            unbatched_source_budget_id,
        )
        await session.execute(statement)
        await session.commit()


async def _guard_sync_write(
    session: AsyncSession,
    active_sync_batch: tuple[str, str] | None,
    unbatched_source_budget_id: str | None,
) -> None:
    if active_sync_batch is None and unbatched_source_budget_id is None:
        return
    if active_sync_batch is None:
        await session.execute(
            text(
                """
                INSERT INTO calibration_sync_batch_heads (budget_id, batch_id)
                VALUES (:budget_id, NULL)
                ON CONFLICT (budget_id) DO NOTHING
                """
            ),
            {"budget_id": unbatched_source_budget_id},
        )
        result = await session.execute(
            text(
                """
                UPDATE calibration_sync_batch_heads
                SET batch_id = batch_id
                WHERE budget_id = :budget_id
                """
            ),
            {"budget_id": unbatched_source_budget_id},
        )
        if getattr(result, "rowcount", 0) != 1:
            raise RuntimeError("could not lock sync batch head")
        await session.execute(
            text(
                """
                UPDATE calibration_sync_batches
                SET unowned_source_dirty = TRUE
                WHERE id = (
                  SELECT batch_id
                  FROM calibration_sync_batch_heads
                  WHERE budget_id = :budget_id
                )
                  AND state = 'started'
                """
            ),
            {"budget_id": unbatched_source_budget_id},
        )
        return
    budget_id, batch_id = active_sync_batch
    result = await session.execute(
        text(
            """
            UPDATE calibration_sync_batch_heads
            SET batch_id = batch_id
            WHERE budget_id = :budget_id
              AND batch_id = :batch_id
              AND EXISTS (
                SELECT 1
                FROM calibration_sync_batches b
                WHERE b.id = calibration_sync_batch_heads.batch_id
                  AND b.budget_id = calibration_sync_batch_heads.budget_id
                  AND b.state = 'started'
              )
            """
        ),
        {"budget_id": budget_id, "batch_id": batch_id},
    )
    if getattr(result, "rowcount", 0) == 1:
        return
    head = await session.execute(
        text(
            """
            SELECT batch_id
            FROM calibration_sync_batch_heads
            WHERE budget_id = :budget_id
            """
        ),
        {"budget_id": budget_id},
    )
    if head.first() is not None:
        raise RuntimeError("sync batch has been superseded by a newer writer")
    raise RuntimeError("sync batch has no durable lifecycle ownership")
