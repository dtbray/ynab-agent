"""Dialect-aware bulk upsert used by focused cache adapters."""

from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from ynab_agent.db.models import Base
from ynab_agent.services.sync import Row


async def bulk_upsert(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    model: type[Base],
    rows: list[Row],
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
        await session.execute(statement)
        await session.commit()
