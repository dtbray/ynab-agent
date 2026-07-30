from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest


def _config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url",
        f"sqlite+aiosqlite:///{database_path}",
    )
    return config


def _insert_job(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    request_hash: str,
    state: str,
    deduplication_key: str | None = None,
) -> None:
    columns = [
        "id",
        "request_hash",
        "state",
        "request_json",
        "cancellation_requested",
        "required_working_bytes",
        "created_at",
        "updated_at",
    ]
    values: list[object] = [
        job_id,
        request_hash,
        state,
        "{}",
        0,
        1,
        "2026-07-29T00:00:00+00:00",
        "2026-07-29T00:00:00+00:00",
    ]
    if deduplication_key is not None:
        columns.append("deduplication_key")
        values.append(deduplication_key)
    placeholders = ", ".join("?" for _ in values)
    connection.execute(
        f"""
        INSERT INTO planner_jobs ({", ".join(columns)})
        VALUES ({placeholders})
        """,
        values,
    )


def test_migration_backfills_active_deduplication_and_allows_terminal_retry(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "planner-migration.db"
    config = _config(database_path)
    command.upgrade(config, "f4c2a9d87103")
    failed_hash = "a" * 64
    active_hash = "b" * 64
    with sqlite3.connect(database_path) as connection:
        _insert_job(
            connection,
            job_id="00000000-0000-4000-8000-000000000041",
            request_hash=failed_hash,
            state="failed",
        )
        _insert_job(
            connection,
            job_id="00000000-0000-4000-8000-000000000042",
            request_hash=active_hash,
            state="accepted",
        )
        connection.commit()

    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        keys = dict(
            connection.execute(
                """
                SELECT request_hash, deduplication_key
                FROM planner_jobs
                """
            )
        )
        assert keys == {failed_hash: None, active_hash: active_hash}

        _insert_job(
            connection,
            job_id="00000000-0000-4000-8000-000000000043",
            request_hash=failed_hash,
            deduplication_key=failed_hash,
            state="accepted",
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="deduplication_key",
        ):
            _insert_job(
                connection,
                job_id="00000000-0000-4000-8000-000000000044",
                request_hash=active_hash,
                deduplication_key=active_hash,
                state="accepted",
            )

    command.downgrade(config, "f4c2a9d87103")
    command.upgrade(config, "head")
