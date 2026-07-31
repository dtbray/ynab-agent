from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config


def test_calibration_migration_is_idempotent_and_has_append_only_graph(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "calibration-migration.db"
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url",
        f"sqlite+aiosqlite:///{database_path}",
    )
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {
            "calibration_sync_batches",
            "calibration_sync_batch_heads",
            "calibration_profiles",
            "calibration_profile_events",
            "calibration_profile_heads",
            "calibration_capture_failures",
            "calibration_snapshots",
            "calibration_observations",
            "calibration_drift_events",
            "calibration_runs",
            "calibration_attribution_reports",
            "calibration_alerts",
            "calibration_alert_events",
            "calibration_alert_heads",
        }.issubset(tables)
