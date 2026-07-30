from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config


def _config(database_path: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url",
        f"sqlite+aiosqlite:///{database_path}",
    )
    return config


def test_migration_adds_durable_composite_identity_mapping(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "spending-tier-migration.db"
    config = _config(database_path)

    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(spending_tier_mappings)"
            )
        }
        assert columns == {
            "budget_id",
            "category_id",
            "tier",
            "essential_floor_milliunits",
            "note",
            "updated_at",
        }
        primary_key = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(spending_tier_mappings)"
            )
            if row[5]
        ]
        assert primary_key == ["budget_id", "category_id"]
        indexes = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(spending_tier_mappings)"
            )
        }
        assert "ix_spending_tier_mappings_budget_tier" in indexes

    command.downgrade(config, "ba6d9c8f31e4")
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "spending_tier_mappings" not in tables
