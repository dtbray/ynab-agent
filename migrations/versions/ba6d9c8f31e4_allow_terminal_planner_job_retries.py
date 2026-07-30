"""allow terminal planner job retries

Revision ID: ba6d9c8f31e4
Revises: f4c2a9d87103
Create Date: 2026-07-29 00:00:01.000000
"""

from __future__ import annotations

import hashlib

from alembic import op
import sqlalchemy as sa


revision = "ba6d9c8f31e4"
down_revision = "f4c2a9d87103"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "planner_jobs",
        sa.Column(
            "deduplication_key",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.execute(
        """
        UPDATE planner_jobs
        SET deduplication_key = request_hash
        WHERE state NOT IN ('failed', 'cancelled')
        """
    )
    op.drop_index(
        op.f("ix_planner_jobs_request_hash"),
        table_name="planner_jobs",
    )
    op.create_index(
        op.f("ix_planner_jobs_request_hash"),
        "planner_jobs",
        ["request_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_planner_jobs_deduplication_key"),
        "planner_jobs",
        ["deduplication_key"],
        unique=True,
    )


def downgrade() -> None:
    connection = op.get_bind()
    duplicate_rows = (
        connection.execute(
            sa.text(
                """
            SELECT id, request_hash
            FROM planner_jobs
            ORDER BY request_hash, created_at, id
            """
            )
        )
        .mappings()
        .all()
    )
    seen_hashes: set[str] = set()
    for row in duplicate_rows:
        request_hash = str(row["request_hash"])
        if request_hash in seen_hashes:
            replacement = hashlib.sha256(f"{request_hash}:{row['id']}".encode()).hexdigest()
            connection.execute(
                sa.text(
                    """
                    UPDATE planner_jobs
                    SET request_hash = :replacement
                    WHERE id = :job_id
                    """
                ),
                {
                    "replacement": replacement,
                    "job_id": str(row["id"]),
                },
            )
        else:
            seen_hashes.add(request_hash)

    op.drop_index(
        op.f("ix_planner_jobs_deduplication_key"),
        table_name="planner_jobs",
    )
    op.drop_column("planner_jobs", "deduplication_key")
    op.drop_index(
        op.f("ix_planner_jobs_request_hash"),
        table_name="planner_jobs",
    )
    op.create_index(
        op.f("ix_planner_jobs_request_hash"),
        "planner_jobs",
        ["request_hash"],
        unique=True,
    )
