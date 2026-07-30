"""add durable planner jobs

Revision ID: f4c2a9d87103
Revises: e1c5d3b7a902
Create Date: 2026-07-29 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "f4c2a9d87103"
down_revision = "e1c5d3b7a902"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "planner_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("request_json", sa.Text(), nullable=False),
        sa.Column("manifest_json", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("required_working_bytes", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.Column("updated_at", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.String(length=64), nullable=True),
        sa.Column("completed_at", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_planner_jobs"),
    )
    op.create_index(
        op.f("ix_planner_jobs_request_hash"),
        "planner_jobs",
        ["request_hash"],
        unique=True,
    )
    op.create_index(
        op.f("ix_planner_jobs_state"),
        "planner_jobs",
        ["state"],
        unique=False,
    )
    op.create_index(
        "ix_planner_jobs_state_created",
        "planner_jobs",
        ["state", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_planner_jobs_state_created",
        table_name="planner_jobs",
    )
    op.drop_index(
        op.f("ix_planner_jobs_state"),
        table_name="planner_jobs",
    )
    op.drop_index(
        op.f("ix_planner_jobs_request_hash"),
        table_name="planner_jobs",
    )
    op.drop_table("planner_jobs")
