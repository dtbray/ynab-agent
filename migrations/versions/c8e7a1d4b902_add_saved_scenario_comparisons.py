"""add saved scenario comparisons

Revision ID: c8e7a1d4b902
Revises: c7d8e9f0a123
Create Date: 2026-07-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "c8e7a1d4b902"
down_revision = "c7d8e9f0a123"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scenario_revisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("scenario_name", sa.String(length=200), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_scenario_revisions"),
        sa.UniqueConstraint(
            "scenario_name",
            "revision",
            name="uq_scenario_revision_name_number",
        ),
    )
    op.create_index(
        "ix_scenario_revisions_name_created",
        "scenario_revisions",
        ["scenario_name", "created_at"],
        unique=False,
    )
    op.create_table(
        "scenario_comparisons",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("baseline_revision_id", sa.String(length=36), nullable=False),
        sa.Column(
            "alternative_revision_ids_json",
            sa.Text(),
            nullable=False,
        ),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("result_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_json", sa.Text(), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["baseline_revision_id"],
            ["scenario_revisions.id"],
            name="fk_scenario_comparison_baseline",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_scenario_comparisons"),
    )
    op.create_index(
        op.f("ix_scenario_comparisons_baseline_revision_id"),
        "scenario_comparisons",
        ["baseline_revision_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_scenario_comparisons_baseline_revision_id"),
        table_name="scenario_comparisons",
    )
    op.drop_table("scenario_comparisons")
    op.drop_index(
        "ix_scenario_revisions_name_created",
        table_name="scenario_revisions",
    )
    op.drop_table("scenario_revisions")
