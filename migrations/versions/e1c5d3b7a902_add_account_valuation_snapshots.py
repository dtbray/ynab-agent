"""add account valuation snapshots

Revision ID: e1c5d3b7a902
Revises: a4a3b7e2f190
Create Date: 2026-07-29 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "e1c5d3b7a902"
down_revision = "a4a3b7e2f190"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "account_valuation_snapshots",
        sa.Column("account_id", sa.String(length=64), nullable=False),
        sa.Column("valuation_date", sa.String(length=10), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("balance_milliunits", sa.BigInteger(), nullable=False),
        sa.Column("observed_at", sa.String(length=64), nullable=False),
        sa.Column(
            "reviewed",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "account_id",
            "valuation_date",
            "source",
            name="pk_account_valuation_snapshots",
        ),
    )
    op.create_index(
        "ix_account_valuation_snapshot_lookup",
        "account_valuation_snapshots",
        ["valuation_date", "account_id", "reviewed"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_account_valuation_snapshot_lookup",
        table_name="account_valuation_snapshots",
    )
    op.drop_table("account_valuation_snapshots")
