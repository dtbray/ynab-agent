"""add spending tier mappings

Revision ID: c7d8e9f0a123
Revises: ba6d9c8f31e4
Create Date: 2026-07-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "c7d8e9f0a123"
down_revision = "ba6d9c8f31e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "spending_tier_mappings",
        sa.Column("budget_id", sa.String(length=64), nullable=False),
        sa.Column("category_id", sa.String(length=64), nullable=False),
        sa.Column("tier", sa.String(length=32), nullable=False),
        sa.Column(
            "essential_floor_milliunits",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column("updated_at", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint(
            "budget_id",
            "category_id",
            name="pk_spending_tier_mappings",
        ),
    )
    op.create_index(
        "ix_spending_tier_mappings_budget_tier",
        "spending_tier_mappings",
        ["budget_id", "tier"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_spending_tier_mappings_budget_tier",
        table_name="spending_tier_mappings",
    )
    op.drop_table("spending_tier_mappings")
