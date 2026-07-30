"""widen imported payee metadata

Revision ID: a4a3b7e2f190
Revises: 6d6cf09d802a
Create Date: 2026-07-29 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "a4a3b7e2f190"
down_revision = "6d6cf09d802a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("transactions") as batch_op:
        batch_op.alter_column(
            "import_payee_name",
            existing_type=sa.String(length=255),
            type_=sa.Text(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "import_payee_name_original",
            existing_type=sa.String(length=255),
            type_=sa.Text(),
            existing_nullable=True,
        )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE transactions
            SET import_payee_name = substr(import_payee_name, 1, 255),
                import_payee_name_original = substr(import_payee_name_original, 1, 255)
            """
        )
    )
    with op.batch_alter_table("transactions") as batch_op:
        batch_op.alter_column(
            "import_payee_name_original",
            existing_type=sa.Text(),
            type_=sa.String(length=255),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "import_payee_name",
            existing_type=sa.Text(),
            type_=sa.String(length=255),
            existing_nullable=True,
        )
