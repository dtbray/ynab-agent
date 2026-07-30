"""expand YNAB API cache

Revision ID: 6d6cf09d802a
Revises: 794607ba49dc
Create Date: 2026-07-28 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "6d6cf09d802a"
down_revision = "794607ba49dc"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("budgets", sa.Column("currency_decimal_digits", sa.BigInteger()))
    op.add_column("budgets", sa.Column("currency_decimal_separator", sa.String(length=8)))
    op.add_column("budgets", sa.Column("currency_symbol_first", sa.Boolean()))
    op.add_column("budgets", sa.Column("currency_group_separator", sa.String(length=8)))
    op.add_column("budgets", sa.Column("currency_symbol", sa.String(length=16)))
    op.add_column("budgets", sa.Column("currency_display_symbol", sa.Boolean()))

    op.add_column(
        "categories",
        sa.Column("internal", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("categories", sa.Column("goal_target_date", sa.String(length=10)))
    op.add_column("categories", sa.Column("goal_snoozed_at", sa.String(length=64)))

    op.add_column("month_categories", sa.Column("category_group_id", sa.String(length=64)))
    op.create_index(
        op.f("ix_month_categories_category_group_id"),
        "month_categories",
        ["category_group_id"],
        unique=False,
    )
    op.add_column("month_categories", sa.Column("category_group_name", sa.String(length=255)))
    op.add_column("month_categories", sa.Column("name", sa.String(length=255)))
    op.add_column(
        "month_categories",
        sa.Column("hidden", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "month_categories",
        sa.Column("internal", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("month_categories", sa.Column("goal_type", sa.String(length=64)))
    op.add_column("month_categories", sa.Column("goal_needs_whole_amount", sa.Boolean()))
    op.add_column("month_categories", sa.Column("goal_day", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_cadence", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_cadence_frequency", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_creation_month", sa.String(length=10)))
    op.add_column("month_categories", sa.Column("goal_target", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_target_month", sa.String(length=10)))
    op.add_column("month_categories", sa.Column("goal_target_date", sa.String(length=10)))
    op.add_column("month_categories", sa.Column("goal_percentage_complete", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_months_to_budget", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_under_funded", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_overall_funded", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_overall_left", sa.BigInteger()))
    op.add_column("month_categories", sa.Column("goal_snoozed_at", sa.String(length=64)))
    op.add_column(
        "month_categories",
        sa.Column("deleted", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.add_column("transactions", sa.Column("import_payee_name", sa.String(length=255)))
    op.add_column(
        "transactions",
        sa.Column("import_payee_name_original", sa.String(length=255)),
    )
    op.add_column("transactions", sa.Column("debt_transaction_type", sa.String(length=64)))
    op.add_column("transactions", sa.Column("account_name", sa.String(length=255)))
    op.add_column("transactions", sa.Column("payee_name", sa.String(length=255)))
    op.add_column("transactions", sa.Column("category_name", sa.String(length=255)))
    op.add_column("subtransactions", sa.Column("category_name", sa.String(length=255)))

    op.add_column(
        "scheduled_transactions",
        sa.Column("account_name", sa.String(length=255)),
    )
    op.add_column(
        "scheduled_transactions",
        sa.Column("payee_name", sa.String(length=255)),
    )
    op.add_column(
        "scheduled_transactions",
        sa.Column("category_name", sa.String(length=255)),
    )
    op.create_table(
        "scheduled_subtransactions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("budget_id", sa.String(length=64), nullable=False),
        sa.Column("scheduled_transaction_id", sa.String(length=64), nullable=False),
        sa.Column("amount", sa.BigInteger()),
        sa.Column("memo", sa.Text()),
        sa.Column("payee_id", sa.String(length=64)),
        sa.Column("payee_name", sa.String(length=255)),
        sa.Column("category_id", sa.String(length=64)),
        sa.Column("category_name", sa.String(length=255)),
        sa.Column("transfer_account_id", sa.String(length=64)),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_scheduled_subtransactions_budget_id"),
        "scheduled_subtransactions",
        ["budget_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_scheduled_subtransactions_scheduled_transaction_id"),
        "scheduled_subtransactions",
        ["scheduled_transaction_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_scheduled_subtransactions_scheduled_transaction_id"),
        table_name="scheduled_subtransactions",
    )
    op.drop_index(
        op.f("ix_scheduled_subtransactions_budget_id"),
        table_name="scheduled_subtransactions",
    )
    op.drop_table("scheduled_subtransactions")
    op.drop_column("scheduled_transactions", "category_name")
    op.drop_column("scheduled_transactions", "payee_name")
    op.drop_column("scheduled_transactions", "account_name")
    op.drop_column("subtransactions", "category_name")
    op.drop_column("transactions", "category_name")
    op.drop_column("transactions", "payee_name")
    op.drop_column("transactions", "account_name")
    op.drop_column("transactions", "debt_transaction_type")
    op.drop_column("transactions", "import_payee_name_original")
    op.drop_column("transactions", "import_payee_name")

    op.drop_column("month_categories", "deleted")
    op.drop_column("month_categories", "goal_snoozed_at")
    op.drop_column("month_categories", "goal_overall_left")
    op.drop_column("month_categories", "goal_overall_funded")
    op.drop_column("month_categories", "goal_under_funded")
    op.drop_column("month_categories", "goal_months_to_budget")
    op.drop_column("month_categories", "goal_percentage_complete")
    op.drop_column("month_categories", "goal_target_date")
    op.drop_column("month_categories", "goal_target_month")
    op.drop_column("month_categories", "goal_target")
    op.drop_column("month_categories", "goal_creation_month")
    op.drop_column("month_categories", "goal_cadence_frequency")
    op.drop_column("month_categories", "goal_cadence")
    op.drop_column("month_categories", "goal_day")
    op.drop_column("month_categories", "goal_needs_whole_amount")
    op.drop_column("month_categories", "goal_type")
    op.drop_column("month_categories", "internal")
    op.drop_column("month_categories", "hidden")
    op.drop_column("month_categories", "name")
    op.drop_column("month_categories", "category_group_name")
    op.drop_index(
        op.f("ix_month_categories_category_group_id"),
        table_name="month_categories",
    )
    op.drop_column("month_categories", "category_group_id")

    op.drop_column("categories", "goal_snoozed_at")
    op.drop_column("categories", "goal_target_date")
    op.drop_column("categories", "internal")

    op.drop_column("budgets", "currency_display_symbol")
    op.drop_column("budgets", "currency_symbol")
    op.drop_column("budgets", "currency_group_separator")
    op.drop_column("budgets", "currency_symbol_first")
    op.drop_column("budgets", "currency_decimal_separator")
    op.drop_column("budgets", "currency_decimal_digits")
