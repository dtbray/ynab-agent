"""Database adapter for checkpointed bulk synchronization."""

from __future__ import annotations

import json
from typing import cast

from ynab_agent.db._upsert import bulk_upsert
from ynab_agent.db.change_tracking import SqlChangeLog
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.models import (
    Account,
    Budget,
    BudgetMonth,
    Category,
    CategoryGroup,
    MonthCategory,
    Payee,
    PayeeLocation,
    ScheduledSubTransaction,
    ScheduledTransaction,
    SubTransaction,
    SyncState,
    Transaction,
)
from ynab_agent.services.change_tracking import ChangeLog
from ynab_agent.services.sync import Row


class SqlSyncStore:
    """Implement synchronization persistence over the SQL cache."""

    def __init__(
        self,
        database: DatabaseManager,
        *,
        change_log: ChangeLog | None = None,
    ) -> None:
        self._database = database
        self._changes = change_log or SqlChangeLog(database)

    async def get_server_knowledge(
        self,
        plan_id: str,
        resource: str,
    ) -> int | None:
        """Return the last YNAB server knowledge checkpoint."""
        rows = await self._database.fetch_all(
            """
            SELECT server_knowledge
            FROM sync_state
            WHERE budget_id = :plan_id AND resource = :resource
            """,
            {"plan_id": plan_id, "resource": resource},
        )
        if not rows:
            return None
        return cast(int, rows[0]["server_knowledge"])

    async def save_server_knowledge(
        self,
        plan_id: str,
        resource: str,
        server_knowledge: int | None,
    ) -> None:
        """Persist a YNAB server knowledge checkpoint."""
        if server_knowledge is None:
            return
        await self._upsert(
            SyncState,
            [
                {
                    "budget_id": plan_id,
                    "resource": resource,
                    "server_knowledge": server_knowledge,
                }
            ],
        )

    async def save_budgets(
        self,
        budgets: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        rows: list[Row] = [
            {
                "id": _str_id(budget.get("id")),
                "name": budget.get("name"),
                "first_month": _str_date(budget.get("first_month")),
                "last_month": _str_date(budget.get("last_month")),
                "last_modified_on": _str_date(
                    budget.get("last_modified_on")
                ),
                "date_format": budget.get("date_format"),
                "currency_format_iso_code": budget.get(
                    "currency_format_iso_code"
                ),
                "currency_format_example": budget.get(
                    "currency_format_example"
                ),
                "currency_decimal_digits": budget.get(
                    "currency_decimal_digits"
                ),
                "currency_decimal_separator": budget.get(
                    "currency_decimal_separator"
                ),
                "currency_symbol_first": budget.get(
                    "currency_symbol_first"
                ),
                "currency_group_separator": budget.get(
                    "currency_group_separator"
                ),
                "currency_symbol": budget.get("currency_symbol"),
                "currency_display_symbol": budget.get(
                    "currency_display_symbol"
                ),
            }
            for budget in budgets
        ]
        if change_batch_id:
            for row in rows:
                await self._changes.record_changes(
                    change_batch_id,
                    str(row["id"]),
                    "budgets",
                    [row],
                )
        await self._upsert(Budget, rows)
        return len(rows)

    async def save_accounts(
        self,
        plan_id: str,
        accounts: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        rows: list[Row] = [
            {
                "id": _str_id(account.get("id")),
                "budget_id": plan_id,
                "name": account.get("name"),
                "type": _enum_value(account.get("type")),
                "on_budget": account.get("on_budget"),
                "closed": account.get("closed"),
                "note": account.get("note"),
                "balance": account.get("balance"),
                "cleared_balance": account.get("cleared_balance"),
                "uncleared_balance": account.get("uncleared_balance"),
                "transfer_payee_id": _str_id(
                    account.get("transfer_payee_id")
                ),
                "direct_import_linked": account.get(
                    "direct_import_linked"
                ),
                "direct_import_in_error": account.get(
                    "direct_import_in_error"
                ),
                "last_reconciled_at": _str_date(
                    account.get("last_reconciled_at")
                ),
                "debt_original_balance": account.get(
                    "debt_original_balance"
                ),
                "debt_interest_rates": _json_or_none(
                    account.get("debt_interest_rates")
                ),
                "debt_minimum_payments": _json_or_none(
                    account.get("debt_minimum_payments")
                ),
                "debt_escrow_amounts": _json_or_none(
                    account.get("debt_escrow_amounts")
                ),
                "deleted": account.get("deleted", False),
            }
            for account in accounts
        ]
        if change_batch_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "accounts",
                rows,
            )
        await self._upsert(Account, rows)
        return len(rows)

    async def save_payees(
        self,
        payees: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        rows: list[Row] = [
            {
                "id": _str_id(payee.get("id")),
                "budget_id": payee.get("budget_id"),
                "name": payee.get("name"),
                "transfer_account_id": _str_id(
                    payee.get("transfer_account_id")
                ),
                "deleted": payee.get("deleted", False),
            }
            for payee in payees
        ]
        plan_id = _single_plan_id(rows)
        if change_batch_id and plan_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "payees",
                rows,
            )
        await self._upsert(Payee, rows)
        return len(rows)

    async def save_payee_locations(
        self,
        payee_locations: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        rows: list[Row] = [
            {
                "id": _str_id(location.get("id")),
                "budget_id": location.get("budget_id"),
                "payee_id": _str_id(location.get("payee_id")),
                "latitude": (
                    str(location["latitude"])
                    if location.get("latitude") is not None
                    else None
                ),
                "longitude": (
                    str(location["longitude"])
                    if location.get("longitude") is not None
                    else None
                ),
                "deleted": location.get("deleted", False),
            }
            for location in payee_locations
        ]
        plan_id = _single_plan_id(rows)
        if change_batch_id and plan_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "payee_locations",
                rows,
            )
        await self._upsert(PayeeLocation, rows)
        return len(rows)

    async def save_categories(
        self,
        category_groups: list[Row],
        categories: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]:
        group_rows: list[Row] = [
            {
                "id": _str_id(group.get("id")),
                "budget_id": group.get("budget_id"),
                "name": group.get("name"),
                "hidden": group.get("hidden", False),
                "deleted": group.get("deleted", False),
            }
            for group in category_groups
        ]
        category_rows: list[Row] = [
            {
                "id": _str_id(category.get("id")),
                "budget_id": category.get("budget_id"),
                "category_group_id": _str_id(
                    category.get("category_group_id")
                ),
                "name": category.get("name"),
                "hidden": category.get("hidden", False),
                "internal": category.get("internal", False),
                "original_category_group_id": _str_id(
                    category.get("original_category_group_id")
                ),
                "note": category.get("note"),
                "budgeted": category.get("budgeted"),
                "activity": category.get("activity"),
                "balance": category.get("balance"),
                "goal_type": category.get("goal_type"),
                "goal_needs_whole_amount": category.get(
                    "goal_needs_whole_amount"
                ),
                "goal_day": category.get("goal_day"),
                "goal_cadence": category.get("goal_cadence"),
                "goal_cadence_frequency": category.get(
                    "goal_cadence_frequency"
                ),
                "goal_creation_month": _str_date(
                    category.get("goal_creation_month")
                ),
                "goal_target": category.get("goal_target"),
                "goal_target_month": _str_date(
                    category.get("goal_target_month")
                ),
                "goal_target_date": _str_date(
                    category.get("goal_target_date")
                ),
                "goal_percentage_complete": category.get(
                    "goal_percentage_complete"
                ),
                "goal_months_to_budget": category.get(
                    "goal_months_to_budget"
                ),
                "goal_under_funded": category.get("goal_under_funded"),
                "goal_overall_funded": category.get(
                    "goal_overall_funded"
                ),
                "goal_overall_left": category.get("goal_overall_left"),
                "goal_snoozed_at": _str_date(
                    category.get("goal_snoozed_at")
                ),
                "deleted": category.get("deleted", False),
            }
            for category in categories
        ]
        plan_id = _single_plan_id(group_rows) or _single_plan_id(
            category_rows
        )
        if change_batch_id and plan_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "category_groups",
                group_rows,
            )
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "categories",
                category_rows,
            )
        await self._upsert(CategoryGroup, group_rows)
        await self._upsert(Category, category_rows)
        return len(group_rows), len(category_rows)

    async def save_months(
        self,
        months: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        rows = [_month_row(month) for month in months]
        plan_id = _single_plan_id(rows)
        if change_batch_id and plan_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "months",
                rows,
            )
        await self._upsert(BudgetMonth, rows)
        return len(rows)

    async def save_month(
        self,
        month: Row,
        categories: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]:
        month_row = _month_row(month)
        category_rows = [_month_category_row(category) for category in categories]
        plan_id_value = month_row.get("budget_id")
        if change_batch_id and plan_id_value:
            plan_id = str(plan_id_value)
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "month_detail",
                [month_row],
            )
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "month_categories",
                category_rows,
            )
        await self._upsert(BudgetMonth, [month_row])
        await self._upsert(MonthCategory, category_rows)
        return 1, len(category_rows)

    async def save_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        rows = [_transaction_row(plan_id, transaction) for transaction in transactions]
        if change_batch_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "transactions",
                rows,
            )
        await self._upsert(Transaction, rows)
        subtransactions: list[Row] = [
            {
                **subtransaction,
                "transaction_id": transaction.get("id"),
            }
            for transaction in transactions
            for subtransaction in _nested_rows(transaction, "subtransactions")
        ]
        await self.save_subtransactions(
            plan_id,
            subtransactions,
            change_batch_id=change_batch_id,
        )
        return len(rows)

    async def save_subtransactions(
        self,
        plan_id: str,
        subtransactions: list[Row],
        *,
        change_batch_id: str,
    ) -> int:
        rows = [
            _subtransaction_row(plan_id, subtransaction)
            for subtransaction in subtransactions
        ]
        if change_batch_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "subtransactions",
                rows,
            )
        await self._upsert(SubTransaction, rows)
        return len(rows)

    async def save_scheduled_transactions(
        self,
        plan_id: str,
        transactions: list[Row],
        *,
        change_batch_id: str,
    ) -> tuple[int, int]:
        rows = [
            _scheduled_transaction_row(plan_id, transaction)
            for transaction in transactions
        ]
        if change_batch_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "scheduled_transactions",
                rows,
            )
        await self._upsert(ScheduledTransaction, rows)

        subtransaction_rows: list[Row] = [
            _scheduled_subtransaction_row(
                plan_id,
                transaction,
                subtransaction,
            )
            for transaction in transactions
            for subtransaction in _nested_rows(transaction, "subtransactions")
        ]
        if change_batch_id:
            await self._changes.record_changes(
                change_batch_id,
                plan_id,
                "scheduled_subtransactions",
                subtransaction_rows,
            )
        await self._upsert(ScheduledSubTransaction, subtransaction_rows)
        return len(rows), len(subtransaction_rows)

    async def _upsert(
        self,
        model: type[Budget]
        | type[Account]
        | type[Payee]
        | type[PayeeLocation]
        | type[CategoryGroup]
        | type[Category]
        | type[BudgetMonth]
        | type[MonthCategory]
        | type[Transaction]
        | type[SubTransaction]
        | type[ScheduledTransaction]
        | type[ScheduledSubTransaction]
        | type[SyncState],
        rows: list[Row],
    ) -> None:
        await bulk_upsert(
            self._database.engine,
            self._database.session_factory,
            model,
            rows,
        )


def _transaction_row(plan_id: str, transaction: Row) -> Row:
    return {
        "id": _str_id(transaction.get("id")),
        "budget_id": plan_id,
        "account_id": _str_id(transaction.get("account_id")),
        "category_id": _str_id(transaction.get("category_id")),
        "payee_id": _str_id(transaction.get("payee_id")),
        "transfer_account_id": _str_id(
            transaction.get("transfer_account_id")
        ),
        "transfer_transaction_id": _str_id(
            transaction.get("transfer_transaction_id")
        ),
        "matched_transaction_id": _str_id(
            transaction.get("matched_transaction_id")
        ),
        "import_id": transaction.get("import_id"),
        "import_payee_name": transaction.get("import_payee_name"),
        "import_payee_name_original": transaction.get(
            "import_payee_name_original"
        ),
        "debt_transaction_type": _enum_value(
            transaction.get("debt_transaction_type")
        ),
        "account_name": transaction.get("account_name"),
        "payee_name": transaction.get("payee_name"),
        "category_name": transaction.get("category_name"),
        "date": _str_date(transaction.get("date")),
        "amount": transaction.get("amount"),
        "memo": transaction.get("memo"),
        "cleared": _enum_value(transaction.get("cleared")),
        "approved": transaction.get("approved", True),
        "flag_color": _enum_value(transaction.get("flag_color")),
        "flag_name": transaction.get("flag_name"),
        "foreign_amount": transaction.get("foreign_amount"),
        "foreign_currency_code": transaction.get(
            "foreign_currency_code"
        ),
        "deleted": transaction.get("deleted", False),
    }


def _subtransaction_row(plan_id: str, subtransaction: Row) -> Row:
    return {
        "id": _str_id(subtransaction.get("id")),
        "budget_id": plan_id,
        "transaction_id": _str_id(subtransaction.get("transaction_id")),
        "amount": subtransaction.get("amount"),
        "memo": subtransaction.get("memo"),
        "payee_id": _str_id(subtransaction.get("payee_id")),
        "payee_name": subtransaction.get("payee_name"),
        "category_id": _str_id(subtransaction.get("category_id")),
        "category_name": subtransaction.get("category_name"),
        "transfer_account_id": _str_id(
            subtransaction.get("transfer_account_id")
        ),
        "transfer_transaction_id": _str_id(
            subtransaction.get("transfer_transaction_id")
        ),
        "deleted": subtransaction.get("deleted", False),
    }


def _scheduled_transaction_row(plan_id: str, transaction: Row) -> Row:
    return {
        "id": _str_id(transaction.get("id")),
        "budget_id": plan_id,
        "account_id": _str_id(transaction.get("account_id")),
        "payee_id": _str_id(transaction.get("payee_id")),
        "category_id": _str_id(transaction.get("category_id")),
        "transfer_account_id": _str_id(
            transaction.get("transfer_account_id")
        ),
        "date_first": _str_date(transaction.get("date_first")),
        "date_next": _str_date(transaction.get("date_next")),
        "frequency": _enum_value(transaction.get("frequency")),
        "amount": transaction.get("amount"),
        "memo": transaction.get("memo"),
        "flag_color": _enum_value(transaction.get("flag_color")),
        "flag_name": transaction.get("flag_name"),
        "account_name": transaction.get("account_name"),
        "payee_name": transaction.get("payee_name"),
        "category_name": transaction.get("category_name"),
        "deleted": transaction.get("deleted", False),
    }


def _scheduled_subtransaction_row(
    plan_id: str,
    transaction: Row,
    subtransaction: Row,
) -> Row:
    return {
        "id": _str_id(subtransaction.get("id")),
        "budget_id": plan_id,
        "scheduled_transaction_id": _str_id(transaction.get("id")),
        "amount": subtransaction.get("amount"),
        "memo": subtransaction.get("memo"),
        "payee_id": _str_id(subtransaction.get("payee_id")),
        "payee_name": subtransaction.get("payee_name"),
        "category_id": _str_id(subtransaction.get("category_id")),
        "category_name": subtransaction.get("category_name"),
        "transfer_account_id": _str_id(
            subtransaction.get("transfer_account_id")
        ),
        "deleted": subtransaction.get("deleted", False),
    }


def _month_row(month: Row) -> Row:
    return {
        "id": month.get("id"),
        "budget_id": month.get("budget_id"),
        "month": month.get("month"),
        "note": month.get("note"),
        "income": month.get("income"),
        "budgeted": month.get("budgeted"),
        "activity": month.get("activity"),
        "to_be_budgeted": month.get("to_be_budgeted"),
        "age_of_money": month.get("age_of_money"),
        "deleted": month.get("deleted", False),
    }


def _month_category_row(category: Row) -> Row:
    return {
        "id": category.get("id"),
        "budget_id": category.get("budget_id"),
        "month": category.get("month"),
        "category_id": _str_id(category.get("category_id")),
        "category_group_id": _str_id(
            category.get("category_group_id")
        ),
        "category_group_name": category.get("category_group_name"),
        "name": category.get("name"),
        "hidden": category.get("hidden", False),
        "internal": category.get("internal", False),
        "budgeted": category.get("budgeted"),
        "activity": category.get("activity"),
        "balance": category.get("balance"),
        "goal_type": _enum_value(category.get("goal_type")),
        "goal_needs_whole_amount": category.get(
            "goal_needs_whole_amount"
        ),
        "goal_day": category.get("goal_day"),
        "goal_cadence": category.get("goal_cadence"),
        "goal_cadence_frequency": category.get(
            "goal_cadence_frequency"
        ),
        "goal_creation_month": _str_date(
            category.get("goal_creation_month")
        ),
        "goal_target": category.get("goal_target"),
        "goal_target_month": _str_date(category.get("goal_target_month")),
        "goal_target_date": _str_date(category.get("goal_target_date")),
        "goal_percentage_complete": category.get(
            "goal_percentage_complete"
        ),
        "goal_months_to_budget": category.get("goal_months_to_budget"),
        "goal_under_funded": category.get("goal_under_funded"),
        "goal_overall_funded": category.get("goal_overall_funded"),
        "goal_overall_left": category.get("goal_overall_left"),
        "goal_snoozed_at": _str_date(category.get("goal_snoozed_at")),
        "deleted": category.get("deleted", False),
    }


def _nested_rows(row: Row, key: str) -> list[Row]:
    value = row.get(key)
    if not isinstance(value, list):
        return []
    return [nested for nested in value if isinstance(nested, dict)]


def _str_id(value: object) -> str | None:
    return str(value) if value is not None else None


def _str_date(value: object) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else _str_id(value)


def _enum_value(value: object) -> str | None:
    enum_value = getattr(value, "value", value)
    return _str_id(enum_value)


def _json_or_none(value: object) -> str | None:
    if value in (None, ""):
        return None
    return json.dumps(value, default=str)


def _single_plan_id(rows: list[Row]) -> str | None:
    plan_ids = {row.get("budget_id") for row in rows if row.get("budget_id")}
    if len(plan_ids) == 1:
        return str(next(iter(plan_ids)))
    return None
