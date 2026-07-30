"""YNAB API client wrapper with credential management."""

from dataclasses import dataclass
from datetime import date
from typing import Generic, Optional, TypeVar
from uuid import UUID
import ynab
from ynab.models.patch_transactions_wrapper import PatchTransactionsWrapper
from ynab.models.save_transaction_with_id_or_import_id import SaveTransactionWithIdOrImportId
from ynab.models.new_transaction import NewTransaction
from ynab.models.patch_month_category_wrapper import PatchMonthCategoryWrapper
from ynab.models.post_transactions_wrapper import PostTransactionsWrapper
from ynab.models.save_month_category import SaveMonthCategory
from ynab.models.save_transactions_response import SaveTransactionsResponse
from ynab.models.transaction_cleared_status import TransactionClearedStatus
from ynab.models.transactions_response import TransactionsResponse

from ynab_agent.config import settings
from ynab_agent.credentials.oauth import YnabOAuthManager
from ynab_agent.credentials.onepassword import OnePasswordCredentialManager

T = TypeVar("T")

NON_BUDGETABLE_CATEGORY_GROUPS = frozenset(
    {
        "Credit Card Payments",
        "Internal Master Category",
    }
)


@dataclass(frozen=True)
class SyncResult(Generic[T]):
    """Rows returned by a delta-capable YNAB endpoint plus its checkpoint."""

    data: T
    server_knowledge: int | None


class YnabClient:
    """Async-capable YNAB API client."""
    
    def __init__(self, access_token: Optional[str] = None):
        self._token = access_token
        self._client: Optional[ynab.ApiClient] = None
        self._accounts_api: Optional[ynab.AccountsApi] = None
        self._categories_api: Optional[ynab.CategoriesApi] = None
        self._months_api: Optional[ynab.MonthsApi] = None
        self._payees_api: Optional[ynab.PayeesApi] = None
        self._payee_locations_api: Optional[ynab.PayeeLocationsApi] = None
        self._scheduled_transactions_api: Optional[ynab.ScheduledTransactionsApi] = None
        self._plans_api: Optional[ynab.PlansApi] = None
        self._transactions_api: Optional[ynab.TransactionsApi] = None
    
    def _get_token(self) -> str:
        """Get access token from OAuth, 1Password, or PAT config."""
        if self._token:
            return self._token

        auth_mode = settings.ynab_auth_mode.lower()
        if auth_mode == "oauth":
            return YnabOAuthManager().access_token()
        
        if auth_mode == "onepassword" and settings.op_service_account_token:
            op_manager = OnePasswordCredentialManager(
                vault=settings.op_vault_name,
                item=settings.op_item_name
            )
            return op_manager.get_ynab_token()
        
        if auth_mode == "pat" and settings.ynab_access_token:
            return settings.ynab_access_token.get_secret_value()

        if auth_mode == "pat":
            raise ValueError("YNAB_AUTH_MODE=pat requires YNAB_ACCESS_TOKEN")
        if auth_mode == "onepassword":
            raise ValueError("YNAB_AUTH_MODE=onepassword requires OP_SERVICE_ACCOUNT_TOKEN")
        if auth_mode == "oauth":
            raise ValueError(
                "YNAB_AUTH_MODE=oauth requires OAuth setup. Run `ynab oauth url`, "
                "`ynab oauth exchange`, then `ynab oauth status`."
            )
        raise ValueError(
            "YNAB_AUTH_MODE must be one of: oauth, onepassword, pat "
            f"(got {auth_mode!r})"
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        configuration = ynab.Configuration(
            access_token=self._get_token()
        )
        self._client = ynab.ApiClient(configuration)
        self._accounts_api = ynab.AccountsApi(self._client)
        self._categories_api = ynab.CategoriesApi(self._client)
        self._months_api = ynab.MonthsApi(self._client)
        self._payees_api = ynab.PayeesApi(self._client)
        self._payee_locations_api = ynab.PayeeLocationsApi(self._client)
        self._scheduled_transactions_api = ynab.ScheduledTransactionsApi(self._client)
        self._plans_api = ynab.PlansApi(self._client)
        self._transactions_api = ynab.TransactionsApi(self._client)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        close = getattr(self._client, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result
        return False

    @property
    def accounts(self) -> ynab.AccountsApi:
        """Return the authenticated YNAB accounts API."""
        if not self._accounts_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._accounts_api

    @property
    def categories(self) -> ynab.CategoriesApi:
        """Return the authenticated YNAB categories API."""
        if not self._categories_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._categories_api

    @property
    def months(self) -> ynab.MonthsApi:
        """Return the authenticated YNAB months API."""
        if not self._months_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._months_api

    @property
    def payees(self) -> ynab.PayeesApi:
        """Return the authenticated YNAB payees API."""
        if not self._payees_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._payees_api

    @property
    def payee_locations(self) -> ynab.PayeeLocationsApi:
        """Return the authenticated YNAB payee locations API."""
        if not self._payee_locations_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._payee_locations_api

    @property
    def plans(self) -> ynab.PlansApi:
        """Return the authenticated YNAB plans API."""
        if not self._plans_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._plans_api

    @property
    def transactions(self) -> ynab.TransactionsApi:
        """Return the authenticated YNAB transactions API."""
        if not self._transactions_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._transactions_api

    @property
    def scheduled_transactions(self) -> ynab.ScheduledTransactionsApi:
        """Return the authenticated YNAB scheduled transactions API."""
        if not self._scheduled_transactions_api:
            raise RuntimeError("YnabClient must be used as an async context manager")
        return self._scheduled_transactions_api

    async def get_plans(self, include_accounts: bool = False) -> list[dict]:
        """Return available plans as dictionaries."""
        response = self.plans.get_plans(include_accounts=include_accounts)
        rows = [_plan_to_dict(plan) for plan in (response.data.plans or [])]
        if include_accounts:
            for row, plan in zip(rows, response.data.plans or [], strict=True):
                row["account_ids"] = [
                    str(account.id) for account in (getattr(plan, "accounts", None) or [])
                ]
        return rows

    async def get_accounts(
        self,
        plan_id: Optional[str] = None,
        last_knowledge_of_server: int | None = None,
        include_server_knowledge: bool = False,
    ) -> list[dict] | SyncResult[list[dict]]:
        """Return accounts for a plan as dictionaries."""
        plan_id = plan_id or settings.ynab_plan_id
        response = self.accounts.get_accounts(
            plan_id=plan_id,
            last_knowledge_of_server=last_knowledge_of_server,
        )
        rows = [
            {
                "id": account.id,
                "name": account.name,
                "type": account.type,
                "on_budget": account.on_budget,
                "closed": account.closed,
                "note": account.note,
                "balance": account.balance,
                "cleared_balance": account.cleared_balance,
                "uncleared_balance": account.uncleared_balance,
                "transfer_payee_id": account.transfer_payee_id,
                "direct_import_linked": account.direct_import_linked,
                "direct_import_in_error": account.direct_import_in_error,
                "last_reconciled_at": account.last_reconciled_at,
                "debt_original_balance": account.debt_original_balance,
                "debt_interest_rates": account.debt_interest_rates,
                "debt_minimum_payments": account.debt_minimum_payments,
                "debt_escrow_amounts": account.debt_escrow_amounts,
                "deleted": account.deleted,
            }
            for account in (response.data.accounts or [])
        ]
        return _sync_result(rows, response, include_server_knowledge)

    async def get_payees(
        self,
        plan_id: Optional[str] = None,
        last_knowledge_of_server: int | None = None,
        include_server_knowledge: bool = False,
    ) -> list[dict] | SyncResult[list[dict]]:
        """Return payees for a plan as dictionaries."""
        plan_id = plan_id or settings.ynab_plan_id
        response = self.payees.get_payees(
            plan_id=plan_id,
            last_knowledge_of_server=last_knowledge_of_server,
        )
        rows = [
            {
                "id": payee.id,
                "budget_id": plan_id,
                "name": payee.name,
                "transfer_account_id": payee.transfer_account_id,
                "deleted": payee.deleted,
            }
            for payee in (response.data.payees or [])
        ]
        return _sync_result(rows, response, include_server_knowledge)

    async def get_payee_locations(self, plan_id: Optional[str] = None) -> list[dict]:
        """Return all payee locations for a plan."""
        plan_id = plan_id or settings.ynab_plan_id
        response = self.payee_locations.get_payee_locations(plan_id=plan_id)
        return [
            {
                "id": location.id,
                "budget_id": plan_id,
                "payee_id": location.payee_id,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "deleted": location.deleted,
            }
            for location in (response.data.payee_locations or [])
        ]

    async def get_categories(
        self,
        plan_id: Optional[str] = None,
        last_knowledge_of_server: int | None = None,
        include_server_knowledge: bool = False,
    ) -> tuple[list[dict], list[dict]] | SyncResult[tuple[list[dict], list[dict]]]:
        """Return category groups and categories for a plan as dictionaries."""
        plan_id = plan_id or settings.ynab_plan_id
        response = self.categories.get_categories(
            plan_id=plan_id,
            last_knowledge_of_server=last_knowledge_of_server,
        )
        groups = []
        categories = []
        for group in response.data.category_groups or []:
            groups.append(
                {
                    "id": group.id,
                    "budget_id": plan_id,
                    "name": group.name,
                    "hidden": group.hidden,
                    "deleted": group.deleted,
                }
            )
            for category in group.categories or []:
                categories.append(_category_to_dict(category, plan_id))
        return _sync_result((groups, categories), response, include_server_knowledge)

    async def get_months(
        self,
        plan_id: Optional[str] = None,
        last_knowledge_of_server: int | None = None,
        include_server_knowledge: bool = False,
    ) -> list[dict] | SyncResult[list[dict]]:
        """Return delta-capable budget month summaries as dictionaries."""
        plan_id = plan_id or settings.ynab_plan_id
        response = self.months.get_plan_months(
            plan_id=plan_id,
            last_knowledge_of_server=last_knowledge_of_server,
        )
        rows = [_month_summary_to_dict(month, plan_id) for month in (response.data.months or [])]
        return _sync_result(rows, response, include_server_knowledge)

    async def get_month(self, plan_id: Optional[str], month: date) -> tuple[dict, list[dict]]:
        """Return one YNAB month and its categories as dictionaries."""
        plan_id = plan_id or settings.ynab_plan_id
        response = self.months.get_plan_month(plan_id=plan_id, month=month)
        month_detail = response.data.month
        month_row = {
            "id": f"{plan_id}:{month_detail.month.isoformat()}",
            "budget_id": plan_id,
            "month": month_detail.month.isoformat(),
            "note": month_detail.note,
            "income": month_detail.income,
            "budgeted": month_detail.budgeted,
            "activity": month_detail.activity,
            "to_be_budgeted": month_detail.to_be_budgeted,
            "age_of_money": month_detail.age_of_money,
            "deleted": month_detail.deleted,
        }
        categories = [
            _month_category_to_dict(category, plan_id, month_detail.month.isoformat())
            for category in (month_detail.categories or [])
        ]
        return month_row, categories

    async def get_transactions(
        self,
        plan_id: Optional[str] = None,
        since_date: Optional[date] = None,
        transaction_type: Optional[str] = None,
        last_knowledge_of_server: int | None = None,
        include_server_knowledge: bool = False,
    ) -> list[dict] | SyncResult[list[dict]]:
        """Return transactions for a plan as dictionaries."""
        plan_id = plan_id or settings.ynab_plan_id
        response: TransactionsResponse = self.transactions.get_transactions(
            plan_id=plan_id,
            since_date=since_date,
            type=transaction_type,
            last_knowledge_of_server=last_knowledge_of_server,
        )
        rows = [_transaction_to_dict(txn) for txn in (response.data.transactions or [])]
        return _sync_result(rows, response, include_server_knowledge)

    async def get_scheduled_transactions(
        self,
        plan_id: Optional[str] = None,
        last_knowledge_of_server: int | None = None,
        include_server_knowledge: bool = False,
    ) -> list[dict] | SyncResult[list[dict]]:
        """Return scheduled transactions and their split detail for a plan."""
        plan_id = plan_id or settings.ynab_plan_id
        response = self.scheduled_transactions.get_scheduled_transactions(
            plan_id=plan_id,
            last_knowledge_of_server=last_knowledge_of_server,
        )
        rows = [
            _scheduled_transaction_to_dict(transaction)
            for transaction in (response.data.scheduled_transactions or [])
        ]
        return _sync_result(rows, response, include_server_knowledge)

    async def get_transactions_by_account(
        self,
        *,
        plan_id: Optional[str] = None,
        account_id: str,
        since_date: Optional[date] = None,
        transaction_type: Optional[str] = None,
        last_knowledge_of_server: int | None = None,
        include_server_knowledge: bool = False,
    ) -> list[dict] | SyncResult[list[dict]]:
        """Return transactions for one account as dictionaries."""
        plan_id = plan_id or settings.ynab_plan_id
        response: TransactionsResponse = self.transactions.get_transactions_by_account(
            plan_id=plan_id,
            account_id=account_id,
            since_date=since_date,
            type=transaction_type,
            last_knowledge_of_server=last_knowledge_of_server,
        )
        rows = [_transaction_to_dict(txn) for txn in (response.data.transactions or [])]
        return _sync_result(rows, response, include_server_knowledge)
    
    async def get_unapproved_transactions(
        self,
        plan_id: Optional[str] = None
    ) -> list[dict]:
        """Get all unapproved transactions."""
        plan_id = plan_id or settings.ynab_plan_id
        
        response: TransactionsResponse = (
            self.transactions.get_transactions(
                plan_id=plan_id,
                type="unapproved"  # Key filter for reminders
            )
        )
        
        # Convert to dicts for easier handling
        return [_transaction_to_dict(txn) for txn in (response.data.transactions or [])]

    async def create_transaction(
        self,
        *,
        plan_id: Optional[str] = None,
        account_id: str,
        transaction_date: date,
        amount: int,
        payee_id: str | None = None,
        payee_name: str | None = None,
        category_id: str | None = None,
        memo: str | None = None,
        cleared: str = "uncleared",
        approved: bool = False,
        import_id: str | None = None,
    ) -> dict:
        """Create a single transaction in YNAB and return response metadata."""
        plan_id = plan_id or settings.ynab_plan_id
        transaction = NewTransaction(
            account_id=UUID(account_id),
            date=transaction_date,
            amount=amount,
            payee_id=UUID(payee_id) if payee_id else None,
            payee_name=payee_name,
            category_id=UUID(category_id) if category_id else None,
            memo=memo,
            cleared=TransactionClearedStatus(cleared),
            approved=approved,
            import_id=import_id,
        )
        response: SaveTransactionsResponse = self.transactions.create_transaction(
            plan_id=plan_id,
            data=PostTransactionsWrapper(transaction=transaction),
        )
        return {
            "transaction_ids": response.data.transaction_ids,
            "duplicate_import_ids": response.data.duplicate_import_ids,
            "server_knowledge": response.data.server_knowledge,
            "transaction": (
                _transaction_to_dict(response.data.transaction)
                if response.data.transaction
                else None
            ),
        }

    async def reconcile_transactions(
        self,
        *,
        plan_id: Optional[str] = None,
        transaction_ids: list[str],
    ) -> dict:
        """Mark existing YNAB transactions reconciled and return response metadata."""
        plan_id = plan_id or settings.ynab_plan_id
        if not transaction_ids:
            return {
                "transaction_ids": [],
                "duplicate_import_ids": [],
                "server_knowledge": None,
                "transactions": [],
            }

        response: SaveTransactionsResponse = self.transactions.update_transactions(
            plan_id=plan_id,
            data=PatchTransactionsWrapper(
                transactions=[
                    SaveTransactionWithIdOrImportId(
                        id=transaction_id,
                        cleared=TransactionClearedStatus("reconciled"),
                    )
                    for transaction_id in transaction_ids
                ],
            ),
        )
        if response is None or response.data is None:
            return {
                "transaction_ids": transaction_ids,
                "duplicate_import_ids": [],
                "server_knowledge": None,
                "transactions": [],
            }
        return {
            "transaction_ids": response.data.transaction_ids,
            "duplicate_import_ids": response.data.duplicate_import_ids,
            "server_knowledge": response.data.server_knowledge,
            "transactions": [
                _transaction_to_dict(transaction)
                for transaction in (response.data.transactions or [])
            ],
        }

    async def update_month_category_budgeted(
        self,
        *,
        plan_id: Optional[str] = None,
        month: date,
        category_id: str,
        budgeted: int,
    ) -> dict:
        """Update a month category budget after rejecting non-budgetable categories."""
        plan_id = plan_id or settings.ynab_plan_id
        await self._validate_budgetable_category(plan_id, category_id)
        response = self.categories.update_month_category(
            plan_id=plan_id,
            month=month,
            category_id=category_id,
            data=PatchMonthCategoryWrapper(
                category=SaveMonthCategory(budgeted=budgeted),
            ),
        )
        return _month_category_to_dict(
            response.data.category,
            plan_id,
            month.isoformat(),
        )

    async def _validate_budgetable_category(self, plan_id: str, category_id: str) -> None:
        category_groups, categories = await self.get_categories(plan_id=plan_id)
        group_names_by_id = {str(group["id"]): group["name"] for group in category_groups}
        category = next(
            (category for category in categories if str(category["id"]) == category_id),
            None,
        )

        if category is None:
            raise ValueError(f"Category {category_id} was not found in budget {plan_id}")

        group_name = group_names_by_id.get(str(category.get("category_group_id")), "Unknown")
        if group_name in NON_BUDGETABLE_CATEGORY_GROUPS:
            raise ValueError(
                f"Refusing to budget category {category['name']!r} in "
                f"non-budgetable group {group_name!r}"
            )
        if category.get("hidden"):
            raise ValueError(f"Refusing to budget hidden category {category['name']!r}")
        if category.get("deleted"):
            raise ValueError(f"Refusing to budget deleted category {category['name']!r}")


def _transaction_to_dict(txn) -> dict:
    """Convert a YNAB SDK transaction model into a plain dictionary."""
    return {
        "id": txn.id,
        "account_id": txn.account_id,
        "category_id": txn.category_id,
        "payee_id": txn.payee_id,
        "transfer_account_id": getattr(txn, "transfer_account_id", None),
        "transfer_transaction_id": getattr(txn, "transfer_transaction_id", None),
        "matched_transaction_id": getattr(txn, "matched_transaction_id", None),
        "import_id": getattr(txn, "import_id", None),
        "import_payee_name": getattr(txn, "import_payee_name", None),
        "import_payee_name_original": getattr(txn, "import_payee_name_original", None),
        "debt_transaction_type": _model_value(getattr(txn, "debt_transaction_type", None)),
        "account_name": getattr(txn, "account_name", None),
        "payee_name": getattr(txn, "payee_name", None),
        "category_name": getattr(txn, "category_name", None),
        "date": _model_date(txn, "date"),
        "amount": txn.amount,
        "memo": txn.memo,
        "cleared": txn.cleared,
        "approved": txn.approved,
        "flag_color": txn.flag_color,
        "flag_name": txn.flag_name,
        "foreign_amount": getattr(txn, "foreign_amount", None),
        "foreign_currency_code": getattr(txn, "foreign_currency_code", None),
        "deleted": getattr(txn, "deleted", False),
        "subtransactions": [
            _subtransaction_to_dict(subtransaction)
            for subtransaction in (getattr(txn, "subtransactions", None) or [])
        ],
    }


def _subtransaction_to_dict(subtransaction) -> dict:
    return {
        "id": subtransaction.id,
        "transaction_id": getattr(subtransaction, "transaction_id", None),
        "amount": subtransaction.amount,
        "memo": subtransaction.memo,
        "payee_id": subtransaction.payee_id,
        "payee_name": getattr(subtransaction, "payee_name", None),
        "category_id": subtransaction.category_id,
        "category_name": getattr(subtransaction, "category_name", None),
        "transfer_account_id": getattr(subtransaction, "transfer_account_id", None),
        "transfer_transaction_id": getattr(subtransaction, "transfer_transaction_id", None),
        "deleted": getattr(subtransaction, "deleted", False),
    }


def _scheduled_transaction_to_dict(transaction) -> dict:
    return {
        "id": transaction.id,
        "date_first": _model_date(transaction, "date_first"),
        "date_next": _model_date(transaction, "date_next"),
        "frequency": _model_value(transaction.frequency),
        "amount": transaction.amount,
        "memo": transaction.memo,
        "flag_color": _model_value(transaction.flag_color),
        "flag_name": transaction.flag_name,
        "account_id": transaction.account_id,
        "payee_id": transaction.payee_id,
        "category_id": transaction.category_id,
        "transfer_account_id": transaction.transfer_account_id,
        "account_name": getattr(transaction, "account_name", None),
        "payee_name": getattr(transaction, "payee_name", None),
        "category_name": getattr(transaction, "category_name", None),
        "deleted": transaction.deleted,
        "subtransactions": [
            {
                "id": subtransaction.id,
                "scheduled_transaction_id": getattr(
                    subtransaction,
                    "scheduled_transaction_id",
                    transaction.id,
                ),
                "amount": subtransaction.amount,
                "memo": subtransaction.memo,
                "payee_id": subtransaction.payee_id,
                "payee_name": getattr(subtransaction, "payee_name", None),
                "category_id": subtransaction.category_id,
                "category_name": getattr(subtransaction, "category_name", None),
                "transfer_account_id": getattr(subtransaction, "transfer_account_id", None),
                "deleted": getattr(subtransaction, "deleted", False),
            }
            for subtransaction in (getattr(transaction, "subtransactions", None) or [])
        ],
    }


def _sync_result(data: T, response, include_server_knowledge: bool) -> T | SyncResult[T]:
    if not include_server_knowledge:
        return data
    return SyncResult(
        data=data,
        server_knowledge=getattr(response.data, "server_knowledge", None),
    )


def _month_summary_to_dict(month, plan_id: str) -> dict:
    return {
        "id": f"{plan_id}:{month.month.isoformat()}",
        "budget_id": plan_id,
        "month": month.month.isoformat(),
        "note": month.note,
        "income": month.income,
        "budgeted": month.budgeted,
        "activity": month.activity,
        "to_be_budgeted": month.to_be_budgeted,
        "age_of_money": month.age_of_money,
        "deleted": getattr(month, "deleted", False),
    }


def _model_date(model, field: str) -> str | None:
    value = getattr(model, field, None) or getattr(model, f"var_{field}", None)
    return value.isoformat() if value else None


def _model_value(value):
    if hasattr(value, "value"):
        return value.value
    return value


def _model_json(value) -> str | None:
    if value in (None, ""):
        return None
    import json

    return json.dumps(value, default=str)


def _plan_to_dict(plan) -> dict:
    date_format = getattr(plan, "date_format", None)
    currency_format = getattr(plan, "currency_format", None)
    return {
        "id": plan.id,
        "name": plan.name,
        "last_modified_on": _model_date(plan, "last_modified_on"),
        "first_month": _model_date(plan, "first_month"),
        "last_month": _model_date(plan, "last_month"),
        "date_format": getattr(date_format, "format", None),
        "currency_format_iso_code": getattr(currency_format, "iso_code", None),
        "currency_format_example": getattr(currency_format, "example_format", None),
        "currency_decimal_digits": getattr(currency_format, "decimal_digits", None),
        "currency_decimal_separator": getattr(currency_format, "decimal_separator", None),
        "currency_symbol_first": getattr(currency_format, "symbol_first", None),
        "currency_group_separator": getattr(currency_format, "group_separator", None),
        "currency_symbol": getattr(currency_format, "currency_symbol", None),
        "currency_display_symbol": getattr(currency_format, "display_symbol", None),
    }


def _category_to_dict(category, plan_id: str) -> dict:
    return {
        "id": category.id,
        "budget_id": plan_id,
        "category_group_id": category.category_group_id,
        "name": category.name,
        "hidden": category.hidden,
        "internal": getattr(category, "internal", False),
        "original_category_group_id": category.original_category_group_id,
        "note": category.note,
        "budgeted": category.budgeted,
        "activity": category.activity,
        "balance": category.balance,
        "goal_type": _model_value(category.goal_type),
        "goal_needs_whole_amount": category.goal_needs_whole_amount,
        "goal_day": category.goal_day,
        "goal_cadence": category.goal_cadence,
        "goal_cadence_frequency": category.goal_cadence_frequency,
        "goal_creation_month": _model_date(category, "goal_creation_month"),
        "goal_target": category.goal_target,
        "goal_target_month": _model_date(category, "goal_target_month"),
        "goal_target_date": _model_date(category, "goal_target_date"),
        "goal_percentage_complete": category.goal_percentage_complete,
        "goal_months_to_budget": category.goal_months_to_budget,
        "goal_under_funded": category.goal_under_funded,
        "goal_overall_funded": category.goal_overall_funded,
        "goal_overall_left": category.goal_overall_left,
        "goal_snoozed_at": _model_date(category, "goal_snoozed_at"),
        "deleted": category.deleted,
    }


def _month_category_to_dict(category, plan_id: str, month: str) -> dict:
    return {
        "id": f"{plan_id}:{month}:{category.id}",
        "budget_id": plan_id,
        "month": month,
        "category_id": category.id,
        "category_group_id": getattr(category, "category_group_id", None),
        "category_group_name": getattr(category, "category_group_name", None),
        "name": getattr(category, "name", None),
        "hidden": getattr(category, "hidden", False),
        "internal": getattr(category, "internal", False),
        "deleted": getattr(category, "deleted", False),
        "budgeted": category.budgeted,
        "activity": category.activity,
        "balance": category.balance,
        "goal_type": _model_value(getattr(category, "goal_type", None)),
        "goal_needs_whole_amount": getattr(category, "goal_needs_whole_amount", None),
        "goal_day": getattr(category, "goal_day", None),
        "goal_cadence": getattr(category, "goal_cadence", None),
        "goal_cadence_frequency": getattr(category, "goal_cadence_frequency", None),
        "goal_creation_month": _model_date(category, "goal_creation_month"),
        "goal_target": getattr(category, "goal_target", None),
        "goal_target_month": _model_date(category, "goal_target_month"),
        "goal_target_date": _model_date(category, "goal_target_date"),
        "goal_percentage_complete": getattr(category, "goal_percentage_complete", None),
        "goal_months_to_budget": getattr(category, "goal_months_to_budget", None),
        "goal_under_funded": getattr(category, "goal_under_funded", None),
        "goal_overall_funded": getattr(category, "goal_overall_funded", None),
        "goal_overall_left": getattr(category, "goal_overall_left", None),
        "goal_snoozed_at": _model_date(category, "goal_snoozed_at"),
    }
