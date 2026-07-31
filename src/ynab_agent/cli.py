"""Command-line interface composition root for YNAB Agent."""

from decimal import Decimal, ROUND_HALF_UP

import typer
from ynab_agent.cli_commands.accounts import accounts_app
from ynab_agent.cli_commands.allocation import allocation_app
from ynab_agent.cli_commands.budget_rollover import rollover_app
from ynab_agent.cli_commands.budget_activity_reports import (
    budget_activity_reports_app,
)
from ynab_agent.cli_commands.cost_to_be_me import cost_to_be_me_app
from ynab_agent.cli_commands.database import database_app
from ynab_agent.cli_commands.debt_reports import debt_reports_app
from ynab_agent.cli_commands.natural_language import natural_language_app
from ynab_agent.cli_commands.obligation_reports import obligation_reports_app
from ynab_agent.cli_commands.oauth import oauth_app
from ynab_agent.cli_commands.overview_reports import overview_reports_app
from ynab_agent.cli_commands.plans import plans_app
from ynab_agent.cli_commands.polling import polling_app
from ynab_agent.cli_commands.reconciliation import (
    reconciliation_status_compat,
)
from ynab_agent.cli_commands.sync import sync_app
from ynab_agent.cli_commands.spending_reports import spending_reports_app
from ynab_agent.cli_commands.spending_guardrails import (
    spending_guardrails_app,
)
from ynab_agent.cli_commands.transactions import transactions_app
from ynab_agent.cli_commands.wealth import wealth_app
from ynab_agent.cli_commands.housing import housing_app
from ynab_agent.config import settings as settings
from ynab_agent.services.sync import _resolve_plan_id as _resolve_plan_id

app = typer.Typer(help="YNAB Agent - Transaction reminders and budget queries")
reports_app = typer.Typer(help="Analyze cached YNAB data")
app.add_typer(oauth_app, name="oauth")
app.add_typer(plans_app, name="plans")
app.add_typer(accounts_app, name="accounts")
app.add_typer(transactions_app, name="transactions")
app.add_typer(reports_app, name="reports")
reports_app.add_typer(debt_reports_app)
reports_app.add_typer(cost_to_be_me_app)
reports_app.add_typer(overview_reports_app)
reports_app.add_typer(budget_activity_reports_app)
reports_app.add_typer(spending_reports_app)
reports_app.add_typer(obligation_reports_app)
app.add_typer(wealth_app, name="wealth")
wealth_app.add_typer(allocation_app, name="allocation")
wealth_app.add_typer(spending_guardrails_app, name="spending")
wealth_app.add_typer(housing_app, name="housing")
app.add_typer(sync_app)
app.add_typer(rollover_app)
app.add_typer(database_app)
app.add_typer(polling_app)
app.add_typer(natural_language_app)


def _reconciliation_status(
    account: dict[str, object],
    transactions: list[dict[str, object]],
) -> dict[str, object]:
    """Compatibility wrapper for callers of the former CLI-local helper."""
    return reconciliation_status_compat(account, transactions)


def _dollars_to_milliunits(amount: str) -> int:
    """Convert a decimal dollar amount to YNAB milliunits."""
    decimal_amount = Decimal(amount)
    return int(
        (decimal_amount * Decimal("1000")).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


if __name__ == "__main__":
    app()
