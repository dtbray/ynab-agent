from __future__ import annotations

import inspect
import json
from typing import Annotated, get_origin, get_type_hints

import pytest
from typer.testing import CliRunner

import ynab_agent.api.client as api_client_module
import ynab_agent.cli_commands.accounts as account_commands
from ynab_agent.cli import app
from ynab_agent.config import settings


runner = CliRunner()


def _account(
    account_id: str,
    *,
    closed: bool = False,
    deleted: bool = False,
) -> dict[str, object]:
    return {
        "id": account_id,
        "name": account_id.title(),
        "type": "checking",
        "on_budget": True,
        "closed": closed,
        "note": None,
        "balance": 1_250_000,
        "cleared_balance": 1_000_000,
        "uncleared_balance": 250_000,
        "transfer_payee_id": None,
        "direct_import_linked": True,
        "direct_import_in_error": False,
        "last_reconciled_at": None,
        "debt_original_balance": None,
        "debt_interest_rates": None,
        "debt_minimum_payments": None,
        "debt_escrow_amounts": None,
        "deleted": deleted,
    }


def test_accounts_list_resolves_runtime_plan_and_filters_closed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    requested_plans: list[str] = []

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            events.append("enter")
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> bool:
            events.append("exit")
            return False

        async def get_accounts(
            self,
            *,
            plan_id: str,
        ) -> list[dict[str, object]]:
            requested_plans.append(plan_id)
            return [
                _account("checking"),
                _account("closed", closed=True),
                _account("deleted", deleted=True),
            ]

    monkeypatch.setattr(api_client_module, "YnabClient", FakeClient)
    monkeypatch.setattr(settings, "ynab_plan_id", "runtime-plan")

    result = runner.invoke(app, ["accounts", "list", "--json"])

    assert result.exit_code == 0, result.output
    assert requested_plans == ["runtime-plan"]
    assert events == ["enter", "exit"]
    rows = json.loads(result.output)
    assert [row["id"] for row in rows] == ["checking"]
    assert rows[0]["balance"] == "1,250.00"
    assert rows[0]["cleared_balance"] == "1,000.00"
    assert rows[0]["uncleared_balance"] == "250.00"


def test_accounts_list_preserves_include_closed_and_table_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> bool:
            return False

        async def get_accounts(
            self,
            *,
            plan_id: str,
        ) -> list[dict[str, object]]:
            assert plan_id == "plan-1"
            return [_account("checking"), _account("closed", closed=True)]

    monkeypatch.setattr(api_client_module, "YnabClient", FakeClient)

    result = runner.invoke(
        app,
        [
            "accounts",
            "list",
            "--plan-id",
            "plan-1",
            "--include-closed",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Name" in result.output
    assert "Balance" in result.output
    assert "Checking" in result.output
    assert "Closed" in result.output


def test_accounts_list_closes_client_when_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FailingClient:
        async def __aenter__(self) -> FailingClient:
            events.append("enter")
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> bool:
            events.append("exit")
            return False

        async def get_accounts(
            self,
            *,
            plan_id: str,
        ) -> list[dict[str, object]]:
            raise RuntimeError(f"failed to fetch {plan_id}")

    monkeypatch.setattr(api_client_module, "YnabClient", FailingClient)

    result = runner.invoke(
        app,
        ["accounts", "list", "--plan-id", "plan-1"],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert events == ["enter", "exit"]


def test_accounts_list_callback_uses_annotated_options() -> None:
    callback = account_commands.list_accounts
    hints = get_type_hints(callback, include_extras=True)

    for parameter in inspect.signature(callback).parameters.values():
        assert get_origin(hints[parameter.name]) is Annotated


def test_accounts_app_registers_reconcile_and_list() -> None:
    callbacks = {command.name for command in account_commands.accounts_app.registered_commands}

    assert callbacks == {"list"}
    assert len(account_commands.accounts_app.registered_groups) == 1
