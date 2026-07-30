from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date
import inspect
import json
from types import SimpleNamespace
from typing import Annotated, get_origin, get_type_hints

import pytest
from typer.testing import CliRunner

import ynab_agent.api.client as api_client_module
import ynab_agent.cli_commands.database as database_commands
import ynab_agent.cli_commands.natural_language as query_commands
import ynab_agent.cli_commands.oauth as oauth_commands
import ynab_agent.cli_commands.plans as plan_commands
import ynab_agent.cli_commands.polling as polling_commands
import ynab_agent.cli_commands.transactions as transaction_commands
import ynab_agent.credentials.oauth as oauth_module
from ynab_agent.cli import app
from ynab_agent.config import Settings, settings
from ynab_agent.queries.natural_language import BudgetQueryResult


runner = CliRunner()


class FixedDate(date):
    @classmethod
    def today(cls) -> FixedDate:
        return cls(2026, 7, 29)


def test_transaction_defaults_resolve_at_each_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transaction_commands, "date", FixedDate)
    monkeypatch.setattr(
        settings,
        "ynab_plan_id",
        "runtime-plan",
    )

    result = runner.invoke(
        app,
        [
            "transactions",
            "create",
            "--account-id",
            "account-1",
            "--amount=-4.25",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)["transaction"]
    assert payload["plan_id"] == "runtime-plan"
    assert payload["date"] == "2026-07-29"


def test_transaction_execute_crosses_client_boundary_only_after_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

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

        async def create_transaction(
            self,
            **kwargs: object,
        ) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "transaction_ids": ["transaction-1"],
                "server_knowledge": 42,
            }

    monkeypatch.setattr(api_client_module, "YnabClient", FakeClient)

    rejected = runner.invoke(
        app,
        [
            "transactions",
            "create",
            "--account-id",
            "account-1",
            "--amount=-4.25",
            "--execute",
        ],
    )
    accepted = runner.invoke(
        app,
        [
            "transactions",
            "create",
            "--plan-id",
            "plan-1",
            "--account-id",
            "account-1",
            "--amount=-4.25",
            "--date",
            "2026-05-20",
            "--execute",
            "--yes",
            "--json",
        ],
    )

    assert rejected.exit_code == 2
    assert calls == [
        {
            "plan_id": "plan-1",
            "account_id": "account-1",
            "transaction_date": date(2026, 5, 20),
            "amount": -4_250,
            "payee_id": None,
            "payee_name": None,
            "category_id": None,
            "memo": None,
            "cleared": "uncleared",
            "approved": False,
            "import_id": None,
        }
    ]
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(accepted.output)["transaction_ids"] == ["transaction-1"]


def test_plan_list_uses_api_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[bool] = []

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

        async def get_plans(
            self,
            include_accounts: bool = False,
        ) -> list[dict[str, object]]:
            requested.append(include_accounts)
            return [
                {
                    "id": "plan-1",
                    "name": "Household",
                    "first_month": "2026-01-01",
                    "last_month": "2026-12-01",
                    "last_modified_on": "2026-07-29",
                }
            ]

    monkeypatch.setattr(api_client_module, "YnabClient", FakeClient)

    result = runner.invoke(
        app,
        ["plans", "list", "--include-accounts", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert requested == [True]
    assert json.loads(result.output)[0]["name"] == "Household"


def test_init_uses_managed_initialized_database_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    @asynccontextmanager
    async def fake_open_database(
        app_settings: Settings,
        *,
        initialize: bool = False,
    ) -> AsyncIterator[object]:
        events.extend(("enter", app_settings, initialize))
        try:
            yield object()
        finally:
            events.append("exit")

    monkeypatch.setattr(
        database_commands,
        "open_database",
        fake_open_database,
    )

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    assert events == [
        "enter",
        settings,
        True,
        "exit",
    ]
    assert "Database initialized" in result.output


def test_poll_once_uses_runtime_interval_and_managed_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    database = object()

    @asynccontextmanager
    async def fake_open_database(
        app_settings: Settings,
        *,
        initialize: bool = False,
    ) -> AsyncIterator[object]:
        events.extend(("enter", app_settings, initialize))
        try:
            yield database
        finally:
            events.append("exit")

    class FakePoller:
        def __init__(self, received_database: object) -> None:
            events.append(("poller", received_database))

        async def run_once(self) -> int:
            events.append("once")
            return 3

    monkeypatch.setattr(polling_commands, "open_database", fake_open_database)
    monkeypatch.setattr(polling_commands, "TransactionPoller", FakePoller)
    monkeypatch.setattr(
        settings,
        "poll_interval_minutes",
        27,
    )

    result = runner.invoke(app, ["poll", "--once"])

    assert result.exit_code == 0, result.output
    assert events == [
        "enter",
        settings,
        True,
        ("poller", database),
        "once",
        "exit",
    ]
    assert "Sent 3 reminders" in result.output


def test_poll_interval_default_resolves_at_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intervals: list[int] = []

    @asynccontextmanager
    async def fake_open_database(
        app_settings: Settings,
        *,
        initialize: bool = False,
    ) -> AsyncIterator[object]:
        assert app_settings is settings
        assert initialize is True
        yield object()

    class FakePoller:
        def __init__(self, received_database: object) -> None:
            self.database = received_database

        async def run_continuous(self, interval: int) -> None:
            intervals.append(interval)

    monkeypatch.setattr(polling_commands, "open_database", fake_open_database)
    monkeypatch.setattr(polling_commands, "TransactionPoller", FakePoller)
    monkeypatch.setattr(settings, "poll_interval_minutes", 27)

    result = runner.invoke(app, ["poll"])

    assert result.exit_code == 0, result.output
    assert intervals == [27]
    assert "interval: 27 min" in result.output


def test_natural_language_query_uses_managed_database_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    database = object()

    @asynccontextmanager
    async def fake_open_database(
        app_settings: Settings,
        *,
        initialize: bool = False,
    ) -> AsyncIterator[object]:
        events.extend(("enter", app_settings, initialize))
        try:
            yield database
        finally:
            events.append("exit")

    class FakeQuerier:
        def __init__(self, received_database: object) -> None:
            events.append(("querier", received_database))

        async def query(self, question: str) -> BudgetQueryResult:
            events.append(("question", question))
            return BudgetQueryResult(
                query=question,
                sql="SELECT 1",
                summary="One matching row",
                data=[{"category": "Groceries", "amount": 25_000}],
            )

    monkeypatch.setattr(
        query_commands,
        "open_database",
        fake_open_database,
    )
    monkeypatch.setattr(
        query_commands,
        "NaturalLanguageQuerier",
        FakeQuerier,
    )

    result = runner.invoke(app, ["query", "top spending"])

    assert result.exit_code == 0, result.output
    assert events == [
        "enter",
        settings,
        True,
        ("querier", database),
        ("question", "top spending"),
        "exit",
    ]
    assert "One matching row" in result.output
    assert "Groceries" in result.output


def test_oauth_quiet_status_preserves_nonzero_unusable_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeManager:
        def status(self) -> SimpleNamespace:
            return SimpleNamespace(
                path=".token.json",
                exists=True,
                has_access_token=True,
                has_refresh_token=False,
                expires_at=None,
                seconds_until_expiry=None,
                refresh_recommended=True,
            )

    monkeypatch.setattr(oauth_module, "YnabOAuthManager", FakeManager)

    result = runner.invoke(app, ["oauth", "status", "--quiet"])

    assert result.exit_code == 1
    assert result.output == ""


@pytest.mark.parametrize(
    "callback",
    (
        oauth_commands.oauth_url,
        oauth_commands.oauth_exchange,
        oauth_commands.oauth_refresh,
        oauth_commands.oauth_status,
        plan_commands.list_plans,
        transaction_commands.list_transactions,
        transaction_commands.create_transaction,
        polling_commands.poll,
        query_commands.query,
    ),
)
def test_moved_typer_parameters_use_annotated(
    callback: Callable[..., object],
) -> None:
    signature = inspect.signature(callback)
    hints = get_type_hints(callback, include_extras=True)

    assert signature.parameters
    for name in signature.parameters:
        assert get_origin(hints[name]) is Annotated
