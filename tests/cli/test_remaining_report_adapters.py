from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import date
import inspect
import json
from types import ModuleType
from typing import Annotated, get_origin, get_type_hints

import pytest
from typer.testing import CliRunner

import ynab_agent.cli_commands.budget_activity_reports as budget_commands
import ynab_agent.cli_commands.obligation_reports as obligation_commands
import ynab_agent.cli_commands.overview_reports as overview_commands
import ynab_agent.cli_commands.spending_reports as spending_commands
from ynab_agent.cli import app
from ynab_agent.config import Settings, settings
from ynab_agent.services.reports.budget_activity import (
    BudgetActivityRequest,
    CategoryActivity,
    OverspendingCategory,
)
from ynab_agent.services.reports.obligations import (
    ObligationRequest,
    ScheduledObligation,
)
from ynab_agent.services.reports.overview import (
    CachedChange,
    HygieneCheck,
    NetWorthBucket,
)
from ynab_agent.services.reports.spending import (
    BurnRateBucket,
    BurnRateGrouping,
    CashflowMonth,
    DateRangeRequest,
    PayeeSpend,
)


runner = CliRunner()


class FixedDate(date):
    @classmethod
    def today(cls) -> FixedDate:
        return cls(2027, 2, 18)


def _managed_database(events: list[tuple[str, bool]]) -> Callable[..., object]:
    @asynccontextmanager
    async def manager(
        app_settings: Settings,
        *,
        initialize: bool = False,
    ) -> AsyncIterator[object]:
        assert app_settings is settings
        events.append(("enter", initialize))
        try:
            yield object()
        finally:
            events.append(("exit", initialize))

    return manager


@asynccontextmanager
async def _unexpected_database(
    app_settings: Settings,
    *,
    initialize: bool = False,
) -> AsyncIterator[object]:
    raise AssertionError(f"database opened during dry validation: {app_settings}, {initialize}")
    yield object()


def test_overview_commands_translate_typed_rows_and_manage_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, bool]] = []

    class FakeService:
        def __init__(self, repository: object) -> None:
            self.repository = repository

        async def net_worth(self) -> tuple[NetWorthBucket, ...]:
            return (
                NetWorthBucket(
                    bucket="on_budget",
                    account_type="checking",
                    account_count=2,
                    balance_milliunits=4_000_000,
                ),
            )

        async def latest_changes(
            self,
            plan_id: str | None = None,
        ) -> tuple[CachedChange, ...]:
            assert plan_id == "plan-1"
            return (
                CachedChange(
                    batch_id="batch-2",
                    budget_id="plan-1",
                    resource="accounts",
                    action="updated",
                    entity_name="Checking",
                    entity_id="account-1",
                    recorded_at="2027-02-18 10:00:00",
                ),
            )

        async def hygiene(
            self,
            *,
            since: date,
        ) -> tuple[HygieneCheck, ...]:
            assert since == date(2026, 1, 1)
            return (HygieneCheck(check_name="unapproved", count=3),)

    monkeypatch.setattr(
        overview_commands,
        "open_database",
        _managed_database(events),
    )
    monkeypatch.setattr(overview_commands, "OverviewReportService", FakeService)

    net_worth = runner.invoke(app, ["reports", "net-worth", "--json"])
    changes = runner.invoke(
        app,
        ["reports", "changes", "--plan-id", "plan-1", "--json"],
    )
    hygiene = runner.invoke(
        app,
        ["reports", "hygiene", "--since", "2026-01-01", "--json"],
    )

    assert net_worth.exit_code == 0, net_worth.output
    assert json.loads(net_worth.output)[0]["balance"] == "4,000.00"
    assert changes.exit_code == 0, changes.output
    change_row = json.loads(changes.output)[0]
    assert change_row["batch_id"] == "batch-2"
    assert change_row["budget_id"] == "plan-1"
    assert change_row["entity_id"] == "account-1"
    assert hygiene.exit_code == 0, hygiene.output
    assert json.loads(hygiene.output)[0] == {
        "check_name": "unapproved",
        "count": 3,
    }
    assert events == [
        ("enter", False),
        ("exit", False),
        ("enter", True),
        ("exit", True),
        ("enter", False),
        ("exit", False),
    ]


def test_budget_activity_commands_use_invocation_month_and_legacy_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[BudgetActivityRequest] = []

    class FakeService:
        def __init__(self, repository: object) -> None:
            self.repository = repository

        async def overspending(
            self,
            request: BudgetActivityRequest,
        ) -> tuple[OverspendingCategory, ...]:
            requests.append(request)
            return (
                OverspendingCategory(
                    group_name="Bills",
                    category="Groceries",
                    budgeted_milliunits=200_000,
                    activity_milliunits=-350_000,
                    balance_milliunits=-150_000,
                    overspent_milliunits=150_000,
                ),
            )

        async def hidden_funds(
            self,
            request: BudgetActivityRequest,
        ) -> tuple[CategoryActivity, ...]:
            requests.append(request)
            return ()

        async def month_activity(
            self,
            request: BudgetActivityRequest,
        ) -> tuple[CategoryActivity, ...]:
            requests.append(request)
            return ()

    monkeypatch.setattr(budget_commands, "date", FixedDate)
    monkeypatch.setattr(
        budget_commands,
        "open_database",
        _managed_database([]),
    )
    monkeypatch.setattr(
        budget_commands,
        "BudgetActivityService",
        FakeService,
    )

    overspending = runner.invoke(
        app,
        ["reports", "overspending", "--json"],
    )
    hidden = runner.invoke(
        app,
        ["reports", "hidden-funds", "--month", "2026-05-01", "--json"],
    )
    month = runner.invoke(
        app,
        ["reports", "month", "--month", "2026-05-01", "--json"],
    )

    assert overspending.exit_code == 0, overspending.output
    assert json.loads(overspending.output)[0]["overspent"] == "150.00"
    assert hidden.exit_code == 0, hidden.output
    assert month.exit_code == 0, month.output
    assert requests[0].month == date(2027, 2, 1)
    assert requests[1].month == date(2026, 5, 1)
    assert requests[2].limit == 15


def test_spending_commands_build_typed_requests_and_render_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    burn_requests: list[tuple[DateRangeRequest, BurnRateGrouping, date]] = []

    class FakeService:
        def __init__(self, repository: object) -> None:
            self.repository = repository

        async def cashflow(
            self,
            request: DateRangeRequest,
        ) -> tuple[CashflowMonth, ...]:
            assert request.since == date(2026, 1, 1)
            return (
                CashflowMonth(
                    month="2026-05",
                    inflow_milliunits=100_000,
                    outflow_milliunits=50_000,
                    net_milliunits=50_000,
                    transaction_count=2,
                ),
            )

        async def burn_rate(
            self,
            request: DateRangeRequest,
            *,
            grouping: BurnRateGrouping,
            current_date: date,
        ) -> tuple[BurnRateBucket, ...]:
            burn_requests.append((request, grouping, current_date))
            return (
                BurnRateBucket(
                    bucket="Bills",
                    outflow_milliunits=1_650_000,
                    transaction_count=3,
                    active_months=2,
                    monthly_burn_milliunits=825_000,
                ),
            )

        async def top_spend(
            self,
            request: DateRangeRequest,
        ) -> tuple[PayeeSpend, ...]:
            assert request.limit == 4
            return (
                PayeeSpend(
                    payee="Market",
                    outflow_milliunits=650_000,
                    transaction_count=2,
                ),
            )

    monkeypatch.setattr(spending_commands, "date", FixedDate)
    monkeypatch.setattr(
        spending_commands,
        "open_database",
        _managed_database([]),
    )
    monkeypatch.setattr(
        spending_commands,
        "SpendingReportService",
        FakeService,
    )

    cashflow = runner.invoke(
        app,
        ["reports", "cashflow", "--since", "2026-01-01", "--json"],
    )
    burn = runner.invoke(
        app,
        [
            "reports",
            "burn-rate",
            "--since",
            "2026-12-01",
            "--group-by",
            "group",
            "--json",
        ],
    )
    top = runner.invoke(
        app,
        ["reports", "top-spend", "--limit", "4", "--json"],
    )

    assert cashflow.exit_code == 0, cashflow.output
    assert json.loads(cashflow.output)[0]["net"] == "50.00"
    assert burn.exit_code == 0, burn.output
    assert json.loads(burn.output)[0]["monthly_burn"] == "825.00"
    assert top.exit_code == 0, top.output
    assert json.loads(top.output)[0]["payee"] == "Market"
    request, grouping, current_date = burn_requests[0]
    assert request.through is None
    assert grouping is BurnRateGrouping.GROUP
    assert current_date == date(2027, 2, 18)


def test_obligations_uses_invocation_date_and_preserves_null_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[ObligationRequest] = []

    class FakeService:
        def __init__(self, repository: object) -> None:
            self.repository = repository

        async def list(
            self,
            request: ObligationRequest,
        ) -> tuple[ScheduledObligation, ...]:
            requests.append(request)
            return (
                ScheduledObligation(
                    due_date="2027-02-18",
                    payee="Unknown",
                    group_name="Unknown",
                    category=None,
                    account=None,
                    frequency=None,
                    amount_milliunits=25_000,
                    memo=None,
                ),
            )

    monkeypatch.setattr(obligation_commands, "date", FixedDate)
    monkeypatch.setattr(
        obligation_commands,
        "open_database",
        _managed_database([]),
    )
    monkeypatch.setattr(
        obligation_commands,
        "ObligationService",
        FakeService,
    )

    result = runner.invoke(
        app,
        ["reports", "obligations", "--include-inflows", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert requests[0].through == date(2027, 2, 18)
    assert requests[0].include_inflows is True
    assert json.loads(result.output)[0]["category"] is None


@pytest.mark.parametrize(
    ("module", "args", "message"),
    (
        (
            overview_commands,
            ["reports", "hygiene", "--since", "bad-date"],
            "must use YYYY-MM-DD",
        ),
        (
            budget_commands,
            ["reports", "overspending", "--month", "bad-date"],
            "must use YYYY-MM-DD",
        ),
        (
            spending_commands,
            ["reports", "burn-rate", "--group-by", "payee"],
            'group-by must be "category" or "group"',
        ),
        (
            obligation_commands,
            ["reports", "obligations", "--through", "bad-date"],
            "must use YYYY-MM-DD",
        ),
    ),
)
def test_dry_validation_errors_do_not_open_database(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    args: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(module, "open_database", _unexpected_database)

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize(
    "module",
    (
        overview_commands,
        budget_commands,
        spending_commands,
        obligation_commands,
    ),
)
def test_report_command_modules_contain_no_sql(module: ModuleType) -> None:
    source = inspect.getsource(module)
    assert "SELECT " not in source
    assert " FROM " not in source


@pytest.mark.parametrize(
    "callback",
    tuple(
        command.callback
        for command_app in (
            overview_commands.overview_reports_app,
            budget_commands.budget_activity_reports_app,
            spending_commands.spending_reports_app,
            obligation_commands.obligation_reports_app,
        )
        for command in command_app.registered_commands
    ),
)
def test_report_callbacks_use_annotated_options(
    callback: Callable[..., object],
) -> None:
    hints = get_type_hints(callback, include_extras=True)
    for parameter in inspect.signature(callback).parameters.values():
        assert get_origin(hints[parameter.name]) is Annotated
