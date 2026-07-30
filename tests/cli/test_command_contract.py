"""Compatibility contracts for the public Typer command surface."""

from dataclasses import dataclass
import json
import re

import pytest
from rich.text import Text
from typer.testing import CliRunner

from ynab_agent.cli import app


runner = CliRunner()

COMMAND_TREE = {
    (): {
        "init",
        "poll",
        "query",
        "sync",
        "yeet-the-red",
        "oauth",
        "plans",
        "accounts",
        "transactions",
        "reports",
        "wealth",
    },
    ("oauth",): {"url", "exchange", "refresh", "status"},
    ("plans",): {"list"},
    ("accounts",): {"list", "reconcile"},
    ("transactions",): {"list", "create"},
    ("reports",): {
        "net-worth",
        "credit-cards",
        "debt-drag",
        "debt-plan",
        "changes",
        "cashflow",
        "overspending",
        "hidden-funds",
        "obligations",
        "burn-rate",
        "top-spend",
        "month",
        "cost-to-be-me",
        "hygiene",
    },
    ("wealth",): {
        "accounts",
        "optimize-social-security",
        "simulate",
        "solve",
        "performance",
        "spending",
        "scenarios",
        "tax",
        "tax-strategy",
    },
    ("wealth", "spending"): {"map", "list", "baseline", "preview"},
    ("wealth", "scenarios"): {"save", "compare"},
}

LEAF_COMMANDS = (
    ("init",),
    ("poll",),
    ("query",),
    ("sync",),
    ("yeet-the-red",),
    ("oauth", "url"),
    ("oauth", "exchange"),
    ("oauth", "refresh"),
    ("oauth", "status"),
    ("plans", "list"),
    ("accounts", "list"),
    ("accounts", "reconcile"),
    ("transactions", "list"),
    ("transactions", "create"),
    ("reports", "net-worth"),
    ("reports", "credit-cards"),
    ("reports", "debt-drag"),
    ("reports", "debt-plan"),
    ("reports", "changes"),
    ("reports", "cashflow"),
    ("reports", "overspending"),
    ("reports", "hidden-funds"),
    ("reports", "obligations"),
    ("reports", "burn-rate"),
    ("reports", "top-spend"),
    ("reports", "month"),
    ("reports", "cost-to-be-me"),
    ("reports", "hygiene"),
    ("wealth", "accounts"),
    ("wealth", "tax"),
    ("wealth", "tax-strategy"),
    ("wealth", "optimize-social-security"),
    ("wealth", "simulate"),
    ("wealth", "solve"),
    ("wealth", "performance"),
    ("wealth", "spending", "map"),
    ("wealth", "spending", "list"),
    ("wealth", "spending", "baseline"),
    ("wealth", "spending", "preview"),
    ("wealth", "scenarios", "save"),
    ("wealth", "scenarios", "compare"),
)

CRITICAL_OPTIONS = {
    ("sync",): {
        "--plan-id",
        "--since",
        "--month",
        "--full",
        "--skip-month-detail",
    },
    ("accounts", "reconcile"): {
        "--plan-id",
        "--account-id",
        "--since",
        "--include-closed",
        "--execute",
        "--yes",
        "--json",
    },
    ("transactions", "create"): {
        "--account-id",
        "--amount",
        "--date",
        "--plan-id",
        "--execute",
        "--yes",
        "--json",
    },
    ("wealth", "simulate"): {
        "--scenario",
        "--returns",
        "--html",
        "--json",
    },
    ("wealth", "optimize-social-security"): {
        "--scenario",
        "--json",
    },
    ("wealth", "solve"): {
        "--scenario",
        "--for",
        "--lower",
        "--upper",
        "--target-success",
        "--returns",
        "--html",
        "--json",
    },
    ("wealth", "performance"): {
        "--account-id",
        "--as-of",
        "--ending-balance",
        "--json",
    },
    ("wealth", "scenarios", "save"): {
        "--scenario",
        "--returns",
        "--json",
    },
    ("wealth", "scenarios", "compare"): {
        "--baseline",
        "--alternative",
        "--json",
    },
    ("wealth", "tax"): {"--input", "--json"},
    ("wealth", "tax-strategy"): {"--scenario", "--returns", "--json"},
}

_COMMAND_ROW = re.compile(r"^│ ([a-z][a-z0-9-]*)\s{2,}", re.MULTILINE)


@dataclass(frozen=True)
class InvocationResult:
    """Environment-independent subset of Click's invocation result."""

    exit_code: int
    output: str


def _invoke(args: list[str]) -> InvocationResult:
    result = runner.invoke(
        app,
        args,
        color=False,
        terminal_width=120,
    )
    return InvocationResult(
        exit_code=result.exit_code,
        output=Text.from_ansi(result.output).plain,
    )


def _command_names(help_output: str) -> set[str]:
    return set(_COMMAND_ROW.findall(help_output))


@pytest.mark.parametrize(
    ("path", "expected_commands"),
    COMMAND_TREE.items(),
    ids=lambda value: "root" if not value else "-".join(sorted(value)),
)
def test_command_tree_is_stable(path, expected_commands) -> None:
    result = _invoke([*path, "--help"])

    assert result.exit_code == 0, result.output
    assert _command_names(result.output) == expected_commands


@pytest.mark.parametrize("path", LEAF_COMMANDS, ids=lambda path: "-".join(path))
def test_every_leaf_command_has_help(path) -> None:
    result = _invoke([*path, "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "--help" in result.output


@pytest.mark.parametrize(
    ("path", "expected_options"),
    CRITICAL_OPTIONS.items(),
    ids=lambda value: "-".join(value) if isinstance(value, tuple) else None,
)
def test_critical_option_names_are_stable(path, expected_options) -> None:
    result = _invoke([*path, "--help"])

    assert result.exit_code == 0, result.output
    for option in expected_options:
        assert option in result.output


@pytest.mark.parametrize(
    "path",
    COMMAND_TREE,
    ids=lambda path: "root" if not path else "-".join(path),
)
def test_command_groups_reject_missing_subcommand(path) -> None:
    result = _invoke(list(path))

    assert result.exit_code == 2
    assert "Missing command." in result.output


def test_unknown_command_has_usage_error_exit_code() -> None:
    result = _invoke(["does-not-exist"])

    assert result.exit_code == 2
    assert "No such command 'does-not-exist'." in result.output


@pytest.mark.parametrize(
    ("path", "missing_option"),
    (
        (("transactions", "create"), "--account-id"),
        (("wealth", "simulate"), "--scenario"),
        (("wealth", "solve"), "--scenario"),
        (("wealth", "performance"), "--account-id"),
        (("wealth", "scenarios", "save"), "--scenario"),
        (("wealth", "scenarios", "compare"), "--baseline"),
    ),
    ids=lambda value: "-".join(value) if isinstance(value, tuple) else None,
)
def test_required_options_have_usage_error_exit_code(path, missing_option) -> None:
    result = _invoke(list(path))

    assert result.exit_code == 2
    assert f"Missing option '{missing_option}'." in result.output


def test_transaction_dry_run_json_contract() -> None:
    result = _invoke(
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
            "--payee-name",
            "Coffee Shop",
            "--memo",
            "dry run",
            "--json",
        ]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "dry_run": True,
        "transaction": {
            "plan_id": "plan-1",
            "account_id": "account-1",
            "date": "2026-05-20",
            "amount": -4250,
            "amount_display": "-4.25",
            "payee_id": None,
            "payee_name": "Coffee Shop",
            "category_id": None,
            "memo": "dry run",
            "cleared": "uncleared",
            "approved": False,
            "import_id": None,
        },
    }
