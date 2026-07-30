from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path

from typer.testing import CliRunner

from ynab_agent.cli import app
from ynab_agent.cli_commands import spending_guardrails as command_module
from ynab_agent.planning.spending_guardrails import (
    RetirementSpendingPlan,
    SpendingTierAmounts,
)


runner = CliRunner()


def test_preview_json_uses_the_same_audit_shape_as_the_domain(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan = RetirementSpendingPlan.model_validate(
        {
            "baseline": {
                "essential": 30_000,
                "lifestyle": 20_000,
                "discretionary": 10_000,
            },
            "essential_floor": 24_000,
            "policy": {
                "kind": "withdrawal_rate_guardrails",
                "lower_withdrawal_rate": 0.03,
                "upper_withdrawal_rate": 0.05,
            },
        }
    )
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "wealth",
            "spending",
            "preview",
            "--plan",
            str(plan_path),
            "--opening-portfolio",
            "1000000",
            "--opening-portfolio",
            "2000000",
            "--start-age",
            "60",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["annual_path"][0]["action"] == "reduced"
    assert payload["summary"]["annual_path"][0]["spending_delta"][
        "discretionary"
    ] == -6000
    assert payload["summary"]["annual_path"][1]["action"] == "restored"


def test_baseline_passes_exclusive_month_and_emits_plan_json(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeService:
        async def derive_plan(self, **kwargs: object) -> RetirementSpendingPlan:
            calls.append(kwargs)
            return RetirementSpendingPlan(
                baseline=SpendingTierAmounts(essential=12_000),
                essential_floor=10_000,
            )

    @asynccontextmanager
    async def fake_scope(*args: object, **kwargs: object):
        yield FakeService()

    monkeypatch.setattr(
        command_module,
        "open_spending_guardrail_service",
        fake_scope,
    )

    result = runner.invoke(
        app,
        [
            "wealth",
            "spending",
            "baseline",
            "--budget-id",
            "budget-1",
            "--through-month",
            "2026-08",
            "--months",
            "6",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert str(calls[0]["through_month"]) == "2026-08-01"
    assert calls[0]["lookback_months"] == 6
    payload = json.loads(result.output)
    assert payload["plan"]["baseline"]["essential"] == 12_000
    assert payload["plan"]["policy"]["kind"] == "fixed_real"
