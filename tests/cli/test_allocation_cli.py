from __future__ import annotations

import json

from typer.testing import CliRunner

from ynab_agent.cli import app
from ynab_agent.http_api.routers.allocation import (
    list_named_stresses as list_http_stresses,
    validate_allocation as validate_http_allocation,
)
from ynab_agent.planning.allocation import PortfolioAllocationPlan


runner = CliRunner()


def _plan() -> dict[str, object]:
    return {
        "market": {
            "us_equity": {
                "expected_return": 0.08,
                "volatility": 0.18,
            },
            "international_equity": {
                "expected_return": 0.07,
                "volatility": 0.20,
            },
            "bonds": {
                "expected_return": 0.04,
                "volatility": 0.07,
            },
            "cash": {
                "expected_return": 0.025,
                "volatility": 0.01,
            },
            "correlation": {
                "values": [
                    [1, 0.75, -0.1, 0],
                    [0.75, 1, -0.1, 0],
                    [-0.1, -0.1, 1, 0.2],
                    [0, 0, 0.2, 1],
                ]
            },
        },
        "accounts": [
            {
                "account_id": "401k",
                "portfolio_weight": 0.8,
                "annual_fee_rate": 0.003,
                "target": {
                    "us_equity": 0.7,
                    "international_equity": 0.2,
                    "bonds": 0.1,
                    "cash": 0,
                },
            },
            {
                "account_id": "brokerage",
                "portfolio_weight": 0.2,
                "target": {
                    "us_equity": 0.5,
                    "international_equity": 0.3,
                    "bonds": 0.1,
                    "cash": 0.1,
                },
            },
        ],
        "rebalancing": {
            "frequency_years": 2,
            "drift_threshold": 0.05,
        },
    }


def test_cli_and_http_allocation_representations_are_identical(
    tmp_path,
) -> None:
    plan_path = tmp_path / "allocation.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "wealth",
            "allocation",
            "validate",
            "--plan",
            str(plan_path),
            "--json",
        ],
    )
    http_response = validate_http_allocation(
        PortfolioAllocationPlan.model_validate(_plan())
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == http_response.model_dump(mode="json")


def test_cli_and_http_named_stress_catalogs_are_identical() -> None:
    result = runner.invoke(
        app,
        ["wealth", "allocation", "stresses", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == list_http_stresses().model_dump(
        mode="json"
    )["stresses"]
