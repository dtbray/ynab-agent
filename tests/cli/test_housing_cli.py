import json

from typer.testing import CliRunner

from ynab_agent.cli import app


runner = CliRunner()


def test_housing_projection_json_matches_public_representation(tmp_path) -> None:
    scenario = {
        "name": "CLI housing",
        "current_age": 65,
        "retirement_age": 65,
        "end_age": 67,
        "accounts": [{"id": "cash-reserve", "role": "cash"}],
        "starting_portfolio": 100_000,
        "annual_spending": 1,
        "tax_buckets": [
            {
                "account_id": "cash-reserve",
                "tax_treatment": "cash",
                "starting_balance": 100_000,
            }
        ],
        "tax_assumptions": {
            "ordinary_income_tax_rate": 0.2,
            "long_term_capital_gains_tax_rate": 0.15,
            "withdrawal_order": ["cash"],
            "retirement_surplus_destination": "cash",
            "apply_required_minimum_distributions": False,
        },
        "housing_plan": {
            "home": {
                "current_value": 500_000,
                "cost_basis": 500_000,
                "annual_appreciation_rate": 0,
            },
            "decision": {
                "kind": "sell",
                "event_age": 65,
                "proceeds_destination_account_id": "cash-reserve",
            },
        },
        "trials": 100,
    }
    path = tmp_path / "housing.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")

    result = runner.invoke(
        app,
        ["wealth", "housing", "project", "--scenario", str(path), "--json"],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["annual_housing"][0]["action"] == "sell"
    assert body["annual_housing"][0]["liquid_deposit"] > 0
    assert body["annual_housing"][0][
        "proceeds_destination_account_id"
    ] == "cash-reserve"
    assert body["manifest"]["schema_version"] == 3
