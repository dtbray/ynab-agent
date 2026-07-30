import asyncio
import json
from pathlib import Path

import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from ynab_agent import cli
from ynab_agent.cli import app
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.scenario_comparison import ScenarioComparison


runner = CliRunner()


def _scenario(**overrides) -> WealthScenario:
    values = {
        "name": "test",
        "current_age": 40,
        "retirement_age": 42,
        "end_age": 45,
        "starting_portfolio": 100_000,
        "annual_contribution": 10_000,
        "annual_spending": 20_000,
        "inflation_rate": 0,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 7,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def test_cli_saves_and_compares_immutable_scenario_revisions(
    monkeypatch,
    tmp_path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'scenario-cli.db'}"

    async def _initialize() -> None:
        database = DatabaseManager(database_url)
        try:
            await database.initialize()
        finally:
            await database.close()

    asyncio.run(_initialize())
    monkeypatch.setattr(cli.settings, "database_url", database_url)
    baseline_path = tmp_path / "baseline.local.json"
    alternative_path = tmp_path / "alternative.local.json"
    baseline_path.write_text(
        _scenario(name="Baseline", starting_portfolio=100_000).model_dump_json(),
        encoding="utf-8",
    )
    alternative_path.write_text(
        _scenario(name="Alternative", starting_portfolio=80_000).model_dump_json(),
        encoding="utf-8",
    )

    baseline_result = runner.invoke(
        app,
        [
            "wealth",
            "scenarios",
            "save",
            "--scenario",
            str(baseline_path),
            "--json",
        ],
    )
    alternative_result = runner.invoke(
        app,
        [
            "wealth",
            "scenarios",
            "save",
            "--scenario",
            str(alternative_path),
            "--json",
        ],
    )
    assert baseline_result.exit_code == 0, baseline_result.output
    assert alternative_result.exit_code == 0, alternative_result.output
    baseline = json.loads(baseline_result.output)
    alternative = json.loads(alternative_result.output)

    comparison_result = runner.invoke(
        app,
        [
            "wealth",
            "scenarios",
            "compare",
            "--baseline",
            baseline["id"],
            "--alternative",
            alternative["id"],
            "--json",
        ],
    )

    assert comparison_result.exit_code == 0, comparison_result.output
    comparison = ScenarioComparison.model_validate_json(
        comparison_result.output
    )
    assert comparison.baseline.revision_id == baseline["id"]
    assert comparison.alternatives[0].revision_id == alternative["id"]

    table_result = runner.invoke(
        app,
        [
            "wealth",
            "scenarios",
            "compare",
            "--baseline",
            baseline["id"],
            "--alternative",
            alternative["id"],
        ],
        terminal_width=300,
    )
    assert table_result.exit_code == 0, table_result.output
    assert "Funded" in table_result.output
    assert "Shortfall" in table_result.output
    assert "Guardrail" in table_result.output
    assert "After-Tax" in table_result.output


def test_wealth_tax_uses_domain_calculator(tmp_path: Path) -> None:
    request_path = tmp_path / "tax.json"
    request_path.write_text(
        json.dumps(
            {
                "filing_status": "single",
                "taxpayer_birth_year": 1980,
                "ordinary_income": 100_000,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["wealth", "tax", "--input", str(request_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_income_tax"] == 16_090.50
    assert payload["policy_manifest"]["policy_id"] == "us_in_2026_v1"


def test_cli_resolves_only_liquid_accounts(monkeypatch, tmp_path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'wealth.db'}"

    async def _seed() -> None:
        db = DatabaseManager(database_url)
        await db.initialize()
        async with db.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO accounts (
                      id, budget_id, name, type, on_budget, closed, balance,
                      cleared_balance, uncleared_balance, deleted
                    )
                    VALUES
                      ('brokerage', 'budget-1', 'Brokerage', 'otherAsset', 0, 0,
                       250000000, 250000000, 0, 0),
                      ('house', 'budget-1', 'House', 'otherAsset', 0, 0,
                       500000000, 500000000, 0, 0)
                    """
                )
            )
            await session.commit()
        await db.close()

    asyncio.run(_seed())
    monkeypatch.setattr(cli.settings, "database_url", database_url)
    scenario_path = tmp_path / "scenario.local.json"
    scenario_values = _scenario().model_dump(mode="json")
    scenario_values["starting_portfolio"] = None
    scenario_values["accounts"] = [
        {"id": "brokerage", "role": "taxable"},
        {"id": "house", "role": "real_estate"},
    ]
    scenario_path.write_text(
        json.dumps(scenario_values),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["wealth", "simulate", "--scenario", str(scenario_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["starting_portfolio"] == 250_000
    assert payload["seed"] == 7
    assert payload["funded_spending_ratio"]["p50"] == 1
    assert payload["cumulative_shortfall_real"]["p50"] == 0
    assert payload["goal_outcomes"][0]["kind"] == "planned_spending"
    assert payload["goal_outcomes"][0]["attainment_probability"] == 1
    assert payload["reproducibility"]["outcome_semantics"]["schema_version"] == 2
    assert payload["reproducibility"]["valuation"]["source"] == ("cached_liquid_accounts")
    assert payload["reproducibility"]["valuation"]["account_ids"] == ["brokerage"]
    assert payload["reproducibility"]["valuation"]["as_of"] is None
    assert len(payload["reproducibility"]["valuation"]["source_sha256"]) == 64


def test_cli_historical_simulation_writes_reproducible_report(tmp_path) -> None:
    scenario_path = tmp_path / "historical.local.json"
    scenario_path.write_text(
        json.dumps(
            _scenario(
                return_model="historical_bootstrap",
                historical_block_size=2,
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )
    returns_path = tmp_path / "returns.local.csv"
    returns_path.write_text(
        "year,nominal_return,inflation_rate\n"
        "2018,-0.04,0.02\n"
        "2019,0.30,0.018\n"
        "2020,0.21,0.012\n"
        "2021,0.25,0.047\n"
        "2022,-0.19,0.08\n"
        "2023,0.26,0.041\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "historical.local.html"

    result = runner.invoke(
        app,
        [
            "wealth",
            "simulate",
            "--scenario",
            str(scenario_path),
            "--returns",
            str(returns_path),
            "--html",
            str(report_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["assumptions"]["historical_source"] == str(returns_path.resolve())
    assert len(payload["assumptions"]["historical_sha256"]) == 64
    assert payload["reproducibility"]["historical"]["date_span"] == {
        "first_year": 2018,
        "last_year": 2023,
    }
    assert payload["reproducibility"]["historical"]["order_policy"] == "require_ascending"
    assert payload["reproducibility"]["historical"]["gap_policy"] == "reject"
    assert payload["reproducibility"]["valuation"]["source"] == "explicit_scenario_input"
    assert payload["html_path"] == str(report_path.resolve())
    assert report_path.exists()


def test_cli_historical_rows_must_be_in_ascending_order(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "historical.local.json"
    scenario_path.write_text(
        json.dumps(
            _scenario(
                return_model="historical_bootstrap",
                historical_block_size=1,
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )
    returns_path = tmp_path / "reordered.local.csv"
    returns_path.write_text(
        "year,nominal_return,inflation_rate\n2023,0.08,0.03\n2022,-0.04,0.05\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "wealth",
            "simulate",
            "--scenario",
            str(scenario_path),
            "--returns",
            str(returns_path),
            "--json",
        ],
        terminal_width=240,
    )

    assert result.exit_code == 2
    assert "historical years must be in strictly ascending" in result.output
    assert "require_ascending" in result.output


def test_cli_solve_preserves_historical_and_valuation_provenance(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "solve.local.json"
    scenario_path.write_text(
        json.dumps(
            _scenario(
                current_age=60,
                retirement_age=60,
                end_age=64,
                annual_contribution=0,
                annual_spending=20_000,
                return_model="historical_bootstrap",
                historical_block_size=1,
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )
    returns_path = tmp_path / "solve-returns.local.csv"
    returns_path.write_text(
        "year,nominal_return,inflation_rate\n2021,0,0\n2022,0,0\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "wealth",
            "solve",
            "--scenario",
            str(scenario_path),
            "--for",
            "annual_spending",
            "--lower",
            "1",
            "--upper",
            "50000",
            "--resolution",
            "100",
            "--target-success",
            "0.9",
            "--returns",
            str(returns_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    manifest = payload["simulation"]["reproducibility"]
    assert payload["value"] == 25_000
    assert manifest["historical"]["observation_count"] == 2
    assert manifest["historical"]["date_span"] == {
        "first_year": 2021,
        "last_year": 2022,
    }
    assert manifest["valuation"]["source"] == "explicit_scenario_input"


def test_wealth_accounts_shows_freshness(monkeypatch, tmp_path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'accounts.db'}"

    async def _seed() -> None:
        db = DatabaseManager(database_url)
        await db.initialize()
        async with db.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO accounts (
                      id, budget_id, name, type, on_budget, closed, balance,
                      cleared_balance, uncleared_balance, deleted
                    )
                    VALUES ('asset', 'budget-1', 'Asset', 'otherAsset', 0, 0,
                            1000000, 1000000, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO transactions (
                      id, budget_id, account_id, date, amount, approved, deleted
                    )
                    VALUES ('txn-1', 'budget-1', 'asset', '2026-07-01', 1000, 1, 0)
                    """
                )
            )
            await session.commit()
        await db.close()

    asyncio.run(_seed())
    monkeypatch.setattr(cli.settings, "database_url", database_url)

    result = runner.invoke(app, ["wealth", "accounts", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["account_id"] == "asset"
    assert payload[0]["latest_transaction"] == "2026-07-01"
    assert payload[0]["role"] == "unassigned"


def test_wealth_performance_uses_cached_tracking_flows(monkeypatch, tmp_path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'performance.db'}"

    async def _seed() -> None:
        db = DatabaseManager(database_url)
        await db.initialize()
        async with db.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO accounts (
                      id, budget_id, name, type, on_budget, closed, balance,
                      cleared_balance, uncleared_balance, deleted
                    )
                    VALUES ('brokerage', 'budget-1', 'Brokerage', 'otherAsset', 0, 0,
                            1100000, 1100000, 0, 0)
                    """
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO transactions (
                      id, budget_id, account_id, transfer_account_id,
                      payee_name, date, amount, approved, deleted
                    )
                    VALUES
                      ('contribution', 'budget-1', 'brokerage', 'checking',
                       'Transfer : Checking', '2021-01-01', 1000000, 1, 0),
                      ('market-update', 'budget-1', 'brokerage', NULL,
                       'Reconciliation Balance Adjustment',
                       '2021-12-31', 100000, 1, 0)
                    """
                )
            )
            await session.commit()
        await db.close()

    asyncio.run(_seed())
    monkeypatch.setattr(cli.settings, "database_url", database_url)

    result = runner.invoke(
        app,
        [
            "wealth",
            "performance",
            "--account-id",
            "brokerage",
            "--as-of",
            "2022-01-01",
            "--ending-balance",
            "1100",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ending_balance"] == 1_100
    assert payload["xirr"] == pytest.approx(0.10)
    assert payload["cash_flow_policy"] == "external_transfers_and_starting_balances"
    assert payload["excluded_transaction_count"] == 1
