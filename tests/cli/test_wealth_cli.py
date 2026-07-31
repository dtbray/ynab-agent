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


def _allocation() -> dict[str, object]:
    return {
        "market": {
            "us_equity": {"expected_return": 0.08, "volatility": 0.18},
            "international_equity": {
                "expected_return": 0.07,
                "volatility": 0.2,
            },
            "bonds": {"expected_return": 0.04, "volatility": 0.07},
            "cash": {"expected_return": 0.02, "volatility": 0.01},
            "correlation": {
                "values": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            },
        },
        "accounts": [
            {
                "account_id": "portfolio",
                "portfolio_weight": 1,
                "target": {
                    "us_equity": 0.6,
                    "international_equity": 0.2,
                    "bonds": 0.15,
                    "cash": 0.05,
                },
            }
        ],
    }


def test_cli_runs_named_multi_asset_stress(tmp_path: Path) -> None:
    scenario_path = tmp_path / "stress.local.json"
    scenario_path.write_text(
        json.dumps(
            _scenario(
                portfolio_allocation=_allocation(),
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "wealth",
            "simulate",
            "--scenario",
            str(scenario_path),
            "--stress",
            "equity_crash",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["assumptions"]["named_stress"] == "equity_crash"
    assert payload["reproducibility"]["multi_asset_paths"][
        "source_name"
    ] == "named_stress_catalog_v1:equity_crash"


def test_cli_emits_healthcare_ltc_metrics_and_manifest(
    tmp_path: Path,
) -> None:
    scenario = _scenario(
        current_age=64,
        retirement_age=64,
        end_age=67,
        annual_contribution=0,
        starting_portfolio=100_000,
        accounts=[
            {
                "id": "home-reserve",
                "role": "taxable",
                "owner_person_id": "alex",
            },
            {
                "id": "cash",
                "role": "cash",
                "owner_person_id": "alex",
            },
        ],
        tax_buckets=[
            {
                "account_id": "home-reserve",
                "owner_person_id": "alex",
                "tax_treatment": "taxable",
                "starting_balance": 0,
                "taxable_basis": 0,
            },
            {
                "account_id": "cash",
                "owner_person_id": "alex",
                "tax_treatment": "cash",
                "starting_balance": 100_000,
            },
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["taxable", "cash"],
            "retirement_surplus_destination": "cash",
        },
        housing_plan={
            "home": {
                "current_value": 5_000,
                "cost_basis": 5_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {
                "kind": "sell",
                "event_age": 64,
                "proceeds_destination_account_id": "home-reserve",
            },
            "care": {"funding_account_id": "home-reserve"},
        },
        household={
            "plan_start_date": "2026-01-02",
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1962-01-02",
                    "retirement_age_months": 64 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 90,
                    },
                }
            ],
        },
        healthcare={
            "medical_inflation_rate": 0,
            "ltc_funding_source": "home_equity",
            "home_equity_available_for_ltc_real": 5_000,
            "people": [
                {
                    "person_id": "alex",
                    "pre_medicare_aca_annual_premium_real": 2_000,
                    "long_term_care": {
                        "lifetime_incidence_probability": 1,
                        "minimum_onset_age": 64,
                        "maximum_onset_age": 64,
                        "mean_duration_years": 1,
                        "duration_standard_deviation_years": 0,
                        "maximum_duration_years": 1,
                        "annual_cost_real": 5_000,
                    },
                }
            ],
        },
    )
    scenario_path = tmp_path / "healthcare.local.json"
    scenario_path.write_text(
        scenario.model_dump_json(),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "wealth",
            "simulate",
            "--scenario",
            str(scenario_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["healthcare_metrics"][
        "ltc_in_plan_incidence_probability"
    ] == 1
    assert payload["annual_healthcare_real"][0]["ltc_active_trials"] == 100
    assert payload["annual_housing"][0][
        "care_funded_from_home_equity_nominal"
    ]["p50"] == 5_000
    assert payload["healthcare_metrics"][
        "lifetime_ltc_home_equity_used_real"
    ]["p50"] == 5_000
    assert payload["reproducibility"]["healthcare"]["assumptions"] == (
        scenario.healthcare.model_dump(mode="json")
    )


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
        _scenario(
            name="Baseline",
            starting_portfolio=100_000,
            portfolio_allocation=_allocation(),
        ).model_dump_json(),
        encoding="utf-8",
    )
    alternative_path.write_text(
        _scenario(
            name="Alternative",
            starting_portfolio=80_000,
            portfolio_allocation=_allocation(),
        ).model_dump_json(),
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
            "--named-stress",
            "equity_crash",
            "--json",
        ],
    )

    assert comparison_result.exit_code == 0, comparison_result.output
    comparison = ScenarioComparison.model_validate_json(
        comparison_result.output
    )
    assert comparison.baseline.revision_id == baseline["id"]
    assert comparison.alternatives[0].revision_id == alternative["id"]
    assert comparison.manifest.common_paths.named_stress == "equity_crash"

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


def test_tax_strategy_cli_matches_simulation_json_and_prints_schedule(
    tmp_path: Path,
) -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "CLI strategy",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 63,
            "starting_portfolio": 100_000,
            "annual_spending": 10_000,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "seed": 92,
            "tax_buckets": [
                {"tax_treatment": "tax_deferred", "starting_balance": 70_000},
                {"tax_treatment": "roth", "starting_balance": 20_000},
                {
                    "tax_treatment": "taxable",
                    "starting_balance": 10_000,
                    "taxable_basis": 5_000,
                },
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "single",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1966,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["taxable", "tax_deferred", "roth"],
                "retirement_surplus_destination": "taxable",
                "strategy": {
                    "withdrawal_policy": "ordered",
                    "roth_conversion": {
                        "start_age": 60,
                        "end_age": 61,
                        "target_federal_ordinary_bracket_rate": 0.10,
                        "max_annual_conversion_real": 20_000,
                    },
                },
            },
        }
    )
    scenario_path = tmp_path / "strategy.local.json"
    scenario_path.write_text(scenario.model_dump_json(), encoding="utf-8")

    strategy_json = runner.invoke(
        app,
        [
            "wealth",
            "tax-strategy",
            "--scenario",
            str(scenario_path),
            "--json",
        ],
    )
    simulation_json = runner.invoke(
        app,
        [
            "wealth",
            "simulate",
            "--scenario",
            str(scenario_path),
            "--json",
        ],
    )
    table = runner.invoke(
        app,
        ["wealth", "tax-strategy", "--scenario", str(scenario_path)],
        terminal_width=500,
    )

    assert strategy_json.exit_code == 0, strategy_json.output
    assert simulation_json.exit_code == 0, simulation_json.output
    assert json.loads(strategy_json.output) == json.loads(simulation_json.output)
    payload = json.loads(strategy_json.output)
    assert payload["annual_tax_strategy_actions"][0]["policy_id"] == (
        "tax_strategy_v1"
    )
    assert table.exit_code == 0, table.output
    assert "Roth Conversion" in table.output
    assert "Tax Treatment" in table.output
    assert "tax_deferred" in table.output
    assert "roth" in table.output
    assert "taxable" in table.output
    assert "hsa" in table.output
    assert "cash" in table.output
    first_withdrawals = payload["annual_tax_strategy_actions"][0][
        "withdrawals_nominal"
    ]
    assert format(first_withdrawals["taxable"]["p50"], ",.2f") in table.output
    assert "IRMAA" in table.output
    assert "After-Tax" in table.output


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
