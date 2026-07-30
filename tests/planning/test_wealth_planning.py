from datetime import date

import pytest

from ynab_agent.planning.historical import load_historical_series
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.performance import AccountCashFlow, calculate_performance
from ynab_agent.planning.reporting import write_simulation_html
from ynab_agent.planning.simulation import prepare_simulation_paths, simulate
from ynab_agent.planning.solver import SolveVariable, solve_scenario

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


def test_simulation_is_seeded_and_uses_real_dollars() -> None:
    scenario = _scenario(return_mean=0.06, return_volatility=0.12, inflation_rate=0.025)

    first = simulate(scenario, 100_000)
    second = simulate(scenario, 100_000)

    assert first == second
    assert first.trials == 100
    assert 0 <= first.success_rate <= 1
    assert first.ending_balance_real["p10"] <= first.ending_balance_real["p50"]
    assert first.ending_balance_real["p50"] <= first.ending_balance_real["p90"]


def test_deterministic_simulation_marks_depletion() -> None:
    result = simulate(
        _scenario(
            current_age=60,
            retirement_age=60,
            end_age=64,
            starting_portfolio=50_000,
            annual_contribution=0,
            annual_spending=20_000,
        ),
        50_000,
    )

    assert result.success_rate == 0
    assert result.median_depletion_age == 62
    assert result.ending_balance_real == {"p10": 0.0, "p50": 0.0, "p90": 0.0}


def test_age_bounded_cash_flows_change_contributions_and_expenses() -> None:
    result = simulate(
        _scenario(
            current_age=40,
            retirement_age=43,
            end_age=45,
            starting_portfolio=100_000,
            annual_contribution=10_000,
            annual_spending=20_000,
            cash_flow_streams=[
                {
                    "name": "Freed mortgage payment",
                    "flow_type": "contribution",
                    "start_age": 41,
                    "end_age": 42,
                    "annual_amount": 5_000,
                },
                {
                    "name": "Temporary mortgage expense",
                    "flow_type": "expense",
                    "start_age": 43,
                    "end_age": 43,
                    "annual_amount": 5_000,
                },
            ],
        ),
        100_000,
    )

    assert result.retirement_balance_real["p50"] == 140_000
    assert result.ending_balance_real["p50"] == 95_000
    assert result.assumptions["cash_flow_stream_count"] == 2


def test_historical_simulation_is_seeded_and_preserves_annual_path() -> None:
    scenario = _scenario(
        return_model="historical_bootstrap",
        historical_block_size=2,
        return_mean=0,
        return_volatility=0,
    )
    returns = [0.10, -0.08, 0.15, -0.03, 0.06, 0.04]
    inflation = [0.02, 0.03, 0.01, 0.04, 0.02, 0.025]

    first = simulate(
        scenario,
        100_000,
        historical_returns=returns,
        historical_inflation=inflation,
    )
    second = simulate(
        scenario,
        100_000,
        historical_returns=returns,
        historical_inflation=inflation,
    )

    assert first == second
    assert first.assumptions["return_model"] == "historical_bootstrap"
    assert len(first.annual_balance_real) == scenario.end_age - scenario.current_age + 1
    assert first.annual_balance_real[0]["age"] == scenario.current_age
    assert first.annual_balance_real[-1]["age"] == scenario.end_age
    assert first.success_rate_ci_95["low"] <= first.success_rate
    assert first.success_rate_ci_95["high"] >= first.success_rate
    assert first.engine["arch"] != "not-used"

    paths = prepare_simulation_paths(
        scenario,
        historical_returns=returns,
        historical_inflation=inflation,
    )
    paired_inflation = {
        round(return_rate, 10): inflation_rate
        for return_rate, inflation_rate in zip(returns, inflation)
    }
    sampled_returns = paths.gross_returns - 1
    sampled_inflation = (
        paths.inflation_factors[1:] / paths.inflation_factors[:-1]
    ) - 1
    for sampled_return, sampled_rate in zip(
        sampled_returns.ravel(),
        sampled_inflation.ravel(),
    ):
        assert sampled_rate == pytest.approx(paired_inflation[round(sampled_return, 10)])


def test_historical_csv_loads_paired_inflation(tmp_path) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "year,nominal_return,inflation_rate\n"
        "2023,0.26,0.034\n"
        "2022,-0.18,0.065\n",
        encoding="utf-8",
    )

    series = load_historical_series(returns_path)

    assert series.years == (2022, 2023)
    assert series.nominal_returns == (-0.18, 0.26)
    assert series.inflation_rates == (0.065, 0.034)


def test_solver_finds_maximum_deterministic_spending() -> None:
    scenario = _scenario(
        current_age=60,
        retirement_age=60,
        end_age=64,
        starting_portfolio=100_000,
        annual_contribution=0,
        annual_spending=20_000,
    )

    result = solve_scenario(
        scenario,
        100_000,
        variable=SolveVariable.ANNUAL_SPENDING,
        lower=1,
        upper=50_000,
        target_success_rate=0.9,
    )

    assert result.value == pytest.approx(25_000, abs=0.01)
    assert result.achieved_success_rate == 1


def test_solver_finds_earliest_successful_retirement_age() -> None:
    scenario = _scenario(
        current_age=60,
        retirement_age=60,
        end_age=65,
        starting_portfolio=100_000,
        annual_contribution=0,
        annual_spending=30_000,
    )

    result = solve_scenario(
        scenario,
        100_000,
        variable=SolveVariable.RETIREMENT_AGE,
        lower=60,
        upper=64,
        target_success_rate=0.9,
    )

    assert result.value == 62
    assert result.achieved_success_rate == 1


def test_performance_calculates_irregular_money_weighted_return() -> None:
    result = calculate_performance(
        account_ids=["brokerage"],
        cash_flows=[AccountCashFlow(date=date(2021, 1, 1), amount=1_000)],
        ending_balance=1_100,
        valuation_date=date(2022, 1, 1),
    )

    assert result.contributions == 1_000
    assert result.withdrawals == 0
    assert result.xirr == pytest.approx(0.10)


def test_simulation_report_is_self_contained(tmp_path) -> None:
    result = simulate(_scenario(), 100_000)
    output_path = tmp_path / "wealth.html"

    written = write_simulation_html(result, output_path)

    assert written == output_path.resolve()
    html = output_path.read_text(encoding="utf-8")
    assert "plotly.js" in html
    assert "Retirement" in html
    assert "median funded spending" in html
    assert "median cumulative shortfall" in html
    assert "Goal attainment" in html


def test_scenario_requires_liquid_source() -> None:
    with pytest.raises(ValueError, match="starting_portfolio"):
        _scenario(
            starting_portfolio=None,
            accounts=[{"id": "house", "role": "real_estate"}],
        )
