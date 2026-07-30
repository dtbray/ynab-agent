from __future__ import annotations

import pytest
import scipy.stats  # type: ignore[import-untyped]

from ynab_agent.planning import simulation
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.simulation import (
    prepare_simulation_paths,
    score_simulation,
    simulate,
)


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "score-test",
        "current_age": 40,
        "retirement_age": 42,
        "end_age": 45,
        "starting_portfolio": 100_000,
        "annual_contribution": 10_000,
        "annual_spending": 20_000,
        "inflation_rate": 0.025,
        "return_mean": 0.06,
        "return_volatility": 0.12,
        "annual_fee_rate": 0.002,
        "trials": 100,
        "seed": 7,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def test_score_matches_full_seeded_simulation() -> None:
    scenario = _scenario()
    paths = prepare_simulation_paths(scenario)

    score = score_simulation(scenario, 100_000, prepared_paths=paths)
    result = simulate(scenario, 100_000, prepared_paths=paths)

    assert score.trials == result.trials
    assert score.depleted_trials == result.depleted_trials
    assert score.successful_trials == result.trials - result.depleted_trials
    assert score.success_rate == result.success_rate
    assert result.retirement_balance_real == pytest.approx(
        {
            "p10": 107_571.02405739742,
            "p50": 120_148.75816256992,
            "p90": 144_250.77625364574,
        }
    )
    assert result.ending_balance_real == pytest.approx(
        {
            "p10": 38_792.410346870216,
            "p50": 68_814.9958266854,
            "p90": 100_865.13103944216,
        }
    )


def test_score_does_not_run_reporting_or_metadata_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario()
    paths = prepare_simulation_paths(scenario)

    def fail_percentiles(_values: object) -> dict[str, float]:
        raise AssertionError("score_simulation must not calculate percentiles")

    def fail_package_version(_distribution: str) -> str:
        raise AssertionError("score_simulation must not collect report metadata")

    def fail_confidence_interval(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("score_simulation must not calculate confidence intervals")

    monkeypatch.setattr(simulation, "_percentiles", fail_percentiles)
    monkeypatch.setattr(simulation, "_package_version", fail_package_version)
    monkeypatch.setattr(scipy.stats, "binomtest", fail_confidence_interval)

    score = score_simulation(scenario, 100_000, prepared_paths=paths)

    assert score.success_rate == 1.0
