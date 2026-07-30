from __future__ import annotations

from typing import Any

import pytest

from ynab_agent.planning import solver
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.simulation import (
    score_simulation as real_score_simulation,
)
from ynab_agent.planning.simulation import (
    simulate as real_simulate,
)
from ynab_agent.planning.solver import SolveVariable, solve_scenario


def _scenario() -> WealthScenario:
    return WealthScenario.model_validate(
        {
            "name": "solver-score-test",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 64,
            "starting_portfolio": 100_000,
            "annual_contribution": 0,
            "annual_spending": 20_000,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "seed": 7,
        }
    )


def test_solver_scores_candidates_and_summarizes_only_the_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_paths: list[object] = []
    summary_paths: list[object] = []

    def tracking_score(*args: Any, **kwargs: Any) -> Any:
        score_paths.append(kwargs["prepared_paths"])
        return real_score_simulation(*args, **kwargs)

    def tracking_simulate(*args: Any, **kwargs: Any) -> Any:
        summary_paths.append(kwargs["prepared_paths"])
        return real_simulate(*args, **kwargs)

    monkeypatch.setattr(solver, "score_simulation", tracking_score)
    monkeypatch.setattr(solver, "simulate", tracking_simulate)

    result = solve_scenario(
        _scenario(),
        100_000,
        variable=SolveVariable.ANNUAL_SPENDING,
        lower=1,
        upper=50_000,
        target_success_rate=0.9,
        resolution=100,
    )

    assert result.value == 25_000
    assert len(score_paths) > 2
    assert len(summary_paths) == 1
    assert len({id(paths) for paths in [*score_paths, *summary_paths]}) == 1
    assert len(result.simulation.annual_balance_real) == 5
