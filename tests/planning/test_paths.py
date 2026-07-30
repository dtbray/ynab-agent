from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.paths import (
    PathSource,
    PathSpec,
    PreparedExperiment,
    ResourceEstimate,
    ResourceLimitError,
    RunPolicy,
    estimate_path_resources,
)
from ynab_agent.planning.simulation import prepare_simulation_paths, simulate


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "path-test",
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


def test_parametric_experiment_uses_one_dimensional_inflation() -> None:
    scenario = _scenario()

    experiment = prepare_simulation_paths(scenario)

    assert experiment.inflation_factors.shape == (6,)
    np.testing.assert_allclose(
        experiment.inflation_factors,
        np.power(1.025, np.arange(6)),
    )
    assert experiment.spec == PathSpec.from_scenario(
        scenario,
        paired_historical_inflation=False,
    )
    assert experiment.generator == "numpy_lognormal_v1"
    assert experiment.resource_estimate.retained_path_bytes == (5 * 100 * 8) + (6 * 8)

    result = simulate(scenario, 100_000, prepared_paths=experiment)
    assert result.engine["path_generator"] == "numpy_lognormal_v1"


def test_vectorized_historical_paths_are_seeded_paired_and_can_exceed_history() -> None:
    scenario = _scenario(
        current_age=40,
        retirement_age=42,
        end_age=50,
        return_model="historical_bootstrap",
        historical_block_size=2,
        inflation_rate=0,
        annual_fee_rate=0,
    )
    returns = (0.11, -0.07, 0.19, 0.03)
    inflation = (0.01, 0.04, 0.02, 0.06)

    first = prepare_simulation_paths(
        scenario,
        historical_returns=returns,
        historical_inflation=inflation,
    )
    second = prepare_simulation_paths(
        scenario,
        historical_returns=returns,
        historical_inflation=inflation,
    )

    assert first.gross_returns.shape == (10, 100)
    assert first.inflation_factors.shape == (11, 100)
    assert first.generator == "vectorized_stationary_bootstrap_v1"
    np.testing.assert_array_equal(first.gross_returns, second.gross_returns)
    np.testing.assert_array_equal(first.inflation_factors, second.inflation_factors)

    paired_inflation = dict(zip(returns, inflation))
    sampled_returns = first.gross_returns - 1
    sampled_inflation = (
        first.inflation_factors[1:] / first.inflation_factors[:-1]
    ) - 1
    for sampled_return, sampled_rate in zip(
        sampled_returns.ravel(),
        sampled_inflation.ravel(),
    ):
        assert sampled_rate == pytest.approx(paired_inflation[round(sampled_return, 2)])

    result = simulate(scenario, 100_000, prepared_paths=first)
    assert len(result.annual_balance_real) == 11
    assert result.engine["path_generator"] == "vectorized_stationary_bootstrap_v1"


class _FailingPathSource:
    called: bool

    def __init__(self) -> None:
        self.called = False

    def prepare(
        self,
        spec: PathSpec,
        *,
        historical_returns: Sequence[float] | None,
        historical_inflation: Sequence[float] | None,
        resource_estimate: ResourceEstimate,
    ) -> PreparedExperiment:
        self.called = True
        raise AssertionError("path source must not run after resource rejection")


def test_resource_policy_rejects_before_path_source_allocation() -> None:
    scenario = _scenario()
    source = _FailingPathSource()
    path_source: PathSource = source

    with pytest.raises(ResourceLimitError, match="exceeds the configured limit"):
        prepare_simulation_paths(
            scenario,
            path_source=path_source,
            run_policy=RunPolicy(maximum_working_bytes=1),
        )

    assert not source.called


def test_paired_historical_estimate_accounts_for_indices_and_inflation_matrix() -> None:
    scenario = _scenario(return_model="historical_bootstrap")
    unpaired = PathSpec.from_scenario(
        scenario,
        paired_historical_inflation=False,
    )
    paired = PathSpec.from_scenario(
        scenario,
        paired_historical_inflation=True,
    )

    unpaired_estimate = estimate_path_resources(unpaired)
    paired_estimate = estimate_path_resources(paired)

    assert paired_estimate.retained_path_bytes > unpaired_estimate.retained_path_bytes
    assert paired_estimate.peak_working_bytes > paired_estimate.retained_path_bytes


def test_prepared_experiment_rejects_different_path_assumptions() -> None:
    scenario = _scenario()
    experiment = prepare_simulation_paths(scenario)
    changed_returns = _scenario(return_mean=0.08)

    with pytest.raises(ValueError, match="path assumptions"):
        simulate(changed_returns, 100_000, prepared_paths=experiment)
