from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from ynab_agent.planning import simulation, solver
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.outcomes import estimate_outcome_state_bytes
from ynab_agent.planning.paths import (
    BoundedPathSource,
    NumpyPathSource,
    PathSpec,
    PathSource,
    PreparedExperiment,
    ProcessResourceBudget,
    ResourceEstimate,
    ResourceLimitError,
    ResourceReservation,
    RunPolicy,
    StorageKind,
    estimate_path_resources,
)
from ynab_agent.planning.simulation import prepare_simulation_paths, simulate
from ynab_agent.planning.solver import SolveVariable, solve_scenario


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "bounded-test",
        "current_age": 40,
        "retirement_age": 42,
        "end_age": 48,
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


def _policy(
    *,
    batch_size: int,
    process_budget: ProcessResourceBudget | None = None,
    temporary_directory: Path | None = None,
    force_memmap: bool = False,
) -> RunPolicy:
    return RunPolicy(
        batch_size=batch_size,
        in_memory_path_bytes=0 if force_memmap else 1024**3,
        temporary_directory=temporary_directory,
        process_budget=process_budget or ProcessResourceBudget(),
    )


def test_exact_results_are_invariant_across_batch_size_and_storage(tmp_path: Path) -> None:
    scenario = _scenario()
    memory_policy = _policy(batch_size=7)
    memmap_policy = _policy(
        batch_size=31,
        temporary_directory=tmp_path,
        force_memmap=True,
    )

    with prepare_simulation_paths(
        scenario,
        run_policy=memory_policy,
    ) as memory_experiment:
        memory_result = simulate(
            scenario,
            100_000,
            prepared_paths=memory_experiment,
        )
        assert memory_experiment.storage is StorageKind.MEMORY
        assert [part.stop - part.start for part in memory_experiment.trial_slices()] == [
            *([7] * 14),
            2,
        ]

    with prepare_simulation_paths(
        scenario,
        run_policy=memmap_policy,
    ) as memmap_experiment:
        temporary_path = memmap_experiment.temporary_path
        assert temporary_path is not None and temporary_path.exists()
        memmap_result = simulate(
            scenario,
            100_000,
            prepared_paths=memmap_experiment,
        )
        assert memmap_experiment.storage is StorageKind.MEMMAP

    assert memory_result == memmap_result
    assert temporary_path is not None and not temporary_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_internal_memmap_is_cleaned_after_success(tmp_path: Path) -> None:
    budget = ProcessResourceBudget()
    policy = _policy(
        batch_size=13,
        process_budget=budget,
        temporary_directory=tmp_path,
        force_memmap=True,
    )

    result = simulate(
        _scenario(),
        100_000,
        run_policy=policy,
    )

    assert 0 <= result.success_rate <= 1
    assert budget.reserved_bytes == 0
    assert list(tmp_path.iterdir()) == []


def test_full_outcome_state_is_reserved_and_released_for_reusable_paths() -> None:
    scenario = _scenario(
        spending_tiers=[
            {"name": "essential", "annual_amount": 15_000},
        ]
    )
    spec = PathSpec.from_scenario(
        scenario,
        paired_historical_inflation=False,
    )
    estimate = estimate_path_resources(spec, batch_size=10)
    outcome_bytes = estimate_outcome_state_bytes(scenario)
    maximum_working_bytes = estimate.peak_working_bytes + outcome_bytes - 1
    budget = ProcessResourceBudget(maximum_bytes=1024**3)
    policy = RunPolicy(
        maximum_working_bytes=maximum_working_bytes,
        in_memory_path_bytes=1024**3,
        batch_size=10,
        process_budget=budget,
    )

    experiment = prepare_simulation_paths(scenario, run_policy=policy)
    base_reserved = budget.reserved_bytes
    try:
        with pytest.raises(ResourceLimitError, match="evaluation memory"):
            simulate(
                scenario,
                100_000,
                prepared_paths=experiment,
            )
        assert budget.reserved_bytes == base_reserved
    finally:
        experiment.close()
    assert budget.reserved_bytes == 0

    with pytest.raises(ResourceLimitError, match="evaluation memory"):
        simulate(
            scenario,
            100_000,
            run_policy=policy,
        )
    assert budget.reserved_bytes == 0


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("failed"), asyncio.CancelledError()],
)
def test_internal_memmap_is_cleaned_after_failure_or_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    budget = ProcessResourceBudget()
    policy = _policy(
        batch_size=11,
        process_budget=budget,
        temporary_directory=tmp_path,
        force_memmap=True,
    )

    def fail_evaluation(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(simulation, "_run_trial_outcomes", fail_evaluation)

    with pytest.raises(type(failure), match=str(failure) or None):
        simulate(
            _scenario(),
            100_000,
            run_policy=policy,
        )

    assert budget.reserved_bytes == 0
    assert list(tmp_path.iterdir()) == []


class _BlockingPathSource:
    def __init__(self, entered: Event, release: Event) -> None:
        self.entered = entered
        self.release = release

    def prepare_bounded(
        self,
        spec: PathSpec,
        *,
        historical_returns: Sequence[float] | None,
        historical_inflation: Sequence[float] | None,
        resource_estimate: ResourceEstimate,
        policy: RunPolicy,
        storage: StorageKind,
        reservation: ResourceReservation,
    ) -> PreparedExperiment:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("timed out waiting to release path source")
        return NumpyPathSource().prepare_bounded(
            spec,
            historical_returns=historical_returns,
            historical_inflation=historical_inflation,
            resource_estimate=resource_estimate,
            policy=policy,
            storage=storage,
            reservation=reservation,
        )


def test_process_budget_rejects_a_concurrent_experiment() -> None:
    scenario = _scenario()
    spec = PathSpec.from_scenario(
        scenario,
        paired_historical_inflation=False,
    )
    estimate = estimate_path_resources(spec, batch_size=10)
    required = estimate.peak_working_bytes
    budget = ProcessResourceBudget(maximum_bytes=required)
    policy = _policy(batch_size=10, process_budget=budget)
    entered = Event()
    release = Event()
    source: BoundedPathSource = _BlockingPathSource(entered, release)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            prepare_simulation_paths,
            scenario,
            path_source=source,
            run_policy=policy,
        )
        assert entered.wait(timeout=5)
        with pytest.raises(ResourceLimitError, match="process-wide"):
            prepare_simulation_paths(
                scenario,
                run_policy=policy,
            )
        release.set()
        experiment = future.result(timeout=5)

    assert budget.reserved_bytes == required
    experiment.close()
    assert budget.reserved_bytes == 0


def test_solver_reuses_and_closes_one_memmap_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared: list[PreparedExperiment] = []

    def tracking_prepare(
        scenario: WealthScenario,
        *,
        historical_returns: Sequence[float] | None = None,
        historical_inflation: Sequence[float] | None = None,
        path_source: PathSource | BoundedPathSource | None = None,
        run_policy: RunPolicy | None = None,
    ) -> PreparedExperiment:
        experiment = prepare_simulation_paths(
            scenario,
            historical_returns=historical_returns,
            historical_inflation=historical_inflation,
            path_source=path_source,
            run_policy=run_policy,
        )
        prepared.append(experiment)
        return experiment

    monkeypatch.setattr(solver, "prepare_simulation_paths", tracking_prepare)
    policy = _policy(
        batch_size=9,
        temporary_directory=tmp_path,
        force_memmap=True,
    )

    result = solve_scenario(
        _scenario(
            current_age=60,
            retirement_age=60,
            end_age=64,
            annual_contribution=0,
            inflation_rate=0,
            return_mean=0,
            return_volatility=0,
            annual_fee_rate=0,
        ),
        100_000,
        variable=SolveVariable.ANNUAL_SPENDING,
        lower=1,
        upper=50_000,
        target_success_rate=0.9,
        resolution=100,
        run_policy=policy,
    )

    assert result.value == 25_000
    assert len(prepared) == 1
    assert prepared[0].closed
    assert list(tmp_path.iterdir()) == []
