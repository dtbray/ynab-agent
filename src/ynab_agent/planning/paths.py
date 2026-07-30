"""Seeded market-path generation with bounded storage and resource accounting."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self, cast

from ynab_agent.planning.models import ReturnModel, WealthScenario

if TYPE_CHECKING:
    import numpy as np


_FLOAT_BYTES = 8
_DEFAULT_MAXIMUM_WORKING_BYTES = 512 * 1024 * 1024
_DEFAULT_IN_MEMORY_PATH_BYTES = 128 * 1024 * 1024
_DEFAULT_MAXIMUM_TEMPORARY_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_PROCESS_WORKING_BYTES = 1024 * 1024 * 1024
_DEFAULT_BATCH_SIZE = 10_000


class ResourceLimitError(ValueError):
    """Raised before a run would exceed a configured resource budget."""


class StorageKind(StrEnum):
    """Backing storage selected for prepared economic paths."""

    MEMORY = "memory"
    MEMMAP = "memmap"


@dataclass(frozen=True)
class PathSpec:
    """Economic path inputs independent of contribution and withdrawal values."""

    years: int
    trials: int
    seed: int
    return_model: ReturnModel
    return_mean: float
    return_volatility: float
    inflation_rate: float
    annual_fee_rate: float
    historical_block_size: int
    paired_historical_inflation: bool

    def __post_init__(self) -> None:
        if self.years <= 0:
            raise ValueError("path years must be positive")
        if self.trials <= 0:
            raise ValueError("path trials must be positive")
        if self.historical_block_size <= 0:
            raise ValueError("historical_block_size must be positive")

    @classmethod
    def from_scenario(
        cls,
        scenario: WealthScenario,
        *,
        paired_historical_inflation: bool,
    ) -> PathSpec:
        return cls(
            years=scenario.end_age - scenario.current_age,
            trials=scenario.trials,
            seed=scenario.seed,
            return_model=scenario.return_model,
            return_mean=scenario.return_mean,
            return_volatility=scenario.return_volatility,
            inflation_rate=scenario.inflation_rate,
            annual_fee_rate=scenario.annual_fee_rate,
            historical_block_size=scenario.historical_block_size,
            paired_historical_inflation=paired_historical_inflation,
        )


@dataclass(frozen=True)
class ResourceEstimate:
    """Estimated storage and peak resident bytes for one experiment."""

    retained_path_bytes: int
    peak_working_bytes: int
    memmap_peak_working_bytes: int
    temporary_path_bytes: int


class ResourceReservation:
    """Idempotent reservation against a process-wide working-memory budget."""

    def __init__(self, budget: ProcessResourceBudget, reserved_bytes: int) -> None:
        self._budget = budget
        self.reserved_bytes = reserved_bytes
        self._released = False
        self._lock = Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._budget._release(self.reserved_bytes)


class ProcessResourceBudget:
    """Atomic process-wide ceiling shared by concurrent planning runs."""

    def __init__(self, maximum_bytes: int = _DEFAULT_PROCESS_WORKING_BYTES) -> None:
        if maximum_bytes <= 0:
            raise ValueError("process maximum_bytes must be positive")
        self.maximum_bytes = maximum_bytes
        self._reserved_bytes = 0
        self._lock = Lock()

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved_bytes

    def reserve(self, required_bytes: int) -> ResourceReservation:
        if required_bytes <= 0:
            raise ValueError("required_bytes must be positive")
        with self._lock:
            projected = self._reserved_bytes + required_bytes
            if projected > self.maximum_bytes:
                required_mib = required_bytes / 1024**2
                available_mib = (self.maximum_bytes - self._reserved_bytes) / 1024**2
                raise ResourceLimitError(
                    "process-wide simulation memory limit exceeded: "
                    f"run requires {required_mib:.1f} MiB but only "
                    f"{available_mib:.1f} MiB is available"
                )
            self._reserved_bytes = projected
        return ResourceReservation(self, required_bytes)

    def _release(self, released_bytes: int) -> None:
        with self._lock:
            self._reserved_bytes -= released_bytes
            if self._reserved_bytes < 0:  # pragma: no cover - defensive invariant
                self._reserved_bytes = 0
                raise RuntimeError("process resource reservation underflow")


_PROCESS_RESOURCE_BUDGET = ProcessResourceBudget()


@dataclass(frozen=True)
class RunPolicy:
    """Per-run batching, storage, and memory ceilings."""

    maximum_working_bytes: int = _DEFAULT_MAXIMUM_WORKING_BYTES
    in_memory_path_bytes: int = _DEFAULT_IN_MEMORY_PATH_BYTES
    maximum_temporary_bytes: int = _DEFAULT_MAXIMUM_TEMPORARY_BYTES
    batch_size: int = _DEFAULT_BATCH_SIZE
    temporary_directory: Path | None = None
    process_budget: ProcessResourceBudget = field(
        default_factory=lambda: _PROCESS_RESOURCE_BUDGET,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.maximum_working_bytes <= 0:
            raise ValueError("maximum_working_bytes must be positive")
        if self.in_memory_path_bytes < 0:
            raise ValueError("in_memory_path_bytes must not be negative")
        if self.maximum_temporary_bytes <= 0:
            raise ValueError("maximum_temporary_bytes must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def storage_for(self, estimate: ResourceEstimate) -> StorageKind:
        if (
            estimate.retained_path_bytes <= self.in_memory_path_bytes
            and estimate.peak_working_bytes <= self.maximum_working_bytes
        ):
            return StorageKind.MEMORY
        return StorageKind.MEMMAP

    def required_working_bytes(self, estimate: ResourceEstimate) -> int:
        if self.storage_for(estimate) is StorageKind.MEMORY:
            return estimate.peak_working_bytes
        return estimate.memmap_peak_working_bytes

    def enforce(self, estimate: ResourceEstimate) -> StorageKind:
        storage = self.storage_for(estimate)
        required_bytes = self.required_working_bytes(estimate)
        if required_bytes > self.maximum_working_bytes:
            required_mib = required_bytes / 1024**2
            allowed_mib = self.maximum_working_bytes / 1024**2
            raise ResourceLimitError(
                "estimated simulation working memory "
                f"({required_mib:.1f} MiB) exceeds the configured limit "
                f"({allowed_mib:.1f} MiB); reduce trials, batch size, or the horizon"
            )
        if (
            storage is StorageKind.MEMMAP
            and estimate.temporary_path_bytes > self.maximum_temporary_bytes
        ):
            required_mib = estimate.temporary_path_bytes / 1024**2
            allowed_mib = self.maximum_temporary_bytes / 1024**2
            raise ResourceLimitError(
                "estimated temporary path storage "
                f"({required_mib:.1f} MiB) exceeds the configured limit "
                f"({allowed_mib:.1f} MiB)"
            )
        return storage

    def trial_slices(self, trials: int) -> Iterator[slice]:
        for start in range(0, trials, self.batch_size):
            yield slice(start, min(start + self.batch_size, trials))


@dataclass(frozen=True)
class RunPolicySnapshot:
    """Serializable resource policy captured when an experiment is prepared."""

    maximum_working_bytes: int
    in_memory_path_bytes: int
    maximum_temporary_bytes: int
    process_maximum_working_bytes: int
    configured_batch_size: int

    @classmethod
    def from_policy(cls, policy: RunPolicy) -> RunPolicySnapshot:
        return cls(
            maximum_working_bytes=policy.maximum_working_bytes,
            in_memory_path_bytes=policy.in_memory_path_bytes,
            maximum_temporary_bytes=policy.maximum_temporary_bytes,
            process_maximum_working_bytes=policy.process_budget.maximum_bytes,
            configured_batch_size=policy.batch_size,
        )


@dataclass(frozen=True)
class SimulationPaths:
    """Prepared return and inflation paths reusable across scenario evaluations."""

    gross_returns: np.ndarray
    inflation_factors: np.ndarray


class _ExperimentResources:
    """Mutable lifecycle state kept outside the frozen experiment value."""

    def __init__(
        self,
        *,
        reservation: ResourceReservation,
        temporary_directory: TemporaryDirectory[str] | None,
    ) -> None:
        self.reservations = [reservation]
        self.temporary_directory = temporary_directory
        self.closed = False
        self.lock = Lock()

    def add_reservation(self, reservation: ResourceReservation) -> None:
        with self.lock:
            if self.closed:
                reservation.release()
                raise RuntimeError("cannot reserve resources for a closed experiment")
            self.reservations.append(reservation)

    def reserve_evaluation(
        self,
        required_bytes: int,
        *,
        maximum_working_bytes: int,
    ) -> ResourceReservation:
        """Reserve transient evaluation state against the experiment budget."""
        with self.lock:
            if self.closed:
                raise RuntimeError("cannot evaluate a closed experiment")
            base_reserved_bytes = sum(
                reservation.reserved_bytes for reservation in self.reservations
            )
            projected = base_reserved_bytes + required_bytes
            if projected > maximum_working_bytes:
                required_mib = projected / 1024**2
                allowed_mib = maximum_working_bytes / 1024**2
                raise ResourceLimitError(
                    "estimated simulation evaluation memory "
                    f"({required_mib:.1f} MiB) exceeds the configured limit "
                    f"({allowed_mib:.1f} MiB); reduce trials, goals, or tax buckets"
                )
            # The preparation policy reservation is last for compatibility
            # path sources and the only reservation for bounded sources.
            budget = self.reservations[-1]._budget
        return budget.reserve(required_bytes)

    def close(self, arrays: Sequence[np.ndarray]) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
        try:
            import numpy as np

            for array in arrays:
                if not isinstance(array, np.memmap):
                    continue
                array.flush()
                mapped_file = getattr(array, "_mmap", None)
                if mapped_file is not None:
                    mapped_file.close()
            if self.temporary_directory is not None:
                self.temporary_directory.cleanup()
        finally:
            for reservation in self.reservations:
                reservation.release()


@dataclass(frozen=True)
class PreparedExperiment(SimulationPaths):
    """Prepared paths plus immutable specification and owned resources."""

    spec: PathSpec
    resource_estimate: ResourceEstimate
    generator: str
    storage: StorageKind
    batch_size: int
    _resources: _ExperimentResources = field(repr=False, compare=False)
    run_policy: RunPolicySnapshot | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    historical_observation_count: int | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    historical_values_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def closed(self) -> bool:
        return self._resources.closed

    @property
    def temporary_path(self) -> Path | None:
        temporary_directory = self._resources.temporary_directory
        return Path(temporary_directory.name) if temporary_directory is not None else None

    def trial_slices(self) -> Iterator[slice]:
        for start in range(0, self.spec.trials, self.batch_size):
            yield slice(start, min(start + self.batch_size, self.spec.trials))

    def reserve_evaluation(
        self,
        required_bytes: int,
    ) -> ResourceReservation:
        """Reserve transient full-report state until the caller releases it."""
        maximum_working_bytes = (
            self.run_policy.maximum_working_bytes
            if self.run_policy is not None
            else _DEFAULT_MAXIMUM_WORKING_BYTES
        )
        return self._resources.reserve_evaluation(
            required_bytes,
            maximum_working_bytes=maximum_working_bytes,
        )

    def close(self) -> None:
        self._resources.close((self.gross_returns, self.inflation_factors))

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("prepared experiment is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class PathSource(Protocol):
    """Compatibility port for custom deterministic path generators."""

    def prepare(
        self,
        spec: PathSpec,
        *,
        historical_returns: Sequence[float] | None,
        historical_inflation: Sequence[float] | None,
        resource_estimate: ResourceEstimate,
    ) -> PreparedExperiment: ...


class BoundedPathSource(Protocol):
    """Path source that accepts an owned bounded-storage preparation context."""

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
    ) -> PreparedExperiment: ...


def estimate_path_resources(
    spec: PathSpec,
    *,
    batch_size: int | None = None,
) -> ResourceEstimate:
    """Estimate storage and resident peaks for memory or memmap backing."""
    effective_batch_size = min(batch_size or spec.trials, spec.trials)
    matrix_bytes = spec.years * spec.trials * _FLOAT_BYTES
    if spec.return_model is ReturnModel.HISTORICAL_BOOTSTRAP and spec.paired_historical_inflation:
        inflation_bytes = (spec.years + 1) * spec.trials * _FLOAT_BYTES
    else:
        inflation_bytes = (spec.years + 1) * _FLOAT_BYTES

    retained_bytes = matrix_bytes + inflation_bytes
    # Three persistent trial-state arrays, one exact-quantile vector, and batch temporaries.
    evaluation_work_bytes = spec.trials * 4 * _FLOAT_BYTES + effective_batch_size * 4 * _FLOAT_BYTES
    if spec.return_model is ReturnModel.HISTORICAL_BOOTSTRAP:
        generation_work_bytes = spec.trials * 6 * _FLOAT_BYTES
    else:
        generation_work_bytes = spec.trials * _FLOAT_BYTES
    non_path_peak = max(evaluation_work_bytes, generation_work_bytes)
    return ResourceEstimate(
        retained_path_bytes=retained_bytes,
        peak_working_bytes=retained_bytes + non_path_peak,
        memmap_peak_working_bytes=non_path_peak,
        temporary_path_bytes=retained_bytes,
    )


def _lognormal_parameters(
    arithmetic_mean: float,
    volatility: float,
) -> tuple[float, float]:
    gross_mean = 1 + arithmetic_mean
    variance = volatility**2
    sigma_squared = math.log1p(variance / gross_mean**2)
    return math.log(gross_mean) - sigma_squared / 2, math.sqrt(sigma_squared)


def _historical_arrays(
    returns: Sequence[float] | None,
    inflation: Sequence[float] | None,
    *,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    import numpy as np

    if returns is None:
        raise ValueError(
            "historical_returns are required when return_model is historical_bootstrap"
        )
    try:
        historical_extra_available = find_spec("arch.bootstrap") is not None
    except ModuleNotFoundError:
        historical_extra_available = False
    if not historical_extra_available:
        raise RuntimeError(
            "historical simulation requires the historical extra: "
            'pip install "ynab-agent[historical]"'
        )

    return_values = np.asarray(returns, dtype=float)
    if return_values.ndim != 1 or return_values.size == 0:
        raise ValueError("historical returns must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(return_values)) or np.any(return_values <= -1):
        raise ValueError("historical returns must be finite and greater than -1")
    if block_size > return_values.size:
        raise ValueError("historical_block_size cannot exceed the observation count")

    inflation_values: np.ndarray | None = None
    if inflation is not None:
        inflation_values = np.asarray(inflation, dtype=float)
        if inflation_values.shape != return_values.shape:
            raise ValueError("historical inflation must have one value per return")
        if np.any(~np.isfinite(inflation_values)) or np.any(inflation_values <= -1):
            raise ValueError("historical inflation must be finite and greater than -1")
    return return_values, inflation_values


def _historical_values_sha256(
    returns: Sequence[float],
    inflation: Sequence[float] | None,
) -> str:
    canonical = json.dumps(
        {
            "inflation_rates": (
                [float(value) for value in inflation] if inflation is not None else None
            ),
            "nominal_returns": [float(value) for value in returns],
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _allocate_path_arrays(
    spec: PathSpec,
    *,
    storage: StorageKind,
    policy: RunPolicy,
) -> tuple[np.ndarray, np.ndarray, TemporaryDirectory[str] | None]:
    import numpy as np

    inflation_shape = (
        (spec.years + 1, spec.trials)
        if spec.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
        and spec.paired_historical_inflation
        else (spec.years + 1,)
    )
    if storage is StorageKind.MEMORY:
        return (
            np.empty((spec.years, spec.trials), dtype=float),
            np.empty(inflation_shape, dtype=float),
            None,
        )

    parent = (
        str(policy.temporary_directory.expanduser().resolve())
        if policy.temporary_directory is not None
        else None
    )
    temporary_directory = TemporaryDirectory(
        prefix="ynab-planner-",
        dir=parent,
    )
    root = Path(temporary_directory.name)
    try:
        gross_returns = np.memmap(
            root / "gross-returns.bin",
            mode="w+",
            dtype=float,
            shape=(spec.years, spec.trials),
        )
        inflation_factors = np.memmap(
            root / "inflation-factors.bin",
            mode="w+",
            dtype=float,
            shape=inflation_shape,
        )
    except BaseException:
        temporary_directory.cleanup()
        raise
    return gross_returns, inflation_factors, temporary_directory


def _fill_lognormal_paths(
    gross_returns: np.ndarray,
    inflation_factors: np.ndarray,
    spec: PathSpec,
) -> None:
    import numpy as np

    rng = np.random.default_rng(spec.seed)
    log_mean, log_volatility = _lognormal_parameters(
        spec.return_mean,
        spec.return_volatility,
    )
    for year in range(spec.years):
        gross_returns[year] = rng.lognormal(
            log_mean,
            log_volatility,
            size=spec.trials,
        )
    inflation_factors[:] = np.power(
        1 + spec.inflation_rate,
        np.arange(spec.years + 1, dtype=float),
    )


def _fill_historical_paths(
    gross_returns: np.ndarray,
    inflation_factors: np.ndarray,
    spec: PathSpec,
    *,
    historical_returns: Sequence[float] | None,
    historical_inflation: Sequence[float] | None,
) -> None:
    import numpy as np

    return_values, inflation_values = _historical_arrays(
        historical_returns,
        historical_inflation,
        block_size=spec.historical_block_size,
    )
    rng = np.random.default_rng(spec.seed)
    indices = rng.integers(0, return_values.size, size=spec.trials)
    restart_probability = 1.0 / spec.historical_block_size
    inflation_factors[0] = 1

    for year in range(spec.years):
        if year:
            restart = rng.random(spec.trials) < restart_probability
            indices = (indices + 1) % return_values.size
            restart_count = int(np.count_nonzero(restart))
            if restart_count:
                indices[restart] = rng.integers(
                    0,
                    return_values.size,
                    size=restart_count,
                )
        np.take(return_values, indices, out=gross_returns[year])
        gross_returns[year] += 1
        if inflation_values is not None:
            np.take(inflation_values, indices, out=inflation_factors[year + 1])
            inflation_factors[year + 1] += 1
            inflation_factors[year + 1] *= inflation_factors[year]

    if inflation_values is None:
        inflation_factors[:] = np.power(
            1 + spec.inflation_rate,
            np.arange(spec.years + 1, dtype=float),
        )


class NumpyPathSource:
    """Generate lognormal or stationary-bootstrap paths in bounded storage."""

    def prepare(
        self,
        spec: PathSpec,
        *,
        historical_returns: Sequence[float] | None,
        historical_inflation: Sequence[float] | None,
        resource_estimate: ResourceEstimate,
    ) -> PreparedExperiment:
        """Prepare directly with the default policy for compatibility."""
        policy = RunPolicy()
        storage = policy.enforce(resource_estimate)
        reservation = policy.process_budget.reserve(
            policy.required_working_bytes(resource_estimate)
        )
        try:
            return self.prepare_bounded(
                spec,
                historical_returns=historical_returns,
                historical_inflation=historical_inflation,
                resource_estimate=resource_estimate,
                policy=policy,
                storage=storage,
                reservation=reservation,
            )
        except BaseException:
            reservation.release()
            raise

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
        gross_returns, inflation_factors, temporary_directory = _allocate_path_arrays(
            spec,
            storage=storage,
            policy=policy,
        )
        try:
            if spec.return_model is ReturnModel.HISTORICAL_BOOTSTRAP:
                _fill_historical_paths(
                    gross_returns,
                    inflation_factors,
                    spec,
                    historical_returns=historical_returns,
                    historical_inflation=historical_inflation,
                )
                generator = "vectorized_stationary_bootstrap_v1"
            else:
                _fill_lognormal_paths(
                    gross_returns,
                    inflation_factors,
                    spec,
                )
                generator = "numpy_lognormal_v1"
            gross_returns *= 1 - spec.annual_fee_rate
        except BaseException:
            _ExperimentResources(
                reservation=reservation,
                temporary_directory=temporary_directory,
            ).close((gross_returns, inflation_factors))
            raise

        return PreparedExperiment(
            gross_returns=gross_returns,
            inflation_factors=inflation_factors,
            spec=spec,
            resource_estimate=resource_estimate,
            generator=generator,
            storage=storage,
            batch_size=min(policy.batch_size, spec.trials),
            _resources=_ExperimentResources(
                reservation=reservation,
                temporary_directory=temporary_directory,
            ),
            run_policy=RunPolicySnapshot.from_policy(policy),
            historical_observation_count=(
                len(historical_returns)
                if spec.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
                and historical_returns is not None
                else None
            ),
            historical_values_sha256=(
                _historical_values_sha256(
                    historical_returns,
                    historical_inflation,
                )
                if spec.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
                and historical_returns is not None
                else None
            ),
        )


def prepare_experiment(
    scenario: WealthScenario,
    *,
    historical_returns: Sequence[float] | None = None,
    historical_inflation: Sequence[float] | None = None,
    path_source: PathSource | BoundedPathSource | None = None,
    run_policy: RunPolicy | None = None,
) -> PreparedExperiment:
    """Reserve resources, then prepare deterministic reusable paths."""
    policy = run_policy or RunPolicy()
    paired_inflation = (
        scenario.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
        and historical_inflation is not None
    )
    spec = PathSpec.from_scenario(
        scenario,
        paired_historical_inflation=paired_inflation,
    )
    estimate = estimate_path_resources(spec, batch_size=policy.batch_size)
    storage = policy.enforce(estimate)
    reservation = policy.process_budget.reserve(policy.required_working_bytes(estimate))
    try:
        source = path_source or NumpyPathSource()
        if hasattr(source, "prepare_bounded"):
            bounded_source = cast(BoundedPathSource, source)
            return bounded_source.prepare_bounded(
                spec,
                historical_returns=historical_returns,
                historical_inflation=historical_inflation,
                resource_estimate=estimate,
                policy=policy,
                storage=storage,
                reservation=reservation,
            )
        experiment = source.prepare(
            spec,
            historical_returns=historical_returns,
            historical_inflation=historical_inflation,
            resource_estimate=estimate,
        )
        experiment._resources.add_reservation(reservation)
        if experiment.run_policy is None:
            object.__setattr__(
                experiment,
                "run_policy",
                RunPolicySnapshot.from_policy(policy),
            )
        if spec.return_model is ReturnModel.HISTORICAL_BOOTSTRAP and historical_returns is not None:
            if experiment.historical_observation_count is None:
                object.__setattr__(
                    experiment,
                    "historical_observation_count",
                    len(historical_returns),
                )
            if experiment.historical_values_sha256 is None:
                object.__setattr__(
                    experiment,
                    "historical_values_sha256",
                    _historical_values_sha256(
                        historical_returns,
                        historical_inflation,
                    ),
                )
        return experiment
    except BaseException:
        reservation.release()
        raise
