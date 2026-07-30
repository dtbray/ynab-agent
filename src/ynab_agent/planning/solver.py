"""Solve retirement scenario thresholds with common seeded return paths."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json

from ynab_agent.planning.historical import HistoricalSeries
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.planning.paths import RunPolicy, SimulationPaths
from ynab_agent.planning.simulation import (
    SimulationResult,
    prepare_simulation_paths,
    score_simulation,
    simulate,
)


class SolveVariable(StrEnum):
    """Supported scenario values that can be solved for."""

    ANNUAL_SPENDING = "annual_spending"
    ANNUAL_CONTRIBUTION = "annual_contribution"
    STARTING_PORTFOLIO = "starting_portfolio"
    RETIREMENT_AGE = "retirement_age"


@dataclass(frozen=True)
class SolveResult:
    """A scenario threshold and the simulation at that threshold."""

    variable: str
    value: float
    resolution: float
    target_success_rate: float
    achieved_success_rate: float
    simulation: SimulationResult

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def solve_scenario(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    variable: SolveVariable,
    lower: float,
    upper: float,
    target_success_rate: float,
    resolution: float = 100.0,
    historical_returns: Sequence[float] | None = None,
    historical_inflation: Sequence[float] | None = None,
    historical_source: str | None = None,
    historical_fingerprint: str | None = None,
    historical_series: HistoricalSeries | None = None,
    valuation_provenance: ValuationProvenance | None = None,
    run_policy: RunPolicy | None = None,
) -> SolveResult:
    """Find a conservative monotone threshold using common seeded return paths."""
    if not 0 < target_success_rate < 1:
        raise ValueError("target_success_rate must be between 0 and 1")
    if lower < 0 or upper <= lower:
        raise ValueError("solve bounds must satisfy 0 <= lower < upper")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if historical_series is not None:
        if (
            historical_returns is not None
            and tuple(float(value) for value in historical_returns)
            != historical_series.nominal_returns
        ):
            raise ValueError("historical_returns do not match the supplied historical_series")
        if (
            historical_inflation is not None
            and tuple(float(value) for value in historical_inflation)
            != historical_series.inflation_rates
        ):
            raise ValueError("historical_inflation does not match the supplied historical_series")
        if historical_source is not None and historical_source != historical_series.source:
            raise ValueError("historical_source does not match the supplied historical_series")
        if (
            historical_fingerprint is not None
            and historical_fingerprint != historical_series.sha256
        ):
            raise ValueError("historical_fingerprint does not match the supplied historical_series")
        historical_returns = historical_series.nominal_returns
        historical_inflation = historical_series.inflation_rates
        historical_source = historical_series.source
        historical_fingerprint = historical_series.sha256

    paths = prepare_simulation_paths(
        scenario,
        historical_returns=historical_returns,
        historical_inflation=historical_inflation,
        run_policy=run_policy,
    )
    try:
        return _solve_prepared(
            scenario,
            starting_portfolio,
            paths=paths,
            variable=variable,
            lower=lower,
            upper=upper,
            target_success_rate=target_success_rate,
            resolution=resolution,
            historical_source=historical_source,
            historical_fingerprint=historical_fingerprint,
            historical_series=historical_series,
            valuation_provenance=valuation_provenance,
        )
    finally:
        paths.close()


def _solve_prepared(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    paths: SimulationPaths,
    variable: SolveVariable,
    lower: float,
    upper: float,
    target_success_rate: float,
    resolution: float,
    historical_source: str | None,
    historical_fingerprint: str | None,
    historical_series: HistoricalSeries | None,
    valuation_provenance: ValuationProvenance | None,
) -> SolveResult:
    """Search candidates against one prepared experiment."""

    def candidate_inputs(value: float) -> tuple[WealthScenario, float]:
        candidate = scenario
        portfolio = starting_portfolio
        if variable is SolveVariable.STARTING_PORTFOLIO:
            portfolio = value
            if scenario.tax_buckets:
                if starting_portfolio <= 0:
                    raise ValueError(
                        "tax-aware starting-portfolio solves require a positive baseline"
                    )
                scale = value / starting_portfolio
                tax_buckets = [
                    {
                        **bucket.model_dump(mode="json"),
                        "starting_balance": bucket.starting_balance * scale,
                        **(
                            {
                                "taxable_basis": (
                                    bucket.taxable_basis * scale
                                    if bucket.taxable_basis is not None
                                    else None
                                )
                            }
                            if bucket.taxable_basis is not None
                            else {}
                        ),
                    }
                    for bucket in scenario.tax_buckets
                ]
                candidate = WealthScenario.model_validate(
                    {
                        **scenario.model_dump(mode="json"),
                        "starting_portfolio": value,
                        "tax_buckets": tax_buckets,
                    }
                )
        elif variable is SolveVariable.RETIREMENT_AGE:
            candidate = WealthScenario.model_validate(
                {**scenario.model_dump(), "retirement_age": int(value)}
            )
        else:
            candidate = WealthScenario.model_validate(
                {**scenario.model_dump(), variable.value: value}
            )
        return candidate, portfolio

    def score(value: float) -> float:
        candidate, portfolio = candidate_inputs(value)
        return score_simulation(
            candidate,
            portfolio,
            prepared_paths=paths,
        ).success_rate

    def summarize(value: float) -> SimulationResult:
        candidate, portfolio = candidate_inputs(value)
        provenance = valuation_provenance
        if variable is SolveVariable.STARTING_PORTFOLIO:
            canonical_value = json.dumps(
                {"starting_portfolio": float(portfolio)},
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            provenance = ValuationProvenance(
                source="solved_starting_portfolio",
                source_sha256=hashlib.sha256(canonical_value).hexdigest(),
            )
        return simulate(
            candidate,
            portfolio,
            prepared_paths=paths,
            historical_source=historical_source,
            historical_fingerprint=historical_fingerprint,
            historical_series=historical_series,
            valuation_provenance=provenance,
        )

    if variable is SolveVariable.RETIREMENT_AGE:
        first_age = max(scenario.current_age, math.ceil(lower))
        final_age = min(scenario.end_age - 1, math.floor(upper))
        if final_age < first_age:
            raise ValueError("retirement-age bounds do not contain a valid age")
        for age in range(first_age, final_age + 1):
            if score(float(age)) >= target_success_rate:
                result = summarize(float(age))
                return SolveResult(
                    variable=variable.value,
                    value=float(age),
                    resolution=1.0,
                    target_success_rate=target_success_rate,
                    achieved_success_rate=result.success_rate,
                    simulation=result,
                )
        raise ValueError("target success rate is not reached within the retirement-age bounds")

    lower_step = math.ceil(lower / resolution)
    upper_step = math.floor(upper / resolution)
    if lower_step >= upper_step:
        raise ValueError("solve bounds do not contain two values at the requested resolution")
    lower_value = lower_step * resolution
    upper_value = upper_step * resolution
    lower_success_rate = score(lower_value)
    upper_success_rate = score(upper_value)
    if variable is SolveVariable.ANNUAL_SPENDING:
        if lower_success_rate < target_success_rate:
            raise ValueError("lower bound does not reach the target success rate")
        if upper_success_rate >= target_success_rate:
            raise ValueError("upper bound still reaches the target; increase it")
    else:
        if lower_success_rate >= target_success_rate:
            raise ValueError("lower bound already reaches the target; decrease it")
        if upper_success_rate < target_success_rate:
            raise ValueError("upper bound does not reach the target success rate")

    if variable is SolveVariable.ANNUAL_SPENDING:
        feasible = lower_step
        infeasible = upper_step
        while feasible + 1 < infeasible:
            candidate = (feasible + infeasible) // 2
            if score(candidate * resolution) >= target_success_rate:
                feasible = candidate
            else:
                infeasible = candidate
        value = feasible * resolution
    else:
        infeasible = lower_step
        feasible = upper_step
        while infeasible + 1 < feasible:
            candidate = (infeasible + feasible) // 2
            if score(candidate * resolution) >= target_success_rate:
                feasible = candidate
            else:
                infeasible = candidate
        value = feasible * resolution

    result = summarize(value)
    return SolveResult(
        variable=variable.value,
        value=value,
        resolution=resolution,
        target_success_rate=target_success_rate,
        achieved_success_rate=result.success_rate,
        simulation=result,
    )
