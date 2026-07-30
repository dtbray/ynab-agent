"""Seeded Monte Carlo retirement simulation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from typing import TYPE_CHECKING

from ynab_agent.planning.historical import HistoricalSeries
from ynab_agent.planning.household import (
    HouseholdState,
    estimate_household_state_bytes,
    prepare_household_state,
)
from ynab_agent.planning.models import (
    CashFlowType,
    IncomeTaxTreatment,
    ReturnModel,
    TaxTreatment,
    ValuationProvenance,
    WealthScenario,
)
from ynab_agent.planning.outcomes import (
    GoalOutcome,
    OutcomeAccumulator,
    estimate_outcome_state_bytes,
    outcome_semantics_manifest,
)
from ynab_agent.planning.paths import (
    BoundedPathSource,
    PathSpec,
    PathSource,
    PreparedExperiment,
    RunPolicy,
    SimulationPaths,
    prepare_experiment,
)
from ynab_agent.planning.progressive_tax import (
    HEALTHCARE_POLICY_PROJECTION,
    calculate_irmaa_surcharge_arrays,
    load_tax_policy,
)
from ynab_agent.planning.spending_guardrails import (
    GuardrailBatchDecision,
    RetirementSpendingPlan,
    estimate_guardrail_state_bytes,
    evaluate_guardrail_batch,
    spending_plan_manifest,
)
from ynab_agent.planning.taxes import TaxAwarePortfolio, estimate_tax_state_bytes
from ynab_agent.planning.tax_strategies import tax_strategy_manifest
from ynab_agent.planning.tax_engine import (
    HouseholdTaxEngine,
    household_tax_engine_for,
)
from ynab_agent.planning.social_security import social_security_policy_manifest

if TYPE_CHECKING:
    import numpy as np


SIMULATION_RESULT_SCHEMA_VERSION = 7
REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION = 6
SIMULATION_ENGINE_VERSION = "wealth_simulation_v8"
_SCENARIO_CANONICALIZATION = "json_sort_keys_v1"


@dataclass(frozen=True)
class SimulationResult:
    """JSON-friendly summary of a retirement simulation."""

    scenario: str
    starting_portfolio: float
    trials: int
    seed: int
    success_rate: float
    success_rate_ci_95: dict[str, float]
    depleted_trials: int
    median_depletion_age: float | None
    retirement_balance_real: dict[str, float]
    ending_balance_real: dict[str, float]
    lifetime_tax_real: dict[str, float] | None
    lifetime_irmaa_surcharge_real: dict[str, float] | None
    irmaa_exposure_probability: float | None
    annual_tax_audit: list[dict[str, object]]
    annual_tax_strategy_actions: list[dict[str, object]]
    annual_balance_real: list[dict[str, float | int]]
    funded_spending_ratio: dict[str, float]
    funded_spending_real: dict[str, float]
    cumulative_shortfall_real: dict[str, float]
    failure_duration_years: dict[str, float] | None
    longest_failure_streak_years: dict[str, float] | None
    recovered_trials: int
    recovery_probability: float | None
    after_tax_ending_balance_real: dict[str, float] | None
    legacy_target_probability: float | None
    goal_outcomes: tuple[GoalOutcome, ...]
    annual_spending_real: list[dict[str, object]]
    guardrail_metrics: dict[str, object] | None
    household_cash_flow_audit: list[dict[str, object]]
    assumptions: dict[str, bool | float | int | str]
    engine: dict[str, str | int]
    reproducibility: dict[str, object] = field(compare=False)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationScore:
    """Lightweight success score without reporting or statistical summaries."""

    trials: int
    successful_trials: int
    depleted_trials: int

    @property
    def success_rate(self) -> float:
        return self.successful_trials / self.trials


@dataclass(frozen=True)
class _TrialOutcomes:
    """Per-trial state needed to score or summarize one scenario evaluation."""

    ending_balances: np.ndarray
    retirement_balances: np.ndarray
    depletion_ages: np.ndarray
    cumulative_tax_real: np.ndarray | None = None
    after_tax_ending_balances: np.ndarray | None = None
    guardrail_reduction_events: np.ndarray | None = None
    guardrail_restoration_events: np.ndarray | None = None
    cumulative_spending_reduction_real: np.ndarray | None = None
    cumulative_spending_restoration_real: np.ndarray | None = None
    annual_tax_audit: list[dict[str, object]] = field(default_factory=list)
    annual_tax_strategy_actions: list[dict[str, object]] = field(default_factory=list)
    cumulative_irmaa_surcharge_real: np.ndarray | None = None
    irmaa_exposed: np.ndarray | None = None


@dataclass
class _GuardrailTrialState:
    """Mutable vector state retained across retirement years."""

    current_total_real: np.ndarray
    essential_real: np.ndarray
    lifestyle_real: np.ndarray
    discretionary_real: np.ndarray
    one_time_real: np.ndarray
    reduction_events: np.ndarray
    restoration_events: np.ndarray
    cumulative_reduction_real: np.ndarray
    cumulative_restoration_real: np.ndarray


def _percentiles(values: np.ndarray) -> dict[str, float]:
    import numpy as np

    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "uninstalled"


def canonical_scenario_sha256(scenario: WealthScenario) -> str:
    """Hash the versioned canonical JSON representation of a scenario."""
    canonical = json.dumps(
        scenario.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _tax_policy_manifest(scenario: WealthScenario) -> dict[str, object] | None:
    assumptions = scenario.tax_assumptions
    if assumptions is None:
        return None
    if assumptions.progressive is None:
        return {
            "model": "account_aware_effective_rates_v1",
            "ordinary_income_tax_rate": assumptions.ordinary_income_tax_rate,
            "long_term_capital_gains_tax_rate": (assumptions.long_term_capital_gains_tax_rate),
            "social_security_taxable_fraction": (assumptions.social_security_taxable_fraction),
            "strategy": tax_strategy_manifest(assumptions.strategy),
        }
    policy, digest = load_tax_policy()
    progressive = assumptions.progressive
    return {
        "model": "progressive_us_indiana",
        "policy_id": policy["policy_id"],
        "effective_year": policy["effective_year"],
        "published_as_of": policy["published_as_of"],
        "resource_sha256": digest,
        "filing_status": progressive.filing_status.value,
        "future_policy_mode": progressive.future_policy_mode.value,
        "bracket_inflation_rate": progressive.bracket_inflation_rate,
        "indiana_resident": progressive.indiana_resident,
        "aca_hook": {
            "household_size": progressive.aca_household_size,
            "benchmark_annual_premium": (progressive.aca_benchmark_annual_premium),
        },
        "irmaa_hook": {
            "lookback_years": 2,
            "historical_magi": [
                value.model_dump(mode="json")
                for value in progressive.irmaa_lookback_magi
            ],
        },
        "strategy": tax_strategy_manifest(assumptions.strategy),
        "healthcare_policy_projection": {
            "mode": HEALTHCARE_POLICY_PROJECTION,
            "healthcare_inflation_rate_applied": False,
            "supplied_healthcare_inflation_rate": (progressive.healthcare_inflation_rate),
            "static_policy_fields": [
                "aca_federal_poverty_levels",
                "aca_applicable_percentages",
                "irmaa_thresholds",
                "irmaa_surcharges",
            ],
        },
        "sources": policy["sources"],
    }


def _canonical_observation_sha256(
    historical_returns: Sequence[float],
    historical_inflation: Sequence[float] | None,
) -> str:
    canonical = json.dumps(
        {
            "inflation_rates": (
                [float(value) for value in historical_inflation]
                if historical_inflation is not None
                else None
            ),
            "nominal_returns": [float(value) for value in historical_returns],
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _resource_policy_manifest(
    paths: SimulationPaths,
    *,
    evaluation_state_bytes: int,
) -> dict[str, object]:
    if not isinstance(paths, PreparedExperiment):
        return {
            "batch_size": None,
            "evaluation_state_bytes": evaluation_state_bytes,
            "storage": "provided_simulation_paths",
            "total_estimated_working_bytes": None,
        }

    policy = (
        asdict(paths.run_policy)
        if paths.run_policy is not None
        else {
            "configured_batch_size": None,
            "in_memory_path_bytes": None,
            "maximum_temporary_bytes": None,
            "maximum_working_bytes": None,
            "process_maximum_working_bytes": None,
        }
    )
    return {
        **policy,
        "batch_size": paths.batch_size,
        "evaluation_state_bytes": evaluation_state_bytes,
        "storage": paths.storage.value,
        "total_estimated_working_bytes": (
            (
                paths.resource_estimate.peak_working_bytes
                if paths.storage.value == "memory"
                else paths.resource_estimate.memmap_peak_working_bytes
            )
            + evaluation_state_bytes
        ),
        "resource_estimate": asdict(paths.resource_estimate),
    }


def _historical_manifest(
    scenario: WealthScenario,
    *,
    historical_returns: Sequence[float] | None,
    historical_inflation: Sequence[float] | None,
    historical_series: HistoricalSeries | None,
    historical_source: str | None,
    historical_fingerprint: str | None,
    paths: SimulationPaths,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if scenario.return_model is not ReturnModel.HISTORICAL_BOOTSTRAP:
        return None, None

    observation_count = (
        len(historical_returns)
        if historical_returns is not None
        else (paths.historical_observation_count if isinstance(paths, PreparedExperiment) else None)
    )
    paired_inflation = historical_inflation is not None or (
        isinstance(paths, PreparedExperiment) and paths.spec.paired_historical_inflation
    )
    observations_sha256 = (
        historical_series.observations_sha256
        if historical_series is not None
        else (
            _canonical_observation_sha256(
                historical_returns,
                historical_inflation,
            )
            if historical_returns is not None
            else (paths.historical_values_sha256 if isinstance(paths, PreparedExperiment) else None)
        )
    )
    sampled_values_sha256 = (
        _canonical_observation_sha256(
            historical_returns,
            historical_inflation,
        )
        if historical_returns is not None
        else (paths.historical_values_sha256 if isinstance(paths, PreparedExperiment) else None)
    )
    historical: dict[str, object] = {
        "source": historical_source,
        "source_sha256": historical_fingerprint,
        "observations_sha256": observations_sha256,
        "sampled_values_sha256": sampled_values_sha256,
        "observation_count": observation_count,
        "date_span": (
            {
                "first_year": historical_series.first_year,
                "last_year": historical_series.last_year,
            }
            if historical_series is not None
            else None
        ),
        "order_policy": (
            historical_series.order_policy.value if historical_series is not None else None
        ),
        "gap_policy": (
            historical_series.gap_policy.value if historical_series is not None else None
        ),
        "paired_inflation": paired_inflation,
    }
    bootstrap: dict[str, object] = {
        "method": "stationary_bootstrap",
        "block_size": scenario.historical_block_size,
        "observation_count": observation_count,
        "paired_return_inflation": paired_inflation,
    }
    return historical, bootstrap


def prepare_simulation_paths(
    scenario: WealthScenario,
    *,
    historical_returns: Sequence[float] | None = None,
    historical_inflation: Sequence[float] | None = None,
    path_source: PathSource | BoundedPathSource | None = None,
    run_policy: RunPolicy | None = None,
) -> PreparedExperiment:
    """Generate seeded paths independently from cash-flow scenario values."""
    try:
        import numpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by an install without the extra
        raise RuntimeError(
            'wealth simulation requires the planner extra: pip install "ynab-agent[planner]"'
        ) from exc

    return prepare_experiment(
        scenario,
        historical_returns=historical_returns,
        historical_inflation=historical_inflation,
        path_source=path_source,
        run_policy=run_policy,
    )


def _new_guardrail_state(
    plan: RetirementSpendingPlan | None,
    trials: int,
) -> _GuardrailTrialState | None:
    if plan is None:
        return None
    import numpy as np

    return _GuardrailTrialState(
        current_total_real=np.full(trials, plan.baseline.total, dtype=float),
        essential_real=np.full(
            trials,
            plan.baseline.essential,
            dtype=float,
        ),
        lifestyle_real=np.full(
            trials,
            plan.baseline.lifestyle,
            dtype=float,
        ),
        discretionary_real=np.full(
            trials,
            plan.baseline.discretionary,
            dtype=float,
        ),
        one_time_real=np.full(
            trials,
            plan.baseline.one_time,
            dtype=float,
        ),
        reduction_events=np.zeros(trials, dtype=np.int32),
        restoration_events=np.zeros(trials, dtype=np.int32),
        cumulative_reduction_real=np.zeros(trials, dtype=float),
        cumulative_restoration_real=np.zeros(trials, dtype=float),
    )


def _record_guardrail_decision(
    state: _GuardrailTrialState,
    trial_slice: slice,
    decision: GuardrailBatchDecision,
    *,
    baseline_total: float,
    spending_scale: float | np.ndarray = 1.0,
) -> None:
    import numpy as np

    previous = state.current_total_real[trial_slice].copy()
    state.current_total_real[trial_slice] = decision.applied_total_real
    state.essential_real[trial_slice] = decision.essential_real
    state.lifestyle_real[trial_slice] = decision.lifestyle_real
    state.discretionary_real[trial_slice] = decision.discretionary_real
    state.one_time_real[trial_slice] = decision.one_time_real
    state.reduction_events[trial_slice] += decision.action == -1
    state.restoration_events[trial_slice] += decision.action == 1
    state.cumulative_reduction_real[trial_slice] += np.maximum(
        0.0,
        (baseline_total - decision.applied_total_real) * spending_scale,
    )
    state.cumulative_restoration_real[trial_slice] += np.maximum(
        0.0,
        (decision.applied_total_real - previous) * spending_scale,
    )


def _run_blended_trial_outcomes(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    paths: SimulationPaths,
    annual_observer: Callable[[int, np.ndarray], None] | None = None,
    outcome_accumulator: OutcomeAccumulator | None = None,
    spending_observer: (
        Callable[
            [
                int,
                _GuardrailTrialState,
                np.ndarray,
                np.ndarray,
            ],
            None,
        ]
        | None
    ) = None,
    household_state: HouseholdState | None = None,
) -> _TrialOutcomes:
    """Evolve trial balances without computing report statistics."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by an install without the extra
        raise RuntimeError(
            'wealth simulation requires the planner extra: pip install "ynab-agent[planner]"'
        ) from exc

    if starting_portfolio < 0:
        raise ValueError("starting_portfolio must not be negative")

    years = scenario.end_age - scenario.current_age
    retirement_offset = scenario.retirement_age - scenario.current_age
    expected_shape = (years, scenario.trials)
    if paths.gross_returns.shape != expected_shape:
        raise ValueError("prepared return paths do not match the scenario")
    valid_inflation_shapes = {
        (years + 1,),
        (years + 1, scenario.trials),
    }
    if paths.inflation_factors.shape not in valid_inflation_shapes:
        raise ValueError("prepared inflation paths do not match the scenario")
    if isinstance(paths, PreparedExperiment):
        if paths.closed:
            raise RuntimeError("prepared experiment is closed")
        expected_spec = PathSpec.from_scenario(
            scenario,
            paired_historical_inflation=paths.spec.paired_historical_inflation,
        )
        if paths.spec != expected_spec:
            raise ValueError("prepared experiment does not match the scenario path assumptions")

    balances = np.full(scenario.trials, starting_portfolio, dtype=float)
    retirement_balances = balances.copy() if retirement_offset == 0 else None
    depletion_ages = np.full(scenario.trials, np.nan)
    spending_plan = scenario.retirement_spending_plan
    guardrail_state = _new_guardrail_state(
        spending_plan,
        scenario.trials,
    )
    if annual_observer is not None:
        annual_observer(scenario.current_age, balances)

    trial_slices = (
        paths.trial_slices()
        if isinstance(paths, PreparedExperiment)
        else iter((slice(0, scenario.trials),))
    )
    batches = tuple(trial_slices)
    for offset in range(years):
        age = scenario.current_age + offset
        annual_action = (
            np.zeros(scenario.trials, dtype=np.int8)
            if age >= scenario.retirement_age
            and guardrail_state is not None
            else None
        )
        annual_withdrawal_rate = (
            np.full(scenario.trials, np.nan, dtype=float)
            if annual_action is not None
            else None
        )
        for trial_slice in batches:
            batch_balances = balances[trial_slice]
            batch_balances *= paths.gross_returns[offset, trial_slice]
            inflation_factor = (
                paths.inflation_factors[offset]
                if paths.inflation_factors.ndim == 1
                else paths.inflation_factors[offset, trial_slice]
            )

            if age < scenario.retirement_age:
                contribution: float | np.ndarray = scenario.annual_contribution
                if scenario.contribution_inflation_adjusted:
                    contribution = contribution * inflation_factor
                for cash_flow in scenario.cash_flow_streams:
                    if cash_flow.flow_type is not CashFlowType.CONTRIBUTION:
                        continue
                    if age < cash_flow.start_age:
                        continue
                    if cash_flow.end_age is not None and age > cash_flow.end_age:
                        continue
                    contribution_amount: float | np.ndarray = cash_flow.annual_amount
                    if cash_flow.inflation_adjusted:
                        contribution_amount = contribution_amount * inflation_factor
                    contribution = contribution + contribution_amount
                batch_balances += contribution
            else:
                if spending_plan is not None and guardrail_state is not None:
                    spending_scale = (
                        household_state.spending_fraction[offset, trial_slice]
                        if household_state is not None
                        else 1.0
                    )
                    decision = evaluate_guardrail_batch(
                        spending_plan,
                        opening_portfolio_real=(
                            batch_balances
                            / inflation_factor
                            / np.where(spending_scale > 0, spending_scale, 1.0)
                        ),
                        previous_total_real=(
                            guardrail_state.current_total_real[trial_slice]
                        ),
                    )
                    _record_guardrail_decision(
                        guardrail_state,
                        trial_slice,
                        decision,
                        baseline_total=spending_plan.baseline.total,
                        spending_scale=spending_scale,
                    )
                    if (
                        annual_action is None
                        or annual_withdrawal_rate is None
                    ):  # pragma: no cover - retirement initialization invariant
                        raise RuntimeError(
                            "guardrail annual audit was not initialized"
                        )
                    annual_action[trial_slice] = decision.action
                    annual_withdrawal_rate[trial_slice] = (
                        decision.withdrawal_rate_before
                    )
                    spending: float | np.ndarray = (
                        decision.applied_total_real * inflation_factor
                    )
                else:
                    spending = (
                        scenario.annual_spending * inflation_factor
                    )
                if household_state is not None:
                    spending = (
                        spending
                        * household_state.spending_fraction[offset, trial_slice]
                    )
                for cash_flow in scenario.cash_flow_streams:
                    if cash_flow.flow_type is not CashFlowType.EXPENSE:
                        continue
                    if age < cash_flow.start_age:
                        continue
                    if cash_flow.end_age is not None and age > cash_flow.end_age:
                        continue
                    expense_amount: float | np.ndarray = cash_flow.annual_amount
                    if cash_flow.inflation_adjusted:
                        expense_amount = expense_amount * inflation_factor
                    spending = spending + expense_amount
                income: float | np.ndarray = 0.0
                if household_state is not None:
                    income = (
                        household_state.ordinary_income[offset, trial_slice]
                        + household_state.social_security_income[
                            offset, trial_slice
                        ]
                        + household_state.tax_free_income[offset, trial_slice]
                    )
                for income_stream in scenario.income_streams:
                    if age < income_stream.start_age:
                        continue
                    if income_stream.end_age is not None and age > income_stream.end_age:
                        continue
                    amount: float | np.ndarray = income_stream.annual_amount
                    if income_stream.inflation_adjusted:
                        amount = amount * inflation_factor
                    income = income + amount
                net_spending = np.maximum(0.0, spending - income)
                withdrawal = net_spending / (1 - scenario.withdrawal_tax_rate)
                batch_depletion_ages = depletion_ages[trial_slice]
                failed = batch_balances < withdrawal
                newly_depleted = np.isnan(batch_depletion_ages) & failed
                batch_depletion_ages[newly_depleted] = age
                if outcome_accumulator is not None:
                    shortfall = np.maximum(0.0, withdrawal - batch_balances) * (
                        1 - scenario.withdrawal_tax_rate
                    )
                    outcome_accumulator.record_retirement_year(
                        trial_slice,
                        required_real=spending / inflation_factor,
                        shortfall_real=shortfall / inflation_factor,
                        failed=failed,
                        active=(
                            household_state.household_alive[
                                offset, trial_slice
                            ]
                            if household_state is not None
                            else True
                        ),
                        spending_scale=(
                            household_state.spending_fraction[
                                offset, trial_slice
                            ]
                            if household_state is not None
                            else 1.0
                        ),
                    )
                np.maximum(
                    0.0,
                    batch_balances - withdrawal,
                    out=batch_balances,
                )

        if offset + 1 == retirement_offset:
            retirement_balances = balances.copy()
        if (
            spending_observer is not None
            and guardrail_state is not None
            and annual_action is not None
            and annual_withdrawal_rate is not None
        ):
            spending_observer(
                age,
                guardrail_state,
                annual_action,
                annual_withdrawal_rate,
            )
        if annual_observer is not None:
            annual_observer(
                age + 1,
                balances / paths.inflation_factors[offset + 1],
            )

    if retirement_balances is None:
        raise RuntimeError("retirement balance was not captured")

    return _TrialOutcomes(
        ending_balances=balances,
        retirement_balances=retirement_balances,
        depletion_ages=depletion_ages,
        guardrail_reduction_events=(
            guardrail_state.reduction_events
            if guardrail_state is not None
            else None
        ),
        guardrail_restoration_events=(
            guardrail_state.restoration_events
            if guardrail_state is not None
            else None
        ),
        cumulative_spending_reduction_real=(
            guardrail_state.cumulative_reduction_real
            if guardrail_state is not None
            else None
        ),
        cumulative_spending_restoration_real=(
            guardrail_state.cumulative_restoration_real
            if guardrail_state is not None
            else None
        ),
    )


def _run_tax_aware_trial_outcomes(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    paths: SimulationPaths,
    annual_observer: Callable[[int, np.ndarray], None] | None = None,
    outcome_accumulator: OutcomeAccumulator | None = None,
    spending_observer: (
        Callable[
            [
                int,
                _GuardrailTrialState,
                np.ndarray,
                np.ndarray,
            ],
            None,
        ]
        | None
    ) = None,
    household_state: HouseholdState | None = None,
    tax_engine: HouseholdTaxEngine | None = None,
) -> _TrialOutcomes:
    """Evolve explicit tax buckets with account-aware retirement withdrawals."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by an install without the extra
        raise RuntimeError(
            'wealth simulation requires the planner extra: pip install "ynab-agent[planner]"'
        ) from exc

    if scenario.tax_assumptions is None or not scenario.tax_buckets:
        raise ValueError("tax-aware simulation requires tax buckets and assumptions")
    years = scenario.end_age - scenario.current_age
    retirement_offset = scenario.retirement_age - scenario.current_age
    expected_shape = (years, scenario.trials)
    if paths.gross_returns.shape != expected_shape:
        raise ValueError("prepared return paths do not match the scenario")
    valid_inflation_shapes = {
        (years + 1,),
        (years + 1, scenario.trials),
    }
    if paths.inflation_factors.shape not in valid_inflation_shapes:
        raise ValueError("prepared inflation paths do not match the scenario")
    if isinstance(paths, PreparedExperiment):
        if paths.closed:
            raise RuntimeError("prepared experiment is closed")
        expected_spec = PathSpec.from_scenario(
            scenario,
            paired_historical_inflation=paths.spec.paired_historical_inflation,
        )
        if paths.spec != expected_spec:
            raise ValueError("prepared experiment does not match the scenario path assumptions")

    portfolio = TaxAwarePortfolio.from_scenario(
        scenario,
        starting_portfolio=starting_portfolio,
    )
    initial_balances = portfolio.total(slice(0, scenario.trials))
    retirement_balances = initial_balances.copy() if retirement_offset == 0 else None
    depletion_ages = np.full(scenario.trials, np.nan)
    spending_plan = scenario.retirement_spending_plan
    guardrail_state = _new_guardrail_state(
        spending_plan,
        scenario.trials,
    )
    if annual_observer is not None:
        annual_observer(scenario.current_age, initial_balances)

    trial_slices = (
        paths.trial_slices()
        if isinstance(paths, PreparedExperiment)
        else iter((slice(0, scenario.trials),))
    )
    batches = tuple(trial_slices)
    annual_tax_audit: list[dict[str, object]] = []
    annual_tax_strategy_actions: list[dict[str, object]] = []
    simulated_magi_by_year: dict[int, np.ndarray] = {}
    for offset in range(years):
        age = scenario.current_age + offset
        portfolio.reset_annual_audit()
        annual_action = (
            np.zeros(scenario.trials, dtype=np.int8)
            if age >= scenario.retirement_age
            and guardrail_state is not None
            else None
        )
        annual_withdrawal_rate = (
            np.full(scenario.trials, np.nan, dtype=float)
            if annual_action is not None
            else None
        )
        for trial_slice in batches:
            inflation_factor = (
                paths.inflation_factors[offset]
                if paths.inflation_factors.ndim == 1
                else paths.inflation_factors[offset, trial_slice]
            )
            opening_tax_deferred = portfolio.opening_tax_deferred(trial_slice)
            portfolio.apply_growth(
                trial_slice,
                paths.gross_returns[offset, trial_slice],
                inflation_factor=inflation_factor,
                assumptions=scenario.tax_assumptions,
            )

            if age < scenario.retirement_age:
                contribution: float | np.ndarray = scenario.annual_contribution
                if scenario.contribution_inflation_adjusted:
                    contribution = contribution * inflation_factor
                portfolio.add_contribution(
                    trial_slice,
                    contribution,
                    scenario=scenario,
                    destination=None,
                )
                for cash_flow in scenario.cash_flow_streams:
                    if cash_flow.flow_type is not CashFlowType.CONTRIBUTION:
                        continue
                    if age < cash_flow.start_age:
                        continue
                    if cash_flow.end_age is not None and age > cash_flow.end_age:
                        continue
                    contribution_amount: float | np.ndarray = cash_flow.annual_amount
                    if cash_flow.inflation_adjusted:
                        contribution_amount = contribution_amount * inflation_factor
                    portfolio.add_contribution(
                        trial_slice,
                        contribution_amount,
                        scenario=scenario,
                        destination=cash_flow.destination_tax_treatment,
                    )
            else:
                if spending_plan is not None and guardrail_state is not None:
                    spending_scale = (
                        household_state.spending_fraction[offset, trial_slice]
                        if household_state is not None
                        else 1.0
                    )
                    decision = evaluate_guardrail_batch(
                        spending_plan,
                        opening_portfolio_real=(
                            portfolio.total(trial_slice)
                            / inflation_factor
                            / np.where(spending_scale > 0, spending_scale, 1.0)
                        ),
                        previous_total_real=(
                            guardrail_state.current_total_real[trial_slice]
                        ),
                    )
                    _record_guardrail_decision(
                        guardrail_state,
                        trial_slice,
                        decision,
                        baseline_total=spending_plan.baseline.total,
                        spending_scale=spending_scale,
                    )
                    if (
                        annual_action is None
                        or annual_withdrawal_rate is None
                    ):  # pragma: no cover - retirement initialization invariant
                        raise RuntimeError(
                            "guardrail annual audit was not initialized"
                        )
                    annual_action[trial_slice] = decision.action
                    annual_withdrawal_rate[trial_slice] = (
                        decision.withdrawal_rate_before
                    )
                    spending: float | np.ndarray = (
                        decision.applied_total_real * inflation_factor
                    )
                else:
                    spending = (
                        scenario.annual_spending * inflation_factor
                    )
                if household_state is not None:
                    spending = (
                        spending
                        * household_state.spending_fraction[offset, trial_slice]
                    )
                for cash_flow in scenario.cash_flow_streams:
                    if cash_flow.flow_type is not CashFlowType.EXPENSE:
                        continue
                    if age < cash_flow.start_age:
                        continue
                    if cash_flow.end_age is not None and age > cash_flow.end_age:
                        continue
                    expense_amount: float | np.ndarray = cash_flow.annual_amount
                    if cash_flow.inflation_adjusted:
                        expense_amount = expense_amount * inflation_factor
                    spending = spending + expense_amount

                ordinary_income: float | np.ndarray = 0.0
                social_security_income: float | np.ndarray = 0.0
                tax_free_income: float | np.ndarray = 0.0
                if household_state is not None:
                    ordinary_income = household_state.ordinary_income[
                        offset, trial_slice
                    ]
                    social_security_income = (
                        household_state.social_security_income[
                            offset, trial_slice
                        ]
                    )
                    tax_free_income = household_state.tax_free_income[
                        offset, trial_slice
                    ]
                for income_stream in scenario.income_streams:
                    if age < income_stream.start_age:
                        continue
                    if income_stream.end_age is not None and age > income_stream.end_age:
                        continue
                    amount: float | np.ndarray = income_stream.annual_amount
                    if income_stream.inflation_adjusted:
                        amount = amount * inflation_factor
                    if income_stream.tax_treatment is IncomeTaxTreatment.ORDINARY:
                        ordinary_income = ordinary_income + amount
                    elif income_stream.tax_treatment is IncomeTaxTreatment.SOCIAL_SECURITY:
                        social_security_income = social_security_income + amount
                    elif income_stream.tax_treatment is IncomeTaxTreatment.TAX_FREE:
                        tax_free_income = tax_free_income + amount
                    else:  # pragma: no cover - scenario validation rejects this
                        raise RuntimeError("tax-aware income stream is unspecified")

                unmet = portfolio.fund_retirement_spending(
                    trial_slice,
                    age=age,
                    tax_year=(
                        scenario.tax_assumptions.progressive.simulation_start_year + offset
                        if scenario.tax_assumptions.progressive is not None
                        else 2026 + offset
                    ),
                    spending=spending,
                    ordinary_income=ordinary_income,
                    social_security_income=social_security_income,
                    tax_free_income=tax_free_income,
                    opening_tax_deferred=opening_tax_deferred,
                    inflation_factor=inflation_factor,
                    assumptions=scenario.tax_assumptions,
                    joint_filing=(
                        household_state.joint_filing[offset, trial_slice]
                        if household_state is not None
                        else None
                    ),
                    tax_engine=tax_engine,
                    owner_birth_years=(
                        {
                            person.id: person.birth_date.year
                            for person in scenario.household.people
                        }
                        if scenario.household is not None
                        else None
                    ),
                )
                batch_depletion_ages = depletion_ages[trial_slice]
                failed = unmet > 0.005
                newly_depleted = np.isnan(batch_depletion_ages) & failed
                batch_depletion_ages[newly_depleted] = age
                if outcome_accumulator is not None:
                    outcome_accumulator.record_retirement_year(
                        trial_slice,
                        required_real=spending / inflation_factor,
                        shortfall_real=unmet / inflation_factor,
                        failed=failed,
                        active=(
                            household_state.household_alive[
                                offset, trial_slice
                            ]
                            if household_state is not None
                            else True
                        ),
                        spending_scale=(
                            household_state.spending_fraction[
                                offset, trial_slice
                            ]
                            if household_state is not None
                            else 1.0
                        ),
                    )

        owner_flows = (
            [
                {
                    "owner_person_id": key.owner_person_id,
                    "tax_treatment": key.tax_treatment.value,
                    "required_minimum_distribution_nominal": (
                        _percentiles(portfolio.annual_rmd_nominal[key])
                    ),
                    "withdrawal_nominal": _percentiles(
                        portfolio.annual_withdrawal_nominal[key]
                    ),
                }
                for key in portfolio.balances
            ]
            if age >= scenario.retirement_age
            else []
        )
        if (
            age >= scenario.retirement_age
            and scenario.tax_assumptions.progressive is not None
        ):
            progressive = scenario.tax_assumptions.progressive
            tax_year = progressive.simulation_start_year + offset
            lookback_year = tax_year - 2
            lookback_magi: float | np.ndarray | None = simulated_magi_by_year.get(
                lookback_year
            )
            if lookback_magi is None:
                lookback_magi = next(
                    (
                        value.magi
                        for value in progressive.irmaa_lookback_magi
                        if value.tax_year == lookback_year
                    ),
                    None,
                )
            surcharge = np.broadcast_to(
                calculate_irmaa_surcharge_arrays(
                    progressive,
                    tax_year=tax_year,
                    lookback_magi=lookback_magi,
                ),
                (scenario.trials,),
            ).astype(float, copy=True)
            portfolio.annual_irmaa_surcharge_nominal[:] = surcharge
            portfolio.cumulative_irmaa_surcharge_real += (
                surcharge / paths.inflation_factors[offset]
            )
            portfolio.irmaa_exposed |= surcharge > 0
            simulated_magi_by_year[tax_year] = (
                portfolio.annual_modified_adjusted_gross_income_nominal.copy()
            )
            annual_tax_audit.append(
                {
                    "tax_year": tax_year,
                    "age": age,
                    "owner_flows": owner_flows,
                    "total_income_tax_nominal": _percentiles(portfolio.annual_tax_nominal),
                    "federal_income_tax_nominal": _percentiles(
                        portfolio.annual_federal_tax_nominal
                    ),
                    "indiana_income_tax_nominal": _percentiles(portfolio.annual_state_tax_nominal),
                    "taxable_social_security_nominal": _percentiles(
                        portfolio.annual_taxable_social_security_nominal
                    ),
                    "federal_deduction_nominal": _percentiles(
                        portfolio.annual_federal_deduction_nominal
                    ),
                    "realized_long_term_capital_gains_nominal": _percentiles(
                        portfolio.annual_realized_long_term_capital_gains_nominal
                    ),
                    "early_distribution_penalty_nominal": _percentiles(
                        portfolio.annual_early_distribution_penalty_nominal
                    ),
                    "modified_adjusted_gross_income_nominal": _percentiles(
                        portfolio.annual_modified_adjusted_gross_income_nominal
                    ),
                    "irmaa_annual_surcharge_nominal": _percentiles(
                        portfolio.annual_irmaa_surcharge_nominal
                    ),
                    "effective_income_tax_rate": _percentiles(portfolio.annual_effective_rate),
                    "marginal_ordinary_income_tax_rate": _percentiles(
                        portfolio.annual_marginal_ordinary_rate
                    ),
                    "marginal_long_term_capital_gains_tax_rate": _percentiles(
                        portfolio.annual_marginal_ltcg_rate
                    ),
                }
            )
        elif age >= scenario.retirement_age:
            annual_tax_audit.append(
                {
                    "tax_year": 2026 + offset,
                    "age": age,
                    "owner_flows": owner_flows,
                }
            )
        if (
            age >= scenario.retirement_age
            and scenario.tax_assumptions.strategy is not None
        ):
            aggregate_withdrawals: dict[str, dict[str, float]] = {}
            for treatment in TaxTreatment:
                keys = [
                    key
                    for key in portfolio.balances
                    if key.tax_treatment is treatment
                ]
                if not keys:
                    continue
                values = np.zeros(scenario.trials, dtype=float)
                for key in keys:
                    values += portfolio.annual_withdrawal_nominal[key]
                    values += portfolio.annual_rmd_nominal[key]
                aggregate_withdrawals[treatment.value] = _percentiles(
                    values
                )
            annual_tax_strategy_actions.append(
                {
                    "schema_version": 1,
                    "policy_id": scenario.tax_assumptions.strategy.policy_id,
                    "tax_year": (
                        scenario.tax_assumptions.progressive.simulation_start_year
                        + offset
                        if scenario.tax_assumptions.progressive is not None
                        else 2026 + offset
                    ),
                    "age": age,
                    "decision_information": "current_year_only",
                    "roth_conversion_nominal": _percentiles(
                        portfolio.annual_roth_conversion_nominal
                    ),
                    "harvested_long_term_capital_gains_nominal": _percentiles(
                        portfolio.annual_harvested_long_term_capital_gains_nominal
                    ),
                    "withdrawals_nominal": aggregate_withdrawals,
                }
            )
        total_balances = portfolio.total(slice(0, scenario.trials))
        if offset + 1 == retirement_offset:
            retirement_balances = total_balances.copy()
        if (
            spending_observer is not None
            and guardrail_state is not None
            and annual_action is not None
            and annual_withdrawal_rate is not None
        ):
            spending_observer(
                age,
                guardrail_state,
                annual_action,
                annual_withdrawal_rate,
            )
        if annual_observer is not None:
            annual_observer(
                age + 1,
                total_balances / paths.inflation_factors[offset + 1],
            )

    if retirement_balances is None:
        raise RuntimeError("retirement balance was not captured")
    after_tax_ending_balances = portfolio.after_tax_estate_value(
        slice(0, scenario.trials),
        assumptions=scenario.tax_assumptions,
        tax_year=(
            scenario.tax_assumptions.progressive.simulation_start_year + years
            if scenario.tax_assumptions.progressive is not None
            else 2026 + years
        ),
        joint_filing=(
            household_state.joint_filing[-1]
            if household_state is not None
            else None
        ),
    )
    return _TrialOutcomes(
        ending_balances=portfolio.total(slice(0, scenario.trials)),
        retirement_balances=retirement_balances,
        depletion_ages=depletion_ages,
        cumulative_tax_real=portfolio.cumulative_tax_real,
        after_tax_ending_balances=after_tax_ending_balances,
        guardrail_reduction_events=(
            guardrail_state.reduction_events
            if guardrail_state is not None
            else None
        ),
        guardrail_restoration_events=(
            guardrail_state.restoration_events
            if guardrail_state is not None
            else None
        ),
        cumulative_spending_reduction_real=(
            guardrail_state.cumulative_reduction_real
            if guardrail_state is not None
            else None
        ),
        cumulative_spending_restoration_real=(
            guardrail_state.cumulative_restoration_real
            if guardrail_state is not None
            else None
        ),
        annual_tax_audit=annual_tax_audit,
        annual_tax_strategy_actions=annual_tax_strategy_actions,
        cumulative_irmaa_surcharge_real=(
            portfolio.cumulative_irmaa_surcharge_real
            if scenario.tax_assumptions.progressive is not None
            else None
        ),
        irmaa_exposed=(
            portfolio.irmaa_exposed
            if scenario.tax_assumptions.progressive is not None
            else None
        ),
    )


def _run_trial_outcomes(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    paths: SimulationPaths,
    annual_observer: Callable[[int, np.ndarray], None] | None = None,
    outcome_accumulator: OutcomeAccumulator | None = None,
    spending_observer: (
        Callable[
            [
                int,
                _GuardrailTrialState,
                np.ndarray,
                np.ndarray,
            ],
            None,
        ]
        | None
    ) = None,
    household_state: HouseholdState | None = None,
    tax_engine: HouseholdTaxEngine | None = None,
) -> _TrialOutcomes:
    if scenario.tax_buckets:
        return _run_tax_aware_trial_outcomes(
            scenario,
            starting_portfolio,
            paths=paths,
            annual_observer=annual_observer,
            outcome_accumulator=outcome_accumulator,
            spending_observer=spending_observer,
            household_state=household_state,
            tax_engine=tax_engine,
        )
    return _run_blended_trial_outcomes(
        scenario,
        starting_portfolio,
        paths=paths,
        annual_observer=annual_observer,
        outcome_accumulator=outcome_accumulator,
        spending_observer=spending_observer,
        household_state=household_state,
    )


def score_simulation(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    historical_returns: Sequence[float] | None = None,
    historical_inflation: Sequence[float] | None = None,
    prepared_paths: SimulationPaths | None = None,
    path_source: PathSource | BoundedPathSource | None = None,
    run_policy: RunPolicy | None = None,
    prepared_household_state: HouseholdState | None = None,
    tax_engine: HouseholdTaxEngine | None = None,
) -> SimulationScore:
    """Evaluate success without percentiles, confidence intervals, or report metadata."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by an install without the extra
        raise RuntimeError(
            'wealth simulation requires the planner extra: pip install "ynab-agent[planner]"'
        ) from exc

    if starting_portfolio < 0:
        raise ValueError("starting_portfolio must not be negative")

    owns_paths = prepared_paths is None
    paths = prepared_paths or prepare_simulation_paths(
        scenario,
        historical_returns=historical_returns,
        historical_inflation=historical_inflation,
        path_source=path_source,
        run_policy=run_policy,
    )
    try:
        household_state = prepared_household_state or prepare_household_state(
            scenario,
            inflation_factors=paths.inflation_factors,
        )
        outcomes = _run_trial_outcomes(
            scenario,
            starting_portfolio,
            paths=paths,
            household_state=household_state,
            tax_engine=tax_engine,
        )
        depleted = int(np.count_nonzero(~np.isnan(outcomes.depletion_ages)))
        return SimulationScore(
            trials=scenario.trials,
            successful_trials=scenario.trials - depleted,
            depleted_trials=depleted,
        )
    finally:
        if owns_paths and isinstance(paths, PreparedExperiment):
            paths.close()


def simulate(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    historical_returns: Sequence[float] | None = None,
    historical_inflation: Sequence[float] | None = None,
    historical_source: str | None = None,
    historical_fingerprint: str | None = None,
    historical_series: HistoricalSeries | None = None,
    valuation_provenance: ValuationProvenance | None = None,
    prepared_paths: SimulationPaths | None = None,
    include_annual_path: bool = True,
    path_source: PathSource | BoundedPathSource | None = None,
    run_policy: RunPolicy | None = None,
    prepared_household_state: HouseholdState | None = None,
    tax_engine: HouseholdTaxEngine | None = None,
) -> SimulationResult:
    """Run a seeded parametric or historical-bootstrap retirement simulation."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by an install without the extra
        raise RuntimeError(
            'wealth simulation requires the planner extra: pip install "ynab-agent[planner]"'
        ) from exc

    if starting_portfolio < 0:
        raise ValueError("starting_portfolio must not be negative")
    if (
        historical_series is not None
        and scenario.return_model is not ReturnModel.HISTORICAL_BOOTSTRAP
    ):
        raise ValueError("historical_series requires return_model='historical_bootstrap'")
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

    years = scenario.end_age - scenario.current_age
    retirement_offset = scenario.retirement_age - scenario.current_age
    owns_paths = prepared_paths is None
    paths = prepared_paths or prepare_simulation_paths(
        scenario,
        historical_returns=historical_returns,
        historical_inflation=historical_inflation,
        path_source=path_source,
        run_policy=run_policy,
    )
    evaluation_state_bytes = (
        estimate_outcome_state_bytes(scenario)
        + estimate_tax_state_bytes(scenario)
        + (
            estimate_guardrail_state_bytes(scenario.trials)
            if scenario.retirement_spending_plan is not None
            else 0
        )
        + estimate_household_state_bytes(scenario)
    )
    evaluation_reservation = None
    try:
        if isinstance(paths, PreparedExperiment):
            evaluation_reservation = paths.reserve_evaluation(evaluation_state_bytes)
        if (
            historical_series is not None
            and isinstance(paths, PreparedExperiment)
            and paths.historical_values_sha256 is not None
            and paths.historical_values_sha256
            != _canonical_observation_sha256(
                historical_series.nominal_returns,
                historical_series.inflation_rates,
            )
        ):
            raise ValueError(
                "prepared historical paths do not match the supplied historical_series"
            )
        annual_balance_real: list[dict[str, float | int]] = []
        annual_spending_real: list[dict[str, object]] = []

        def observe_annual_balance(age: int, real_balances: np.ndarray) -> None:
            annual_balance_real.append({"age": age, **_percentiles(real_balances)})

        outcome_accumulator = OutcomeAccumulator(scenario)
        def observe_annual_spending(
            age: int,
            state: _GuardrailTrialState,
            action: np.ndarray,
            withdrawal_rate: np.ndarray,
        ) -> None:
            import numpy as np

            finite_rates = withdrawal_rate[np.isfinite(withdrawal_rate)]
            spending_scale = (
                household_state.spending_fraction[age - scenario.current_age]
                if household_state is not None
                else 1.0
            )
            annual_spending_real.append(
                {
                    "age": age,
                    "policy": scenario.retirement_spending_plan.policy.kind
                    if scenario.retirement_spending_plan is not None
                    else "fixed_real",
                    "reduced_trials": int(np.count_nonzero(action == -1)),
                    "restored_trials": int(np.count_nonzero(action == 1)),
                    "held_trials": int(np.count_nonzero(action == 0)),
                    "total": _percentiles(
                        state.current_total_real * spending_scale
                    ),
                    "essential": _percentiles(
                        state.essential_real * spending_scale
                    ),
                    "lifestyle": _percentiles(
                        state.lifestyle_real * spending_scale
                    ),
                    "discretionary": _percentiles(
                        state.discretionary_real * spending_scale
                    ),
                    "one_time": _percentiles(
                        state.one_time_real * spending_scale
                    ),
                    "withdrawal_rate": (
                        _percentiles(finite_rates)
                        if finite_rates.size
                        else None
                    ),
                    "essential_floor": (
                        scenario.retirement_spending_plan.essential_floor
                        if scenario.retirement_spending_plan is not None
                        else 0.0
                    ),
                }
            )
        household_state = prepared_household_state or prepare_household_state(
            scenario,
            inflation_factors=paths.inflation_factors,
        )
        outcomes = _run_trial_outcomes(
            scenario,
            starting_portfolio,
            paths=paths,
            annual_observer=observe_annual_balance if include_annual_path else None,
            outcome_accumulator=outcome_accumulator,
            spending_observer=(
                observe_annual_spending
                if include_annual_path
                and scenario.retirement_spending_plan is not None
                else None
            ),
            household_state=household_state,
            tax_engine=tax_engine,
        )
        depleted = int(np.count_nonzero(~np.isnan(outcomes.depletion_ages)))
        finite_depletion_ages = outcomes.depletion_ages[~np.isnan(outcomes.depletion_ages)]
        successful = scenario.trials - depleted
        try:
            from scipy.stats import binomtest  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - install without planner extra
            raise RuntimeError(
                'confidence intervals require the planner extra: pip install "ynab-agent[planner]"'
            ) from exc
        confidence_interval = binomtest(successful, scenario.trials).proportion_ci(
            confidence_level=0.95,
            method="wilson",
        )
        scenario_sha256 = canonical_scenario_sha256(scenario)
        historical_manifest, bootstrap_manifest = _historical_manifest(
            scenario,
            historical_returns=historical_returns,
            historical_inflation=historical_inflation,
            historical_series=historical_series,
            historical_source=historical_source,
            historical_fingerprint=historical_fingerprint,
            paths=paths,
        )
        packages = {
            "ynab-agent": _package_version("ynab-agent"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "arch": (
                _package_version("arch")
                if scenario.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
                else "not-used"
            ),
        }
        bit_generator = "PCG64"
        path_generator = (
            paths.generator
            if isinstance(paths, PreparedExperiment)
            else "provided_simulation_paths"
        )
        after_tax_ending_balance_real = (
            outcomes.after_tax_ending_balances / paths.inflation_factors[years]
            if outcomes.after_tax_ending_balances is not None
            else None
        )
        outcome_summary = outcome_accumulator.summarize(
            successful_trials=successful,
            after_tax_ending_balance_real=after_tax_ending_balance_real,
        )
        spending_plan = scenario.retirement_spending_plan
        spending_manifest = (
            spending_plan_manifest(spending_plan).model_dump(mode="json")
            if spending_plan is not None
            else None
        )
        guardrail_metrics: dict[str, object] | None = None
        if spending_plan is not None:
            reduction_events = outcomes.guardrail_reduction_events
            restoration_events = outcomes.guardrail_restoration_events
            cumulative_reduction = (
                outcomes.cumulative_spending_reduction_real
            )
            cumulative_restoration = (
                outcomes.cumulative_spending_restoration_real
            )
            if (
                reduction_events is None
                or restoration_events is None
                or cumulative_reduction is None
                or cumulative_restoration is None
            ):  # pragma: no cover - simulation integration invariant
                raise RuntimeError("guardrail outcomes were not captured")
            guardrail_metrics = {
                "policy": spending_plan.policy.kind,
                "reduction_events": int(np.sum(reduction_events)),
                "restoration_events": int(np.sum(restoration_events)),
                "trials_with_reduction": int(
                    np.count_nonzero(reduction_events)
                ),
                "trials_with_restoration": int(
                    np.count_nonzero(restoration_events)
                ),
                "cumulative_reduction_real": _percentiles(
                    cumulative_reduction
                ),
                "cumulative_restoration_real": _percentiles(
                    cumulative_restoration
                ),
                "essential_floor": spending_plan.essential_floor,
            }

        return SimulationResult(
            scenario=scenario.name,
            starting_portfolio=float(starting_portfolio),
            trials=scenario.trials,
            seed=scenario.seed,
            success_rate=float(successful / scenario.trials),
            success_rate_ci_95={
                "low": float(confidence_interval.low),
                "high": float(confidence_interval.high),
            },
            depleted_trials=depleted,
            median_depletion_age=(
                float(np.median(finite_depletion_ages)) if finite_depletion_ages.size else None
            ),
            retirement_balance_real=_percentiles(
                outcomes.retirement_balances / paths.inflation_factors[retirement_offset]
            ),
            ending_balance_real=_percentiles(
                outcomes.ending_balances / paths.inflation_factors[years]
            ),
            lifetime_tax_real=(
                _percentiles(outcomes.cumulative_tax_real)
                if outcomes.cumulative_tax_real is not None
                else None
            ),
            lifetime_irmaa_surcharge_real=(
                _percentiles(outcomes.cumulative_irmaa_surcharge_real)
                if outcomes.cumulative_irmaa_surcharge_real is not None
                else None
            ),
            irmaa_exposure_probability=(
                float(np.mean(outcomes.irmaa_exposed))
                if outcomes.irmaa_exposed is not None
                else None
            ),
            annual_tax_audit=outcomes.annual_tax_audit,
            annual_tax_strategy_actions=outcomes.annual_tax_strategy_actions,
            annual_balance_real=annual_balance_real,
            funded_spending_ratio=outcome_summary.funded_spending_ratio,
            funded_spending_real=outcome_summary.funded_spending_real,
            cumulative_shortfall_real=outcome_summary.cumulative_shortfall_real,
            failure_duration_years=outcome_summary.failure_duration_years,
            longest_failure_streak_years=(outcome_summary.longest_failure_streak_years),
            recovered_trials=outcome_summary.recovered_trials,
            recovery_probability=outcome_summary.recovery_probability,
            after_tax_ending_balance_real=(outcome_summary.after_tax_ending_balance_real),
            legacy_target_probability=(outcome_summary.legacy_target_probability),
            goal_outcomes=outcome_summary.goal_outcomes,
            annual_spending_real=annual_spending_real,
            guardrail_metrics=guardrail_metrics,
            household_cash_flow_audit=(
                household_state.audit if household_state is not None else []
            ),
            assumptions={
                "current_age": scenario.current_age,
                "retirement_age": scenario.retirement_age,
                "end_age": scenario.end_age,
                "annual_spending_today": scenario.annual_spending,
                "return_model": scenario.return_model.value,
                "return_mean": scenario.return_mean,
                "return_volatility": scenario.return_volatility,
                "historical_block_size": scenario.historical_block_size,
                "inflation_rate": scenario.inflation_rate,
                "annual_fee_rate": scenario.annual_fee_rate,
                "withdrawal_tax_rate": scenario.withdrawal_tax_rate,
                "cash_flow_stream_count": len(scenario.cash_flow_streams),
                **(
                    {
                        "spending_policy": spending_plan.policy.kind,
                        "essential_spending_floor": (
                            spending_plan.essential_floor
                        ),
                        "essential_spending_baseline": (
                            spending_plan.baseline.essential
                        ),
                        "lifestyle_spending_baseline": (
                            spending_plan.baseline.lifestyle
                        ),
                        "discretionary_spending_baseline": (
                            spending_plan.baseline.discretionary
                        ),
                        "one_time_spending_baseline": (
                            spending_plan.baseline.one_time
                        ),
                    }
                    if spending_plan is not None
                    else {}
                ),
                "tax_model": (
                    (
                        "progressive_us_indiana_v1"
                        if scenario.tax_assumptions.progressive is not None
                        else "account_aware_effective_rates_v1"
                    )
                    if scenario.tax_assumptions is not None
                    else "blended_withdrawal_rate"
                ),
                "tax_bucket_count": len(scenario.tax_buckets),
                "tax_strategy_policy": (
                    scenario.tax_assumptions.strategy.policy_id
                    if scenario.tax_assumptions is not None
                    and scenario.tax_assumptions.strategy is not None
                    else "not_configured"
                ),
                "spending_tier_count": len(scenario.spending_tiers),
                "legacy_target_real": (
                    scenario.legacy_target_real
                    if scenario.legacy_target_real is not None
                    else "not_configured"
                ),
                "household_people": (
                    len(scenario.household.people)
                    if scenario.household is not None
                    else 0
                ),
                **(
                    {
                        "ordinary_income_tax_rate": (
                            scenario.tax_assumptions.ordinary_income_tax_rate
                        ),
                        "long_term_capital_gains_tax_rate": (
                            scenario.tax_assumptions.long_term_capital_gains_tax_rate
                        ),
                        "social_security_taxable_fraction": (
                            scenario.tax_assumptions.social_security_taxable_fraction
                        ),
                        "qualified_hsa_withdrawal_fraction": (
                            scenario.tax_assumptions.qualified_hsa_withdrawal_fraction
                        ),
                        "rmd_start_age": scenario.tax_assumptions.rmd_start_age,
                        "required_minimum_distributions": (
                            scenario.tax_assumptions.apply_required_minimum_distributions
                        ),
                        "tax_withdrawal_order": ",".join(
                            treatment.value
                            for treatment in scenario.tax_assumptions.withdrawal_order
                        ),
                    }
                    if scenario.tax_assumptions is not None
                    else {}
                ),
                "scenario_sha256": scenario_sha256,
                **(
                    {"historical_source": historical_source}
                    if historical_source is not None
                    else {}
                ),
                **(
                    {"historical_sha256": historical_fingerprint}
                    if historical_fingerprint is not None
                    else {}
                ),
            },
            engine={
                "schema_version": 2,
                "python": platform.python_version(),
                "numpy": packages["numpy"],
                "scipy": packages["scipy"],
                "arch": packages["arch"],
                "bit_generator": bit_generator,
                "path_generator": path_generator,
            },
            reproducibility={
                "schema_version": REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION,
                "result_schema_version": SIMULATION_RESULT_SCHEMA_VERSION,
                "engine_version": SIMULATION_ENGINE_VERSION,
                "scenario": {
                    "canonicalization": _SCENARIO_CANONICALIZATION,
                    "sha256": scenario_sha256,
                },
                "random": {
                    "seed": scenario.seed,
                    "bit_generator": bit_generator,
                },
                "packages": {
                    "python": platform.python_version(),
                    **packages,
                },
                "historical": historical_manifest,
                "bootstrap": bootstrap_manifest,
                "run_policy": _resource_policy_manifest(
                    paths,
                    evaluation_state_bytes=evaluation_state_bytes,
                ),
                "outcome_semantics": outcome_semantics_manifest(),
                "tax_policy": _tax_policy_manifest(scenario),
                "household": (
                    {
                        "cash_flow_audit": "per-person annual real-dollar percentiles",
                        "longevity": (
                            "seeded_correlated_bounded_normal_or_deterministic_v1"
                        ),
                        "longevity_random": {
                            "substream": "household_longevity_v1",
                            "seed_derivation": "scenario_seed_xor_constant",
                            "seed_xor_hex": "0x535341",
                            "draw_policy": (
                                "one standard-normal draw per person and trial; "
                                "second-person correlation applied after draw"
                            ),
                        },
                        "assets_after_death": "remain_in_household_portfolio",
                        "pre_retirement_household_income": (
                            "audit_only_unless_an_explicit_cash_flow_contributes_it"
                        ),
                        "post_death_outcomes": (
                            "inactive_for_recovery_and_spending_tier_attainment"
                        ),
                        "survivor_spending_tiers": (
                            "scaled_by_survivor_spending_fraction"
                        ),
                        "tax_engine_port": (
                            household_tax_engine_for(
                                scenario.tax_assumptions,
                                tax_engine,
                            ).engine_id
                            if scenario.tax_assumptions is not None
                            else "not_used"
                        ),
                        "social_security": social_security_policy_manifest(),
                    }
                    if scenario.household is not None
                    else None
                ),
                "valuation": (
                    valuation_provenance.model_dump(mode="json")
                    if valuation_provenance is not None
                    else None
                ),
                "retirement_spending": spending_manifest,
                "path_generator": path_generator,
            },
        )
    finally:
        if evaluation_reservation is not None:
            evaluation_reservation.release()
        if owns_paths and isinstance(paths, PreparedExperiment):
            paths.close()
