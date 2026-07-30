"""Bounded retirement shortfall and goal-attainment semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ynab_agent.planning.models import WealthScenario

if TYPE_CHECKING:
    import numpy as np


OUTCOME_SEMANTICS_SCHEMA_VERSION = 2
_REAL_SHORTFALL_TOLERANCE = 0.005
_FLOAT_BYTES = 8


class GoalKind(StrEnum):
    """Stable categories for scenario outcome goals."""

    PLANNED_SPENDING = "planned_spending"
    SPENDING_TIER = "spending_tier"
    LEGACY_TARGET = "legacy_target"


@dataclass(frozen=True)
class GoalOutcome:
    """Probability and evidence for one bounded scenario goal."""

    name: str
    kind: GoalKind
    target_real: float
    attainment_probability: float | None
    attained_trials: int | None
    evaluation_basis: str


@dataclass(frozen=True)
class OutcomeSummary:
    """Aggregate shortfall severity and goal outcomes."""

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


def estimate_outcome_state_bytes(scenario: WealthScenario) -> int:
    """Conservatively estimate full-report per-trial outcome state."""
    # Required, shortfall, failure count/current/longest streak, recovered flag,
    # one percentile work vector, and one attainment flag per explicit tier.
    array_count = 7 + len(scenario.spending_tiers)
    return scenario.trials * array_count * _FLOAT_BYTES


def outcome_semantics_manifest() -> dict[str, object]:
    """Describe versioned result semantics without scenario-specific values."""
    return {
        "schema_version": OUTCOME_SEMANTICS_SCHEMA_VERSION,
        "success_rate": (
            "probability of no retirement year with unfunded spending after "
            "applying the configured spending policy; fixed-real scenarios "
            "preserve the pre-outcome-semantics depletion definition"
        ),
        "funded_spending_ratio": (
            "per-trial cumulative real spending funded divided by cumulative "
            "real spending required after applying the configured policy"
        ),
        "funded_spending_real": (
            "sum of policy-adjusted spending funded over retirement in today's "
            "dollars"
        ),
        "cumulative_shortfall_real": (
            "sum of annual unfunded policy-adjusted spending in today's dollars"
        ),
        "failure_duration_years": (
            "percentiles among affected trials of retirement years with a shortfall"
        ),
        "longest_failure_streak_years": (
            "percentiles among affected trials of consecutive shortfall years"
        ),
        "recovery_probability": (
            "among affected trials, probability of at least one fully funded "
            "retirement year after a shortfall year"
        ),
        "spending_tier": (
            "attained only when the named real annual amount is funded in every "
            "retirement year"
        ),
        "legacy_target": (
            "attained when tax-adjusted real ending estate meets the target; "
            "requires explicit tax buckets and assumptions"
        ),
    }


class OutcomeAccumulator:
    """Collect full-report metrics with memory bounded by trials and goal count."""

    def __init__(self, scenario: WealthScenario) -> None:
        import numpy as np

        self.scenario = scenario
        self.cumulative_required_real = np.zeros(scenario.trials, dtype=float)
        self.cumulative_shortfall_real = np.zeros(scenario.trials, dtype=float)
        self.failure_years = np.zeros(scenario.trials, dtype=np.int64)
        self.current_failure_streak = np.zeros(scenario.trials, dtype=np.int64)
        self.longest_failure_streak = np.zeros(scenario.trials, dtype=np.int64)
        self.recovered_after_failure = np.zeros(scenario.trials, dtype=bool)
        self.tier_attained = np.ones(
            (len(scenario.spending_tiers), scenario.trials),
            dtype=bool,
        )

    def record_retirement_year(
        self,
        trial_slice: slice,
        *,
        required_real: float | np.ndarray,
        shortfall_real: np.ndarray,
        failed: np.ndarray,
    ) -> None:
        """Record one year's real spending outcomes for one trial batch."""
        import numpy as np

        required = np.broadcast_to(
            np.asarray(required_real, dtype=float),
            shortfall_real.shape,
        )
        funded = np.maximum(0.0, required - shortfall_real)
        self.cumulative_required_real[trial_slice] += required
        self.cumulative_shortfall_real[trial_slice] += shortfall_real

        prior_failures = self.failure_years[trial_slice] > 0
        recovered = prior_failures & ~failed
        self.recovered_after_failure[trial_slice] |= recovered
        self.failure_years[trial_slice] += failed
        current = self.current_failure_streak[trial_slice]
        current[failed] += 1
        current[~failed] = 0
        np.maximum(
            self.longest_failure_streak[trial_slice],
            current,
            out=self.longest_failure_streak[trial_slice],
        )

        for tier_index, tier in enumerate(self.scenario.spending_tiers):
            self.tier_attained[tier_index, trial_slice] &= (
                funded + _REAL_SHORTFALL_TOLERANCE >= tier.annual_amount
            )

    def summarize(
        self,
        *,
        successful_trials: int,
        after_tax_ending_balance_real: np.ndarray | None,
    ) -> OutcomeSummary:
        """Summarize bounded state into stable public outcome primitives."""
        import numpy as np

        funded_ratio = np.divide(
            self.cumulative_required_real - self.cumulative_shortfall_real,
            self.cumulative_required_real,
            out=np.ones_like(self.cumulative_required_real),
            where=self.cumulative_required_real > 0,
        )
        np.clip(funded_ratio, 0.0, 1.0, out=funded_ratio)
        affected = self.failure_years > 0
        affected_trials = int(np.count_nonzero(affected))
        recovered_trials = int(
            np.count_nonzero(self.recovered_after_failure & affected)
        )

        goals = [
            GoalOutcome(
                name="planned_spending",
                kind=GoalKind.PLANNED_SPENDING,
                target_real=self.scenario.annual_spending,
                attainment_probability=successful_trials / self.scenario.trials,
                attained_trials=successful_trials,
                evaluation_basis=(
                    "all_policy_adjusted_retirement_spending_funded"
                    if self.scenario.retirement_spending_plan is not None
                    else "all_planned_retirement_spending_funded"
                ),
            )
        ]
        for tier_index, tier in enumerate(self.scenario.spending_tiers):
            tier_attained_trials = int(
                np.count_nonzero(self.tier_attained[tier_index])
            )
            goals.append(
                GoalOutcome(
                    name=tier.name,
                    kind=GoalKind.SPENDING_TIER,
                    target_real=tier.annual_amount,
                    attainment_probability=(
                        tier_attained_trials / self.scenario.trials
                    ),
                    attained_trials=tier_attained_trials,
                    evaluation_basis="minimum_annual_real_spending_funded",
                )
            )

        after_tax_percentiles = (
            _percentiles(after_tax_ending_balance_real)
            if after_tax_ending_balance_real is not None
            else None
        )
        legacy_probability: float | None = None
        if self.scenario.legacy_target_real is not None:
            attained_trials: int | None = None
            evaluation_basis = "explicit_tax_inputs_required"
            if after_tax_ending_balance_real is not None:
                attained_trials = int(
                    np.count_nonzero(
                        after_tax_ending_balance_real + _REAL_SHORTFALL_TOLERANCE
                        >= self.scenario.legacy_target_real
                    )
                )
                legacy_probability = attained_trials / self.scenario.trials
                evaluation_basis = "tax_adjusted_real_ending_estate"
            goals.append(
                GoalOutcome(
                    name="legacy_target",
                    kind=GoalKind.LEGACY_TARGET,
                    target_real=self.scenario.legacy_target_real,
                    attainment_probability=legacy_probability,
                    attained_trials=attained_trials,
                    evaluation_basis=evaluation_basis,
                )
            )

        return OutcomeSummary(
            funded_spending_ratio=_percentiles(funded_ratio),
            funded_spending_real=_percentiles(
                self.cumulative_required_real
                - self.cumulative_shortfall_real
            ),
            cumulative_shortfall_real=_percentiles(
                self.cumulative_shortfall_real
            ),
            failure_duration_years=(
                _percentiles(self.failure_years[affected])
                if affected_trials
                else None
            ),
            longest_failure_streak_years=(
                _percentiles(self.longest_failure_streak[affected])
                if affected_trials
                else None
            ),
            recovered_trials=recovered_trials,
            recovery_probability=(
                recovered_trials / affected_trials
                if affected_trials
                else None
            ),
            after_tax_ending_balance_real=after_tax_percentiles,
            legacy_target_probability=legacy_probability,
            goal_outcomes=tuple(goals),
        )


def _percentiles(values: np.ndarray) -> dict[str, float]:
    import numpy as np

    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}
