"""Retirement spending tiers and deterministic guardrail policies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import numpy as np


MAX_GUARDRAIL_YEARS = 130


class SpendingTier(StrEnum):
    """Outcome priority attached to a stable YNAB category identity."""

    ESSENTIAL = "essential"
    LIFESTYLE = "lifestyle"
    DISCRETIONARY = "discretionary"
    ONE_TIME = "one_time"


class SpendingTierAmounts(BaseModel):
    """Annual real spending assigned to each outcome tier."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    essential: float = Field(default=0, ge=0)
    lifestyle: float = Field(default=0, ge=0)
    discretionary: float = Field(default=0, ge=0)
    one_time: float = Field(default=0, ge=0)

    @property
    def total(self) -> float:
        return (
            self.essential
            + self.lifestyle
            + self.discretionary
            + self.one_time
        )


class SpendingTierDelta(BaseModel):
    """Signed annual real-dollar change for each spending tier."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    essential: float = 0
    lifestyle: float = 0
    discretionary: float = 0
    one_time: float = 0

    @property
    def total(self) -> float:
        return (
            self.essential
            + self.lifestyle
            + self.discretionary
            + self.one_time
        )


class FixedRealSpendingPolicy(BaseModel):
    """Restore and hold the real baseline every year."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["fixed_real"] = "fixed_real"


class FloorCeilingSpendingPolicy(BaseModel):
    """Target a portfolio percentage with bounded annual changes."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    kind: Literal["floor_ceiling"] = "floor_ceiling"
    target_withdrawal_rate: float = Field(gt=0, lt=1)
    maximum_annual_reduction: float = Field(default=0.05, ge=0, lt=1)
    maximum_annual_restoration: float = Field(default=0.05, ge=0, lt=1)


class WithdrawalRateGuardrailPolicy(BaseModel):
    """Cut or restore spending when its withdrawal rate crosses guardrails."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    kind: Literal["withdrawal_rate_guardrails"] = (
        "withdrawal_rate_guardrails"
    )
    lower_withdrawal_rate: float = Field(gt=0, lt=1)
    upper_withdrawal_rate: float = Field(gt=0, lt=1)
    reduction_rate: float = Field(default=0.10, gt=0, lt=1)
    restoration_rate: float = Field(default=0.10, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_guardrails(self) -> WithdrawalRateGuardrailPolicy:
        if self.lower_withdrawal_rate >= self.upper_withdrawal_rate:
            raise ValueError(
                "lower withdrawal-rate guardrail must be below the upper guardrail"
            )
        return self


SpendingPolicy = Annotated[
    FixedRealSpendingPolicy
    | FloorCeilingSpendingPolicy
    | WithdrawalRateGuardrailPolicy,
    Field(discriminator="kind"),
]


class SpendingPlanSource(BaseModel):
    """Stable YNAB provenance for one derived tier baseline."""

    model_config = ConfigDict(frozen=True)

    budget_id: str = Field(min_length=1, max_length=64)
    through_month: str = Field(pattern=r"^\d{4}-\d{2}$")
    lookback_months: int = Field(ge=1, le=120)
    mapping_sha256: str = Field(min_length=64, max_length=64)
    mapped_category_count: int = Field(ge=0)


class RetirementSpendingPlan(BaseModel):
    """Policy-independent tier cash flows plus a replaceable policy."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    baseline: SpendingTierAmounts
    essential_floor: float = Field(ge=0)
    policy: SpendingPolicy = Field(
        default_factory=FixedRealSpendingPolicy,
    )
    source: SpendingPlanSource | None = None

    @model_validator(mode="after")
    def validate_essential_floor(self) -> RetirementSpendingPlan:
        if self.essential_floor > self.baseline.essential:
            raise ValueError(
                "essential floor cannot exceed baseline essential spending"
            )
        return self


class AnnualSpendingContext(BaseModel):
    """One real-dollar portfolio observation used by a policy."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    age: int = Field(ge=0, le=130)
    opening_portfolio_real: float = Field(ge=0)


class GuardrailAnnualAudit(BaseModel):
    """Auditable annual spending reduction, restoration, or hold."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    age: int
    policy: str
    action: Literal["held", "reduced", "restored"]
    reason: str
    opening_portfolio_real: float
    withdrawal_rate_before: float | None
    previous_spending: SpendingTierAmounts
    applied_spending: SpendingTierAmounts
    spending_delta: SpendingTierDelta
    essential_floor: float


class GuardrailAuditSummary(BaseModel):
    """Annual path plus metrics that #88 can merge into outcome semantics."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    reduction_years: int
    restoration_years: int
    cumulative_reduction_real: float
    cumulative_restoration_real: float
    minimum_essential_spending_real: float
    annual_path: tuple[GuardrailAnnualAudit, ...] = Field(
        max_length=MAX_GUARDRAIL_YEARS,
    )


class SpendingPlanManifest(BaseModel):
    """Versioned material assumptions for scenario reproducibility."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    plan_sha256: str = Field(min_length=64, max_length=64)
    baseline: SpendingTierAmounts
    essential_floor: float
    policy: SpendingPolicy
    source: SpendingPlanSource | None


class AnnualSpendingAuditSink(Protocol):
    """Narrow integration seam for #88 annual outcome rows."""

    def record_spending_audit(self, audit: GuardrailAnnualAudit) -> None: ...


@dataclass(frozen=True)
class GuardrailBatchDecision:
    """Vectorized trial decision consumed by the simulation engine."""

    applied_total_real: np.ndarray
    essential_real: np.ndarray
    lifestyle_real: np.ndarray
    discretionary_real: np.ndarray
    one_time_real: np.ndarray
    withdrawal_rate_before: np.ndarray
    action: np.ndarray


def estimate_guardrail_state_bytes(trials: int) -> int:
    """Conservatively estimate the persistent per-trial guardrail state."""
    if trials < 0:
        raise ValueError("trials must not be negative")
    float_arrays = 9
    integer_arrays = 2
    return trials * (float_arrays * 8 + integer_arrays * 4)


def evaluate_guardrails(
    plan: RetirementSpendingPlan,
    contexts: Iterable[AnnualSpendingContext],
    *,
    audit_sink: AnnualSpendingAuditSink | None = None,
) -> GuardrailAuditSummary:
    """Evaluate one policy without rewriting the tier cash-flow baseline."""
    rows: list[GuardrailAnnualAudit] = []
    previous = plan.baseline
    previous_age: int | None = None
    for context in contexts:
        if previous_age is not None and context.age <= previous_age:
            raise ValueError("guardrail context ages must be strictly increasing")
        if len(rows) >= MAX_GUARDRAIL_YEARS:
            raise ValueError("guardrail path exceeds the supported horizon")
        row = _evaluate_year(plan, context, previous)
        rows.append(row)
        if audit_sink is not None:
            audit_sink.record_spending_audit(row)
        previous = row.applied_spending
        previous_age = context.age

    minimum_essential = min(
        (row.applied_spending.essential for row in rows),
        default=plan.baseline.essential,
    )
    return GuardrailAuditSummary(
        reduction_years=sum(row.action == "reduced" for row in rows),
        restoration_years=sum(row.action == "restored" for row in rows),
        cumulative_reduction_real=sum(
            max(0.0, plan.baseline.total - row.applied_spending.total)
            for row in rows
        ),
        cumulative_restoration_real=sum(
            max(0.0, row.spending_delta.total)
            for row in rows
            if row.action == "restored"
        ),
        minimum_essential_spending_real=minimum_essential,
        annual_path=tuple(rows),
    )


def guardrail_outcome_metrics(
    summary: GuardrailAuditSummary,
) -> dict[str, float | int]:
    """Return fields that #88 can merge into its outcome metric object."""
    return {
        "spending_reduction_years": summary.reduction_years,
        "spending_restoration_years": summary.restoration_years,
        "cumulative_spending_reduction_real": (
            summary.cumulative_reduction_real
        ),
        "cumulative_spending_restoration_real": (
            summary.cumulative_restoration_real
        ),
        "minimum_essential_spending_real": (
            summary.minimum_essential_spending_real
        ),
    }


def spending_plan_manifest(
    plan: RetirementSpendingPlan,
) -> SpendingPlanManifest:
    """Capture every material tier, floor, policy, and YNAB source assumption."""
    canonical = json.dumps(
        plan.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return SpendingPlanManifest(
        plan_sha256=sha256(canonical).hexdigest(),
        baseline=plan.baseline,
        essential_floor=plan.essential_floor,
        policy=plan.policy,
        source=plan.source,
    )


def evaluate_guardrail_batch(
    plan: RetirementSpendingPlan,
    *,
    opening_portfolio_real: np.ndarray,
    previous_total_real: np.ndarray,
) -> GuardrailBatchDecision:
    """Apply one policy across a bounded simulation trial batch."""
    import numpy as np

    if opening_portfolio_real.shape != previous_total_real.shape:
        raise ValueError("guardrail portfolio and spending batches must align")
    withdrawal_rate = np.divide(
        previous_total_real,
        opening_portfolio_real,
        out=np.full_like(previous_total_real, np.inf, dtype=float),
        where=opening_portfolio_real > 0,
    )
    policy = plan.policy
    if isinstance(policy, FixedRealSpendingPolicy):
        target = np.full_like(
            previous_total_real,
            plan.baseline.total,
            dtype=float,
        )
    elif isinstance(policy, FloorCeilingSpendingPolicy):
        desired = (
            opening_portfolio_real * policy.target_withdrawal_rate
        )
        annual_floor = previous_total_real * (
            1 - policy.maximum_annual_reduction
        )
        annual_ceiling = previous_total_real * (
            1 + policy.maximum_annual_restoration
        )
        target = np.minimum(
            plan.baseline.total,
            np.maximum(
                plan.essential_floor,
                np.minimum(annual_ceiling, np.maximum(annual_floor, desired)),
            ),
        )
    else:
        target = previous_total_real.copy()
        reduce = (
            (opening_portfolio_real <= 0)
            | (withdrawal_rate > policy.upper_withdrawal_rate)
        )
        restore = (
            ~reduce
            & (withdrawal_rate < policy.lower_withdrawal_rate)
            & (previous_total_real < plan.baseline.total)
        )
        target[reduce] = np.maximum(
            plan.essential_floor,
            previous_total_real[reduce] * (1 - policy.reduction_rate),
        )
        target[restore] = np.minimum(
            plan.baseline.total,
            previous_total_real[restore] * (1 + policy.restoration_rate),
        )

    target = np.minimum(
        plan.baseline.total,
        np.maximum(plan.essential_floor, target),
    )
    essential = np.minimum(plan.baseline.essential, target)
    essential = np.maximum(plan.essential_floor, essential)
    remaining = np.maximum(0.0, target - essential)
    lifestyle = np.minimum(plan.baseline.lifestyle, remaining)
    remaining = np.maximum(0.0, remaining - lifestyle)
    discretionary = np.minimum(plan.baseline.discretionary, remaining)
    remaining = np.maximum(0.0, remaining - discretionary)
    one_time = np.minimum(plan.baseline.one_time, remaining)
    applied = essential + lifestyle + discretionary + one_time
    action = np.zeros(applied.shape, dtype=np.int8)
    action[applied < previous_total_real - 0.005] = -1
    action[applied > previous_total_real + 0.005] = 1
    return GuardrailBatchDecision(
        applied_total_real=applied,
        essential_real=essential,
        lifestyle_real=lifestyle,
        discretionary_real=discretionary,
        one_time_real=one_time,
        withdrawal_rate_before=withdrawal_rate,
        action=action,
    )


def _evaluate_year(
    plan: RetirementSpendingPlan,
    context: AnnualSpendingContext,
    previous: SpendingTierAmounts,
) -> GuardrailAnnualAudit:
    withdrawal_rate = (
        previous.total / context.opening_portfolio_real
        if context.opening_portfolio_real > 0
        else None
    )
    target_total, reason = _target_total(
        plan,
        previous=previous,
        opening_portfolio=context.opening_portfolio_real,
        withdrawal_rate=withdrawal_rate,
    )
    applied = _allocate_tiers(plan, target_total)
    delta = SpendingTierDelta(
        essential=applied.essential - previous.essential,
        lifestyle=applied.lifestyle - previous.lifestyle,
        discretionary=applied.discretionary - previous.discretionary,
        one_time=applied.one_time - previous.one_time,
    )
    if applied.total < previous.total - 0.005:
        action: Literal["held", "reduced", "restored"] = "reduced"
    elif applied.total > previous.total + 0.005:
        action = "restored"
    else:
        action = "held"
    return GuardrailAnnualAudit(
        age=context.age,
        policy=plan.policy.kind,
        action=action,
        reason=reason,
        opening_portfolio_real=context.opening_portfolio_real,
        withdrawal_rate_before=withdrawal_rate,
        previous_spending=previous,
        applied_spending=applied,
        spending_delta=delta,
        essential_floor=plan.essential_floor,
    )


def _target_total(
    plan: RetirementSpendingPlan,
    *,
    previous: SpendingTierAmounts,
    opening_portfolio: float,
    withdrawal_rate: float | None,
) -> tuple[float, str]:
    policy = plan.policy
    if isinstance(policy, FixedRealSpendingPolicy):
        return plan.baseline.total, "fixed_real_baseline"
    if isinstance(policy, FloorCeilingSpendingPolicy):
        desired = opening_portfolio * policy.target_withdrawal_rate
        annual_floor = previous.total * (
            1 - policy.maximum_annual_reduction
        )
        annual_ceiling = previous.total * (
            1 + policy.maximum_annual_restoration
        )
        bounded = min(annual_ceiling, max(annual_floor, desired))
        return (
            min(plan.baseline.total, max(plan.essential_floor, bounded)),
            "portfolio_target_bounded_by_annual_floor_ceiling",
        )
    if withdrawal_rate is None or (
        withdrawal_rate > policy.upper_withdrawal_rate
    ):
        return (
            max(
                plan.essential_floor,
                previous.total * (1 - policy.reduction_rate),
            ),
            "upper_withdrawal_rate_guardrail",
        )
    if (
        withdrawal_rate < policy.lower_withdrawal_rate
        and previous.total < plan.baseline.total
    ):
        return (
            min(
                plan.baseline.total,
                previous.total * (1 + policy.restoration_rate),
            ),
            "lower_withdrawal_rate_guardrail",
        )
    return previous.total, "within_withdrawal_rate_guardrails"


def _allocate_tiers(
    plan: RetirementSpendingPlan,
    target_total: float,
) -> SpendingTierAmounts:
    target = min(
        plan.baseline.total,
        max(plan.essential_floor, target_total),
    )
    essential = min(plan.baseline.essential, target)
    if essential < plan.essential_floor:
        essential = plan.essential_floor
    remaining = max(0.0, target - essential)
    lifestyle = min(plan.baseline.lifestyle, remaining)
    remaining -= lifestyle
    discretionary = min(plan.baseline.discretionary, remaining)
    remaining -= discretionary
    one_time = min(plan.baseline.one_time, remaining)
    return SpendingTierAmounts(
        essential=essential,
        lifestyle=lifestyle,
        discretionary=discretionary,
        one_time=one_time,
    )
