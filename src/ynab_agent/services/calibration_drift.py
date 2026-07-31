"""Pure, deterministic plan-versus-observation drift detectors."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pydantic import Field, field_validator, model_validator

from ynab_agent.services.calibration_models import (
    ConfidenceAssessment,
    DriftEvent,
    DriftKind,
    FreshnessAssessment,
    FreshnessState,
    FrozenCalibrationModel,
    FrozenFloatMap,
    FrozenJsonObject,
    ImmutableJson,
    MAX_IDENTIFIER_LENGTH,
    MAX_SOURCE_LINKS_PER_OBSERVATION,
    MaterialityThreshold,
    ObservationUnit,
    ThresholdMode,
    _lossless_float,
    canonical_content_sha256,
    classify_freshness,
)


MAX_ALLOCATION_CLASSES = 32
_WEIGHT_TOLERANCE = 1e-6


class ScalarDriftInput(FrozenCalibrationModel):
    """Inputs shared by scalar spending, contribution, and balance drift."""

    subject_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    baseline_value: int | float
    observed_value: int | float
    unit: ObservationUnit
    threshold: MaterialityThreshold
    confidence: ConfidenceAssessment
    observation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_LINKS_PER_OBSERVATION,
    )

    @model_validator(mode="after")
    def validate_observation_ids(self) -> ScalarDriftInput:
        _lossless_float(self.baseline_value, label="scalar drift baseline")
        _lossless_float(self.observed_value, label="scalar drift observation")
        _validate_observation_ids(self.observation_ids)
        return self


class DebtPayoffInput(FrozenCalibrationModel):
    """Normalized non-negative liabilities used to detect a payoff crossing."""

    subject_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    previous_liability: int | float = Field(ge=0)
    current_liability: int | float = Field(ge=0)
    payoff_tolerance: int | float = Field(default=1, ge=0)
    threshold: MaterialityThreshold
    confidence: ConfidenceAssessment
    observation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_LINKS_PER_OBSERVATION,
    )

    @model_validator(mode="after")
    def validate_observation_ids(self) -> DebtPayoffInput:
        _lossless_float(self.previous_liability, label="previous liability")
        _lossless_float(self.current_liability, label="current liability")
        _lossless_float(self.payoff_tolerance, label="payoff tolerance")
        _validate_observation_ids(self.observation_ids)
        return self


class AllocationDriftInput(FrozenCalibrationModel):
    """Target and observed asset weights for one account or portfolio."""

    subject_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    target_weights: FrozenFloatMap
    observed_weights: FrozenFloatMap
    threshold: MaterialityThreshold
    confidence: ConfidenceAssessment
    observation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_LINKS_PER_OBSERVATION,
    )

    @model_validator(mode="after")
    def validate_weights(self) -> AllocationDriftInput:
        if (
            not 1 <= len(self.target_weights) <= MAX_ALLOCATION_CLASSES
            or not 1 <= len(self.observed_weights) <= MAX_ALLOCATION_CLASSES
        ):
            raise ValueError("allocation class count is invalid")
        if set(self.target_weights) != set(self.observed_weights):
            raise ValueError("target and observed allocation classes must match")
        for weights in (self.target_weights, self.observed_weights):
            if any(weight < 0 or weight > 1 for weight in weights.values()):
                raise ValueError("allocation weights must be between zero and one")
            if abs(sum(weights.values()) - 1) > _WEIGHT_TOLERANCE:
                raise ValueError("allocation weights must sum to one")
        if any(
            not asset_class or len(asset_class) > MAX_IDENTIFIER_LENGTH
            for asset_class in self.target_weights
        ):
            raise ValueError("allocation class identifiers are invalid")
        _validate_observation_ids(self.observation_ids)
        return self


class StalenessInput(FrozenCalibrationModel):
    """Evidence timestamps used to classify stale and frozen valuations."""

    subject_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    evaluated_at: datetime
    latest_activity_at: datetime | None = None
    last_reconciled_at: datetime | None = None
    latest_reviewed_at: datetime | None = None
    unchanged_since: datetime | None = None
    stale_after_days: int = Field(gt=0, le=3_650)
    frozen_after_days: int = Field(gt=0, le=3_650)
    threshold: MaterialityThreshold
    confidence: ConfidenceAssessment
    observation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_LINKS_PER_OBSERVATION,
    )

    @field_validator(
        "evaluated_at",
        "latest_activity_at",
        "last_reconciled_at",
        "latest_reviewed_at",
        "unchanged_since",
    )
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("staleness timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_chronology(self) -> StalenessInput:
        if self.frozen_after_days < self.stale_after_days:
            raise ValueError("frozen_after_days must be at least stale_after_days")
        timestamps = (
            self.latest_activity_at,
            self.last_reconciled_at,
            self.latest_reviewed_at,
            self.unchanged_since,
        )
        if any(timestamp is not None and timestamp > self.evaluated_at for timestamp in timestamps):
            raise ValueError("staleness evidence cannot postdate evaluated_at")
        _validate_observation_ids(self.observation_ids)
        return self


def detect_spending_drift(candidate: ScalarDriftInput) -> DriftEvent:
    """Compare annualized observed spending with the resolved plan amount."""
    _require_unit(candidate, ObservationUnit.DOLLARS_PER_YEAR, "spending")
    return _detect_scalar(DriftKind.SPENDING, candidate)


def detect_contribution_drift(candidate: ScalarDriftInput) -> DriftEvent:
    """Compare annualized external portfolio contributions with the plan."""
    _require_unit(
        candidate,
        ObservationUnit.DOLLARS_PER_YEAR,
        "contribution",
    )
    return _detect_scalar(DriftKind.CONTRIBUTION, candidate)


def detect_balance_drift(candidate: ScalarDriftInput) -> DriftEvent:
    """Compare a prior resolved account balance with a copied observation."""
    _require_unit(candidate, ObservationUnit.DOLLARS, "balance")
    return _detect_scalar(DriftKind.BALANCE, candidate)


def detect_debt_payoff(candidate: DebtPayoffInput) -> DriftEvent:
    """Detect a liability crossing the configured paid-off tolerance."""
    previous_liability = _lossless_float(
        candidate.previous_liability,
        label="previous liability",
    )
    current_liability = _lossless_float(
        candidate.current_liability,
        label="current liability",
    )
    payoff_tolerance = _lossless_float(
        candidate.payoff_tolerance,
        label="payoff tolerance",
    )
    signed_delta = current_liability - previous_liability
    absolute_delta = abs(signed_delta)
    relative_delta = _relative_delta(
        signed_delta,
        previous_liability,
        candidate.threshold.relative_floor,
    )
    paid_off = previous_liability > payoff_tolerance and current_liability <= payoff_tolerance
    material = paid_off and _is_material(
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        threshold=candidate.threshold,
        confidence=candidate.confidence,
    )
    return _event(
        kind=DriftKind.DEBT_PAYOFF,
        subject_id=candidate.subject_id,
        baseline_value=candidate.previous_liability,
        observed_value=candidate.current_liability,
        unit=ObservationUnit.DOLLARS,
        absolute_delta=absolute_delta,
        signed_delta=signed_delta,
        relative_delta=relative_delta,
        material=material,
        threshold=candidate.threshold,
        confidence=candidate.confidence,
        observation_ids=candidate.observation_ids,
        details={
            "paid_off": paid_off,
            "payoff_tolerance": candidate.payoff_tolerance,
        },
    )


def detect_allocation_drift(candidate: AllocationDriftInput) -> DriftEvent:
    """Measure observed allocation distance without inferring its cause."""
    ordered_classes = tuple(sorted(candidate.target_weights))
    signed_by_class = {
        asset_class: (
            candidate.observed_weights[asset_class] - candidate.target_weights[asset_class]
        )
        for asset_class in ordered_classes
    }
    max_absolute_drift = max(abs(delta) for delta in signed_by_class.values())
    relative_by_class = {
        asset_class: _relative_delta(
            delta,
            candidate.target_weights[asset_class],
            candidate.threshold.relative_floor,
        )
        for asset_class, delta in signed_by_class.items()
    }
    max_relative_drift = max(abs(delta) for delta in relative_by_class.values())
    total_variation = 0.5 * sum(abs(delta) for delta in signed_by_class.values())
    largest_class = min(
        (
            asset_class
            for asset_class, delta in signed_by_class.items()
            if abs(delta) == max_absolute_drift
        ),
        default=ordered_classes[0],
    )
    largest_relative_class = min(
        (
            asset_class
            for asset_class, delta in relative_by_class.items()
            if abs(delta) == max_relative_drift
        ),
        default=ordered_classes[0],
    )
    material_by_class = {
        asset_class: _is_material(
            absolute_delta=abs(signed_by_class[asset_class]),
            relative_delta=relative_by_class[asset_class],
            threshold=candidate.threshold,
            confidence=candidate.confidence,
        )
        for asset_class in ordered_classes
    }
    material_classes = tuple(
        asset_class for asset_class in ordered_classes if material_by_class[asset_class]
    )
    reported_class = (
        max(
            material_classes,
            key=lambda asset_class: (
                _materiality_score(
                    absolute_delta=abs(signed_by_class[asset_class]),
                    relative_delta=relative_by_class[asset_class],
                    threshold=candidate.threshold,
                ),
                asset_class,
            ),
        )
        if material_classes
        else largest_class
    )
    signed_delta = signed_by_class[reported_class]
    relative_delta = relative_by_class[reported_class]
    absolute_delta = abs(signed_delta)
    material = bool(material_classes)
    return _event(
        kind=DriftKind.ALLOCATION,
        subject_id=candidate.subject_id,
        baseline_value=FrozenJsonObject(candidate.target_weights),
        observed_value=FrozenJsonObject(candidate.observed_weights),
        unit=ObservationUnit.FRACTION,
        absolute_delta=absolute_delta,
        signed_delta=signed_delta,
        relative_delta=relative_delta,
        material=material,
        threshold=candidate.threshold,
        confidence=candidate.confidence,
        observation_ids=candidate.observation_ids,
        details={
            "largest_drift_class": largest_class,
            "largest_relative_drift_class": largest_relative_class,
            "reported_class": reported_class,
            "portfolio_max_absolute_drift": max_absolute_drift,
            "signed_drift_by_class": FrozenJsonObject(signed_by_class),
            "relative_drift_by_class": FrozenJsonObject(relative_by_class),
            "total_variation_distance": total_variation,
        },
    )


def assess_staleness(candidate: StalenessInput) -> FreshnessAssessment:
    """Classify source evidence, treating mechanical syncs as non-review evidence."""
    (
        state,
        latest_evidence,
        age_days,
        unchanged_days,
        evidence,
    ) = classify_freshness(
        evaluated_at=candidate.evaluated_at,
        latest_activity_at=candidate.latest_activity_at,
        last_reconciled_at=candidate.last_reconciled_at,
        latest_reviewed_at=candidate.latest_reviewed_at,
        unchanged_since=candidate.unchanged_since,
        stale_after_days=candidate.stale_after_days,
        frozen_after_days=candidate.frozen_after_days,
    )
    return FreshnessAssessment(
        state=state,
        evaluated_at=candidate.evaluated_at,
        latest_activity_at=candidate.latest_activity_at,
        last_reconciled_at=candidate.last_reconciled_at,
        latest_reviewed_at=candidate.latest_reviewed_at,
        latest_evidence_at=latest_evidence,
        unchanged_since=candidate.unchanged_since,
        age_days=age_days,
        unchanged_days=unchanged_days,
        stale_after_days=candidate.stale_after_days,
        frozen_after_days=candidate.frozen_after_days,
        evidence=evidence,
    )


def detect_staleness(candidate: StalenessInput) -> DriftEvent:
    """Return an alert-ready event for one evidence-based freshness assessment."""
    freshness = assess_staleness(candidate)
    if freshness.state is FreshnessState.FROZEN:
        observed_days = freshness.unchanged_days or 0
    elif freshness.state is FreshnessState.STALE:
        observed_days = freshness.age_days or 0
    else:
        observed_days = 0
    relative_delta = _relative_delta(
        float(observed_days),
        0,
        candidate.threshold.relative_floor,
    )
    material = freshness.state in {FreshnessState.STALE, FreshnessState.FROZEN} and _is_material(
        absolute_delta=float(observed_days),
        relative_delta=relative_delta,
        threshold=candidate.threshold,
        confidence=candidate.confidence,
    )
    return _event(
        kind=DriftKind.STALENESS,
        subject_id=candidate.subject_id,
        baseline_value=FrozenJsonObject(
            {
                "stale_after_days": candidate.stale_after_days,
                "frozen_after_days": candidate.frozen_after_days,
            }
        ),
        observed_value=FrozenJsonObject(
            {
                "state": freshness.state.value,
                "age_days": freshness.age_days,
                "unchanged_days": freshness.unchanged_days,
            }
        ),
        unit=ObservationUnit.DAYS,
        absolute_delta=float(observed_days),
        signed_delta=float(observed_days),
        relative_delta=relative_delta,
        material=material,
        threshold=candidate.threshold,
        confidence=candidate.confidence,
        observation_ids=candidate.observation_ids,
        freshness=freshness,
    )


def _detect_scalar(
    kind: DriftKind,
    candidate: ScalarDriftInput,
) -> DriftEvent:
    baseline = _lossless_float(candidate.baseline_value, label="scalar drift baseline")
    observed = _lossless_float(candidate.observed_value, label="scalar drift observation")
    signed_delta = observed - baseline
    absolute_delta = abs(signed_delta)
    relative_delta = _relative_delta(
        signed_delta,
        baseline,
        candidate.threshold.relative_floor,
    )
    return _event(
        kind=kind,
        subject_id=candidate.subject_id,
        baseline_value=candidate.baseline_value,
        observed_value=candidate.observed_value,
        unit=candidate.unit,
        absolute_delta=absolute_delta,
        signed_delta=signed_delta,
        relative_delta=relative_delta,
        material=_is_material(
            absolute_delta=absolute_delta,
            relative_delta=relative_delta,
            threshold=candidate.threshold,
            confidence=candidate.confidence,
        ),
        threshold=candidate.threshold,
        confidence=candidate.confidence,
        observation_ids=candidate.observation_ids,
    )


def _relative_delta(
    signed_delta: float,
    baseline: float,
    relative_floor: float,
) -> float:
    return signed_delta / max(abs(baseline), relative_floor)


def _is_material(
    *,
    absolute_delta: float,
    relative_delta: float,
    threshold: MaterialityThreshold,
    confidence: ConfidenceAssessment,
) -> bool:
    if confidence.score < threshold.minimum_confidence:
        return False
    tests: list[bool] = []
    if threshold.absolute is not None:
        tests.append(absolute_delta >= threshold.absolute)
    if threshold.relative is not None:
        tests.append(abs(relative_delta) >= threshold.relative)
    if threshold.mode is ThresholdMode.ALL:
        return all(tests)
    return any(tests)


def _materiality_score(
    *,
    absolute_delta: float,
    relative_delta: float,
    threshold: MaterialityThreshold,
) -> float:
    scores: list[float] = []
    if threshold.absolute is not None:
        scores.append(absolute_delta / threshold.absolute)
    if threshold.relative is not None:
        scores.append(abs(relative_delta) / threshold.relative)
    if threshold.mode is ThresholdMode.ALL:
        return min(scores)
    return max(scores)


def _event(
    *,
    kind: DriftKind,
    subject_id: str,
    baseline_value: ImmutableJson,
    observed_value: ImmutableJson,
    unit: ObservationUnit,
    absolute_delta: float,
    signed_delta: float,
    relative_delta: float | None,
    material: bool,
    threshold: MaterialityThreshold,
    confidence: ConfidenceAssessment,
    observation_ids: tuple[str, ...],
    freshness: FreshnessAssessment | None = None,
    details: Mapping[str, object] | None = None,
) -> DriftEvent:
    canonical_observation_ids = tuple(sorted(observation_ids))
    material_value: dict[str, object] = {
        "kind": kind.value,
        "subject_id": subject_id,
        "baseline_value": baseline_value,
        "observed_value": observed_value,
        "unit": unit.value,
        "absolute_delta": absolute_delta,
        "signed_delta": signed_delta,
        "relative_delta": relative_delta,
        "material": material,
        "threshold": threshold.model_dump(mode="python"),
        "confidence": confidence.model_dump(mode="python"),
        "observation_ids": canonical_observation_ids,
        "freshness": (freshness.model_dump(mode="python") if freshness is not None else None),
        "interpretation": "observed_association_not_causal",
        "details": details or {},
    }
    return DriftEvent.model_validate(
        {
            **material_value,
            "content_sha256": canonical_content_sha256(material_value),
        }
    )


def _require_unit(
    candidate: ScalarDriftInput,
    expected: ObservationUnit,
    detector: str,
) -> None:
    if candidate.unit is not expected:
        raise ValueError(f"{detector} drift requires unit {expected.value}")


def _validate_observation_ids(observation_ids: tuple[str, ...]) -> None:
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("drift observation IDs must be unique")
    if any(
        not observation_id or len(observation_id) > MAX_IDENTIFIER_LENGTH
        for observation_id in observation_ids
    ):
        raise ValueError("drift observation IDs are invalid")
