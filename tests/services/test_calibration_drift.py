from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import ValidationError
import pytest

from ynab_agent.services.calibration_drift import (
    AllocationDriftInput,
    DebtPayoffInput,
    ScalarDriftInput,
    StalenessInput,
    assess_staleness,
    detect_allocation_drift,
    detect_balance_drift,
    detect_contribution_drift,
    detect_debt_payoff,
    detect_spending_drift,
    detect_staleness,
)
from ynab_agent.services.calibration_models import (
    ConfidenceAssessment,
    DriftKind,
    FreshnessState,
    MaterialityThreshold,
    ObservationUnit,
    ThresholdMode,
)


NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
CONFIDENT = ConfidenceAssessment(
    score=0.95,
    basis="complete reviewed observation window",
)


def _scalar(
    *,
    baseline: float,
    observed: float,
    threshold: MaterialityThreshold | None = None,
    unit: ObservationUnit = ObservationUnit.DOLLARS_PER_YEAR,
) -> ScalarDriftInput:
    return ScalarDriftInput(
        subject_id="portfolio",
        baseline_value=baseline,
        observed_value=observed,
        unit=unit,
        threshold=threshold or MaterialityThreshold(absolute=1_000, relative=0.05),
        confidence=CONFIDENT,
        observation_ids=("observation-1",),
    )


def test_scalar_detectors_preserve_direction_materiality_and_kind() -> None:
    spending = detect_spending_drift(_scalar(baseline=60_000, observed=66_000))
    contribution = detect_contribution_drift(_scalar(baseline=7_500, observed=6_000))
    balance = detect_balance_drift(
        _scalar(
            baseline=100_000,
            observed=110_000,
            unit=ObservationUnit.DOLLARS,
        )
    )

    assert spending.kind is DriftKind.SPENDING
    assert spending.signed_delta == 6_000
    assert spending.relative_delta == pytest.approx(0.10)
    assert spending.material is True
    assert contribution.kind is DriftKind.CONTRIBUTION
    assert contribution.signed_delta == -1_500
    assert contribution.material is True
    assert balance.kind is DriftKind.BALANCE
    assert balance.unit is ObservationUnit.DOLLARS
    assert balance.interpretation == "observed_association_not_causal"
    assert len(balance.content_sha256) == 64


def test_materiality_all_mode_and_confidence_gate_are_deterministic() -> None:
    all_mode = MaterialityThreshold(
        absolute=10_000,
        relative=0.05,
        mode=ThresholdMode.ALL,
    )
    event = detect_spending_drift(
        _scalar(
            baseline=100_000,
            observed=106_000,
            threshold=all_mode,
        )
    )
    assert event.absolute_delta == 6_000
    assert event.relative_delta == pytest.approx(0.06)
    assert event.material is False

    low_confidence = _scalar(
        baseline=100_000,
        observed=120_000,
        threshold=MaterialityThreshold(
            absolute=1,
            minimum_confidence=0.99,
        ),
    ).model_copy(
        update={
            "confidence": ConfidenceAssessment(
                score=0.5,
                basis="partial month",
                limitations=("incomplete imports",),
            )
        }
    )
    assert detect_contribution_drift(low_confidence).material is False


def test_debt_payoff_requires_a_tolerance_crossing_and_material_balance() -> None:
    paid_off = detect_debt_payoff(
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=15_000,
            current_liability=0.50,
            payoff_tolerance=1,
            threshold=MaterialityThreshold(absolute=5_000),
            confidence=CONFIDENT,
            observation_ids=("mortgage-before", "mortgage-after"),
        )
    )
    nearly_paid = detect_debt_payoff(
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=15_000,
            current_liability=2,
            payoff_tolerance=1,
            threshold=MaterialityThreshold(absolute=5_000),
            confidence=CONFIDENT,
            observation_ids=("mortgage-before", "mortgage-after"),
        )
    )

    assert paid_off.kind is DriftKind.DEBT_PAYOFF
    assert paid_off.material is True
    assert paid_off.signed_delta == pytest.approx(-14_999.50)
    assert paid_off.details["paid_off"] is True
    assert nearly_paid.material is False
    assert nearly_paid.details["paid_off"] is False


def test_allocation_drift_uses_max_weight_and_reports_total_variation() -> None:
    event = detect_allocation_drift(
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights={
                "us_equity": 0.70,
                "international_equity": 0.20,
                "bonds": 0.10,
            },
            observed_weights={
                "us_equity": 0.60,
                "international_equity": 0.25,
                "bonds": 0.15,
            },
            threshold=MaterialityThreshold(
                absolute=0.05,
                relative_floor=0.01,
            ),
            confidence=CONFIDENT,
            observation_ids=("allocation-1",),
        )
    )

    assert event.kind is DriftKind.ALLOCATION
    assert event.absolute_delta == pytest.approx(0.10)
    assert event.material is True
    assert event.details["largest_drift_class"] == "us_equity"
    assert event.details["total_variation_distance"] == pytest.approx(0.10)


def test_allocation_requires_matching_normalized_classes() -> None:
    with pytest.raises(ValidationError, match="classes must match"):
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights={"stocks": 1},
            observed_weights={"cash": 1},
            threshold=MaterialityThreshold(absolute=0.05),
            confidence=CONFIDENT,
            observation_ids=("allocation-1",),
        )
    with pytest.raises(ValidationError, match="sum to one"):
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights={"stocks": 1},
            observed_weights={"stocks": 0.90},
            threshold=MaterialityThreshold(absolute=0.05),
            confidence=CONFIDENT,
            observation_ids=("allocation-1",),
        )


def test_allocation_checks_relative_materiality_for_every_class() -> None:
    event = detect_allocation_drift(
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights={"a_large": 0.99, "z_small": 0.01},
            observed_weights={"a_large": 0.98, "z_small": 0.02},
            threshold=MaterialityThreshold(
                absolute=1,
                relative=0.50,
                relative_floor=0.001,
            ),
            confidence=CONFIDENT,
            observation_ids=("allocation-1",),
        )
    )

    assert event.material is True
    assert event.relative_delta == pytest.approx(1)
    assert event.details["reported_class"] == "z_small"
    assert event.details["largest_drift_class"] == "a_large"
    assert event.details["largest_relative_drift_class"] == "z_small"


def test_allocation_all_mode_requires_one_class_to_pass_both_tests() -> None:
    event = detect_allocation_drift(
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights={"large": 0.90, "middle": 0.09, "small": 0.01},
            observed_weights={"large": 0.85, "middle": 0.13, "small": 0.02},
            threshold=MaterialityThreshold(
                absolute=0.045,
                relative=0.50,
                relative_floor=0.001,
                mode=ThresholdMode.ALL,
            ),
            confidence=CONFIDENT,
            observation_ids=("allocation-1",),
        )
    )

    assert event.absolute_delta == pytest.approx(0.05)
    assert event.relative_delta == pytest.approx(-0.05 / 0.90)
    assert event.details["reported_class"] == "large"
    assert event.material is False


def test_scalar_detectors_enforce_domain_units() -> None:
    wrong_unit = _scalar(
        baseline=100,
        observed=200,
        unit=ObservationUnit.DAYS,
    )

    with pytest.raises(ValueError, match="spending drift requires"):
        detect_spending_drift(wrong_unit)
    with pytest.raises(ValueError, match="contribution drift requires"):
        detect_contribution_drift(wrong_unit)
    with pytest.raises(ValueError, match="balance drift requires"):
        detect_balance_drift(wrong_unit)


def test_relative_drift_uses_floor_for_zero_and_preserves_sign() -> None:
    positive = detect_spending_drift(
        _scalar(
            baseline=0,
            observed=5,
            threshold=MaterialityThreshold(
                relative=0.5,
                relative_floor=10,
            ),
        )
    )
    negative = detect_contribution_drift(
        _scalar(
            baseline=0,
            observed=-5,
            threshold=MaterialityThreshold(
                relative=0.5,
                relative_floor=10,
            ),
        )
    )

    assert positive.relative_delta == 0.5
    assert positive.material is True
    assert negative.relative_delta == -0.5
    assert negative.material is True


def _staleness(**updates: object) -> StalenessInput:
    values: dict[str, object] = {
        "subject_id": "tracking-account",
        "evaluated_at": NOW,
        "latest_activity_at": NOW - timedelta(days=10),
        "last_reconciled_at": NOW - timedelta(days=10),
        "unchanged_since": NOW - timedelta(days=20),
        "stale_after_days": 45,
        "frozen_after_days": 90,
        "threshold": MaterialityThreshold(absolute=45),
        "confidence": CONFIDENT,
        "observation_ids": ("freshness-1",),
    }
    values.update(updates)
    return StalenessInput.model_validate(values)


def test_staleness_classifies_fresh_stale_and_frozen_evidence() -> None:
    fresh = assess_staleness(_staleness())
    stale = assess_staleness(
        _staleness(
            latest_activity_at=NOW - timedelta(days=70),
            last_reconciled_at=NOW - timedelta(days=70),
            unchanged_since=NOW - timedelta(days=70),
        )
    )
    frozen = assess_staleness(
        _staleness(
            latest_activity_at=NOW - timedelta(days=5),
            last_reconciled_at=NOW - timedelta(days=120),
            unchanged_since=NOW - timedelta(days=100),
        )
    )
    reviewed_unchanged = assess_staleness(
        _staleness(
            latest_activity_at=NOW - timedelta(days=5),
            last_reconciled_at=NOW - timedelta(days=120),
            latest_reviewed_at=NOW - timedelta(days=2),
            unchanged_since=NOW - timedelta(days=100),
        )
    )

    assert fresh.state is FreshnessState.FRESH
    assert stale.state is FreshnessState.STALE
    assert stale.age_days == 70
    assert frozen.state is FreshnessState.FROZEN
    assert frozen.unchanged_days == 100
    assert reviewed_unchanged.state is FreshnessState.FRESH


def test_staleness_event_is_material_only_for_stale_or_frozen_state() -> None:
    fresh = detect_staleness(_staleness())
    stale = detect_staleness(
        _staleness(
            latest_activity_at=NOW - timedelta(days=70),
            last_reconciled_at=NOW - timedelta(days=70),
            unchanged_since=NOW - timedelta(days=70),
        )
    )

    assert fresh.material is False
    assert stale.material is True
    assert stale.freshness is not None
    assert stale.freshness.state is FreshnessState.STALE
    assert stale.interpretation == "observed_association_not_causal"


def test_staleness_without_evidence_is_unknown_and_non_material() -> None:
    candidate = _staleness(
        latest_activity_at=None,
        last_reconciled_at=None,
        latest_reviewed_at=None,
        unchanged_since=None,
    )

    freshness = assess_staleness(candidate)
    event = detect_staleness(candidate)

    assert freshness.state is FreshnessState.UNKNOWN
    assert freshness.age_days is None
    assert event.material is False
    assert event.freshness is not None
    assert event.freshness.state is FreshnessState.UNKNOWN


def test_stale_materiality_uses_evidence_age_not_unchanged_window() -> None:
    event = detect_staleness(
        _staleness(
            latest_activity_at=NOW - timedelta(days=50),
            last_reconciled_at=NOW - timedelta(days=50),
            unchanged_since=NOW - timedelta(days=200),
            frozen_after_days=365,
            threshold=MaterialityThreshold(absolute=100),
        )
    )

    assert event.freshness is not None
    assert event.freshness.state is FreshnessState.STALE
    assert event.absolute_delta == 50
    assert event.material is False


def test_staleness_rejects_naive_and_future_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _staleness(evaluated_at=datetime(2026, 7, 30))
    with pytest.raises(ValidationError, match="cannot postdate"):
        _staleness(latest_activity_at=NOW + timedelta(seconds=1))
    with pytest.raises(ValidationError, match="must be at least"):
        _staleness(stale_after_days=90, frozen_after_days=45)
