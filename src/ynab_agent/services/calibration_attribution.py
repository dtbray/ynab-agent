"""Auditable model-counterfactual attribution arithmetic.

This module only accounts for differences between simulation outputs. It does
not claim that an observed financial change caused a real-world outcome.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import hashlib
import math
import unicodedata
from typing import Final, Literal

from pydantic import Field, model_validator

from ynab_agent.services.calibration_models import (
    CalibrationSnapshotManifest,
    ConfidenceAssessment,
    FrozenCalibrationModel,
    MAX_DRIFT_EVENTS_PER_SNAPSHOT,
    MAX_IDENTIFIER_LENGTH,
    SHA256_PATTERN,
    canonical_content_sha256,
)


MAX_ATTRIBUTION_FACTORS = 16
MAX_ATTRIBUTION_PERMUTATIONS = 64
MAX_ATTRIBUTION_DRIFT_HASHES_PER_FACTOR = MAX_DRIFT_EVENTS_PER_SNAPSHOT
MAX_ATTRIBUTION_LINKED_DRIFT_HASHES = (
    MAX_ATTRIBUTION_FACTORS * MAX_ATTRIBUTION_DRIFT_HASHES_PER_FACTOR
)
MAX_ATTRIBUTION_REAL_AMOUNT = 1_000_000_000_000_000
ATTRIBUTION_REPLAY_ALGORITHM: Final = "sha256_fisher_yates_v1"
_PROBABILITY_ABSOLUTE_TOLERANCE = 1e-12
_MONETARY_ABSOLUTE_TOLERANCE = 0.01


class AttributionMethod(StrEnum):
    """Bounded counterfactual methods supported by later simulation workers."""

    ONE_FACTOR_COUNTERFACTUAL = "one_factor_counterfactual"
    ORDERED_COUNTERFACTUAL_WATERFALL = "ordered_counterfactual_waterfall"
    PERMUTATION_AVERAGED_COUNTERFACTUAL = "permutation_averaged_counterfactual"


class OutcomeMetricValues(FrozenCalibrationModel):
    """Outcome levels used by continuous-calibration explanations."""

    success_probability: float = Field(ge=0, le=1)
    cumulative_shortfall_real_p50: float = Field(
        ge=0,
        le=MAX_ATTRIBUTION_REAL_AMOUNT,
    )
    lifetime_tax_real_p50: float | None = Field(
        default=None,
        ge=0,
        le=MAX_ATTRIBUTION_REAL_AMOUNT,
    )
    after_tax_estate_value_real_p50: float | None = Field(
        default=None,
        ge=0,
        le=MAX_ATTRIBUTION_REAL_AMOUNT,
    )


class OutcomeMetricDelta(FrozenCalibrationModel):
    """Signed current-minus-baseline changes in stable outcome metrics."""

    success_probability: float = Field(ge=-1, le=1)
    cumulative_shortfall_real_p50: float = Field(
        ge=-MAX_ATTRIBUTION_REAL_AMOUNT,
        le=MAX_ATTRIBUTION_REAL_AMOUNT,
    )
    lifetime_tax_real_p50: float | None = Field(
        default=None,
        ge=-MAX_ATTRIBUTION_REAL_AMOUNT,
        le=MAX_ATTRIBUTION_REAL_AMOUNT,
    )
    after_tax_estate_value_real_p50: float | None = Field(
        default=None,
        ge=-MAX_ATTRIBUTION_REAL_AMOUNT,
        le=MAX_ATTRIBUTION_REAL_AMOUNT,
    )


class AttributionMethodManifest(FrozenCalibrationModel):
    """Exact method and identities needed to audit an attribution."""

    schema_version: int = Field(default=1, ge=1)
    method: AttributionMethod
    factor_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_ATTRIBUTION_FACTORS,
    )
    permutation_count: int = Field(
        default=0,
        ge=0,
        le=MAX_ATTRIBUTION_PERMUTATIONS,
    )
    seed: int = Field(default=0, ge=0, le=2**64 - 1)
    replay_algorithm: Literal["sha256_fisher_yates_v1"] = ATTRIBUTION_REPLAY_ALGORITHM
    permutation_fingerprint_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    calibration_snapshot_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    drift_event_sha256: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_DRIFT_EVENTS_PER_SNAPSHOT,
    )
    baseline_revision_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    current_revision_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    comparison_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    counterfactual_definition: str = Field(min_length=1, max_length=1_000)
    claim_scope: Literal["model_counterfactual_not_real_world_causation"] = (
        "model_counterfactual_not_real_world_causation"
    )

    @model_validator(mode="after")
    def validate_method(self) -> AttributionMethodManifest:
        if len(set(self.factor_ids)) != len(self.factor_ids):
            raise ValueError("attribution factor IDs must be unique")
        if any(
            not factor_id or len(factor_id) > MAX_IDENTIFIER_LENGTH for factor_id in self.factor_ids
        ):
            raise ValueError("attribution factor IDs are invalid")
        _validate_hashes(
            self.drift_event_sha256,
            "attribution context drift event hashes",
        )
        object.__setattr__(
            self,
            "drift_event_sha256",
            tuple(sorted(self.drift_event_sha256)),
        )
        if (self.method is AttributionMethod.PERMUTATION_AVERAGED_COUNTERFACTUAL) != (
            self.permutation_count > 0
        ):
            raise ValueError("permutation count is required only for permutation attribution")
        if self.method is AttributionMethod.PERMUTATION_AVERAGED_COUNTERFACTUAL:
            expected = permutation_fingerprint_sha256(
                factor_ids=self.factor_ids,
                permutation_count=self.permutation_count,
                seed=self.seed,
            )
            if self.permutation_fingerprint_sha256 != expected:
                raise ValueError("permutation fingerprint does not match replay contract")
        elif self.permutation_fingerprint_sha256 is not None:
            raise ValueError("permutation fingerprint is only valid for permutation attribution")
        elif self.seed != 0:
            raise ValueError("non-permutation attribution must use the default seed")
        return self


class ModelAttributionContribution(FrozenCalibrationModel):
    """One factor's simulation-derived marginal metric contribution."""

    factor_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    label: str = Field(min_length=1, max_length=500)
    drift_event_sha256: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_ATTRIBUTION_DRIFT_HASHES_PER_FACTOR,
    )
    marginal_delta: OutcomeMetricDelta
    evaluation_count: int = Field(gt=0, le=MAX_ATTRIBUTION_PERMUTATIONS)
    confidence: ConfidenceAssessment
    claim_scope: Literal["model_counterfactual_not_real_world_causation"] = (
        "model_counterfactual_not_real_world_causation"
    )

    @model_validator(mode="after")
    def validate_drift_hashes(self) -> ModelAttributionContribution:
        _validate_hashes(
            self.drift_event_sha256,
            "attribution drift event hashes",
        )
        object.__setattr__(
            self,
            "drift_event_sha256",
            tuple(sorted(self.drift_event_sha256)),
        )
        return self


class AttributionReport(FrozenCalibrationModel):
    """Content-addressed outcome delta decomposition with explicit residual."""

    baseline: OutcomeMetricValues
    current: OutcomeMetricValues
    total_delta: OutcomeMetricDelta
    contributions: tuple[ModelAttributionContribution, ...] = Field(
        min_length=1,
        max_length=MAX_ATTRIBUTION_FACTORS,
    )
    explained_delta: OutcomeMetricDelta
    interaction_residual: OutcomeMetricDelta
    method: AttributionMethodManifest
    calibration_snapshot_manifest: CalibrationSnapshotManifest
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    claim_scope: Literal["model_counterfactual_not_real_world_causation"] = (
        "model_counterfactual_not_real_world_causation"
    )

    @model_validator(mode="before")
    @classmethod
    def admit_bounded_link_graph(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        contributions = value.get("contributions")
        if not isinstance(contributions, list | tuple):
            return value
        linked_hash_count = 0
        for contribution in contributions:
            raw_hashes = (
                contribution.drift_event_sha256
                if isinstance(contribution, ModelAttributionContribution)
                else (
                    contribution.get("drift_event_sha256")
                    if isinstance(contribution, Mapping)
                    else None
                )
            )
            if isinstance(raw_hashes, list | tuple):
                linked_hash_count += len(raw_hashes)
                if linked_hash_count > MAX_ATTRIBUTION_LINKED_DRIFT_HASHES:
                    raise ValueError("attribution linked drift hashes exceed the resource limit")
        return value

    @model_validator(mode="after")
    def verify_accounting_and_hash(self) -> AttributionReport:
        snapshot_hash = canonical_content_sha256(self.calibration_snapshot_manifest)
        if self.method.calibration_snapshot_manifest_sha256 != snapshot_hash:
            raise ValueError("attribution method must bind the exact calibration snapshot manifest")
        snapshot_drift_hashes = tuple(
            event.content_sha256 for event in self.calibration_snapshot_manifest.drift_events
        )
        if self.method.drift_event_sha256 != tuple(sorted(snapshot_drift_hashes)):
            raise ValueError(
                "attribution method drift context must match exact snapshot drift events"
            )
        if tuple(row.factor_id for row in self.contributions) != self.method.factor_ids:
            raise ValueError("attribution contributions must match manifest factor order")
        expected_evaluation_count = (
            self.method.permutation_count
            if self.method.method is AttributionMethod.PERMUTATION_AVERAGED_COUNTERFACTUAL
            else 1
        )
        if any(row.evaluation_count != expected_evaluation_count for row in self.contributions):
            raise ValueError("attribution evaluation counts must match the method manifest")
        context_hashes = set(self.method.drift_event_sha256)
        linked_hashes = {
            drift_hash
            for contribution in self.contributions
            for drift_hash in contribution.drift_event_sha256
        }
        if not linked_hashes.issubset(context_hashes):
            raise ValueError("attribution contribution drift links are outside method context")
        if linked_hashes != context_hashes:
            raise ValueError("attribution method context contains unlinked drift events")
        expected_total = metric_delta(self.baseline, self.current)
        expected_explained = sum_metric_deltas(
            tuple(row.marginal_delta for row in self.contributions)
        )
        expected_residual = subtract_metric_deltas(
            expected_total,
            expected_explained,
        )
        _require_close("total", self.total_delta, expected_total)
        _require_close("explained", self.explained_delta, expected_explained)
        _require_close(
            "interaction residual",
            self.interaction_residual,
            expected_residual,
        )
        expected_hash = canonical_content_sha256(
            self,
            exclude=frozenset({"content_sha256"}),
        )
        if self.content_sha256 != expected_hash:
            raise ValueError("attribution report content hash mismatch")
        return self


def metric_delta(
    baseline: OutcomeMetricValues,
    current: OutcomeMetricValues,
) -> OutcomeMetricDelta:
    """Calculate current-minus-baseline outcome movement."""
    return OutcomeMetricDelta(
        success_probability=(current.success_probability - baseline.success_probability),
        cumulative_shortfall_real_p50=(
            current.cumulative_shortfall_real_p50 - baseline.cumulative_shortfall_real_p50
        ),
        lifetime_tax_real_p50=_optional_difference(
            baseline.lifetime_tax_real_p50,
            current.lifetime_tax_real_p50,
            "lifetime tax",
        ),
        after_tax_estate_value_real_p50=_optional_difference(
            baseline.after_tax_estate_value_real_p50,
            current.after_tax_estate_value_real_p50,
            "after-tax estate",
        ),
    )


def sum_metric_deltas(
    deltas: tuple[OutcomeMetricDelta, ...],
) -> OutcomeMetricDelta:
    """Sum factor marginals while preserving unavailable-metric semantics."""
    if not deltas:
        return OutcomeMetricDelta(
            success_probability=0,
            cumulative_shortfall_real_p50=0,
        )
    return OutcomeMetricDelta(
        success_probability=sum(row.success_probability for row in deltas),
        cumulative_shortfall_real_p50=sum(row.cumulative_shortfall_real_p50 for row in deltas),
        lifetime_tax_real_p50=_sum_optional(
            tuple(row.lifetime_tax_real_p50 for row in deltas),
            "lifetime tax",
        ),
        after_tax_estate_value_real_p50=_sum_optional(
            tuple(row.after_tax_estate_value_real_p50 for row in deltas),
            "after-tax estate",
        ),
    )


def subtract_metric_deltas(
    left: OutcomeMetricDelta,
    right: OutcomeMetricDelta,
) -> OutcomeMetricDelta:
    """Subtract two compatible metric vectors."""
    return OutcomeMetricDelta(
        success_probability=left.success_probability - right.success_probability,
        cumulative_shortfall_real_p50=(
            left.cumulative_shortfall_real_p50 - right.cumulative_shortfall_real_p50
        ),
        lifetime_tax_real_p50=_optional_difference(
            right.lifetime_tax_real_p50,
            left.lifetime_tax_real_p50,
            "lifetime tax",
        ),
        after_tax_estate_value_real_p50=_optional_difference(
            right.after_tax_estate_value_real_p50,
            left.after_tax_estate_value_real_p50,
            "after-tax estate",
        ),
    )


def build_attribution_report(
    *,
    baseline: OutcomeMetricValues,
    current: OutcomeMetricValues,
    contributions: tuple[ModelAttributionContribution, ...],
    method: AttributionMethodManifest,
    calibration_snapshot_manifest: CalibrationSnapshotManifest,
) -> AttributionReport:
    """Build and hash an auditable decomposition, retaining interaction residual."""
    if tuple(row.factor_id for row in contributions) != method.factor_ids:
        raise ValueError("attribution contributions must match manifest factor order")
    total = metric_delta(baseline, current)
    explained = sum_metric_deltas(tuple(row.marginal_delta for row in contributions))
    residual = subtract_metric_deltas(total, explained)
    material = {
        "baseline": baseline.model_dump(mode="python"),
        "current": current.model_dump(mode="python"),
        "total_delta": total.model_dump(mode="python"),
        "contributions": [contribution.model_dump(mode="python") for contribution in contributions],
        "explained_delta": explained.model_dump(mode="python"),
        "interaction_residual": residual.model_dump(mode="python"),
        "method": method.model_dump(mode="python"),
        "calibration_snapshot_manifest": calibration_snapshot_manifest.model_dump(mode="python"),
        "claim_scope": "model_counterfactual_not_real_world_causation",
    }
    return AttributionReport.model_validate(
        {
            **material,
            "content_sha256": canonical_content_sha256(material),
        }
    )


def deterministic_attribution_permutations(
    *,
    factor_ids: tuple[str, ...],
    permutation_count: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Generate the versioned permutation sequence committed by a manifest."""
    normalized_factor_ids = tuple(
        unicodedata.normalize("NFC", factor_id) for factor_id in factor_ids
    )
    if not normalized_factor_ids or len(normalized_factor_ids) > MAX_ATTRIBUTION_FACTORS:
        raise ValueError("attribution factor count is invalid")
    if any(
        not factor_id or len(factor_id) > MAX_IDENTIFIER_LENGTH
        for factor_id in normalized_factor_ids
    ):
        raise ValueError("attribution factor IDs are invalid")
    if len(set(normalized_factor_ids)) != len(normalized_factor_ids):
        raise ValueError("attribution factor IDs must be unique")
    if not 1 <= permutation_count <= MAX_ATTRIBUTION_PERMUTATIONS:
        raise ValueError("attribution permutation count is invalid")
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("attribution seed is invalid")
    permutations: list[tuple[str, ...]] = []
    for permutation_index in range(permutation_count):
        values = list(normalized_factor_ids)
        for upper_index in range(len(values) - 1, 0, -1):
            digest = hashlib.sha256(
                (
                    f"{ATTRIBUTION_REPLAY_ALGORITHM}\0{seed}\0{permutation_index}\0{upper_index}"
                ).encode("utf-8")
            ).digest()
            swap_index = int.from_bytes(digest[:8], "big") % (upper_index + 1)
            values[upper_index], values[swap_index] = (
                values[swap_index],
                values[upper_index],
            )
        permutations.append(tuple(values))
    return tuple(permutations)


def permutation_fingerprint_sha256(
    *,
    factor_ids: tuple[str, ...],
    permutation_count: int,
    seed: int,
) -> str:
    """Fingerprint the exact deterministic sequence used for attribution."""
    permutations = deterministic_attribution_permutations(
        factor_ids=factor_ids,
        permutation_count=permutation_count,
        seed=seed,
    )
    return canonical_content_sha256(
        {
            "algorithm": ATTRIBUTION_REPLAY_ALGORITHM,
            "factor_ids": tuple(
                unicodedata.normalize("NFC", factor_id) for factor_id in factor_ids
            ),
            "permutation_count": permutation_count,
            "seed": seed,
            "permutations": permutations,
        }
    )


def _optional_difference(
    baseline: float | None,
    current: float | None,
    metric: str,
) -> float | None:
    if baseline is None and current is None:
        return None
    if baseline is None or current is None:
        raise ValueError(f"{metric} availability must match")
    return current - baseline


def _sum_optional(
    values: tuple[float | None, ...],
    metric: str,
) -> float | None:
    available = tuple(value for value in values if value is not None)
    if not available:
        return None
    if len(available) != len(values):
        raise ValueError(f"{metric} attribution availability must match")
    return sum(available)


def _require_close(
    label: str,
    actual: OutcomeMetricDelta,
    expected: OutcomeMetricDelta,
) -> None:
    for field_name in (
        "success_probability",
        "cumulative_shortfall_real_p50",
        "lifetime_tax_real_p50",
        "after_tax_estate_value_real_p50",
    ):
        actual_value = getattr(actual, field_name)
        expected_value = getattr(expected, field_name)
        if actual_value is None and expected_value is None:
            continue
        tolerance = (
            _PROBABILITY_ABSOLUTE_TOLERANCE
            if field_name == "success_probability"
            else _MONETARY_ABSOLUTE_TOLERANCE
        )
        if (
            actual_value is None
            or expected_value is None
            or not math.isclose(
                actual_value,
                expected_value,
                rel_tol=0,
                abs_tol=tolerance,
            )
        ):
            raise ValueError(f"attribution {label} does not account for {field_name}")


def _validate_hashes(values: tuple[str, ...], label: str) -> None:
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError(f"{label} must be lowercase SHA-256 values")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
