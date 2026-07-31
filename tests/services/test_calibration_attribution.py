from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

from pydantic import ValidationError
import pytest

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.calibration_attribution import (
    AttributionMethod,
    AttributionMethodManifest,
    AttributionReport,
    MAX_ATTRIBUTION_DRIFT_HASHES_PER_FACTOR,
    MAX_ATTRIBUTION_FACTORS,
    MAX_ATTRIBUTION_LINKED_DRIFT_HASHES,
    ModelAttributionContribution,
    OutcomeMetricDelta,
    OutcomeMetricValues,
    build_attribution_report,
    deterministic_attribution_permutations,
    metric_delta,
    permutation_fingerprint_sha256,
    subtract_metric_deltas,
    sum_metric_deltas,
)
from ynab_agent.services.calibration_models import (
    CalibrationObservation,
    CalibrationPolicy,
    CalibrationSnapshotManifest,
    ConfidenceAssessment,
    FreshnessAssessment,
    FreshnessState,
    FrozenJsonObject,
    FrozenWealthScenario,
    ImmutableJson,
    MaterialityThreshold,
    ObservationKind,
    ObservationPayload,
    ObservationSourceKind,
    ObservationUnit,
    SourceObservationLink,
    SourceWatermark,
    canonical_content_sha256,
)
from ynab_agent.services.calibration_drift import (
    ScalarDriftInput,
    detect_balance_drift,
    detect_spending_drift,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
CONFIDENCE = ConfidenceAssessment(
    score=0.80,
    basis="paired-path model counterfactual",
    limitations=("observational input change",),
)
NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
THRESHOLD = MaterialityThreshold(absolute=1)
POLICY = CalibrationPolicy(
    spending=THRESHOLD,
    contribution=THRESHOLD,
    debt_payoff=THRESHOLD,
    balance=THRESHOLD,
    allocation=THRESHOLD,
    staleness=THRESHOLD,
)
WATERMARK = SourceWatermark(
    budget_id="budget-1",
    resource="accounts",
    server_knowledge=42,
    synced_at=NOW,
    change_batch_id="batch-1",
)


def _observation(
    *,
    id: str,
    kind: ObservationKind,
    subject_id: str,
    value: object,
    unit: ObservationUnit,
    role: str,
    previous: bool,
) -> CalibrationObservation:
    plan_field = {
        ObservationKind.SPENDING: "annual_spending",
        ObservationKind.CONTRIBUTION: "annual_contribution",
        ObservationKind.ACCOUNT_BALANCE: "starting_portfolio",
    }.get(kind)
    freshness = None
    if kind is ObservationKind.ACCOUNT_BALANCE:
        reconciled_at = NOW - timedelta(days=2)
        freshness = FreshnessAssessment(
            state=FreshnessState.FRESH,
            evaluated_at=NOW,
            last_reconciled_at=reconciled_at,
            latest_evidence_at=reconciled_at,
            age_days=2,
            stale_after_days=45,
            frozen_after_days=90,
            evidence=("last_reconciled",),
        )
    return CalibrationObservation.create(
        id=id,
        kind=kind,
        subject_id=subject_id,
        effective_date=(NOW - timedelta(days=1) if previous else NOW).date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=unit,
            value=cast(ImmutableJson, value),
            fields=FrozenJsonObject(
                {
                    "drift_role": role,
                    **({"drift_aggregation": "exact"} if kind is ObservationKind.SPENDING else {}),
                    **({"plan_field": plan_field} if previous and plan_field is not None else {}),
                }
            ),
        ),
        freshness=freshness,
        source_links=(
            SourceObservationLink(
                source_kind=ObservationSourceKind.YNAB_SYNC,
                source_id=id,
                source_uri=f"ynab://budget/budget-1/evidence/{id}",
                content_sha256=canonical_content_sha256({"source": id}),
                observed_at=NOW,
                change_batch_id="batch-1",
            ),
        ),
    )


def _snapshot(*, include_spending: bool = True) -> CalibrationSnapshotManifest:
    balance_baseline = _observation(
        id="balance-baseline",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="portfolio",
        value=100_000,
        unit=ObservationUnit.DOLLARS,
        role="baseline",
        previous=True,
    )
    balance_current = _observation(
        id="balance-current",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="portfolio",
        value=110_000,
        unit=ObservationUnit.DOLLARS,
        role="current",
        previous=False,
    )
    balance_event = detect_balance_drift(
        ScalarDriftInput(
            subject_id="portfolio",
            baseline_value=100_000,
            observed_value=110_000,
            unit=ObservationUnit.DOLLARS,
            threshold=THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(balance_baseline.id, balance_current.id),
        )
    )
    observations = [balance_baseline, balance_current]
    events = [balance_event]
    if include_spending:
        spending_baseline = _observation(
            id="spending-baseline",
            kind=ObservationKind.SPENDING,
            subject_id="household",
            value=50_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            role="baseline",
            previous=True,
        )
        spending_current = _observation(
            id="spending-current",
            kind=ObservationKind.SPENDING,
            subject_id="household",
            value=55_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            role="observed",
            previous=False,
        )
        observations.extend((spending_baseline, spending_current))
        events.append(
            detect_spending_drift(
                ScalarDriftInput(
                    subject_id="household",
                    baseline_value=50_000,
                    observed_value=55_000,
                    unit=ObservationUnit.DOLLARS_PER_YEAR,
                    threshold=THRESHOLD,
                    confidence=CONFIDENCE,
                    observation_ids=(spending_baseline.id, spending_current.id),
                )
            )
        )
    scenario = WealthScenario(
        name="attribution evidence",
        current_age=30,
        retirement_age=67,
        end_age=95,
        starting_portfolio=100_000,
        annual_contribution=7_500,
        annual_spending=50_000,
        trials=100,
        seed=97,
    )
    return CalibrationSnapshotManifest(
        profile_id="profile-1",
        scenario_revision_id="00000000-0000-0000-0000-000000000097",
        scenario_manifest_sha256=HASH_A,
        source_sync_batch_id="batch-1",
        as_of=NOW,
        resolved_scenario=FrozenWealthScenario(scenario.model_dump(mode="python")),
        resolved_scenario_sha256=canonical_content_sha256(scenario),
        policy=POLICY,
        observations=tuple(observations),
        drift_events=tuple(events),
        source_watermarks=(WATERMARK,),
    )


SNAPSHOT = _snapshot()
BALANCE_SNAPSHOT = _snapshot(include_spending=False)


def _drift_hash(factor_id: str) -> str:
    kind = {"balance": "balance", "spending": "spending"}.get(factor_id)
    if kind is None:
        return canonical_content_sha256({"factor_id": factor_id})
    return next(event.content_sha256 for event in SNAPSHOT.drift_events if event.kind.value == kind)


def _method(
    *,
    snapshot: CalibrationSnapshotManifest = SNAPSHOT,
    factor_ids: tuple[str, ...] = ("balance", "spending"),
) -> AttributionMethodManifest:
    return AttributionMethodManifest(
        method=AttributionMethod.ONE_FACTOR_COUNTERFACTUAL,
        factor_ids=factor_ids,
        baseline_revision_manifest_sha256=HASH_A,
        current_revision_manifest_sha256=HASH_B,
        comparison_manifest_sha256=HASH_C,
        calibration_snapshot_manifest_sha256=canonical_content_sha256(snapshot),
        drift_event_sha256=tuple(event.content_sha256 for event in snapshot.drift_events),
        counterfactual_definition=(
            "Each factor is evaluated against the saved baseline on identical paths."
        ),
    )


def _contribution(
    factor_id: str,
    *,
    success: float,
    shortfall: float,
    tax: float,
    estate: float,
) -> ModelAttributionContribution:
    return ModelAttributionContribution(
        factor_id=factor_id,
        label=factor_id.title(),
        drift_event_sha256=(_drift_hash(factor_id),),
        marginal_delta=OutcomeMetricDelta(
            success_probability=success,
            cumulative_shortfall_real_p50=shortfall,
            lifetime_tax_real_p50=tax,
            after_tax_estate_value_real_p50=estate,
        ),
        evaluation_count=1,
        confidence=CONFIDENCE,
    )


def test_metric_arithmetic_uses_current_minus_baseline() -> None:
    baseline = OutcomeMetricValues(
        success_probability=0.50,
        cumulative_shortfall_real_p50=20_000,
        lifetime_tax_real_p50=100_000,
        after_tax_estate_value_real_p50=500_000,
    )
    current = OutcomeMetricValues(
        success_probability=0.60,
        cumulative_shortfall_real_p50=15_000,
        lifetime_tax_real_p50=110_000,
        after_tax_estate_value_real_p50=600_000,
    )

    delta = metric_delta(baseline, current)

    assert delta.success_probability == pytest.approx(0.10)
    assert delta.cumulative_shortfall_real_p50 == -5_000
    assert delta.lifetime_tax_real_p50 == 10_000
    assert delta.after_tax_estate_value_real_p50 == 100_000


def test_report_accounts_for_explained_delta_and_interaction_residual() -> None:
    baseline = OutcomeMetricValues(
        success_probability=0.50,
        cumulative_shortfall_real_p50=20_000,
        lifetime_tax_real_p50=100_000,
        after_tax_estate_value_real_p50=500_000,
    )
    current = OutcomeMetricValues(
        success_probability=0.60,
        cumulative_shortfall_real_p50=15_000,
        lifetime_tax_real_p50=110_000,
        after_tax_estate_value_real_p50=600_000,
    )
    contributions = (
        _contribution(
            "balance",
            success=0.08,
            shortfall=-3_000,
            tax=6_000,
            estate=80_000,
        ),
        _contribution(
            "spending",
            success=0.01,
            shortfall=-1_000,
            tax=3_000,
            estate=15_000,
        ),
    )

    report = build_attribution_report(
        baseline=baseline,
        current=current,
        contributions=contributions,
        method=_method(),
        calibration_snapshot_manifest=SNAPSHOT,
    )

    assert report.explained_delta.success_probability == pytest.approx(0.09)
    assert report.interaction_residual.success_probability == pytest.approx(0.01)
    assert report.interaction_residual.cumulative_shortfall_real_p50 == -1_000
    assert report.interaction_residual.lifetime_tax_real_p50 == 1_000
    assert report.interaction_residual.after_tax_estate_value_real_p50 == 5_000
    assert report.claim_scope == "model_counterfactual_not_real_world_causation"
    assert all(
        row.claim_scope == "model_counterfactual_not_real_world_causation"
        for row in report.contributions
    )
    assert len(report.content_sha256) == 64


def test_report_hash_and_accounting_reject_tampering() -> None:
    baseline = OutcomeMetricValues(
        success_probability=0.50,
        cumulative_shortfall_real_p50=20_000,
    )
    current = OutcomeMetricValues(
        success_probability=0.60,
        cumulative_shortfall_real_p50=15_000,
    )
    contributions = (
        _contribution(
            "balance",
            success=0.08,
            shortfall=-3_000,
            tax=0,
            estate=0,
        ).model_copy(
            update={
                "marginal_delta": OutcomeMetricDelta(
                    success_probability=0.08,
                    cumulative_shortfall_real_p50=-3_000,
                )
            }
        ),
        _contribution(
            "spending",
            success=0.02,
            shortfall=-2_000,
            tax=0,
            estate=0,
        ).model_copy(
            update={
                "marginal_delta": OutcomeMetricDelta(
                    success_probability=0.02,
                    cumulative_shortfall_real_p50=-2_000,
                )
            }
        ),
    )
    report = build_attribution_report(
        baseline=baseline,
        current=current,
        contributions=contributions,
        method=_method(),
        calibration_snapshot_manifest=SNAPSHOT,
    )

    with pytest.raises(ValidationError, match="content hash mismatch"):
        AttributionReport.model_validate(
            {
                **report.model_dump(mode="json"),
                "content_sha256": "d" * 64,
            }
        )
    with pytest.raises(ValidationError, match="does not account"):
        AttributionReport.model_validate(
            {
                **report.model_dump(mode="json"),
                "interaction_residual": {
                    "success_probability": 1,
                    "cumulative_shortfall_real_p50": 0,
                },
            }
        )

    arbitrary_context = report.method.model_copy(
        update={
            "calibration_snapshot_manifest_sha256": HASH_D,
            "drift_event_sha256": (HASH_A, HASH_B),
        }
    )
    material = {
        **report.model_dump(mode="python", exclude={"content_sha256"}),
        "method": arbitrary_context,
    }
    with pytest.raises(ValidationError, match="exact calibration snapshot manifest"):
        AttributionReport.model_validate(
            {
                **material,
                "content_sha256": canonical_content_sha256(material),
            }
        )

    wrong_snapshot_material = {
        **report.model_dump(mode="python", exclude={"content_sha256"}),
        "calibration_snapshot_manifest": BALANCE_SNAPSHOT,
    }
    with pytest.raises(ValidationError, match="exact calibration snapshot manifest"):
        AttributionReport.model_validate(
            {
                **wrong_snapshot_material,
                "content_sha256": canonical_content_sha256(wrong_snapshot_material),
            }
        )


def test_manifest_requires_deterministic_factor_order_and_permutation_contract() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        AttributionMethodManifest(
            method=AttributionMethod.ONE_FACTOR_COUNTERFACTUAL,
            factor_ids=("balance", "balance"),
            baseline_revision_manifest_sha256=HASH_A,
            current_revision_manifest_sha256=HASH_B,
            comparison_manifest_sha256=HASH_C,
            calibration_snapshot_manifest_sha256=HASH_D,
            drift_event_sha256=(HASH_A,),
            counterfactual_definition="one factor at a time",
        )
    with pytest.raises(ValidationError, match="permutation count"):
        AttributionMethodManifest(
            method=AttributionMethod.PERMUTATION_AVERAGED_COUNTERFACTUAL,
            factor_ids=("balance",),
            baseline_revision_manifest_sha256=HASH_A,
            current_revision_manifest_sha256=HASH_B,
            comparison_manifest_sha256=HASH_C,
            calibration_snapshot_manifest_sha256=HASH_D,
            drift_event_sha256=(HASH_A,),
            counterfactual_definition="bounded deterministic permutations",
        )
    with pytest.raises(ValidationError, match="must be unique"):
        AttributionMethodManifest(
            method=AttributionMethod.ONE_FACTOR_COUNTERFACTUAL,
            factor_ids=("\N{LATIN SMALL LETTER E WITH ACUTE}", "e\u0301"),
            baseline_revision_manifest_sha256=HASH_A,
            current_revision_manifest_sha256=HASH_B,
            comparison_manifest_sha256=HASH_C,
            calibration_snapshot_manifest_sha256=HASH_D,
            drift_event_sha256=(HASH_A,),
            counterfactual_definition="normalized factor identifiers",
        )


def test_permutation_manifest_fingerprints_exact_versioned_sequence() -> None:
    factor_ids = ("balance", "spending", "contribution")
    fingerprint = permutation_fingerprint_sha256(
        factor_ids=factor_ids,
        permutation_count=4,
        seed=97,
    )
    manifest = AttributionMethodManifest(
        method=AttributionMethod.PERMUTATION_AVERAGED_COUNTERFACTUAL,
        factor_ids=factor_ids,
        permutation_count=4,
        seed=97,
        permutation_fingerprint_sha256=fingerprint,
        baseline_revision_manifest_sha256=HASH_A,
        current_revision_manifest_sha256=HASH_B,
        comparison_manifest_sha256=HASH_C,
        calibration_snapshot_manifest_sha256=canonical_content_sha256(BALANCE_SNAPSHOT),
        drift_event_sha256=(HASH_A,),
        counterfactual_definition="average exact versioned permutations",
    )

    first = deterministic_attribution_permutations(
        factor_ids=factor_ids,
        permutation_count=4,
        seed=97,
    )
    second = deterministic_attribution_permutations(
        factor_ids=factor_ids,
        permutation_count=4,
        seed=97,
    )

    assert first == second
    assert len(first) == 4
    assert manifest.replay_algorithm == "sha256_fisher_yates_v1"
    with pytest.raises(ValidationError, match="fingerprint"):
        AttributionMethodManifest.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "permutation_fingerprint_sha256": "d" * 64,
            }
        )


def test_permutation_helpers_normalize_factor_ids_before_replay() -> None:
    composed = "\N{LATIN SMALL LETTER E WITH ACUTE}"
    decomposed = "e\u0301"

    assert deterministic_attribution_permutations(
        factor_ids=(decomposed, "balance"),
        permutation_count=4,
        seed=97,
    ) == deterministic_attribution_permutations(
        factor_ids=(composed, "balance"),
        permutation_count=4,
        seed=97,
    )
    assert permutation_fingerprint_sha256(
        factor_ids=(decomposed, "balance"),
        permutation_count=4,
        seed=97,
    ) == permutation_fingerprint_sha256(
        factor_ids=(composed, "balance"),
        permutation_count=4,
        seed=97,
    )
    with pytest.raises(ValueError, match="must be unique"):
        deterministic_attribution_permutations(
            factor_ids=(composed, decomposed),
            permutation_count=1,
            seed=97,
        )


def test_permutation_report_rejects_incoherent_evaluation_counts() -> None:
    factor_ids = ("balance",)
    method = AttributionMethodManifest(
        method=AttributionMethod.PERMUTATION_AVERAGED_COUNTERFACTUAL,
        factor_ids=factor_ids,
        permutation_count=4,
        seed=97,
        permutation_fingerprint_sha256=permutation_fingerprint_sha256(
            factor_ids=factor_ids,
            permutation_count=4,
            seed=97,
        ),
        baseline_revision_manifest_sha256=HASH_A,
        current_revision_manifest_sha256=HASH_B,
        comparison_manifest_sha256=HASH_C,
        calibration_snapshot_manifest_sha256=canonical_content_sha256(BALANCE_SNAPSHOT),
        drift_event_sha256=(_drift_hash("balance"),),
        counterfactual_definition="average exact versioned permutations",
    )
    contribution = _contribution(
        "balance",
        success=0,
        shortfall=0,
        tax=0,
        estate=0,
    )

    with pytest.raises(
        ValidationError,
        match="evaluation counts must match",
    ):
        build_attribution_report(
            baseline=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            current=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            contributions=(contribution,),
            method=method,
            calibration_snapshot_manifest=BALANCE_SNAPSHOT,
        )


def test_build_rejects_factor_order_mismatch() -> None:
    contributions = (
        _contribution(
            "spending",
            success=0,
            shortfall=0,
            tax=0,
            estate=0,
        ),
        _contribution(
            "balance",
            success=0,
            shortfall=0,
            tax=0,
            estate=0,
        ),
    )
    with pytest.raises(ValueError, match="factor order"):
        build_attribution_report(
            baseline=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            current=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            contributions=contributions,
            method=_method(),
            calibration_snapshot_manifest=SNAPSHOT,
        )


def test_report_requires_complete_linked_drift_context() -> None:
    balance = _contribution(
        "balance",
        success=0,
        shortfall=0,
        tax=0,
        estate=0,
    )
    with pytest.raises(ValidationError, match="at least 1"):
        ModelAttributionContribution.model_validate(
            {
                **balance.model_dump(mode="json"),
                "drift_event_sha256": (),
            }
        )

    unlinked_method = _method(snapshot=BALANCE_SNAPSHOT, factor_ids=("balance",)).model_copy(
        update={
            "factor_ids": ("balance",),
            "drift_event_sha256": (
                _drift_hash("balance"),
                _drift_hash("unlinked"),
            ),
        }
    )
    with pytest.raises(ValidationError, match="exact snapshot drift events"):
        build_attribution_report(
            baseline=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            current=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            contributions=(balance,),
            method=unlinked_method,
            calibration_snapshot_manifest=BALANCE_SNAPSHOT,
        )

    outside = balance.model_copy(update={"drift_event_sha256": (_drift_hash("outside"),)})
    method = _method(snapshot=BALANCE_SNAPSHOT, factor_ids=("balance",))
    with pytest.raises(ValidationError, match="outside method context"):
        build_attribution_report(
            baseline=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            current=OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=0,
                after_tax_estate_value_real_p50=0,
            ),
            contributions=(outside,),
            method=method,
            calibration_snapshot_manifest=BALANCE_SNAPSHOT,
        )


def test_max_snapshot_drift_context_fits_factor_and_global_link_bounds() -> None:
    hashes = tuple(
        f"{index:064x}" for index in range(1, MAX_ATTRIBUTION_DRIFT_HASHES_PER_FACTOR + 1)
    )
    contribution = ModelAttributionContribution(
        factor_id="max-context",
        label="Maximum snapshot context",
        drift_event_sha256=hashes,
        marginal_delta=OutcomeMetricDelta(
            success_probability=0,
            cumulative_shortfall_real_p50=0,
        ),
        evaluation_count=1,
        confidence=CONFIDENCE,
    )
    method = AttributionMethodManifest(
        method=AttributionMethod.ONE_FACTOR_COUNTERFACTUAL,
        factor_ids=("max-context",),
        baseline_revision_manifest_sha256=HASH_A,
        current_revision_manifest_sha256=HASH_B,
        comparison_manifest_sha256=HASH_C,
        calibration_snapshot_manifest_sha256=HASH_D,
        drift_event_sha256=hashes,
        counterfactual_definition="all maximum-bound snapshot events remain linked",
    )
    maximum_graph = {
        "contributions": tuple(
            contribution.model_copy(update={"factor_id": f"factor-{index}"})
            for index in range(MAX_ATTRIBUTION_FACTORS)
        )
    }

    assert len(contribution.drift_event_sha256) == 2_000
    assert method.drift_event_sha256 == hashes
    assert (
        sum(len(row.drift_event_sha256) for row in maximum_graph["contributions"])
        == MAX_ATTRIBUTION_LINKED_DRIFT_HASHES
    )
    assert AttributionReport.admit_bounded_link_graph(maximum_graph) is maximum_graph

    oversized_graph = {
        "contributions": (
            {
                **contribution.model_dump(mode="python"),
                "drift_event_sha256": (*hashes, HASH_A),
            },
            *maximum_graph["contributions"][1:],
        )
    }
    with pytest.raises(ValueError, match="resource limit"):
        AttributionReport.admit_bounded_link_graph(oversized_graph)
    with pytest.raises(ValidationError, match="at most 2000"):
        ModelAttributionContribution(
            factor_id="oversized",
            label="Oversized factor",
            drift_event_sha256=(*hashes, HASH_A),
            marginal_delta=OutcomeMetricDelta(
                success_probability=0,
                cumulative_shortfall_real_p50=0,
            ),
            evaluation_count=1,
            confidence=CONFIDENCE,
        )


def test_large_monetary_accounting_uses_absolute_cent_tolerance() -> None:
    method = _method(snapshot=BALANCE_SNAPSHOT, factor_ids=("balance",))
    report = build_attribution_report(
        baseline=OutcomeMetricValues(
            success_probability=0.5,
            cumulative_shortfall_real_p50=100_000_000_000_000,
        ),
        current=OutcomeMetricValues(
            success_probability=0.5,
            cumulative_shortfall_real_p50=100_000_000_000_100,
        ),
        contributions=(
            _contribution(
                "balance",
                success=0,
                shortfall=99,
                tax=0,
                estate=0,
            ).model_copy(
                update={
                    "marginal_delta": OutcomeMetricDelta(
                        success_probability=0,
                        cumulative_shortfall_real_p50=99,
                    )
                }
            ),
        ),
        method=method,
        calibration_snapshot_manifest=BALANCE_SNAPSHOT,
    )

    with pytest.raises(ValidationError, match="does not account"):
        AttributionReport.model_validate(
            {
                **report.model_dump(mode="json"),
                "interaction_residual": {
                    "success_probability": 0,
                    "cumulative_shortfall_real_p50": 1.02,
                    "lifetime_tax_real_p50": None,
                    "after_tax_estate_value_real_p50": None,
                },
            }
        )
    with pytest.raises(ValidationError, match="less than or equal"):
        OutcomeMetricValues(
            success_probability=0.5,
            cumulative_shortfall_real_p50=1_000_000_000_000_001,
        )


def test_optional_metric_availability_must_match() -> None:
    with pytest.raises(ValueError, match="lifetime tax availability"):
        metric_delta(
            OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
            ),
            OutcomeMetricValues(
                success_probability=0.5,
                cumulative_shortfall_real_p50=0,
                lifetime_tax_real_p50=1,
            ),
        )
    with pytest.raises(ValueError, match="lifetime tax attribution availability"):
        sum_metric_deltas(
            (
                OutcomeMetricDelta(
                    success_probability=0,
                    cumulative_shortfall_real_p50=0,
                    lifetime_tax_real_p50=1,
                ),
                OutcomeMetricDelta(
                    success_probability=0,
                    cumulative_shortfall_real_p50=0,
                ),
            )
        )


def test_subtraction_preserves_metric_signs() -> None:
    residual = subtract_metric_deltas(
        OutcomeMetricDelta(
            success_probability=0.10,
            cumulative_shortfall_real_p50=-5_000,
            lifetime_tax_real_p50=10_000,
            after_tax_estate_value_real_p50=100_000,
        ),
        OutcomeMetricDelta(
            success_probability=0.09,
            cumulative_shortfall_real_p50=-4_000,
            lifetime_tax_real_p50=9_000,
            after_tax_estate_value_real_p50=95_000,
        ),
    )

    assert residual.success_probability == pytest.approx(0.01)
    assert residual.cumulative_shortfall_real_p50 == -1_000
    assert residual.lifetime_tax_real_p50 == 1_000
    assert residual.after_tax_estate_value_real_p50 == 5_000
