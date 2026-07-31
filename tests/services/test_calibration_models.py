from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta, timezone
import multiprocessing
import pickle
from typing import cast

from pydantic import JsonValue, ValidationError
import pytest

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.calibration_drift import (
    ScalarDriftInput,
    StalenessInput,
    detect_balance_drift,
    detect_spending_drift,
    detect_staleness,
)
from ynab_agent.services.calibration_models import (
    CalibrationObservation,
    CalibrationPolicy,
    CalibrationSnapshot,
    CalibrationSnapshotManifest,
    ConfidenceAssessment,
    FreshnessAssessment,
    FreshnessState,
    MaterialityThreshold,
    ObservationKind,
    ObservationPayload,
    ObservationSourceKind,
    ObservationUnit,
    SourceObservationLink,
    SourceWatermark,
    canonical_content_sha256,
)


NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
HASH_A = "a" * 64
HASH_B = "b" * 64


def _cross_process_round_trip(snapshot: CalibrationSnapshot) -> float:
    restored = pickle.loads(pickle.dumps(snapshot))
    return float(restored.manifest.resolved_scenario.annual_spending)


def _link() -> SourceObservationLink:
    return SourceObservationLink(
        source_kind=ObservationSourceKind.YNAB_SYNC,
        source_id="account-1",
        source_uri="ynab://budget/budget-1/account/account-1",
        content_sha256=HASH_A,
        observed_at=NOW,
        change_batch_id="batch-1",
    )


def _watermark() -> SourceWatermark:
    return SourceWatermark(
        budget_id="budget-1",
        resource="accounts",
        server_knowledge=42,
        synced_at=NOW,
        change_batch_id="batch-1",
    )


def _threshold() -> MaterialityThreshold:
    return MaterialityThreshold(absolute=100, relative=0.05)


def _policy() -> CalibrationPolicy:
    threshold = _threshold()
    return CalibrationPolicy(
        spending=threshold,
        contribution=threshold,
        debt_payoff=threshold,
        balance=threshold,
        allocation=MaterialityThreshold(absolute=0.05),
        staleness=MaterialityThreshold(absolute=45),
    )


def _freshness(
    *,
    evaluated_at: datetime = NOW,
    last_reconciled_at: datetime | None = NOW - timedelta(days=2),
    stale_after_days: int = 45,
    frozen_after_days: int = 90,
) -> FreshnessAssessment:
    latest = last_reconciled_at
    age_days = (
        int((evaluated_at - latest).total_seconds() // 86_400) if latest is not None else None
    )
    return FreshnessAssessment(
        state=(
            FreshnessState.FRESH
            if latest is not None and age_days is not None and age_days < stale_after_days
            else FreshnessState.UNKNOWN
        ),
        evaluated_at=evaluated_at,
        last_reconciled_at=last_reconciled_at,
        latest_evidence_at=latest,
        age_days=age_days,
        stale_after_days=stale_after_days,
        frozen_after_days=frozen_after_days,
        evidence=(("last_reconciled",) if last_reconciled_at is not None else ()),
    )


def _observation() -> CalibrationObservation:
    return CalibrationObservation.create(
        id="observation-1",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="account-1",
        effective_date=date(2026, 7, 30),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS,
            value=125_000,
            fields={"balance_milliunits": 125_000_000},
        ),
        freshness=_freshness(),
        source_links=(_link(),),
        source_watermarks=(_watermark(),),
    )


def _scenario() -> WealthScenario:
    return WealthScenario(
        name="Continuous plan",
        current_age=30,
        retirement_age=67,
        end_age=95,
        starting_portfolio=125_000,
        annual_contribution=7_500,
        annual_spending=60_000,
        trials=100,
        seed=97,
    )


def test_canonical_hash_is_order_independent_and_unicode_stable() -> None:
    first = {"z": [1, 2], "a": {"name": "VTSAX"}}
    second = {"a": {"name": "VTSAX"}, "z": [1, 2]}

    assert canonical_content_sha256(cast(JsonValue, first)) == canonical_content_sha256(
        cast(JsonValue, second)
    )
    assert len(canonical_content_sha256({"fund": "世界"})) == 64


def test_canonical_hash_normalizes_numbers_unicode_and_utc_instants() -> None:
    assert canonical_content_sha256({"value": 1}) == canonical_content_sha256({"value": 1.0})
    assert canonical_content_sha256({"value": 0.0}) == canonical_content_sha256({"value": -0.0})
    assert canonical_content_sha256(
        {"fund": "\N{LATIN SMALL LETTER E WITH ACUTE}"}
    ) == canonical_content_sha256({"fund": "e\u0301"})
    utc = _watermark()
    same_instant = utc.model_copy(
        update={"synced_at": NOW.astimezone(timezone(timedelta(hours=-4)))}
    )
    assert canonical_content_sha256(utc) == canonical_content_sha256(same_instant)
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_content_sha256(cast(JsonValue, {1: "not JSON"}))
    assert len(canonical_content_sha256(tuple(range(10_000)))) == 64


def test_nfc_normalization_precedes_uniqueness_and_sorting() -> None:
    composed = "\N{LATIN SMALL LETTER E WITH ACUTE}"
    decomposed = "e\u0301"

    with pytest.raises(ValidationError, match="must be unique"):
        ConfidenceAssessment(
            score=0.5,
            basis="normalized evidence",
            limitations=(composed, decomposed),
        )
    with pytest.raises(ValidationError, match="collide after normalization"):
        ObservationPayload(
            unit=ObservationUnit.STRUCTURED,
            value={composed: 1, decomposed: 2},
        )


def test_observation_is_frozen_extra_forbid_and_content_addressed() -> None:
    observation = _observation()

    assert observation.content_sha256 == canonical_content_sha256(
        observation.model_dump(mode="json", exclude={"content_sha256"})
    )
    with pytest.raises(ValidationError, match="frozen"):
        observation.subject_id = "changed"
    with pytest.raises(ValidationError, match="Extra inputs"):
        CalibrationObservation.model_validate(
            {
                **observation.model_dump(mode="json"),
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError, match="content hash mismatch"):
        CalibrationObservation.model_validate(
            {
                **observation.model_dump(mode="json"),
                "payload": {
                    "unit": "dollars",
                    "value": 999_999,
                    "fields": {},
                },
            }
        )


def test_nested_observation_values_are_deeply_immutable_and_bounded() -> None:
    observation = _observation()

    with pytest.raises(TypeError):
        dict.__setitem__(observation.payload.fields, "balance_milliunits", 1)
    nested = ObservationPayload(
        unit=ObservationUnit.STRUCTURED,
        value={"nested": [1]},
    )
    with pytest.raises(TypeError):
        dict.__setitem__(nested.value, "other", 2)
    with pytest.raises(TypeError):
        list.append(nested.value["nested"], 2)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="array exceeds the item limit"):
        ObservationPayload(
            unit=ObservationUnit.STRUCTURED,
            value={"huge": list(range(20_000))},
        )
    with pytest.raises(ValidationError, match="node limit"):
        ObservationPayload(
            unit=ObservationUnit.STRUCTURED,
            value={"branches": tuple(tuple(range(100)) for _ in range(50))},
        )
    with pytest.raises(ValidationError, match="limitations are invalid"):
        ConfidenceAssessment(
            score=0.5,
            basis="partial evidence",
            limitations=("x" * 501,),
        )


def test_source_evidence_order_has_one_canonical_observation_hash() -> None:
    first = _link()
    second = _link().model_copy(
        update={
            "source_id": "account-2",
            "source_uri": "ynab://budget/budget-1/account/account-2",
            "content_sha256": HASH_B,
        }
    )
    common = {
        "id": "observation-ordered",
        "kind": ObservationKind.ACCOUNT_BALANCE,
        "subject_id": "account-1",
        "effective_date": date(2026, 7, 30),
        "observed_at": NOW,
        "payload": ObservationPayload(
            unit=ObservationUnit.DOLLARS,
            value=125_000,
        ),
        "freshness": _freshness(),
    }

    forward = CalibrationObservation.create(
        **common,
        source_links=(first, second),
    )
    reverse = CalibrationObservation.create(
        **common,
        source_links=(second, first),
    )
    offset_now = NOW.astimezone(timezone(timedelta(hours=-4)))
    equivalent_instant = CalibrationObservation.create(
        **{
            **common,
            "observed_at": offset_now,
        },
        source_links=(
            first.model_copy(update={"observed_at": offset_now}),
            second.model_copy(update={"observed_at": offset_now}),
        ),
    )

    assert forward.source_links == reverse.source_links
    assert forward.content_sha256 == reverse.content_sha256
    assert forward.content_sha256 == equivalent_instant.content_sha256


def test_source_timestamps_require_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        SourceWatermark(
            budget_id="budget-1",
            resource="accounts",
            server_knowledge=42,
            synced_at=datetime(2026, 7, 30),
            change_batch_id="batch-1",
        )
    with pytest.raises(ValidationError, match="timezone"):
        SourceObservationLink(
            source_kind=ObservationSourceKind.YNAB_SYNC,
            source_id="account-1",
            source_uri="ynab://budget/budget-1/account/account-1",
            content_sha256=HASH_A,
            observed_at=datetime(2026, 7, 30),
        )
    with pytest.raises(ValidationError, match="change_batch_id"):
        SourceObservationLink(
            source_kind=ObservationSourceKind.YNAB_SYNC,
            source_id="account-1",
            source_uri="ynab://budget/budget-1/account/account-1",
            content_sha256=HASH_A,
            observed_at=NOW,
        )


def test_observation_kind_unit_freshness_and_utc_date_contracts() -> None:
    with pytest.raises(ValidationError, match="unit does not match"):
        CalibrationObservation.create(
            id="wrong-unit",
            kind=ObservationKind.SPENDING,
            subject_id="portfolio",
            effective_date=NOW.date(),
            observed_at=NOW,
            payload=ObservationPayload(
                unit=ObservationUnit.DOLLARS,
                value=60_000,
            ),
            source_links=(_link(),),
        )
    with pytest.raises(ValidationError, match="require freshness"):
        CalibrationObservation.create(
            id="missing-freshness",
            kind=ObservationKind.ASSET_ALLOCATION,
            subject_id="portfolio",
            effective_date=NOW.date(),
            observed_at=NOW,
            payload=ObservationPayload(
                unit=ObservationUnit.FRACTION,
                value={"stocks": 1},
            ),
            source_links=(_link(),),
        )

    offset_instant = datetime(
        2026,
        7,
        29,
        20,
        30,
        tzinfo=timezone(timedelta(hours=-4)),
    )
    observation = CalibrationObservation.create(
        id="utc-boundary",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="account-1",
        effective_date=date(2026, 7, 30),
        observed_at=offset_instant,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS,
            value=125_000,
        ),
        freshness=_freshness(
            evaluated_at=offset_instant,
            last_reconciled_at=offset_instant - timedelta(days=1),
        ),
        source_links=(_link().model_copy(update={"observed_at": offset_instant}),),
    )

    assert observation.effective_date == date(2026, 7, 30)


def test_materiality_thresholds_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        MaterialityThreshold(absolute=0)
    with pytest.raises(ValidationError, match="greater than 0"):
        MaterialityThreshold(relative=0)


def test_snapshot_freezes_scenario_observations_policy_and_watermarks() -> None:
    scenario = _scenario()
    observation = _observation()
    manifest = CalibrationSnapshotManifest(
        profile_id="profile-1",
        scenario_revision_id="00000000-0000-0000-0000-000000000097",
        scenario_manifest_sha256=HASH_A,
        source_sync_batch_id="batch-1",
        as_of=NOW,
        resolved_scenario=scenario,
        resolved_scenario_sha256=canonical_content_sha256(scenario),
        policy=_policy(),
        observations=(observation,),
        source_watermarks=(_watermark(),),
    )
    snapshot = CalibrationSnapshot(
        id="00000000-0000-0000-0000-000000000098",
        manifest=manifest,
        manifest_sha256=canonical_content_sha256(manifest),
        created_at=NOW,
    )

    assert snapshot.manifest.observations[0].source_links[0].source_uri.startswith("ynab://")
    assert snapshot.manifest.resolved_scenario.starting_portfolio == 125_000
    with pytest.raises(AttributeError):
        snapshot.manifest.resolved_scenario.annual_spending = 999
    with pytest.raises(TypeError):
        list.append(
            snapshot.manifest.resolved_scenario.accounts,  # type: ignore[arg-type]
            {"id": "mutable"},
        )
    scenario.annual_spending = 999
    assert snapshot.manifest.resolved_scenario.annual_spending == 60_000
    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored.manifest.resolved_scenario.annual_spending == 60_000
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        assert executor.submit(_cross_process_round_trip, snapshot).result() == 60_000
    with pytest.raises(ValidationError, match="manifest hash mismatch"):
        CalibrationSnapshot.model_validate(
            {
                **snapshot.model_dump(mode="json"),
                "manifest_sha256": HASH_B,
            }
        )


def test_freshness_contract_rejects_incoherent_state_and_chronology() -> None:
    with pytest.raises(ValidationError, match="state does not match"):
        FreshnessAssessment(
            state=FreshnessState.FRESH,
            evaluated_at=NOW,
            last_reconciled_at=NOW - timedelta(days=60),
            latest_evidence_at=NOW - timedelta(days=60),
            age_days=60,
            stale_after_days=30,
            frozen_after_days=90,
            evidence=("last_reconciled",),
        )
    with pytest.raises(ValidationError, match="cannot postdate"):
        FreshnessAssessment(
            state=FreshnessState.FRESH,
            evaluated_at=NOW,
            last_reconciled_at=NOW + timedelta(seconds=1),
            latest_evidence_at=NOW + timedelta(seconds=1),
            age_days=0,
            stale_after_days=30,
            frozen_after_days=90,
            evidence=("last_reconciled",),
        )
    with pytest.raises(ValidationError, match="must be at least"):
        FreshnessAssessment(
            state=FreshnessState.UNKNOWN,
            evaluated_at=NOW,
            stale_after_days=90,
            frozen_after_days=30,
        )


def test_observation_freshness_is_evaluated_at_observation_time() -> None:
    evaluated_at = NOW - timedelta(days=1)

    with pytest.raises(ValidationError, match="evaluated at observation observed_at"):
        CalibrationObservation.create(
            id="carried-freshness",
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="account-1",
            effective_date=evaluated_at.date(),
            observed_at=NOW,
            payload=ObservationPayload(
                unit=ObservationUnit.DOLLARS,
                value=125_000,
            ),
            freshness=_freshness(
                evaluated_at=evaluated_at,
                last_reconciled_at=evaluated_at - timedelta(days=2),
            ),
            source_links=(_link(),),
        )


def test_snapshot_rejects_scenario_hash_drift_and_future_observations() -> None:
    scenario = _scenario()
    observation = _observation()
    common = {
        "profile_id": "profile-1",
        "scenario_revision_id": "00000000-0000-0000-0000-000000000097",
        "scenario_manifest_sha256": HASH_A,
        "source_sync_batch_id": "batch-1",
        "as_of": NOW,
        "resolved_scenario": scenario,
        "policy": _policy(),
        "observations": (observation,),
        "source_watermarks": (_watermark(),),
    }
    with pytest.raises(ValidationError, match="resolved scenario content hash mismatch"):
        CalibrationSnapshotManifest.model_validate(
            {
                **common,
                "resolved_scenario_sha256": HASH_B,
            }
        )

    future_observation = CalibrationObservation.create(
        id="future",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="account-1",
        effective_date=date(2026, 7, 31),
        observed_at=NOW + timedelta(days=1),
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS,
            value=125_000,
        ),
        freshness=_freshness(evaluated_at=NOW + timedelta(days=1)),
        source_links=(_link().model_copy(update={"observed_at": NOW + timedelta(days=1)}),),
    )
    with pytest.raises(ValidationError, match="cannot postdate"):
        CalibrationSnapshotManifest.model_validate(
            {
                **common,
                "observations": (future_observation,),
                "resolved_scenario_sha256": canonical_content_sha256(scenario),
            },
        )

    with pytest.raises(ValidationError, match="watermarks cannot postdate"):
        CalibrationSnapshotManifest.model_validate(
            {
                **common,
                "resolved_scenario_sha256": canonical_content_sha256(scenario),
                "source_watermarks": (
                    _watermark().model_copy(update={"synced_at": NOW + timedelta(seconds=1)}),
                ),
            }
        )


def test_snapshot_rejects_future_nested_source_evidence() -> None:
    scenario = _scenario()
    observation = _observation()
    common = {
        "profile_id": "profile-1",
        "scenario_revision_id": "00000000-0000-0000-0000-000000000097",
        "scenario_manifest_sha256": HASH_A,
        "source_sync_batch_id": "batch-1",
        "as_of": NOW,
        "resolved_scenario": scenario,
        "resolved_scenario_sha256": canonical_content_sha256(scenario),
        "policy": _policy(),
        "source_watermarks": (_watermark(),),
    }
    future_link_observation = observation.model_copy(
        update={
            "source_links": (_link().model_copy(update={"observed_at": NOW + timedelta(days=1)}),)
        }
    )
    future_watermark_observation = observation.model_copy(
        update={
            "source_watermarks": (
                _watermark().model_copy(update={"synced_at": NOW + timedelta(days=1)}),
            )
        }
    )
    future_effective_observation = observation.model_copy(
        update={"effective_date": date(2030, 1, 1)}
    )

    with pytest.raises(ValidationError, match="source links cannot postdate"):
        CalibrationSnapshotManifest(
            **common,
            observations=(future_link_observation,),
        )
    with pytest.raises(
        ValidationError,
        match="observation watermarks cannot postdate",
    ):
        CalibrationSnapshotManifest(
            **common,
            observations=(future_watermark_observation,),
        )
    with pytest.raises(ValidationError, match="effective_date cannot postdate"):
        CalibrationSnapshotManifest(
            **common,
            observations=(future_effective_observation,),
        )


def test_snapshot_enforces_references_watermarks_and_sync_batch() -> None:
    scenario = _scenario()
    observation = _observation()
    common = {
        "profile_id": "profile-1",
        "scenario_revision_id": "00000000-0000-0000-0000-000000000097",
        "scenario_manifest_sha256": HASH_A,
        "source_sync_batch_id": "batch-1",
        "as_of": NOW,
        "resolved_scenario": scenario,
        "resolved_scenario_sha256": canonical_content_sha256(scenario),
        "policy": _policy(),
        "source_watermarks": (_watermark(),),
    }
    orphan = detect_balance_drift(
        ScalarDriftInput(
            subject_id="account-1",
            baseline_value=100_000,
            observed_value=125_000,
            unit=ObservationUnit.DOLLARS,
            threshold=_threshold(),
            confidence=ConfidenceAssessment(
                score=1,
                basis="reviewed balance",
            ),
            observation_ids=("missing-observation",),
        )
    )

    with pytest.raises(ValidationError, match="IDs must be unique"):
        CalibrationSnapshotManifest(
            **common,
            observations=(observation, observation),
        )
    with pytest.raises(ValidationError, match="must reference"):
        CalibrationSnapshotManifest(
            **common,
            observations=(observation,),
            drift_events=(orphan,),
        )
    with pytest.raises(ValidationError, match="batches must match"):
        CalibrationSnapshotManifest(
            **{
                **common,
                "source_watermarks": (
                    _watermark().model_copy(update={"change_batch_id": "other-batch"}),
                ),
            },
            observations=(observation,),
        )


def test_snapshot_binds_policy_and_drift_reference_context_exactly() -> None:
    scenario = _scenario()
    current_observation = _observation().model_copy(
        update={
            "payload": ObservationPayload(
                unit=ObservationUnit.DOLLARS,
                value=125_000,
                fields={"drift_role": "current"},
            )
        }
    )
    current_observation = CalibrationObservation.create(
        id=current_observation.id,
        kind=current_observation.kind,
        subject_id="portfolio",
        effective_date=current_observation.effective_date,
        observed_at=current_observation.observed_at,
        payload=current_observation.payload,
        freshness=current_observation.freshness,
        source_links=current_observation.source_links,
        source_watermarks=current_observation.source_watermarks,
    )
    baseline_observation = CalibrationObservation.create(
        id="observation-baseline",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="portfolio",
        effective_date=date(2026, 7, 29),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS,
            value=125_000,
            fields={
                "drift_role": "baseline",
                "plan_field": "starting_portfolio",
            },
        ),
        freshness=_freshness(),
        source_links=(
            _link().model_copy(
                update={
                    "source_id": "account-1-baseline",
                    "source_uri": "ynab://budget/budget-1/account/account-1/baseline",
                    "content_sha256": HASH_B,
                }
            ),
        ),
        source_watermarks=(_watermark(),),
    )
    observations = (baseline_observation, current_observation)
    common = {
        "profile_id": "profile-1",
        "scenario_revision_id": "00000000-0000-0000-0000-000000000097",
        "scenario_manifest_sha256": HASH_A,
        "source_sync_batch_id": "batch-1",
        "as_of": NOW,
        "resolved_scenario": scenario,
        "resolved_scenario_sha256": canonical_content_sha256(scenario),
        "policy": _policy(),
        "observations": observations,
        "source_watermarks": (_watermark(),),
    }
    valid_event = detect_balance_drift(
        ScalarDriftInput(
            subject_id="portfolio",
            baseline_value=125_000,
            observed_value=125_000,
            unit=ObservationUnit.DOLLARS,
            threshold=_threshold(),
            confidence=ConfidenceAssessment(score=1, basis="reviewed"),
            observation_ids=tuple(row.id for row in observations),
        )
    )

    assert CalibrationSnapshotManifest(
        **common,
        drift_events=(valid_event,),
    ).drift_events == (valid_event,)

    wrong_threshold = detect_balance_drift(
        ScalarDriftInput(
            subject_id="portfolio",
            baseline_value=125_000,
            observed_value=125_000,
            unit=ObservationUnit.DOLLARS,
            threshold=MaterialityThreshold(absolute=200, relative=0.05),
            confidence=ConfidenceAssessment(score=1, basis="reviewed"),
            observation_ids=tuple(row.id for row in observations),
        )
    )
    with pytest.raises(ValidationError, match="threshold must match"):
        CalibrationSnapshotManifest(
            **common,
            drift_events=(wrong_threshold,),
        )

    wrong_kind = detect_spending_drift(
        ScalarDriftInput(
            subject_id="account-1",
            baseline_value=60_000,
            observed_value=61_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            threshold=_threshold(),
            confidence=ConfidenceAssessment(score=1, basis="reviewed"),
            observation_ids=tuple(row.id for row in observations),
        )
    )
    with pytest.raises(ValidationError, match="context does not match"):
        CalibrationSnapshotManifest(
            **common,
            drift_events=(wrong_kind,),
        )

    wrong_subject = detect_balance_drift(
        ScalarDriftInput(
            subject_id="another-account",
            baseline_value=100_000,
            observed_value=125_000,
            unit=ObservationUnit.DOLLARS,
            threshold=_threshold(),
            confidence=ConfidenceAssessment(score=1, basis="reviewed"),
            observation_ids=tuple(row.id for row in observations),
        )
    )
    with pytest.raises(ValidationError, match="context does not match"):
        CalibrationSnapshotManifest(
            **common,
            drift_events=(wrong_subject,),
        )


def test_snapshot_binds_observation_and_drift_freshness_windows_to_policy() -> None:
    scenario = _scenario()
    mismatched_observation = CalibrationObservation.create(
        id="freshness-mismatch",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="account-1",
        effective_date=NOW.date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS,
            value=125_000,
        ),
        freshness=_freshness(stale_after_days=30),
        source_links=(_link(),),
    )
    common = {
        "profile_id": "profile-1",
        "scenario_revision_id": "00000000-0000-0000-0000-000000000097",
        "scenario_manifest_sha256": HASH_A,
        "source_sync_batch_id": "batch-1",
        "as_of": NOW,
        "resolved_scenario": scenario,
        "resolved_scenario_sha256": canonical_content_sha256(scenario),
        "policy": _policy(),
        "source_watermarks": (_watermark(),),
    }
    with pytest.raises(ValidationError, match="freshness windows must match"):
        CalibrationSnapshotManifest(
            **common,
            observations=(mismatched_observation,),
        )

    freshness_observation = CalibrationObservation.create(
        id="freshness-1",
        kind=ObservationKind.ACCOUNT_FRESHNESS,
        subject_id="account-1",
        effective_date=NOW.date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DAYS,
            value=2,
        ),
        freshness=_freshness(),
        source_links=(_link(),),
    )
    staleness_event = detect_staleness(
        StalenessInput(
            subject_id="account-1",
            evaluated_at=NOW,
            last_reconciled_at=NOW - timedelta(days=2),
            stale_after_days=30,
            frozen_after_days=90,
            threshold=_policy().staleness,
            confidence=ConfidenceAssessment(score=1, basis="reviewed"),
            observation_ids=(freshness_observation.id,),
        )
    )
    with pytest.raises(ValidationError, match="drift freshness windows must match"):
        CalibrationSnapshotManifest(
            **common,
            observations=(freshness_observation,),
            drift_events=(staleness_event,),
        )


def test_snapshot_rejects_carried_fresh_assessment_at_later_as_of() -> None:
    evaluated_at = NOW - timedelta(days=120)
    source_evidence_at = evaluated_at - timedelta(days=2)
    carried_fresh = CalibrationObservation.create(
        id="carried-freshness",
        kind=ObservationKind.ACCOUNT_FRESHNESS,
        subject_id="account-1",
        effective_date=evaluated_at.date(),
        observed_at=evaluated_at,
        payload=ObservationPayload(
            unit=ObservationUnit.DAYS,
            value=2,
        ),
        freshness=_freshness(
            evaluated_at=evaluated_at,
            last_reconciled_at=source_evidence_at,
            stale_after_days=45,
        ),
        source_links=(_link().model_copy(update={"observed_at": evaluated_at}),),
    )
    scenario = _scenario()

    with pytest.raises(ValidationError, match="evaluated at snapshot as_of"):
        CalibrationSnapshotManifest(
            profile_id="profile-1",
            scenario_revision_id="00000000-0000-0000-0000-000000000097",
            scenario_manifest_sha256=HASH_A,
            source_sync_batch_id="batch-1",
            as_of=NOW,
            resolved_scenario=scenario,
            resolved_scenario_sha256=canonical_content_sha256(scenario),
            policy=_policy(),
            observations=(carried_fresh,),
            source_watermarks=(_watermark(),),
        )


def test_snapshot_rejects_staleness_drift_with_contradictory_freshness() -> None:
    freshness_observation = CalibrationObservation.create(
        id="freshness-1",
        kind=ObservationKind.ACCOUNT_FRESHNESS,
        subject_id="account-1",
        effective_date=NOW.date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DAYS,
            value=2,
        ),
        freshness=_freshness(),
        source_links=(_link(),),
    )
    contradictory_event = detect_staleness(
        StalenessInput(
            subject_id="account-1",
            evaluated_at=NOW,
            last_reconciled_at=NOW - timedelta(days=60),
            stale_after_days=45,
            frozen_after_days=90,
            threshold=_policy().staleness,
            confidence=ConfidenceAssessment(score=1, basis="reviewed"),
            observation_ids=(freshness_observation.id,),
        )
    )
    scenario = _scenario()

    with pytest.raises(ValidationError, match="must match referenced observations"):
        CalibrationSnapshotManifest(
            profile_id="profile-1",
            scenario_revision_id="00000000-0000-0000-0000-000000000097",
            scenario_manifest_sha256=HASH_A,
            source_sync_batch_id="batch-1",
            as_of=NOW,
            resolved_scenario=scenario,
            resolved_scenario_sha256=canonical_content_sha256(scenario),
            policy=_policy(),
            observations=(freshness_observation,),
            drift_events=(contradictory_event,),
            source_watermarks=(_watermark(),),
        )


def test_snapshot_cannot_be_created_before_its_manifest_as_of() -> None:
    scenario = _scenario()
    manifest = CalibrationSnapshotManifest(
        profile_id="profile-1",
        scenario_revision_id="00000000-0000-0000-0000-000000000097",
        scenario_manifest_sha256=HASH_A,
        source_sync_batch_id="batch-1",
        as_of=NOW,
        resolved_scenario=scenario,
        resolved_scenario_sha256=canonical_content_sha256(scenario),
        policy=_policy(),
        observations=(_observation(),),
        source_watermarks=(_watermark(),),
    )

    with pytest.raises(ValidationError, match="created before as_of"):
        CalibrationSnapshot(
            id="00000000-0000-0000-0000-000000000098",
            manifest=manifest,
            manifest_sha256=canonical_content_sha256(manifest),
            created_at=NOW - timedelta(seconds=1),
        )


def test_drift_event_hash_cannot_be_reused_for_changed_materiality() -> None:
    threshold = _threshold()
    confidence = ConfidenceAssessment(score=1, basis="reviewed balance")
    event = detect_balance_drift(
        ScalarDriftInput(
            subject_id="account-1",
            baseline_value=100_000,
            observed_value=125_000,
            unit=ObservationUnit.DOLLARS,
            threshold=threshold,
            confidence=confidence,
            observation_ids=("observation-1",),
        )
    )

    with pytest.raises(TypeError):
        dict.__setitem__(event.details, "tampered", True)
    with pytest.raises(ValidationError, match="content hash mismatch"):
        type(event).model_validate(
            {
                **event.model_dump(mode="json"),
                "material": False,
            }
        )
    with pytest.raises(ValidationError, match="content hash mismatch"):
        CalibrationSnapshotManifest(
            profile_id="profile-1",
            scenario_revision_id="00000000-0000-0000-0000-000000000097",
            scenario_manifest_sha256=HASH_A,
            source_sync_batch_id="batch-1",
            as_of=NOW,
            resolved_scenario=_scenario(),
            resolved_scenario_sha256=canonical_content_sha256(_scenario()),
            policy=_policy(),
            observations=(_observation(),),
            drift_events=(event.model_copy(update={"material": False}),),
            source_watermarks=(_watermark(),),
        )
