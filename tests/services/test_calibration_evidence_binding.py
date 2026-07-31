from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable, cast

from pydantic import ValidationError
import pytest

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.calibration_drift import (
    AllocationDriftInput,
    DebtPayoffInput,
    ScalarDriftInput,
    StalenessInput,
    detect_allocation_drift,
    detect_balance_drift,
    detect_contribution_drift,
    detect_debt_payoff,
    detect_spending_drift,
    detect_staleness,
)
from ynab_agent.services.calibration_models import (
    CalibrationObservation,
    CalibrationPolicy,
    CalibrationSnapshotManifest,
    ConfidenceAssessment,
    DriftEvent,
    FreshnessAssessment,
    FreshnessState,
    FrozenFloatMap,
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


NOW = datetime(2026, 7, 30, 12, tzinfo=timezone.utc)
SOURCE_HASH = "a" * 64
SCENARIO_HASH = "b" * 64
CONFIDENCE = ConfidenceAssessment(score=1, basis="copied source evidence")
SCALAR_THRESHOLD = MaterialityThreshold(absolute=100, relative=0.05)
ALLOCATION_THRESHOLD = MaterialityThreshold(absolute=0.05)
STALENESS_THRESHOLD = MaterialityThreshold(absolute=45)
POLICY = CalibrationPolicy(
    spending=SCALAR_THRESHOLD,
    contribution=SCALAR_THRESHOLD,
    debt_payoff=SCALAR_THRESHOLD,
    balance=SCALAR_THRESHOLD,
    allocation=ALLOCATION_THRESHOLD,
    staleness=STALENESS_THRESHOLD,
)
WATERMARK = SourceWatermark(
    budget_id="budget-1",
    resource="accounts",
    server_knowledge=42,
    synced_at=NOW,
    change_batch_id="batch-1",
)


def _freshness() -> FreshnessAssessment:
    reconciled_at = NOW - timedelta(days=2)
    return FreshnessAssessment(
        state=FreshnessState.FRESH,
        evaluated_at=NOW,
        last_reconciled_at=reconciled_at,
        latest_evidence_at=reconciled_at,
        age_days=2,
        stale_after_days=45,
        frozen_after_days=90,
        evidence=("last_reconciled",),
    )


def _observation(
    *,
    id: str,
    kind: ObservationKind,
    subject_id: str,
    value: object,
    unit: ObservationUnit,
    fields: dict[str, object] | None = None,
    effective_date: date = NOW.date(),
) -> CalibrationObservation:
    resolved_fields = dict(fields or {})
    if resolved_fields.get("drift_role") == "baseline":
        plan_field = {
            ObservationKind.SPENDING: "annual_spending",
            ObservationKind.CONTRIBUTION: "annual_contribution",
            ObservationKind.ACCOUNT_BALANCE: "starting_portfolio",
        }.get(kind)
        if plan_field is not None:
            resolved_fields["plan_field"] = plan_field
    freshness = (
        _freshness()
        if kind
        in {
            ObservationKind.DEBT_BALANCE,
            ObservationKind.ACCOUNT_BALANCE,
            ObservationKind.ASSET_ALLOCATION,
            ObservationKind.ACCOUNT_FRESHNESS,
        }
        else None
    )
    return CalibrationObservation.create(
        id=id,
        kind=kind,
        subject_id=subject_id,
        effective_date=effective_date,
        observed_at=NOW,
        payload=ObservationPayload(
            unit=unit,
            value=cast(ImmutableJson, value),
            fields=FrozenJsonObject(resolved_fields),
        ),
        freshness=freshness,
        source_links=(
            SourceObservationLink(
                source_kind=ObservationSourceKind.YNAB_SYNC,
                source_id=id,
                source_uri=f"ynab://budget/budget-1/evidence/{id}",
                content_sha256=SOURCE_HASH,
                observed_at=NOW,
                change_batch_id="batch-1",
            ),
        ),
    )


def _manifest(
    observations: tuple[CalibrationObservation, ...],
    event: DriftEvent | None,
) -> CalibrationSnapshotManifest:
    starting_portfolio = 125_000
    annual_contribution = 7_500
    annual_spending = 60_000
    if event is not None and isinstance(event.baseline_value, int | float):
        if event.kind.value == "balance":
            starting_portfolio = float(event.baseline_value)
        elif event.kind.value == "contribution":
            annual_contribution = float(event.baseline_value)
        elif event.kind.value == "spending":
            annual_spending = float(event.baseline_value)
    scenario = WealthScenario(
        name="evidence binding",
        current_age=30,
        retirement_age=67,
        end_age=95,
        starting_portfolio=starting_portfolio,
        annual_contribution=annual_contribution,
        annual_spending=annual_spending,
        trials=100,
        seed=97,
    )
    return CalibrationSnapshotManifest(
        profile_id="profile-1",
        scenario_revision_id="00000000-0000-0000-0000-000000000097",
        scenario_manifest_sha256=SCENARIO_HASH,
        source_sync_batch_id="batch-1",
        as_of=NOW,
        resolved_scenario=FrozenWealthScenario(scenario.model_dump(mode="python")),
        resolved_scenario_sha256=canonical_content_sha256(scenario),
        policy=POLICY,
        observations=observations,
        drift_events=(() if event is None else (event,)),
        source_watermarks=(WATERMARK,),
    )


def _with_source_version(
    observation: CalibrationObservation,
    *,
    source_id: str,
    source_uri: str,
    content_sha256: str,
    observed_at: datetime,
) -> CalibrationObservation:
    return CalibrationObservation.create(
        id=observation.id,
        kind=observation.kind,
        subject_id=observation.subject_id,
        effective_date=observation.effective_date,
        observed_at=observation.observed_at,
        payload=observation.payload,
        freshness=observation.freshness,
        source_links=(
            observation.source_links[0].model_copy(
                update={
                    "source_id": source_id,
                    "source_uri": source_uri,
                    "content_sha256": content_sha256,
                    "observed_at": observed_at,
                }
            ),
        ),
        source_watermarks=observation.source_watermarks,
    )


def _scalar_event(
    detector: Callable[[ScalarDriftInput], DriftEvent],
    *,
    subject_id: str,
    baseline: float,
    observed: float,
    unit: ObservationUnit,
    observation_ids: tuple[str, ...],
) -> DriftEvent:
    return detector(
        ScalarDriftInput(
            subject_id=subject_id,
            baseline_value=baseline,
            observed_value=observed,
            unit=unit,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=observation_ids,
        )
    )


def test_scalar_drift_is_bound_to_exact_and_auditable_sum_evidence() -> None:
    spending = (
        _observation(
            id="spending-baseline",
            kind=ObservationKind.SPENDING,
            subject_id="household",
            value=58_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={"drift_role": "baseline", "drift_aggregation": "exact"},
            effective_date=NOW.date() - timedelta(days=1),
        ),
        _observation(
            id="housing",
            kind=ObservationKind.SPENDING,
            subject_id="household",
            value=24_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={
                "drift_role": "observed",
                "drift_aggregation": "sum",
                "drift_group_id": "annual-spending",
            },
        ),
        _observation(
            id="other-spending",
            kind=ObservationKind.SPENDING,
            subject_id="household",
            value=36_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={
                "drift_role": "observed",
                "drift_aggregation": "sum",
                "drift_group_id": "annual-spending",
            },
        ),
    )
    spending_event = _scalar_event(
        detect_spending_drift,
        subject_id="household",
        baseline=58_000,
        observed=60_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=tuple(row.id for row in spending),
    )
    contribution = (
        _observation(
            id="contribution-baseline",
            kind=ObservationKind.CONTRIBUTION,
            subject_id="portfolio",
            value=7_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={"drift_role": "baseline", "drift_aggregation": "exact"},
            effective_date=NOW.date() - timedelta(days=1),
        ),
        _observation(
            id="annual-contribution",
            kind=ObservationKind.CONTRIBUTION,
            subject_id="portfolio",
            value=7_500,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={"drift_role": "observed", "drift_aggregation": "exact"},
        ),
    )
    contribution_event = _scalar_event(
        detect_contribution_drift,
        subject_id="portfolio",
        baseline=7_000,
        observed=7_500,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=tuple(row.id for row in contribution),
    )
    balance = (
        _observation(
            id="account-balance-baseline",
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="portfolio",
            value=120_000,
            unit=ObservationUnit.DOLLARS,
            fields={"drift_role": "baseline"},
            effective_date=NOW.date() - timedelta(days=1),
        ),
        _observation(
            id="account-balance",
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="portfolio",
            value=125_000,
            unit=ObservationUnit.DOLLARS,
            fields={"drift_role": "current"},
        ),
    )
    balance_event = _scalar_event(
        detect_balance_drift,
        subject_id="portfolio",
        baseline=120_000,
        observed=125_000,
        unit=ObservationUnit.DOLLARS,
        observation_ids=tuple(row.id for row in balance),
    )

    assert _manifest(spending, spending_event).drift_events == (spending_event,)
    assert _manifest(contribution, contribution_event).drift_events == (contribution_event,)
    assert _manifest(balance, balance_event).drift_events == (balance_event,)


@pytest.mark.parametrize(
    ("kind", "detector", "unit", "fields"),
    (
        (
            ObservationKind.SPENDING,
            detect_spending_drift,
            ObservationUnit.DOLLARS_PER_YEAR,
            {"drift_aggregation": "exact"},
        ),
        (
            ObservationKind.CONTRIBUTION,
            detect_contribution_drift,
            ObservationUnit.DOLLARS_PER_YEAR,
            {"drift_aggregation": "exact"},
        ),
        (
            ObservationKind.ACCOUNT_BALANCE,
            detect_balance_drift,
            ObservationUnit.DOLLARS,
            {},
        ),
    ),
)
def test_scalar_drift_rejects_fabricated_observed_values(
    kind: ObservationKind,
    detector: Callable[[ScalarDriftInput], DriftEvent],
    unit: ObservationUnit,
    fields: dict[str, object],
) -> None:
    subject_id = "portfolio" if kind is ObservationKind.ACCOUNT_BALANCE else "subject-1"
    baseline = _observation(
        id=f"{kind.value}-baseline",
        kind=kind,
        subject_id=subject_id,
        value=100_000,
        unit=unit,
        fields={
            **fields,
            "drift_role": (
                "baseline" if kind is not ObservationKind.ACCOUNT_BALANCE else "baseline"
            ),
        },
        effective_date=NOW.date() - timedelta(days=1),
    )
    observation = _observation(
        id=f"{kind.value}-observation",
        kind=kind,
        subject_id=subject_id,
        value=125_000,
        unit=unit,
        fields={
            **fields,
            "drift_role": (
                "observed" if kind is not ObservationKind.ACCOUNT_BALANCE else "current"
            ),
        },
    )
    fabricated = _scalar_event(
        detector,
        subject_id=subject_id,
        baseline=100_000,
        observed=999_999,
        unit=unit,
        observation_ids=(baseline.id, observation.id),
    )

    with pytest.raises(ValidationError, match="must match referenced observation evidence"):
        _manifest((baseline, observation), fabricated)


def test_spending_and_contribution_fail_closed_on_ambiguous_aggregation() -> None:
    spending_baseline = _observation(
        id="missing-mode-baseline",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=58_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={"drift_role": "baseline", "drift_aggregation": "exact"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    missing_mode = _observation(
        id="missing-mode",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=60_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={"drift_role": "observed"},
    )
    missing_mode_event = _scalar_event(
        detect_spending_drift,
        subject_id="household",
        baseline=58_000,
        observed=60_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=(spending_baseline.id, missing_mode.id),
    )
    with pytest.raises(ValidationError, match="exact or sum aggregation"):
        _manifest((spending_baseline, missing_mode), missing_mode_event)

    contribution_baseline = _observation(
        id="mismatch-baseline",
        kind=ObservationKind.CONTRIBUTION,
        subject_id="portfolio",
        value=7_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={"drift_role": "baseline", "drift_aggregation": "exact"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    mismatched_current = tuple(
        _observation(
            id=f"component-{index}",
            kind=ObservationKind.CONTRIBUTION,
            subject_id="portfolio",
            value=value,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={
                "drift_role": "observed",
                "drift_aggregation": "sum",
                "drift_group_id": group,
            },
        )
        for index, (value, group) in enumerate(((3_000, "ira"), (4_500, "401k")))
    )
    mismatched_event = _scalar_event(
        detect_contribution_drift,
        subject_id="portfolio",
        baseline=7_000,
        observed=7_500,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=(
            contribution_baseline.id,
            *(row.id for row in mismatched_current),
        ),
    )
    with pytest.raises(ValidationError, match="requires one explicit group"):
        _manifest((contribution_baseline, *mismatched_current), mismatched_event)


def test_allocation_drift_fails_closed_without_typed_scenario_target() -> None:
    target = _observation(
        id="allocation-target",
        kind=ObservationKind.ASSET_ALLOCATION,
        subject_id="portfolio",
        value={"stocks": 0.7, "bonds": 0.3},
        unit=ObservationUnit.FRACTION,
        fields={"drift_role": "target"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    observation = _observation(
        id="allocation",
        kind=ObservationKind.ASSET_ALLOCATION,
        subject_id="portfolio",
        value={"stocks": 0.8, "bonds": 0.2},
        unit=ObservationUnit.FRACTION,
        fields={"drift_role": "current"},
    )
    valid = detect_allocation_drift(
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights=FrozenFloatMap({"stocks": 0.7, "bonds": 0.3}),
            observed_weights=FrozenFloatMap({"stocks": 0.8, "bonds": 0.2}),
            threshold=ALLOCATION_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(target.id, observation.id),
        )
    )
    with pytest.raises(ValidationError, match="typed target-allocation field"):
        _manifest((target, observation), valid)

    duplicate = _observation(
        id="allocation-duplicate",
        kind=ObservationKind.ASSET_ALLOCATION,
        subject_id="portfolio",
        value={"stocks": 0.8, "bonds": 0.2},
        unit=ObservationUnit.FRACTION,
        fields={"drift_role": "current"},
    )
    ambiguous = detect_allocation_drift(
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights=FrozenFloatMap({"stocks": 0.7, "bonds": 0.3}),
            observed_weights=FrozenFloatMap({"stocks": 0.8, "bonds": 0.2}),
            threshold=ALLOCATION_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(target.id, observation.id, duplicate.id),
        )
    )
    with pytest.raises(ValidationError, match="duplicate payload/group evidence"):
        _manifest((target, observation, duplicate), ambiguous)


def test_balance_drift_rejects_ambiguous_multiple_observations() -> None:
    observations = (
        _observation(
            id="balance-baseline",
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="account-1",
            value=120_000,
            unit=ObservationUnit.DOLLARS,
            fields={"drift_role": "baseline"},
            effective_date=NOW.date() - timedelta(days=1),
        ),
        _observation(
            id="balance-current-a",
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="account-1",
            value=125_000,
            unit=ObservationUnit.DOLLARS,
            fields={"drift_role": "current"},
        ),
        _observation(
            id="balance-current-b",
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="account-1",
            value=126_000,
            unit=ObservationUnit.DOLLARS,
            fields={"drift_role": "current"},
        ),
    )
    ambiguous = _scalar_event(
        detect_balance_drift,
        subject_id="account-1",
        baseline=120_000,
        observed=125_000,
        unit=ObservationUnit.DOLLARS,
        observation_ids=tuple(row.id for row in observations),
    )

    with pytest.raises(ValidationError, match="unambiguous baseline/current"):
        _manifest(observations, ambiguous)


def test_debt_payoff_is_bound_to_ordered_previous_and_current_evidence() -> None:
    previous = _observation(
        id="mortgage-previous",
        kind=ObservationKind.DEBT_BALANCE,
        subject_id="mortgage",
        value=1_000,
        unit=ObservationUnit.DOLLARS,
        fields={"drift_role": "previous"},
        effective_date=NOW.date() - timedelta(days=30),
    )
    current = _observation(
        id="mortgage-current",
        kind=ObservationKind.DEBT_BALANCE,
        subject_id="mortgage",
        value=0,
        unit=ObservationUnit.DOLLARS,
        fields={"drift_role": "current"},
    )
    event = detect_debt_payoff(
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=1_000,
            current_liability=0,
            payoff_tolerance=1,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(previous.id, current.id),
        )
    )

    assert _manifest((previous, current), event).drift_events == (event,)

    fabricated = detect_debt_payoff(
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=50_000,
            current_liability=0,
            payoff_tolerance=1,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(previous.id, current.id),
        )
    )
    with pytest.raises(ValidationError, match="must match referenced observation evidence"):
        _manifest((previous, current), fabricated)

    duplicate_roles = _observation(
        id="mortgage-current-duplicate-role",
        kind=ObservationKind.DEBT_BALANCE,
        subject_id="mortgage",
        value=0,
        unit=ObservationUnit.DOLLARS,
        fields={"drift_role": "previous"},
    )
    duplicate_role_event = detect_debt_payoff(
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=1_000,
            current_liability=0,
            payoff_tolerance=1,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(previous.id, duplicate_roles.id),
        )
    )
    with pytest.raises(ValidationError, match="unambiguous previous/current"):
        _manifest((previous, duplicate_roles), duplicate_role_event)

    missing_role = _observation(
        id="mortgage-current-missing-role",
        kind=ObservationKind.DEBT_BALANCE,
        subject_id="mortgage",
        value=0,
        unit=ObservationUnit.DOLLARS,
    )
    missing_role_event = detect_debt_payoff(
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=1_000,
            current_liability=0,
            payoff_tolerance=1,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(previous.id, missing_role.id),
        )
    )
    with pytest.raises(ValidationError, match="unambiguous previous/current"):
        _manifest((previous, missing_role), missing_role_event)

    reversed_dates = _observation(
        id="mortgage-previous-same-date",
        kind=ObservationKind.DEBT_BALANCE,
        subject_id="mortgage",
        value=1_000,
        unit=ObservationUnit.DOLLARS,
        fields={"drift_role": "previous"},
    )
    reversed_date_event = detect_debt_payoff(
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=1_000,
            current_liability=0,
            payoff_tolerance=1,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(reversed_dates.id, current.id),
        )
    )
    with pytest.raises(ValidationError, match="must predate"):
        _manifest((reversed_dates, current), reversed_date_event)


def _shared_source_role_pair(
    kind: ObservationKind,
    *,
    earlier_source_at: datetime,
    later_source_at: datetime,
) -> tuple[tuple[CalibrationObservation, ...], DriftEvent]:
    if kind is ObservationKind.SPENDING:
        subject_id = "household"
        unit = ObservationUnit.DOLLARS_PER_YEAR
        earlier_role, later_role = "baseline", "observed"
        earlier_value, later_value = 60_000, 61_000
    elif kind is ObservationKind.CONTRIBUTION:
        subject_id = "portfolio"
        unit = ObservationUnit.DOLLARS_PER_YEAR
        earlier_role, later_role = "baseline", "observed"
        earlier_value, later_value = 7_500, 8_000
    elif kind is ObservationKind.ACCOUNT_BALANCE:
        subject_id = "portfolio"
        unit = ObservationUnit.DOLLARS
        earlier_role, later_role = "baseline", "current"
        earlier_value, later_value = 125_000, 130_000
    elif kind is ObservationKind.ASSET_ALLOCATION:
        subject_id = "portfolio"
        unit = ObservationUnit.FRACTION
        earlier_role, later_role = "target", "current"
        earlier_value = {"stocks": 0.7, "bonds": 0.3}
        later_value = {"stocks": 0.8, "bonds": 0.2}
    else:
        subject_id = "mortgage"
        unit = ObservationUnit.DOLLARS
        earlier_role, later_role = "previous", "current"
        earlier_value, later_value = 1_000, 0
    aggregation = (
        {"drift_aggregation": "exact"}
        if kind in {ObservationKind.SPENDING, ObservationKind.CONTRIBUTION}
        else {}
    )
    earlier = _with_source_version(
        _observation(
            id=f"{kind.value}-chronology-earlier",
            kind=kind,
            subject_id=subject_id,
            value=earlier_value,
            unit=unit,
            fields={**aggregation, "drift_role": earlier_role},
            effective_date=NOW.date() - timedelta(days=1),
        ),
        source_id=f"shared-{kind.value}",
        source_uri=f"source://{kind.value}/earlier",
        content_sha256="1" * 64,
        observed_at=earlier_source_at,
    )
    later = _with_source_version(
        _observation(
            id=f"{kind.value}-chronology-later",
            kind=kind,
            subject_id=subject_id,
            value=later_value,
            unit=unit,
            fields={**aggregation, "drift_role": later_role},
        ),
        source_id=f"shared-{kind.value}",
        source_uri=f"source://{kind.value}/later",
        content_sha256="2" * 64,
        observed_at=later_source_at,
    )
    observation_ids = (earlier.id, later.id)
    if kind is ObservationKind.SPENDING:
        event = _scalar_event(
            detect_spending_drift,
            subject_id=subject_id,
            baseline=60_000,
            observed=61_000,
            unit=unit,
            observation_ids=observation_ids,
        )
    elif kind is ObservationKind.CONTRIBUTION:
        event = _scalar_event(
            detect_contribution_drift,
            subject_id=subject_id,
            baseline=7_500,
            observed=8_000,
            unit=unit,
            observation_ids=observation_ids,
        )
    elif kind is ObservationKind.ACCOUNT_BALANCE:
        event = _scalar_event(
            detect_balance_drift,
            subject_id=subject_id,
            baseline=125_000,
            observed=130_000,
            unit=unit,
            observation_ids=observation_ids,
        )
    elif kind is ObservationKind.ASSET_ALLOCATION:
        event = detect_allocation_drift(
            AllocationDriftInput(
                subject_id=subject_id,
                target_weights=FrozenFloatMap(cast(dict[str, float], earlier_value)),
                observed_weights=FrozenFloatMap(cast(dict[str, float], later_value)),
                threshold=ALLOCATION_THRESHOLD,
                confidence=CONFIDENCE,
                observation_ids=observation_ids,
            )
        )
    else:
        event = detect_debt_payoff(
            DebtPayoffInput(
                subject_id=subject_id,
                previous_liability=1_000,
                current_liability=0,
                payoff_tolerance=1,
                threshold=SCALAR_THRESHOLD,
                confidence=CONFIDENCE,
                observation_ids=observation_ids,
            )
        )
    return (earlier, later), event


@pytest.mark.parametrize(
    "kind",
    (
        ObservationKind.SPENDING,
        ObservationKind.CONTRIBUTION,
        ObservationKind.ACCOUNT_BALANCE,
        ObservationKind.ASSET_ALLOCATION,
        ObservationKind.DEBT_BALANCE,
    ),
)
@pytest.mark.parametrize(
    ("earlier_offset", "later_offset"),
    ((-1, -1), (-1, -2)),
    ids=("equal", "reversed"),
)
def test_shared_source_role_pairs_require_strict_version_chronology(
    kind: ObservationKind,
    earlier_offset: int,
    later_offset: int,
) -> None:
    observations, event = _shared_source_role_pair(
        kind,
        earlier_source_at=NOW + timedelta(minutes=earlier_offset),
        later_source_at=NOW + timedelta(minutes=later_offset),
    )

    with pytest.raises(ValidationError, match="shared source versions require"):
        _manifest(observations, event)


@pytest.mark.parametrize(
    "kind",
    (
        ObservationKind.SPENDING,
        ObservationKind.CONTRIBUTION,
        ObservationKind.ACCOUNT_BALANCE,
        ObservationKind.ASSET_ALLOCATION,
        ObservationKind.DEBT_BALANCE,
    ),
)
def test_shared_source_role_pairs_accept_strict_version_chronology(
    kind: ObservationKind,
) -> None:
    observations, event = _shared_source_role_pair(
        kind,
        earlier_source_at=NOW - timedelta(minutes=2),
        later_source_at=NOW - timedelta(minutes=1),
    )

    if kind is ObservationKind.ASSET_ALLOCATION:
        with pytest.raises(ValidationError, match="typed target-allocation field"):
            _manifest(observations, event)
    else:
        assert _manifest(observations, event).drift_events == (event,)


@pytest.mark.parametrize(
    ("earlier_source_at", "later_source_at", "valid"),
    (
        (NOW - timedelta(minutes=2), NOW - timedelta(minutes=1), True),
        (NOW - timedelta(minutes=1), NOW - timedelta(minutes=1), False),
        (NOW - timedelta(minutes=1), NOW - timedelta(minutes=2), False),
    ),
    ids=("valid", "equal", "reversed"),
)
def test_sum_group_shared_source_versions_require_strict_chronology(
    earlier_source_at: datetime,
    later_source_at: datetime,
    valid: bool,
) -> None:
    baseline = _with_source_version(
        _observation(
            id="sum-plan",
            kind=ObservationKind.SPENDING,
            subject_id="household",
            value=60_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={"drift_role": "baseline", "drift_aggregation": "exact"},
            effective_date=NOW.date() - timedelta(days=1),
        ),
        source_id="shared-sum-source",
        source_uri="source://sum/earlier",
        content_sha256="3" * 64,
        observed_at=earlier_source_at,
    )
    first = _with_source_version(
        _observation(
            id="sum-current-shared",
            kind=ObservationKind.SPENDING,
            subject_id="household",
            value=20_000,
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            fields={
                "drift_role": "observed",
                "drift_aggregation": "sum",
                "drift_group_id": "annual-spending",
            },
        ),
        source_id="shared-sum-source",
        source_uri="source://sum/later",
        content_sha256="4" * 64,
        observed_at=later_source_at,
    )
    second = _observation(
        id="sum-current-independent",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=41_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={
            "drift_role": "observed",
            "drift_aggregation": "sum",
            "drift_group_id": "annual-spending",
        },
    )
    event = _scalar_event(
        detect_spending_drift,
        subject_id="household",
        baseline=60_000,
        observed=61_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=(baseline.id, first.id, second.id),
    )

    if valid:
        assert _manifest((baseline, first, second), event).drift_events == (event,)
    else:
        with pytest.raises(ValidationError, match="shared source versions require"):
            _manifest((baseline, first, second), event)


def test_staleness_drift_requires_one_exact_freshness_observation() -> None:
    observation = _observation(
        id="freshness",
        kind=ObservationKind.ACCOUNT_FRESHNESS,
        subject_id="account-1",
        value=2,
        unit=ObservationUnit.DAYS,
    )
    valid = detect_staleness(
        StalenessInput(
            subject_id="account-1",
            evaluated_at=NOW,
            last_reconciled_at=NOW - timedelta(days=2),
            stale_after_days=45,
            frozen_after_days=90,
            threshold=STALENESS_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(observation.id,),
        )
    )
    duplicate = _observation(
        id="freshness-duplicate",
        kind=ObservationKind.ACCOUNT_FRESHNESS,
        subject_id="account-1",
        value=2,
        unit=ObservationUnit.DAYS,
    )
    ambiguous = detect_staleness(
        StalenessInput(
            subject_id="account-1",
            evaluated_at=NOW,
            last_reconciled_at=NOW - timedelta(days=2),
            stale_after_days=45,
            frozen_after_days=90,
            threshold=STALENESS_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(observation.id, duplicate.id),
        )
    )

    assert _manifest((observation,), valid).drift_events == (valid,)
    with pytest.raises(ValidationError, match="duplicate payload/group evidence"):
        _manifest((observation, duplicate), ambiguous)


def test_rehashed_fabricated_derivation_cannot_hide_behind_valid_evidence() -> None:
    baseline = _observation(
        id="balance-baseline",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="portfolio",
        value=100_000,
        unit=ObservationUnit.DOLLARS,
        fields={"drift_role": "baseline"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    observation = _observation(
        id="balance",
        kind=ObservationKind.ACCOUNT_BALANCE,
        subject_id="portfolio",
        value=125_000,
        unit=ObservationUnit.DOLLARS,
        fields={"drift_role": "current"},
    )
    valid = _scalar_event(
        detect_balance_drift,
        subject_id="portfolio",
        baseline=100_000,
        observed=125_000,
        unit=ObservationUnit.DOLLARS,
        observation_ids=(baseline.id, observation.id),
    )
    material = {
        **valid.model_dump(mode="python", exclude={"content_sha256"}),
        "signed_delta": 900_000,
        "absolute_delta": 900_000,
        "material": True,
    }
    fabricated = DriftEvent.model_validate(
        {
            **material,
            "content_sha256": canonical_content_sha256(material),
        }
    )

    with pytest.raises(ValidationError, match="derivation must match"):
        _manifest((baseline, observation), fabricated)


@pytest.mark.parametrize(
    ("kind", "detector", "unit", "current_role"),
    (
        (
            ObservationKind.SPENDING,
            detect_spending_drift,
            ObservationUnit.DOLLARS_PER_YEAR,
            "observed",
        ),
        (
            ObservationKind.CONTRIBUTION,
            detect_contribution_drift,
            ObservationUnit.DOLLARS_PER_YEAR,
            "observed",
        ),
        (
            ObservationKind.ACCOUNT_BALANCE,
            detect_balance_drift,
            ObservationUnit.DOLLARS,
            "current",
        ),
    ),
)
def test_scalar_drift_rejects_self_declared_baselines(
    kind: ObservationKind,
    detector: Callable[[ScalarDriftInput], DriftEvent],
    unit: ObservationUnit,
    current_role: str,
) -> None:
    subject_id = "portfolio" if kind is ObservationKind.ACCOUNT_BALANCE else "subject-1"
    aggregation = (
        {"drift_aggregation": "exact"} if kind is not ObservationKind.ACCOUNT_BALANCE else {}
    )
    baseline = _observation(
        id=f"{kind.value}-real-baseline",
        kind=kind,
        subject_id=subject_id,
        value=100_000,
        unit=unit,
        fields={**aggregation, "drift_role": "baseline"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    current = _observation(
        id=f"{kind.value}-current",
        kind=kind,
        subject_id=subject_id,
        value=125_000,
        unit=unit,
        fields={**aggregation, "drift_role": current_role},
    )
    fabricated = _scalar_event(
        detector,
        subject_id=subject_id,
        baseline=99_999,
        observed=125_000,
        unit=unit,
        observation_ids=(baseline.id, current.id),
    )

    with pytest.raises(ValidationError, match="plan baseline must equal"):
        _manifest((baseline, current), fabricated)


def test_external_baseline_role_cannot_replace_typed_plan_selector() -> None:
    baseline = CalibrationObservation.create(
        id="untyped-spending-baseline",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        effective_date=NOW.date() - timedelta(days=1),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            value=60_000,
            fields=FrozenJsonObject({"drift_role": "baseline", "drift_aggregation": "exact"}),
        ),
        source_links=(
            SourceObservationLink(
                source_kind=ObservationSourceKind.USER_REVIEWED,
                source_id="external-baseline",
                source_uri="review://external-baseline",
                content_sha256="e" * 64,
                observed_at=NOW,
            ),
        ),
    )
    observed = _observation(
        id="observed-spending",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=61_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={"drift_role": "observed", "drift_aggregation": "exact"},
    )
    event = _scalar_event(
        detect_spending_drift,
        subject_id="household",
        baseline=60_000,
        observed=61_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=(baseline.id, observed.id),
    )

    with pytest.raises(
        ValidationError,
        match=r"must select resolved_scenario\.annual_spending",
    ):
        _manifest((baseline, observed), event)


def test_allocation_drift_rejects_self_declared_target() -> None:
    target = _observation(
        id="allocation-real-target",
        kind=ObservationKind.ASSET_ALLOCATION,
        subject_id="portfolio",
        value={"stocks": 0.7, "bonds": 0.3},
        unit=ObservationUnit.FRACTION,
        fields={"drift_role": "target"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    current = _observation(
        id="allocation-real-current",
        kind=ObservationKind.ASSET_ALLOCATION,
        subject_id="portfolio",
        value={"stocks": 0.8, "bonds": 0.2},
        unit=ObservationUnit.FRACTION,
        fields={"drift_role": "current"},
    )
    fabricated = detect_allocation_drift(
        AllocationDriftInput(
            subject_id="portfolio",
            target_weights=FrozenFloatMap({"stocks": 0.6, "bonds": 0.4}),
            observed_weights=FrozenFloatMap({"stocks": 0.8, "bonds": 0.2}),
            threshold=ALLOCATION_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=(target.id, current.id),
        )
    )

    with pytest.raises(ValidationError, match="typed target-allocation field"):
        _manifest((target, current), fabricated)


def test_non_lossless_exact_money_never_aliases_through_float() -> None:
    with pytest.raises(ValidationError, match="cannot be represented exactly"):
        ScalarDriftInput(
            subject_id="account-1",
            baseline_value=2**53 + 1,
            observed_value=2**53,
            unit=ObservationUnit.DOLLARS,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=("baseline", "current"),
        )
    with pytest.raises(ValidationError, match="cannot be represented exactly"):
        DebtPayoffInput(
            subject_id="mortgage",
            previous_liability=2**53 + 1,
            current_liability=2**53,
            threshold=SCALAR_THRESHOLD,
            confidence=CONFIDENCE,
            observation_ids=("previous", "current"),
        )


def test_distinct_observation_ids_cannot_reuse_upstream_evidence() -> None:
    first = _observation(
        id="first",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=10_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={"drift_role": "baseline", "drift_aggregation": "exact"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    second = CalibrationObservation.create(
        id="second",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        effective_date=NOW.date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            value=11_000,
            fields=FrozenJsonObject({"drift_role": "observed", "drift_aggregation": "exact"}),
        ),
        source_links=first.source_links,
    )

    with pytest.raises(ValidationError, match="reuse an upstream source link"):
        _manifest((first, second), None)


def test_uri_variants_cannot_split_one_exact_upstream_fact_across_sum_rows() -> None:
    first = _observation(
        id="split-a",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=20_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={
            "drift_role": "observed",
            "drift_aggregation": "sum",
            "drift_group_id": "annual-spending",
        },
    )
    reused = first.source_links[0].model_copy(
        update={"source_uri": "ynab://presentation-alias/same-upstream-fact"}
    )
    second = CalibrationObservation.create(
        id="split-b",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        effective_date=NOW.date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            value=40_000,
            fields=FrozenJsonObject(
                {
                    "drift_role": "observed",
                    "drift_aggregation": "sum",
                    "drift_group_id": "annual-spending",
                }
            ),
        ),
        source_links=(reused,),
    )

    with pytest.raises(ValidationError, match="reuse an upstream source link"):
        _manifest((first, second), None)


def test_sum_side_cannot_reuse_stable_source_across_versions() -> None:
    baseline = _observation(
        id="spending-plan",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=58_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={"drift_role": "baseline", "drift_aggregation": "exact"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    first = _observation(
        id="version-a",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=20_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={
            "drift_role": "observed",
            "drift_aggregation": "sum",
            "drift_group_id": "annual-spending",
        },
    )
    versioned_link = first.source_links[0].model_copy(
        update={
            "content_sha256": "c" * 64,
            "observed_at": NOW - timedelta(minutes=1),
            "source_uri": "ynab://presentation-alias/new-version",
        }
    )
    second = CalibrationObservation.create(
        id="version-b",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        effective_date=NOW.date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            value=40_000,
            fields=FrozenJsonObject(
                {
                    "drift_role": "observed",
                    "drift_aggregation": "sum",
                    "drift_group_id": "annual-spending",
                }
            ),
        ),
        source_links=(versioned_link,),
    )
    event = _scalar_event(
        detect_spending_drift,
        subject_id="household",
        baseline=58_000,
        observed=60_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=(baseline.id, first.id, second.id),
    )

    with pytest.raises(ValidationError, match="cannot reuse a stable source"):
        _manifest((baseline, first, second), event)


def test_stable_source_can_appear_in_distinct_chronological_roles() -> None:
    raw_baseline = _observation(
        id="plan-baseline",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=58_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={"drift_role": "baseline", "drift_aggregation": "exact"},
        effective_date=NOW.date() - timedelta(days=1),
    )
    baseline = CalibrationObservation.create(
        id=raw_baseline.id,
        kind=raw_baseline.kind,
        subject_id=raw_baseline.subject_id,
        effective_date=raw_baseline.effective_date,
        observed_at=raw_baseline.observed_at,
        payload=raw_baseline.payload,
        source_links=(
            raw_baseline.source_links[0].model_copy(
                update={"observed_at": NOW - timedelta(minutes=2)}
            ),
        ),
    )
    next_version = baseline.source_links[0].model_copy(
        update={
            "content_sha256": "d" * 64,
            "observed_at": NOW - timedelta(minutes=1),
            "source_uri": "ynab://presentation-alias/current-version",
        }
    )
    observed = CalibrationObservation.create(
        id="plan-current",
        kind=ObservationKind.SPENDING,
        subject_id="household",
        effective_date=NOW.date(),
        observed_at=NOW,
        payload=ObservationPayload(
            unit=ObservationUnit.DOLLARS_PER_YEAR,
            value=60_000,
            fields=FrozenJsonObject({"drift_role": "observed", "drift_aggregation": "exact"}),
        ),
        source_links=(next_version,),
    )
    event = _scalar_event(
        detect_spending_drift,
        subject_id="household",
        baseline=58_000,
        observed=60_000,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        observation_ids=(baseline.id, observed.id),
    )

    assert _manifest((baseline, observed), event).drift_events == (event,)
