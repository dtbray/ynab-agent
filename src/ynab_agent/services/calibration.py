"""Continuous wealth-plan calibration orchestration and durable contracts."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import logging
from collections.abc import Callable
from typing import Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ynab_agent.planning.models import LIQUID_PORTFOLIO_ROLES, WealthScenario
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
    CalibrationObservation,
    CalibrationPolicy,
    CalibrationSnapshot,
    CalibrationSnapshotManifest,
    ConfidenceAssessment,
    DriftEvent,
    DriftKind,
    FreshnessAssessment,
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
from ynab_agent.services.scenario_comparison import ScenarioRevision
from ynab_agent.services.calibration_attribution import AttributionReport


CALIBRATION_NAMESPACE = UUID("56820f95-d89f-4c17-87ad-2a68f0b7f98d")
MAX_REVIEWED_ALLOCATIONS = 64
logger = logging.getLogger(__name__)


def default_calibration_policy() -> CalibrationPolicy:
    """Conservative defaults that suppress immaterial forecast churn."""
    return CalibrationPolicy(
        spending=MaterialityThreshold(absolute=1_000, relative=0.05),
        contribution=MaterialityThreshold(absolute=500, relative=0.05),
        debt_payoff=MaterialityThreshold(absolute=100, relative=0.01),
        balance=MaterialityThreshold(absolute=2_500, relative=0.05),
        allocation=MaterialityThreshold(absolute=0.05),
        staleness=MaterialityThreshold(absolute=45),
        stale_after_days=45,
        frozen_after_days=90,
        lookback_months=12,
    )


class ReviewedAllocation(BaseModel):
    """User-reviewed current allocation, copied without account display names."""

    model_config = ConfigDict(frozen=True)

    account_id: str = Field(min_length=1, max_length=128)
    weights: FrozenFloatMap
    reviewed_at: datetime
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_weights(self) -> ReviewedAllocation:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("allocation reviewed_at must include a timezone")
        if abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("reviewed allocation weights must sum to one")
        return self


class CalibrationProfile(BaseModel):
    """Immutable configuration binding one saved scenario to one YNAB budget."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=36, max_length=36)
    scenario_revision_id: str = Field(min_length=36, max_length=36)
    budget_id: str = Field(min_length=1, max_length=64)
    policy: CalibrationPolicy = Field(default_factory=default_calibration_policy)
    reviewed_allocations: tuple[ReviewedAllocation, ...] = Field(
        default=(),
        max_length=MAX_REVIEWED_ALLOCATIONS,
    )
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime

    @model_validator(mode="after")
    def verify_profile(self) -> CalibrationProfile:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("profile created_at must include a timezone")
        account_ids = tuple(row.account_id for row in self.reviewed_allocations)
        if len(account_ids) != len(set(account_ids)):
            raise ValueError("reviewed allocation account IDs must be unique")
        expected = canonical_content_sha256(
            self,
            exclude=frozenset({"profile_sha256"}),
        )
        if expected != self.profile_sha256:
            raise ValueError("calibration profile content hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        scenario_revision_id: str,
        budget_id: str,
        policy: CalibrationPolicy,
        reviewed_allocations: tuple[ReviewedAllocation, ...] = (),
        created_at: datetime,
    ) -> CalibrationProfile:
        material: dict[str, object] = {
            "id": id,
            "scenario_revision_id": scenario_revision_id,
            "budget_id": budget_id,
            "policy": policy,
            "reviewed_allocations": tuple(
                sorted(reviewed_allocations, key=lambda row: row.account_id)
            ),
            "created_at": created_at,
        }
        return cls(
            id=id,
            scenario_revision_id=scenario_revision_id,
            budget_id=budget_id,
            policy=policy,
            reviewed_allocations=tuple(
                sorted(reviewed_allocations, key=lambda row: row.account_id)
            ),
            created_at=created_at,
            profile_sha256=canonical_content_sha256(material),
        )


class CalibrationSourceAccount(BaseModel):
    """Privacy-minimized cache facts needed for one account calibration."""

    model_config = ConfigDict(frozen=True)

    account_id: str = Field(min_length=1, max_length=128)
    balance: float
    debt_balance: float | None = Field(default=None, ge=0)
    latest_activity_at: datetime | None = None
    last_reconciled_at: datetime | None = None
    latest_reviewed_at: datetime | None = None
    unchanged_since: datetime | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationSourceState(BaseModel):
    """Bounded, signed aggregate data selected from one completed sync batch."""

    model_config = ConfigDict(frozen=True)

    budget_id: str
    source_sync_batch_id: str
    observed_at: datetime
    annual_spending: float = Field(ge=0)
    annual_contribution: float = Field(ge=0)
    spending_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accounts: tuple[CalibrationSourceAccount, ...]
    watermarks: tuple[SourceWatermark, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_state(self) -> CalibrationSourceState:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("source observed_at must include a timezone")
        if any(row.account_id == "" for row in self.accounts):
            raise ValueError("source account identity is invalid")
        if len({row.account_id for row in self.accounts}) != len(self.accounts):
            raise ValueError("source account identities must be unique")
        if any(
            watermark.change_batch_id != self.source_sync_batch_id for watermark in self.watermarks
        ):
            raise ValueError("source watermarks must match the sync batch")
        return self


class CalibrationRunState(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CalibrationProfileState(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class CalibrationRun(BaseModel):
    """Durable execution identity; result content is separately verified."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=36, max_length=36)
    snapshot_id: str = Field(min_length=36, max_length=36)
    state: CalibrationRunState
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    result: dict[str, object] | None = None
    error_code: str | None = Field(default=None, max_length=64)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AlertState(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class CalibrationAlert(BaseModel):
    """Privacy-bounded alert representation safe for API and notifications."""

    model_config = ConfigDict(frozen=True)

    id: str
    profile_id: str
    drift_kind: DriftKind
    subject_id: str
    state: AlertState
    created_at: datetime
    state_changed_at: datetime


class CalibrationCaptureFailure(BaseModel):
    """Privacy-bounded durable evidence that one post-sync capture failed."""

    model_config = ConfigDict(frozen=True)

    id: str
    profile_id: str
    source_sync_batch_id: str
    error_code: str
    created_at: datetime


class CalibrationStatus(BaseModel):
    """Interface-neutral public status shared by CLI and HTTP."""

    model_config = ConfigDict(frozen=True)

    profile: CalibrationProfile
    profile_state: CalibrationProfileState = CalibrationProfileState.ACTIVE
    runs: tuple[CalibrationRun, ...]
    alerts: tuple[CalibrationAlert, ...]
    capture_failures: tuple[CalibrationCaptureFailure, ...] = ()


class CalibrationRepository(Protocol):
    async def create_profile(self, profile: CalibrationProfile) -> None: ...
    async def get_profile(self, profile_id: str) -> CalibrationProfile | None: ...
    async def list_profiles(
        self, budget_id: str | None = None
    ) -> tuple[CalibrationProfile, ...]: ...
    async def list_active_profiles(
        self,
        budget_id: str,
    ) -> tuple[CalibrationProfile, ...]: ...
    async def set_profile_state(
        self,
        profile_id: str,
        state: CalibrationProfileState,
        *,
        at: datetime,
    ) -> bool: ...
    async def get_profile_state(
        self,
        profile_id: str,
    ) -> CalibrationProfileState | None: ...
    async def record_completed_sync_batch(
        self,
        budget_id: str,
        source_sync_batch_id: str,
        *,
        completed_at: datetime,
    ) -> None: ...
    async def begin_sync_batch(
        self,
        budget_id: str,
        source_sync_batch_id: str,
        *,
        started_at: datetime,
    ) -> None: ...
    async def get_scenario_revision(self, revision_id: str) -> ScenarioRevision | None: ...
    async def load_source_state(
        self,
        profile: CalibrationProfile,
        scenario: WealthScenario,
        *,
        source_sync_batch_id: str,
        as_of: datetime,
    ) -> CalibrationSourceState: ...
    async def get_snapshot_for_batch(
        self,
        profile_id: str,
        source_sync_batch_id: str,
    ) -> CalibrationSnapshot | None: ...
    async def get_snapshot(self, snapshot_id: str) -> CalibrationSnapshot | None: ...
    async def prepare_run_recovery(self) -> None: ...
    async def get_latest_snapshot(self, profile_id: str) -> CalibrationSnapshot | None: ...
    async def create_snapshot(self, snapshot: CalibrationSnapshot) -> None: ...
    async def create_run(self, run: CalibrationRun) -> CalibrationRun: ...
    async def get_run(self, run_id: str) -> CalibrationRun | None: ...
    async def list_runs(
        self,
        *,
        state: CalibrationRunState | None = None,
        limit: int = 100,
    ) -> tuple[CalibrationRun, ...]: ...
    async def list_profile_runs(
        self,
        profile_id: str,
        *,
        limit: int = 100,
    ) -> tuple[CalibrationRun, ...]: ...
    async def record_capture_failure(
        self,
        failure: CalibrationCaptureFailure,
    ) -> None: ...
    async def list_capture_failures(
        self,
        profile_id: str,
        *,
        limit: int = 100,
    ) -> tuple[CalibrationCaptureFailure, ...]: ...
    async def mark_run_running(self, run_id: str, at: datetime) -> CalibrationRun | None: ...
    async def mark_run_succeeded(
        self,
        run_id: str,
        *,
        result: dict[str, object],
        result_sha256: str,
        completed_at: datetime,
    ) -> CalibrationRun | None: ...
    async def mark_run_failed(
        self,
        run_id: str,
        *,
        error_code: str,
        completed_at: datetime,
    ) -> CalibrationRun | None: ...
    async def store_attribution(
        self,
        run_id: str,
        report: AttributionReport,
        *,
        created_at: datetime,
    ) -> None: ...
    async def reconcile_alerts(
        self,
        profile_id: str,
        snapshot: CalibrationSnapshot,
        *,
        at: datetime,
    ) -> tuple[CalibrationAlert, ...]: ...
    async def list_alerts(self, profile_id: str) -> tuple[CalibrationAlert, ...]: ...
    async def transition_alert(
        self,
        alert_id: str,
        profile_id: str,
        snapshot_id: str,
        state: AlertState,
        *,
        at: datetime,
    ) -> CalibrationAlert | None: ...


class CalibrationService:
    """Capture immutable source state and enqueue exactly one run per sync batch."""

    def __init__(
        self,
        repository: CalibrationRepository,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def create_profile(
        self,
        *,
        scenario_revision_id: str,
        budget_id: str,
        policy: CalibrationPolicy | None = None,
        reviewed_allocations: tuple[ReviewedAllocation, ...] = (),
    ) -> CalibrationProfile:
        revision = await self.repository.get_scenario_revision(scenario_revision_id)
        if revision is None:
            raise ValueError("scenario revision was not found")
        scenario = revision.manifest.scenario
        configured = {row.account_id for row in reviewed_allocations}
        expected = (
            {row.account_id for row in scenario.portfolio_allocation.accounts}
            if scenario.portfolio_allocation is not None
            else set()
        )
        if configured != expected:
            raise ValueError("reviewed allocations must exactly cover allocation accounts")
        created_at = self._clock()
        profile_id = str(
            uuid5(
                CALIBRATION_NAMESPACE,
                f"profile:{scenario_revision_id}:{budget_id}:{created_at.isoformat()}",
            )
        )
        profile = CalibrationProfile.create(
            id=profile_id,
            scenario_revision_id=scenario_revision_id,
            budget_id=budget_id,
            policy=policy or default_calibration_policy(),
            reviewed_allocations=reviewed_allocations,
            created_at=created_at,
        )
        await self.repository.create_profile(profile)
        return profile

    async def disable_profile(self, profile_id: str) -> bool:
        """Append a disabled state so obsolete plans stop running after sync."""
        profile = await self.repository.get_profile(profile_id)
        if profile is None:
            return False
        return await self.repository.set_profile_state(
            profile_id,
            CalibrationProfileState.DISABLED,
            at=max(
                self._clock(),
                profile.created_at + timedelta(microseconds=1),
            ),
        )

    async def capture_after_sync(
        self,
        profile_id: str,
        *,
        source_sync_batch_id: str,
    ) -> tuple[CalibrationSnapshot, CalibrationRun, bool]:
        existing = await self.repository.get_snapshot_for_batch(
            profile_id,
            source_sync_batch_id,
        )
        if existing is not None:
            latest = await self.repository.get_latest_snapshot(
                existing.manifest.profile_id
            )
            if latest is not None and latest.id == existing.id:
                await self.repository.reconcile_alerts(
                    existing.manifest.profile_id,
                    existing,
                    at=self._clock(),
                )
            run = await self.repository.create_run(_accepted_run(existing.id, existing.created_at))
            return existing, run, False
        profile = await self._require_profile(profile_id)
        revision = await self.repository.get_scenario_revision(profile.scenario_revision_id)
        if revision is None:
            raise ValueError("calibration scenario revision was not found")
        now = self._clock()
        source = await self.repository.load_source_state(
            profile,
            revision.manifest.scenario,
            source_sync_batch_id=source_sync_batch_id,
            as_of=now,
        )
        previous = await self.repository.get_latest_snapshot(profile.id)
        snapshot = build_calibration_snapshot(
            profile=profile,
            revision=revision,
            source=source,
            previous=previous,
            created_at=now,
        )
        await self.repository.create_snapshot(snapshot)
        await self.repository.reconcile_alerts(profile.id, snapshot, at=now)
        run = await self.repository.create_run(_accepted_run(snapshot.id, now))
        return snapshot, run, True

    async def capture_enabled_after_sync(
        self,
        *,
        budget_id: str,
        source_sync_batch_id: str,
    ) -> tuple[CalibrationRun, ...]:
        runs: list[CalibrationRun] = []
        for profile in await self.repository.list_active_profiles(budget_id):
            try:
                _, run, _ = await self.capture_after_sync(
                    profile.id,
                    source_sync_batch_id=source_sync_batch_id,
                )
                runs.append(run)
            except Exception as error:
                error_code = (
                    "invalid_source"
                    if isinstance(error, ValueError)
                    else "capture_failed"
                )
                await self.repository.record_capture_failure(
                    CalibrationCaptureFailure(
                        id=str(
                            uuid5(
                                CALIBRATION_NAMESPACE,
                                f"capture-failure:{profile.id}:{source_sync_batch_id}",
                            )
                        ),
                        profile_id=profile.id,
                        source_sync_batch_id=source_sync_batch_id,
                        error_code=error_code,
                        created_at=self._clock(),
                    )
                )
                logger.exception(
                    "calibration capture failed for profile %s after sync batch %s",
                    profile.id,
                    source_sync_batch_id,
                )
        return tuple(runs)

    async def after_sync(
        self,
        *,
        budget_id: str,
        source_sync_batch_id: str,
    ) -> None:
        """Sync-service hook: snapshot before any resulting run is admitted."""
        await self.capture_enabled_after_sync(
            budget_id=budget_id,
            source_sync_batch_id=source_sync_batch_id,
        )

    async def _require_profile(self, profile_id: str) -> CalibrationProfile:
        profile = await self.repository.get_profile(profile_id)
        if profile is None:
            raise ValueError("calibration profile was not found")
        return profile


def _accepted_run(snapshot_id: str, at: datetime) -> CalibrationRun:
    return CalibrationRun(
        id=str(uuid5(CALIBRATION_NAMESPACE, f"run:{snapshot_id}")),
        snapshot_id=snapshot_id,
        state=CalibrationRunState.ACCEPTED,
        created_at=at,
    )


def build_calibration_snapshot(
    *,
    profile: CalibrationProfile,
    revision: ScenarioRevision,
    source: CalibrationSourceState,
    previous: CalibrationSnapshot | None,
    created_at: datetime,
) -> CalibrationSnapshot:
    """Build and replay-validate one immutable pre-run snapshot."""
    scenario = revision.manifest.scenario
    observations: list[CalibrationObservation] = []
    events: list[DriftEvent] = []
    confidence = ConfidenceAssessment(
        score=1,
        basis="signed cache rows and immutable saved-plan inputs",
    )
    contribution_confidence = ConfidenceAssessment(
        score=0.75,
        basis=(
            "positive external transfers into selected liquid accounts, excluding "
            "internal transfers and non-transfer valuation activity"
        ),
        limitations=(
            "direct non-transfer contributions cannot be distinguished safely from "
            "income, reconciliation, or market-value adjustments",
        ),
    )
    baseline_at = min(revision.created_at, source.observed_at - timedelta(microseconds=1))
    baseline_date = min(
        baseline_at.astimezone(timezone.utc).date(),
        source.observed_at.astimezone(timezone.utc).date() - timedelta(days=1),
    )

    spending_baseline = _plan_observation(
        profile,
        revision,
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=scenario.annual_spending,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        selector="annual_spending",
        role="baseline",
        effective_date=baseline_date,
        observed_at=source.observed_at,
        source=source,
    )
    spending_current = _cache_observation(
        kind=ObservationKind.SPENDING,
        subject_id="household",
        value=source.annual_spending,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={
            "drift_role": "observed",
            "drift_aggregation": "exact",
        },
        source_id=f"spending:{profile.budget_id}:{profile.policy.lookback_months}",
        source_sha256=source.spending_sha256,
        source=source,
    )
    observations.extend((spending_baseline, spending_current))
    events.append(
        detect_spending_drift(
            ScalarDriftInput(
                subject_id="household",
                baseline_value=scenario.annual_spending,
                observed_value=source.annual_spending,
                unit=ObservationUnit.DOLLARS_PER_YEAR,
                threshold=profile.policy.spending,
                confidence=confidence,
                observation_ids=(spending_baseline.id, spending_current.id),
            )
        )
    )

    contribution_baseline = _plan_observation(
        profile,
        revision,
        kind=ObservationKind.CONTRIBUTION,
        subject_id="portfolio",
        value=scenario.annual_contribution,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        selector="annual_contribution",
        role="baseline",
        effective_date=baseline_date,
        observed_at=source.observed_at,
        source=source,
    )
    contribution_current = _cache_observation(
        kind=ObservationKind.CONTRIBUTION,
        subject_id="portfolio",
        value=source.annual_contribution,
        unit=ObservationUnit.DOLLARS_PER_YEAR,
        fields={
            "drift_role": "observed",
            "drift_aggregation": "exact",
        },
        source_id=f"contribution:{profile.budget_id}:{profile.policy.lookback_months}",
        source_sha256=source.contribution_sha256,
        source=source,
    )
    observations.extend((contribution_baseline, contribution_current))
    events.append(
        detect_contribution_drift(
            ScalarDriftInput(
                subject_id="portfolio",
                baseline_value=scenario.annual_contribution,
                observed_value=source.annual_contribution,
                unit=ObservationUnit.DOLLARS_PER_YEAR,
                threshold=profile.policy.contribution,
                confidence=contribution_confidence,
                observation_ids=(contribution_baseline.id, contribution_current.id),
            )
        )
    )

    tax_balance = {
        bucket.account_id: bucket.starting_balance
        for bucket in scenario.tax_buckets
        if bucket.account_id is not None
    }
    source_accounts = {row.account_id: row for row in source.accounts}
    liquid_ids = {
        account.id for account in scenario.accounts if account.role in LIQUID_PORTFOLIO_ROLES
    }
    if not tax_balance and scenario.starting_portfolio is not None and liquid_ids:
        portfolio_freshness = max(
            (
                _freshness(
                    source_accounts[account_id],
                    profile.policy,
                    source.observed_at,
                )
                for account_id in liquid_ids
                if account_id in source_accounts
            ),
            key=lambda row: (
                {"fresh": 0, "stale": 1, "frozen": 2, "unknown": 3}[
                    row.state.value
                ],
                row.unchanged_days or -1,
                row.age_days or -1,
            ),
        )
        actual_portfolio = max(
            0,
            sum(
                source_accounts[account_id].balance
                for account_id in liquid_ids
                if account_id in source_accounts
            ),
        )
        baseline_observation = _plan_observation(
            profile,
            revision,
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="portfolio",
            value=revision.manifest.starting_portfolio,
            unit=ObservationUnit.DOLLARS,
            selector="starting_portfolio",
            role="baseline",
            effective_date=baseline_date,
            observed_at=source.observed_at,
            source=source,
            freshness=portfolio_freshness,
        )
        current_observation = _cache_observation(
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id="portfolio",
            value=actual_portfolio,
            unit=ObservationUnit.DOLLARS,
            fields={
                "drift_role": "current",
                "account_balances": {
                    account_id: source_accounts[account_id].balance
                    for account_id in sorted(liquid_ids)
                    if account_id in source_accounts
                },
            },
            source_id=f"portfolio-balance:{profile.budget_id}",
            source_sha256=canonical_content_sha256(
                {
                    account_id: source_accounts[account_id].content_sha256
                    for account_id in sorted(liquid_ids)
                    if account_id in source_accounts
                }
            ),
            source=source,
            freshness=portfolio_freshness,
        )
        observations.extend((baseline_observation, current_observation))
        portfolio_balance_event = detect_balance_drift(
            ScalarDriftInput(
                subject_id="portfolio",
                baseline_value=revision.manifest.starting_portfolio,
                observed_value=actual_portfolio,
                unit=ObservationUnit.DOLLARS,
                threshold=profile.policy.balance,
                confidence=confidence,
                observation_ids=(
                    baseline_observation.id,
                    current_observation.id,
                ),
            )
        )
        events.append(
            _event_with_details(
                portfolio_balance_event,
                {
                    "account_balances": {
                        account_id: source_accounts[account_id].balance
                        for account_id in sorted(liquid_ids)
                        if account_id in source_accounts
                    }
                },
            )
        )
    for account_id in sorted(liquid_ids):
        row = source_accounts.get(account_id)
        baseline = tax_balance.get(account_id)
        if row is None or baseline is None:
            continue
        baseline_observation = _plan_observation(
            profile,
            revision,
            kind=ObservationKind.ACCOUNT_BALANCE,
            subject_id=account_id,
            value=baseline,
            unit=ObservationUnit.DOLLARS,
            selector=f"tax_buckets[account_id={account_id}].starting_balance",
            role="baseline",
            effective_date=baseline_date,
            observed_at=source.observed_at,
            source=source,
            freshness=_freshness(row, profile.policy, source.observed_at),
        )
        current_observation = _account_observation(
            profile,
            row,
            source,
            kind=ObservationKind.ACCOUNT_BALANCE,
            value=row.balance,
            unit=ObservationUnit.DOLLARS,
            role="current",
        )
        observations.extend((baseline_observation, current_observation))
        events.append(
            detect_balance_drift(
                ScalarDriftInput(
                    subject_id=account_id,
                    baseline_value=baseline,
                    observed_value=row.balance,
                    unit=ObservationUnit.DOLLARS,
                    threshold=profile.policy.balance,
                    confidence=confidence,
                    observation_ids=(baseline_observation.id, current_observation.id),
                )
            )
        )

    reviewed = {row.account_id: row for row in profile.reviewed_allocations}
    if scenario.portfolio_allocation is not None:
        for account in scenario.portfolio_allocation.accounts:
            reviewed_row = reviewed.get(account.account_id)
            source_account = source_accounts.get(account.account_id)
            if reviewed_row is None or source_account is None:
                continue
            allocation_source_account = source_account.model_copy(
                update={
                    "latest_reviewed_at": max(
                        value
                        for value in (
                            source_account.latest_reviewed_at,
                            reviewed_row.reviewed_at,
                        )
                        if value is not None
                    )
                }
            )
            target_observation = _plan_observation(
                profile,
                revision,
                kind=ObservationKind.ASSET_ALLOCATION,
                subject_id=account.account_id,
                value=FrozenJsonObject(account.target.model_dump(mode="python")),
                unit=ObservationUnit.FRACTION,
                selector=(f"portfolio_allocation.accounts[account_id={account.account_id}].target"),
                role="target",
                effective_date=baseline_date,
                observed_at=source.observed_at,
                source=source,
                freshness=_freshness(
                    allocation_source_account,
                    profile.policy,
                    source.observed_at,
                ),
            )
            current_observation = CalibrationObservation.create(
                id=_observation_id(
                    profile.id,
                    source.source_sync_batch_id,
                    ObservationKind.ASSET_ALLOCATION,
                    account.account_id,
                    "current",
                ),
                kind=ObservationKind.ASSET_ALLOCATION,
                subject_id=account.account_id,
                effective_date=reviewed_row.reviewed_at.date(),
                observed_at=source.observed_at,
                payload=ObservationPayload(
                    unit=ObservationUnit.FRACTION,
                    value=FrozenJsonObject(reviewed_row.weights),
                    fields=FrozenJsonObject({"drift_role": "current"}),
                ),
                freshness=_freshness(
                    allocation_source_account,
                    profile.policy,
                    source.observed_at,
                ),
                source_links=(
                    SourceObservationLink(
                        source_kind=ObservationSourceKind.USER_REVIEWED,
                        source_id=f"allocation:{profile.id}:{account.account_id}",
                        source_uri=f"calibration-profile://{profile.id}/allocation/{account.account_id}",
                        content_sha256=reviewed_row.source_sha256,
                        observed_at=reviewed_row.reviewed_at,
                    ),
                ),
                source_watermarks=source.watermarks,
            )
            observations.extend((target_observation, current_observation))
            events.append(
                detect_allocation_drift(
                    AllocationDriftInput(
                        subject_id=account.account_id,
                        target_weights=FrozenFloatMap(account.target.model_dump(mode="python")),
                        observed_weights=reviewed_row.weights,
                        threshold=profile.policy.allocation,
                        confidence=confidence,
                        observation_ids=(target_observation.id, current_observation.id),
                    )
                )
            )

    previous_debt = _previous_debt_observations(previous)
    for account_id, row in sorted(source_accounts.items()):
        freshness = _freshness(row, profile.policy, source.observed_at)
        freshness_observation = _account_observation(
            profile,
            row,
            source,
            kind=ObservationKind.ACCOUNT_FRESHNESS,
            value=(
                freshness.unchanged_days
                if freshness.state.value == "frozen"
                else freshness.age_days
            ),
            unit=ObservationUnit.DAYS,
            role="current",
            freshness=freshness,
        )
        observations.append(freshness_observation)
        events.append(
            detect_staleness(
                StalenessInput(
                    subject_id=account_id,
                    evaluated_at=source.observed_at,
                    latest_activity_at=row.latest_activity_at,
                    last_reconciled_at=row.last_reconciled_at,
                    latest_reviewed_at=row.latest_reviewed_at,
                    unchanged_since=row.unchanged_since,
                    stale_after_days=profile.policy.stale_after_days,
                    frozen_after_days=profile.policy.frozen_after_days,
                    threshold=profile.policy.staleness,
                    confidence=confidence,
                    observation_ids=(freshness_observation.id,),
                )
            )
        )
        if row.debt_balance is None:
            continue
        current_observation = _account_observation(
            profile,
            row,
            source,
            kind=ObservationKind.DEBT_BALANCE,
            value=row.debt_balance,
            unit=ObservationUnit.DOLLARS,
            role="current",
        )
        observations.append(current_observation)
        prior = previous_debt.get(account_id)
        if prior is not None and prior.effective_date < current_observation.effective_date:
            copied = prior.model_copy(
                update={
                    "id": _observation_id(
                        profile.id,
                        source.source_sync_batch_id,
                        ObservationKind.DEBT_BALANCE,
                        account_id,
                        "previous",
                    ),
                    "payload": ObservationPayload(
                        unit=ObservationUnit.DOLLARS,
                        value=prior.payload.value,
                        fields=FrozenJsonObject({"drift_role": "previous"}),
                    ),
                }
            )
            # Recreate so the content hash covers the new role and identity.
            copied = CalibrationObservation.create(
                id=copied.id,
                kind=copied.kind,
                subject_id=copied.subject_id,
                effective_date=copied.effective_date,
                observed_at=source.observed_at,
                payload=copied.payload,
                freshness=_freshness(
                    row,
                    profile.policy,
                    source.observed_at,
                ),
                source_links=tuple(
                    link.model_copy(
                        update={
                            "source_kind": ObservationSourceKind.DERIVED_CACHE_QUERY,
                            "source_uri": (
                                f"ynab-cache://{source.budget_id}/prior-debt/{account_id}"
                            ),
                            "change_batch_id": source.source_sync_batch_id,
                        }
                    )
                    for link in copied.source_links
                ),
                source_watermarks=source.watermarks,
            )
            observations.append(copied)
            previous_value = prior.payload.value
            if isinstance(previous_value, bool) or not isinstance(
                previous_value,
                int | float,
            ):
                raise ValueError("prior debt observation must be numeric")
            events.append(
                detect_debt_payoff(
                    DebtPayoffInput(
                        subject_id=account_id,
                        previous_liability=previous_value,
                        current_liability=row.debt_balance,
                        threshold=profile.policy.debt_payoff,
                        confidence=confidence,
                        observation_ids=(copied.id, current_observation.id),
                    )
                )
            )

    resolved = scenario.model_copy(
        update={
            "starting_portfolio": revision.manifest.starting_portfolio,
        }
    )
    manifest = CalibrationSnapshotManifest(
        profile_id=profile.id,
        scenario_revision_id=revision.id,
        scenario_manifest_sha256=revision.manifest_sha256,
        previous_snapshot_id=previous.id if previous is not None else None,
        source_sync_batch_id=source.source_sync_batch_id,
        as_of=source.observed_at,
        resolved_scenario=FrozenWealthScenario(resolved.model_dump(mode="python")),
        resolved_scenario_sha256=canonical_content_sha256(resolved),
        historical_dataset=revision.manifest.historical_dataset,
        policy=profile.policy,
        observations=tuple(observations),
        drift_events=tuple(events),
        source_watermarks=source.watermarks,
    )
    snapshot_id = str(
        uuid5(
            CALIBRATION_NAMESPACE,
            f"snapshot:{profile.id}:{source.source_sync_batch_id}",
        )
    )
    return CalibrationSnapshot(
        id=snapshot_id,
        manifest=manifest,
        manifest_sha256=canonical_content_sha256(manifest),
        created_at=created_at,
    )


def _plan_observation(
    profile: CalibrationProfile,
    revision: ScenarioRevision,
    *,
    kind: ObservationKind,
    subject_id: str,
    value: ImmutableJson,
    unit: ObservationUnit,
    selector: str,
    role: str,
    effective_date: date,
    observed_at: datetime,
    source: CalibrationSourceState,
    freshness: FreshnessAssessment | None = None,
) -> CalibrationObservation:
    return CalibrationObservation.create(
        id=_observation_id(
            profile.id,
            source.source_sync_batch_id,
            kind,
            subject_id,
            role,
        ),
        kind=kind,
        subject_id=subject_id,
        effective_date=effective_date,
        observed_at=observed_at,
        payload=ObservationPayload(
            unit=unit,
            value=value,
            fields=FrozenJsonObject(
                {
                    "drift_role": role,
                    "drift_aggregation": "exact",
                    "plan_field": selector,
                }
            ),
        ),
        freshness=freshness,
        source_links=(
            SourceObservationLink(
                source_kind=ObservationSourceKind.USER_REVIEWED,
                source_id=f"scenario:{revision.id}:{selector}",
                source_uri=f"scenario-revision://{revision.id}/{selector}",
                content_sha256=revision.manifest_sha256,
                observed_at=min(revision.created_at, observed_at),
            ),
        ),
        source_watermarks=source.watermarks,
    )


def _cache_observation(
    *,
    kind: ObservationKind,
    subject_id: str,
    value: ImmutableJson,
    unit: ObservationUnit,
    fields: dict[str, object],
    source_id: str,
    source_sha256: str,
    source: CalibrationSourceState,
    freshness: FreshnessAssessment | None = None,
) -> CalibrationObservation:
    return CalibrationObservation.create(
        id=_observation_id(
            source.budget_id,
            source.source_sync_batch_id,
            kind,
            subject_id,
            str(fields["drift_role"]),
        ),
        kind=kind,
        subject_id=subject_id,
        effective_date=source.observed_at.date(),
        observed_at=source.observed_at,
        payload=ObservationPayload(
            unit=unit,
            value=value,
            fields=FrozenJsonObject(fields),
        ),
        freshness=freshness,
        source_links=(
            SourceObservationLink(
                source_kind=ObservationSourceKind.DERIVED_CACHE_QUERY,
                source_id=source_id,
                source_uri=f"ynab-cache://{source.budget_id}/{source_id}",
                content_sha256=source_sha256,
                observed_at=source.observed_at,
                change_batch_id=source.source_sync_batch_id,
            ),
        ),
        source_watermarks=source.watermarks,
    )


def _account_observation(
    profile: CalibrationProfile,
    row: CalibrationSourceAccount,
    source: CalibrationSourceState,
    *,
    kind: ObservationKind,
    value: ImmutableJson,
    unit: ObservationUnit,
    role: str,
    freshness: FreshnessAssessment | None = None,
) -> CalibrationObservation:
    return _cache_observation(
        kind=kind,
        subject_id=row.account_id,
        value=value,
        unit=unit,
        fields={"drift_role": role},
        source_id=f"account:{row.account_id}:{kind.value}",
        source_sha256=row.content_sha256,
        source=source,
        freshness=(freshness or _freshness(row, profile.policy, source.observed_at)),
    )


def _freshness(
    row: CalibrationSourceAccount,
    policy: CalibrationPolicy,
    evaluated_at: datetime,
) -> FreshnessAssessment:
    return assess_staleness(
        StalenessInput(
            subject_id=row.account_id,
            evaluated_at=evaluated_at,
            latest_activity_at=(
                min(row.latest_activity_at, evaluated_at)
                if row.latest_activity_at is not None
                else None
            ),
            last_reconciled_at=(
                min(row.last_reconciled_at, evaluated_at)
                if row.last_reconciled_at is not None
                else None
            ),
            latest_reviewed_at=(
                min(row.latest_reviewed_at, evaluated_at)
                if row.latest_reviewed_at is not None
                else None
            ),
            unchanged_since=(
                min(row.unchanged_since, evaluated_at) if row.unchanged_since is not None else None
            ),
            stale_after_days=policy.stale_after_days,
            frozen_after_days=policy.frozen_after_days,
            threshold=policy.staleness,
            confidence=ConfidenceAssessment(score=1, basis="signed cache timestamps"),
            observation_ids=("freshness-placeholder",),
        )
    )


def _previous_debt_observations(
    previous: CalibrationSnapshot | None,
) -> dict[str, CalibrationObservation]:
    if previous is None:
        return {}
    return {
        row.subject_id: row
        for row in previous.manifest.observations
        if row.kind is ObservationKind.DEBT_BALANCE
        and row.payload.fields.get("drift_role") == "current"
    }


def _observation_id(
    namespace: str,
    batch_id: str,
    kind: ObservationKind,
    subject_id: str,
    role: str,
) -> str:
    digest = hashlib.sha256(
        f"{namespace}:{batch_id}:{kind.value}:{subject_id}:{role}".encode()
    ).hexdigest()
    return f"obs-{digest}"


def _event_with_details(
    event: DriftEvent,
    details: dict[str, object],
) -> DriftEvent:
    material = event.model_dump(
        mode="python",
        exclude={"content_sha256"},
    )
    material["details"] = {
        **event.details.to_dict(),
        **details,
    }
    return DriftEvent.model_validate(
        {
            **material,
            "content_sha256": canonical_content_sha256(material),
        }
    )
