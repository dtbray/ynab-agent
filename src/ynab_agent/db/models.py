"""SQLAlchemy models for the local YNAB cache."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all YNAB cache tables."""


class Budget(Base):
    __tablename__ = "budgets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    first_month: Mapped[str | None] = mapped_column(String(10))
    last_month: Mapped[str | None] = mapped_column(String(10))
    last_modified_on: Mapped[str | None] = mapped_column(String(64))
    date_format: Mapped[str | None] = mapped_column(String(64))
    currency_format_iso_code: Mapped[str | None] = mapped_column(String(16))
    currency_format_example: Mapped[str | None] = mapped_column(String(32))
    currency_decimal_digits: Mapped[int | None] = mapped_column(BigInteger)
    currency_decimal_separator: Mapped[str | None] = mapped_column(String(8))
    currency_symbol_first: Mapped[bool | None] = mapped_column(Boolean)
    currency_group_separator: Mapped[str | None] = mapped_column(String(8))
    currency_symbol: Mapped[str | None] = mapped_column(String(16))
    currency_display_symbol: Mapped[bool | None] = mapped_column(Boolean)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[str | None] = mapped_column(String(64))
    on_budget: Mapped[bool | None] = mapped_column(Boolean)
    closed: Mapped[bool | None] = mapped_column(Boolean)
    note: Mapped[str | None] = mapped_column(Text)
    balance: Mapped[int | None] = mapped_column(BigInteger)
    cleared_balance: Mapped[int | None] = mapped_column(BigInteger)
    uncleared_balance: Mapped[int | None] = mapped_column(BigInteger)
    transfer_payee_id: Mapped[str | None] = mapped_column(String(64))
    direct_import_linked: Mapped[bool | None] = mapped_column(Boolean)
    direct_import_in_error: Mapped[bool | None] = mapped_column(Boolean)
    last_reconciled_at: Mapped[str | None] = mapped_column(String(64))
    debt_original_balance: Mapped[int | None] = mapped_column(BigInteger)
    debt_interest_rates: Mapped[str | None] = mapped_column(Text)
    debt_minimum_payments: Mapped[str | None] = mapped_column(Text)
    debt_escrow_amounts: Mapped[str | None] = mapped_column(Text)
    change_batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class AccountValuationSnapshotRecord(Base):
    """Dated tracking-account valuation with explicit provenance."""

    __tablename__ = "account_valuation_snapshots"
    __table_args__ = (
        Index(
            "ix_account_valuation_snapshot_lookup",
            "valuation_date",
            "account_id",
            "reviewed",
        ),
    )

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    valuation_date: Mapped[str] = mapped_column(String(10), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    balance_milliunits: Mapped[int] = mapped_column(BigInteger)
    observed_at: Mapped[str] = mapped_column(String(64))
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())


class PlannerJobRecord(Base):
    """Durable retirement-simulation request and result."""

    __tablename__ = "planner_jobs"
    __table_args__ = (
        Index(
            "ix_planner_jobs_state_created",
            "state",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_hash: Mapped[str] = mapped_column(
        String(64),
        index=True,
    )
    deduplication_key: Mapped[str | None] = mapped_column(
        String(64),
        unique=True,
        index=True,
    )
    state: Mapped[str] = mapped_column(String(16), index=True)
    request_json: Mapped[str] = mapped_column(Text)
    manifest_json: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    cancellation_requested: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    required_working_bytes: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[str | None] = mapped_column(String(64))
    completed_at: Mapped[str | None] = mapped_column(String(64))


class SpendingTierMappingRecord(Base):
    """Stable YNAB category identity and retirement-spending semantics."""

    __tablename__ = "spending_tier_mappings"
    __table_args__ = (
        Index(
            "ix_spending_tier_mappings_budget_tier",
            "budget_id",
            "tier",
        ),
    )

    budget_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    category_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    tier: Mapped[str] = mapped_column(String(32))
    essential_floor_milliunits: Mapped[int | None] = mapped_column(
        BigInteger,
    )
    note: Mapped[str | None] = mapped_column(String(500))
    updated_at: Mapped[str] = mapped_column(String(64))


class ScenarioRevisionRecord(Base):
    """Append-only resolved inputs for one named planning scenario."""

    __tablename__ = "scenario_revisions"
    __table_args__ = (
        UniqueConstraint(
            "scenario_name",
            "revision",
            name="uq_scenario_revision_name_number",
        ),
        Index(
            "ix_scenario_revisions_name_created",
            "scenario_name",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scenario_name: Mapped[str] = mapped_column(String(200))
    revision: Mapped[int] = mapped_column(Integer)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    manifest_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationProfileRecord(Base):
    """Immutable configuration for one continuously calibrated plan."""

    __tablename__ = "calibration_profiles"
    __table_args__ = (Index("ix_calibration_profiles_budget_created", "budget_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scenario_revision_id: Mapped[str] = mapped_column(
        ForeignKey("scenario_revisions.id"),
        index=True,
    )
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    profile_json: Mapped[str] = mapped_column(Text)
    profile_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationSyncBatchRecord(Base):
    """Durable lifecycle and proof for one cache synchronization batch."""

    __tablename__ = "calibration_sync_batches"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(16), index=True)
    started_at: Mapped[str] = mapped_column(String(64))
    unowned_source_dirty: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    completed_at: Mapped[str | None] = mapped_column(String(64))
    predecessor_id: Mapped[str | None] = mapped_column(
        ForeignKey("calibration_sync_batches.id")
    )
    superseded_by: Mapped[str | None] = mapped_column(
        ForeignKey("calibration_sync_batches.id")
    )
    watermarks_json: Mapped[str | None] = mapped_column(Text)
    watermarks_sha256: Mapped[str | None] = mapped_column(String(64))


class CalibrationSyncBatchHeadRecord(Base):
    """Authoritative cache-writer generation for one budget."""

    __tablename__ = "calibration_sync_batch_heads"

    budget_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str | None] = mapped_column(
        ForeignKey("calibration_sync_batches.id"),
        unique=True,
    )


class CalibrationProfileEventRecord(Base):
    """Append-only active/disabled lifecycle for a calibration profile."""

    __tablename__ = "calibration_profile_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_profiles.id"),
        index=True,
    )
    state: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationCaptureFailureRecord(Base):
    """Privacy-bounded post-sync capture failure."""

    __tablename__ = "calibration_capture_failures"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "source_sync_batch_id",
            name="uq_calibration_capture_failure_profile_batch",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_profiles.id"),
        index=True,
    )
    source_sync_batch_id: Mapped[str] = mapped_column(String(64), index=True)
    error_code: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationSnapshotRecord(Base):
    """Append-only, content-addressed inputs captured before automation."""

    __tablename__ = "calibration_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "source_sync_batch_id",
            name="uq_calibration_snapshot_profile_batch",
        ),
        Index("ix_calibration_snapshots_profile_created", "profile_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_profiles.id"),
        index=True,
    )
    previous_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("calibration_snapshots.id"),
    )
    source_sync_batch_id: Mapped[str] = mapped_column(String(64), index=True)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    manifest_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationProfileHeadRecord(Base):
    """CAS pointer that serializes one profile's immutable snapshot chain."""

    __tablename__ = "calibration_profile_heads"

    profile_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_profiles.id"),
        primary_key=True,
    )
    snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("calibration_snapshots.id"),
        unique=True,
    )


class CalibrationObservationRecord(Base):
    """Queryable copy of one observation embedded in a snapshot."""

    __tablename__ = "calibration_observations"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "content_sha256",
            name="uq_calibration_observation_snapshot_hash",
        ),
        Index(
            "ix_calibration_observations_subject",
            "kind",
            "subject_id",
            "observed_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_snapshots.id"),
        primary_key=True,
    )
    kind: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str] = mapped_column(String(200))
    content_sha256: Mapped[str] = mapped_column(String(64))
    observation_json: Mapped[str] = mapped_column(Text)
    observed_at: Mapped[str] = mapped_column(String(64))


class CalibrationDriftEventRecord(Base):
    """Queryable immutable drift decision embedded in a snapshot."""

    __tablename__ = "calibration_drift_events"

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_snapshots.id"),
        primary_key=True,
    )
    content_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    subject_id: Mapped[str] = mapped_column(String(200), index=True)
    material: Mapped[bool] = mapped_column(Boolean)
    event_json: Mapped[str] = mapped_column(Text)


class CalibrationRunRecord(Base):
    """Durable idempotent execution linked to an immutable snapshot."""

    __tablename__ = "calibration_runs"
    __table_args__ = (Index("ix_calibration_runs_state_created", "state", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_snapshots.id"),
        unique=True,
        index=True,
    )
    state: Mapped[str] = mapped_column(String(16), index=True)
    result_sha256: Mapped[str | None] = mapped_column(String(64))
    result_json: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[str | None] = mapped_column(String(64))
    completed_at: Mapped[str | None] = mapped_column(String(64))


class CalibrationAttributionRecord(Base):
    """Content-addressed explanation for one completed calibration run."""

    __tablename__ = "calibration_attribution_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_runs.id"),
        unique=True,
        index=True,
    )
    content_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    report_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationAlertRecord(Base):
    """Privacy-bounded immutable alert identity."""

    __tablename__ = "calibration_alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_profiles.id"),
        index=True,
    )
    drift_kind: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str] = mapped_column(String(200))
    fingerprint_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationAlertEventRecord(Base):
    """Append-only lifecycle transition for an alert."""

    __tablename__ = "calibration_alert_events"
    __table_args__ = (
        UniqueConstraint(
            "alert_id",
            "snapshot_id",
            "state",
            name="uq_calibration_alert_event_transition",
        ),
        Index("ix_calibration_alert_events_alert_created", "alert_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    alert_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_alerts.id"),
        index=True,
    )
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_snapshots.id"),
        index=True,
    )
    state: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[str] = mapped_column(String(64))


class CalibrationAlertHeadRecord(Base):
    """CAS pointer to the authoritative event in one alert lifecycle."""

    __tablename__ = "calibration_alert_heads"

    alert_id: Mapped[str] = mapped_column(
        ForeignKey("calibration_alerts.id"),
        primary_key=True,
    )
    event_id: Mapped[str | None] = mapped_column(
        ForeignKey("calibration_alert_events.id"),
        unique=True,
    )


class ScenarioComparisonRecord(Base):
    """Persisted result and recipe for one common-path comparison."""

    __tablename__ = "scenario_comparisons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    baseline_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scenario_revisions.id"),
        index=True,
    )
    alternative_revision_ids_json: Mapped[str] = mapped_column(Text)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    result_sha256: Mapped[str] = mapped_column(String(64))
    manifest_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64))


class Payee(Base):
    __tablename__ = "payees"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    transfer_account_id: Mapped[str | None] = mapped_column(String(64))
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class CategoryGroup(Base):
    __tablename__ = "category_groups"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    category_group_id: Mapped[str | None] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    internal: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    original_category_group_id: Mapped[str | None] = mapped_column(String(64))
    note: Mapped[str | None] = mapped_column(Text)
    budgeted: Mapped[int | None] = mapped_column(BigInteger)
    activity: Mapped[int | None] = mapped_column(BigInteger)
    balance: Mapped[int | None] = mapped_column(BigInteger)
    goal_type: Mapped[str | None] = mapped_column(String(64))
    goal_needs_whole_amount: Mapped[bool | None] = mapped_column(Boolean)
    goal_day: Mapped[int | None] = mapped_column(BigInteger)
    goal_cadence: Mapped[int | None] = mapped_column(BigInteger)
    goal_cadence_frequency: Mapped[int | None] = mapped_column(BigInteger)
    goal_creation_month: Mapped[str | None] = mapped_column(String(10))
    goal_target: Mapped[int | None] = mapped_column(BigInteger)
    goal_target_month: Mapped[str | None] = mapped_column(String(10))
    goal_target_date: Mapped[str | None] = mapped_column(String(10))
    goal_percentage_complete: Mapped[int | None] = mapped_column(BigInteger)
    goal_months_to_budget: Mapped[int | None] = mapped_column(BigInteger)
    goal_under_funded: Mapped[int | None] = mapped_column(BigInteger)
    goal_overall_funded: Mapped[int | None] = mapped_column(BigInteger)
    goal_overall_left: Mapped[int | None] = mapped_column(BigInteger)
    goal_snoozed_at: Mapped[str | None] = mapped_column(String(64))
    change_batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class BudgetMonth(Base):
    __tablename__ = "budget_months"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    month: Mapped[str] = mapped_column(String(10), index=True)
    note: Mapped[str | None] = mapped_column(Text)
    income: Mapped[int | None] = mapped_column(BigInteger)
    budgeted: Mapped[int | None] = mapped_column(BigInteger)
    activity: Mapped[int | None] = mapped_column(BigInteger)
    to_be_budgeted: Mapped[int | None] = mapped_column(BigInteger)
    age_of_money: Mapped[int | None] = mapped_column(BigInteger)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class MonthCategory(Base):
    __tablename__ = "month_categories"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    month: Mapped[str] = mapped_column(String(10), index=True)
    category_id: Mapped[str | None] = mapped_column(String(64), index=True)
    category_group_id: Mapped[str | None] = mapped_column(String(64), index=True)
    category_group_name: Mapped[str | None] = mapped_column(String(255))
    name: Mapped[str | None] = mapped_column(String(255))
    hidden: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    internal: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    budgeted: Mapped[int | None] = mapped_column(BigInteger)
    activity: Mapped[int | None] = mapped_column(BigInteger)
    balance: Mapped[int | None] = mapped_column(BigInteger)
    goal_type: Mapped[str | None] = mapped_column(String(64))
    goal_needs_whole_amount: Mapped[bool | None] = mapped_column(Boolean)
    goal_day: Mapped[int | None] = mapped_column(BigInteger)
    goal_cadence: Mapped[int | None] = mapped_column(BigInteger)
    goal_cadence_frequency: Mapped[int | None] = mapped_column(BigInteger)
    goal_creation_month: Mapped[str | None] = mapped_column(String(10))
    goal_target: Mapped[int | None] = mapped_column(BigInteger)
    goal_target_month: Mapped[str | None] = mapped_column(String(10))
    goal_target_date: Mapped[str | None] = mapped_column(String(10))
    goal_percentage_complete: Mapped[int | None] = mapped_column(BigInteger)
    goal_months_to_budget: Mapped[int | None] = mapped_column(BigInteger)
    goal_under_funded: Mapped[int | None] = mapped_column(BigInteger)
    goal_overall_funded: Mapped[int | None] = mapped_column(BigInteger)
    goal_overall_left: Mapped[int | None] = mapped_column(BigInteger)
    goal_snoozed_at: Mapped[str | None] = mapped_column(String(64))
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    account_id: Mapped[str | None] = mapped_column(String(64), index=True)
    category_id: Mapped[str | None] = mapped_column(String(64), index=True)
    payee_id: Mapped[str | None] = mapped_column(String(64), index=True)
    transfer_account_id: Mapped[str | None] = mapped_column(String(64))
    transfer_transaction_id: Mapped[str | None] = mapped_column(String(64))
    matched_transaction_id: Mapped[str | None] = mapped_column(String(64))
    import_id: Mapped[str | None] = mapped_column(String(255))
    import_payee_name: Mapped[str | None] = mapped_column(Text)
    import_payee_name_original: Mapped[str | None] = mapped_column(Text)
    debt_transaction_type: Mapped[str | None] = mapped_column(String(64))
    account_name: Mapped[str | None] = mapped_column(String(255))
    payee_name: Mapped[str | None] = mapped_column(String(255))
    category_name: Mapped[str | None] = mapped_column(String(255))
    date: Mapped[str | None] = mapped_column(String(10), index=True)
    amount: Mapped[int | None] = mapped_column(BigInteger)
    memo: Mapped[str | None] = mapped_column(Text)
    cleared: Mapped[str | None] = mapped_column(String(32))
    approved: Mapped[bool | None] = mapped_column(Boolean)
    flag_color: Mapped[str | None] = mapped_column(String(32))
    flag_name: Mapped[str | None] = mapped_column(String(255))
    foreign_amount: Mapped[int | None] = mapped_column(BigInteger)
    foreign_currency_code: Mapped[str | None] = mapped_column(String(16))
    change_batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class SubTransaction(Base):
    __tablename__ = "subtransactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[int | None] = mapped_column(BigInteger)
    memo: Mapped[str | None] = mapped_column(Text)
    payee_id: Mapped[str | None] = mapped_column(String(64))
    payee_name: Mapped[str | None] = mapped_column(String(255))
    category_id: Mapped[str | None] = mapped_column(String(64))
    category_name: Mapped[str | None] = mapped_column(String(255))
    transfer_account_id: Mapped[str | None] = mapped_column(String(64))
    transfer_transaction_id: Mapped[str | None] = mapped_column(String(64))
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class PayeeLocation(Base):
    __tablename__ = "payee_locations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    payee_id: Mapped[str | None] = mapped_column(String(64), index=True)
    latitude: Mapped[str | None] = mapped_column(String(64))
    longitude: Mapped[str | None] = mapped_column(String(64))
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class ScheduledTransaction(Base):
    __tablename__ = "scheduled_transactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    account_id: Mapped[str | None] = mapped_column(String(64))
    payee_id: Mapped[str | None] = mapped_column(String(64))
    category_id: Mapped[str | None] = mapped_column(String(64))
    transfer_account_id: Mapped[str | None] = mapped_column(String(64))
    date_first: Mapped[str | None] = mapped_column(String(10))
    date_next: Mapped[str | None] = mapped_column(String(10))
    frequency: Mapped[str | None] = mapped_column(String(64))
    amount: Mapped[int | None] = mapped_column(BigInteger)
    memo: Mapped[str | None] = mapped_column(Text)
    flag_color: Mapped[str | None] = mapped_column(String(32))
    flag_name: Mapped[str | None] = mapped_column(String(255))
    account_name: Mapped[str | None] = mapped_column(String(255))
    payee_name: Mapped[str | None] = mapped_column(String(255))
    category_name: Mapped[str | None] = mapped_column(String(255))
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class ScheduledSubTransaction(Base):
    __tablename__ = "scheduled_subtransactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    scheduled_transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[int | None] = mapped_column(BigInteger)
    memo: Mapped[str | None] = mapped_column(Text)
    payee_id: Mapped[str | None] = mapped_column(String(64))
    payee_name: Mapped[str | None] = mapped_column(String(255))
    category_id: Mapped[str | None] = mapped_column(String(64))
    category_name: Mapped[str | None] = mapped_column(String(255))
    transfer_account_id: Mapped[str | None] = mapped_column(String(64))
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class TransactionNotification(Base):
    __tablename__ = "transaction_notifications"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    notified_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())


class BudgetChange(Base):
    __tablename__ = "budget_changes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(36), index=True)
    budget_id: Mapped[str] = mapped_column(String(64), index=True)
    resource: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    action: Mapped[str] = mapped_column(String(16), index=True)
    entity_name: Mapped[str | None] = mapped_column(String(255))
    payload: Mapped[str] = mapped_column(Text)
    recorded_at: Mapped[str] = mapped_column(
        DateTime,
        server_default=func.now(),
        index=True,
    )


class SyncState(Base):
    __tablename__ = "sync_state"

    budget_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    resource: Mapped[str] = mapped_column(String(64), primary_key=True)
    server_knowledge: Mapped[int] = mapped_column(BigInteger)
    change_batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    synced_at: Mapped[str] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
    )
