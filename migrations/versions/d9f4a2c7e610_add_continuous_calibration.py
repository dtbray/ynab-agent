"""add continuous calibration

Revision ID: d9f4a2c7e610
Revises: c8e7a1d4b902
Create Date: 2026-07-31 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "d9f4a2c7e610"
down_revision = "c8e7a1d4b902"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sync_state",
        sa.Column("change_batch_id", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_sync_state_change_batch_id",
        "sync_state",
        ["change_batch_id"],
    )
    op.add_column(
        "accounts",
        sa.Column("change_batch_id", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_accounts_change_batch_id",
        "accounts",
        ["change_batch_id"],
    )
    op.add_column(
        "transactions",
        sa.Column("change_batch_id", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_transactions_change_batch_id",
        "transactions",
        ["change_batch_id"],
    )
    op.add_column(
        "categories",
        sa.Column("change_batch_id", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_categories_change_batch_id",
        "categories",
        ["change_batch_id"],
    )
    op.create_table(
        "calibration_sync_batches",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("budget_id", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("started_at", sa.String(64), nullable=False),
        sa.Column(
            "unowned_source_dirty",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("completed_at", sa.String(64), nullable=True),
        sa.Column("predecessor_id", sa.String(64), nullable=True),
        sa.Column("superseded_by", sa.String(64), nullable=True),
        sa.Column("watermarks_json", sa.Text(), nullable=True),
        sa.Column("watermarks_sha256", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(["predecessor_id"], ["calibration_sync_batches.id"]),
        sa.ForeignKeyConstraint(["superseded_by"], ["calibration_sync_batches.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_calibration_sync_batches_budget_id",
        "calibration_sync_batches",
        ["budget_id"],
    )
    op.create_index(
        "ix_calibration_sync_batches_state",
        "calibration_sync_batches",
        ["state"],
    )
    op.create_table(
        "calibration_sync_batch_heads",
        sa.Column("budget_id", sa.String(64), nullable=False),
        sa.Column("batch_id", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(["batch_id"], ["calibration_sync_batches.id"]),
        sa.PrimaryKeyConstraint("budget_id"),
        sa.UniqueConstraint("batch_id"),
    )
    op.create_table(
        "calibration_profiles",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("scenario_revision_id", sa.String(36), nullable=False),
        sa.Column("budget_id", sa.String(64), nullable=False),
        sa.Column("profile_json", sa.Text(), nullable=False),
        sa.Column("profile_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["scenario_revision_id"], ["scenario_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_calibration_profiles_scenario_revision_id",
        "calibration_profiles",
        ["scenario_revision_id"],
    )
    op.create_index("ix_calibration_profiles_budget_id", "calibration_profiles", ["budget_id"])
    op.create_index(
        "ix_calibration_profiles_budget_created",
        "calibration_profiles",
        ["budget_id", "created_at"],
    )
    op.create_table(
        "calibration_profile_events",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["calibration_profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_calibration_profile_events_profile_id",
        "calibration_profile_events",
        ["profile_id"],
    )
    op.create_table(
        "calibration_capture_failures",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("source_sync_batch_id", sa.String(64), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["calibration_profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_id",
            "source_sync_batch_id",
            name="uq_calibration_capture_failure_profile_batch",
        ),
    )
    op.create_index(
        "ix_calibration_capture_failures_profile_id",
        "calibration_capture_failures",
        ["profile_id"],
    )
    op.create_index(
        "ix_calibration_capture_failures_source_sync_batch_id",
        "calibration_capture_failures",
        ["source_sync_batch_id"],
    )
    op.create_table(
        "calibration_snapshots",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("previous_snapshot_id", sa.String(36), nullable=True),
        sa.Column("source_sync_batch_id", sa.String(64), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("manifest_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["calibration_profiles.id"]),
        sa.ForeignKeyConstraint(["previous_snapshot_id"], ["calibration_snapshots.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "profile_id", "source_sync_batch_id", name="uq_calibration_snapshot_profile_batch"
        ),
    )
    op.create_index("ix_calibration_snapshots_profile_id", "calibration_snapshots", ["profile_id"])
    op.create_index(
        "ix_calibration_snapshots_source_sync_batch_id",
        "calibration_snapshots",
        ["source_sync_batch_id"],
    )
    op.create_index(
        "ix_calibration_snapshots_profile_created",
        "calibration_snapshots",
        ["profile_id", "created_at"],
    )
    op.create_table(
        "calibration_profile_heads",
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("snapshot_id", sa.String(36), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["calibration_profiles.id"]),
        sa.ForeignKeyConstraint(["snapshot_id"], ["calibration_snapshots.id"]),
        sa.PrimaryKeyConstraint("profile_id"),
        sa.UniqueConstraint("snapshot_id"),
    )
    op.create_table(
        "calibration_observations",
        sa.Column("id", sa.String(200), nullable=False),
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(200), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("observation_json", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["calibration_snapshots.id"]),
        sa.PrimaryKeyConstraint("id", "snapshot_id"),
        sa.UniqueConstraint(
            "snapshot_id", "content_sha256", name="uq_calibration_observation_snapshot_hash"
        ),
    )
    op.create_index(
        "ix_calibration_observations_subject",
        "calibration_observations",
        ["kind", "subject_id", "observed_at"],
    )
    op.create_table(
        "calibration_drift_events",
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(200), nullable=False),
        sa.Column("material", sa.Boolean(), nullable=False),
        sa.Column("event_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["calibration_snapshots.id"]),
        sa.PrimaryKeyConstraint("snapshot_id", "content_sha256"),
    )
    op.create_index("ix_calibration_drift_events_kind", "calibration_drift_events", ["kind"])
    op.create_index(
        "ix_calibration_drift_events_subject_id", "calibration_drift_events", ["subject_id"]
    )
    op.create_table(
        "calibration_runs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("result_sha256", sa.String(64), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.Column("started_at", sa.String(64), nullable=True),
        sa.Column("completed_at", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(["snapshot_id"], ["calibration_snapshots.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("snapshot_id"),
    )
    op.create_index(
        "ix_calibration_runs_snapshot_id", "calibration_runs", ["snapshot_id"], unique=True
    )
    op.create_index("ix_calibration_runs_state", "calibration_runs", ["state"])
    op.create_index(
        "ix_calibration_runs_state_created", "calibration_runs", ["state", "created_at"]
    )
    op.create_table(
        "calibration_attribution_reports",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("report_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["calibration_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
        sa.UniqueConstraint("content_sha256"),
    )
    op.create_index(
        "ix_calibration_attribution_reports_run_id",
        "calibration_attribution_reports",
        ["run_id"],
        unique=True,
    )
    op.create_table(
        "calibration_alerts",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("drift_kind", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(200), nullable=False),
        sa.Column("fingerprint_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["calibration_profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fingerprint_sha256"),
    )
    op.create_index("ix_calibration_alerts_profile_id", "calibration_alerts", ["profile_id"])
    op.create_table(
        "calibration_alert_events",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("alert_id", sa.String(36), nullable=False),
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("created_at", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["alert_id"], ["calibration_alerts.id"]),
        sa.ForeignKeyConstraint(["snapshot_id"], ["calibration_snapshots.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "alert_id", "snapshot_id", "state", name="uq_calibration_alert_event_transition"
        ),
    )
    op.create_index(
        "ix_calibration_alert_events_alert_id", "calibration_alert_events", ["alert_id"]
    )
    op.create_index(
        "ix_calibration_alert_events_snapshot_id", "calibration_alert_events", ["snapshot_id"]
    )
    op.create_index(
        "ix_calibration_alert_events_alert_created",
        "calibration_alert_events",
        ["alert_id", "created_at"],
    )
    op.create_table(
        "calibration_alert_heads",
        sa.Column("alert_id", sa.String(36), nullable=False),
        sa.Column("event_id", sa.String(36), nullable=True),
        sa.ForeignKeyConstraint(["alert_id"], ["calibration_alerts.id"]),
        sa.ForeignKeyConstraint(["event_id"], ["calibration_alert_events.id"]),
        sa.PrimaryKeyConstraint("alert_id"),
        sa.UniqueConstraint("event_id"),
    )


def downgrade() -> None:
    op.drop_table("calibration_alert_heads")
    op.drop_table("calibration_alert_events")
    op.drop_table("calibration_alerts")
    op.drop_table("calibration_attribution_reports")
    op.drop_table("calibration_runs")
    op.drop_table("calibration_drift_events")
    op.drop_table("calibration_observations")
    op.drop_table("calibration_profile_heads")
    op.drop_table("calibration_snapshots")
    op.drop_table("calibration_capture_failures")
    op.drop_table("calibration_profile_events")
    op.drop_table("calibration_profiles")
    op.drop_table("calibration_sync_batch_heads")
    op.drop_table("calibration_sync_batches")
    op.drop_index("ix_categories_change_batch_id", table_name="categories")
    op.drop_column("categories", "change_batch_id")
    op.drop_index("ix_transactions_change_batch_id", table_name="transactions")
    op.drop_column("transactions", "change_batch_id")
    op.drop_index("ix_accounts_change_batch_id", table_name="accounts")
    op.drop_column("accounts", "change_batch_id")
    op.drop_index("ix_sync_state_change_batch_id", table_name="sync_state")
    op.drop_column("sync_state", "change_batch_id")
