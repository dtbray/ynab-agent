"""SQL adapter for immutable scenario revisions and comparisons."""

from __future__ import annotations

import json
from typing import cast

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.scenario_comparison import (
    ScenarioComparison,
    ScenarioComparisonError,
    ScenarioComparisonErrorCode,
    ScenarioComparisonManifest,
    ScenarioRevision,
    ScenarioRevisionManifest,
    verify_scenario_comparison,
    verify_scenario_revision,
)


class SqlScenarioRevisionRepository:
    """Append-only persistence for saved planning decisions."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    async def next_revision_number(self, scenario_name: str) -> int:
        rows = await self.database.fetch_all(
            """
            SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision
            FROM scenario_revisions
            WHERE scenario_name = :scenario_name
            """,
            {"scenario_name": scenario_name},
        )
        return cast(int, rows[0]["next_revision"])

    async def create_revision(self, revision: ScenarioRevision) -> None:
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO scenario_revisions (
                          id, scenario_name, revision,
                          manifest_sha256, manifest_json, created_at
                        )
                        VALUES (
                          :id, :scenario_name, :revision,
                          :manifest_sha256, :manifest_json, :created_at
                        )
                        """
                    ),
                    {
                        "id": revision.id,
                        "scenario_name": revision.scenario_name,
                        "revision": revision.revision,
                        "manifest_sha256": revision.manifest_sha256,
                        "manifest_json": revision.manifest.model_dump_json(),
                        "created_at": revision.created_at.isoformat(),
                    },
                )
                await session.commit()
        except IntegrityError as exc:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.PERSISTENCE_CONFLICT,
                "scenario revision identity already exists",
            ) from exc

    async def get_revision(self, revision_id: str) -> ScenarioRevision | None:
        rows = await self.database.fetch_all(
            """
            SELECT id, scenario_name, revision,
                   manifest_sha256, manifest_json, created_at
            FROM scenario_revisions
            WHERE id = :revision_id
            """,
            {"revision_id": revision_id},
        )
        if not rows:
            return None
        row = rows[0]
        revision = ScenarioRevision.model_validate(
            {
                **row,
                "manifest": ScenarioRevisionManifest.model_validate_json(
                    str(row["manifest_json"])
                ),
            }
        )
        verify_scenario_revision(revision)
        return revision

    async def create_comparison(self, comparison: ScenarioComparison) -> None:
        try:
            async with self.database.session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO scenario_comparisons (
                          id, baseline_revision_id,
                          alternative_revision_ids_json,
                          manifest_sha256, result_sha256, manifest_json,
                          result_json, created_at
                        )
                        VALUES (
                          :id, :baseline_revision_id,
                          :alternative_revision_ids_json,
                          :manifest_sha256, :result_sha256, :manifest_json,
                          :result_json, :created_at
                        )
                        """
                    ),
                    {
                        "id": comparison.id,
                        "baseline_revision_id": (
                            comparison.baseline.revision_id
                        ),
                        "alternative_revision_ids_json": (
                            json.dumps(
                                comparison.manifest.alternative_revision_ids,
                                separators=(",", ":"),
                            )
                        ),
                        "manifest_sha256": comparison.manifest_sha256,
                        "result_sha256": comparison.result_sha256,
                        "manifest_json": comparison.manifest.model_dump_json(),
                        "result_json": comparison.model_dump_json(),
                        "created_at": comparison.created_at.isoformat(),
                    },
                )
                await session.commit()
        except IntegrityError as exc:
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.PERSISTENCE_CONFLICT,
                "scenario comparison could not be persisted",
            ) from exc

    async def get_comparison(
        self,
        comparison_id: str,
    ) -> ScenarioComparison | None:
        rows = await self.database.fetch_all(
            """
            SELECT baseline_revision_id, alternative_revision_ids_json,
                   manifest_sha256, result_sha256, manifest_json,
                   result_json, created_at
            FROM scenario_comparisons
            WHERE id = :comparison_id
            """,
            {"comparison_id": comparison_id},
        )
        if not rows:
            return None
        row = rows[0]
        comparison = ScenarioComparison.model_validate_json(
            str(row["result_json"])
        )
        if (
            comparison.manifest_sha256 != row["manifest_sha256"]
            or comparison.result_sha256 != row["result_sha256"]
            or comparison.baseline.revision_id
            != row["baseline_revision_id"]
            or list(comparison.manifest.alternative_revision_ids)
            != json.loads(str(row["alternative_revision_ids_json"]))
            or comparison.created_at.isoformat() != row["created_at"]
            or comparison.manifest
            != ScenarioComparisonManifest.model_validate_json(
                str(row["manifest_json"])
            )
        ):
            raise ScenarioComparisonError(
                ScenarioComparisonErrorCode.PERSISTED_CONTENT_MISMATCH,
                "scenario comparison columns do not match its result",
            )
        verify_scenario_comparison(comparison)
        return comparison
