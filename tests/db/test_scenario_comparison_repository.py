from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest
from sqlalchemy import text

from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.scenario_comparison import SqlScenarioRevisionRepository
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.services.planner_jobs import PlannerExecutionPolicy
from ynab_agent.services.scenario_comparison import (
    ScenarioComparisonError,
    ScenarioComparisonErrorCode,
    ScenarioComparisonService,
)
from ynab_agent.services.wealth import ResolvedStartingPortfolio


class ExplicitPortfolioResolver:
    async def resolve_starting_portfolio_with_provenance(
        self,
        scenario: WealthScenario,
    ) -> ResolvedStartingPortfolio:
        assert scenario.starting_portfolio is not None
        return ResolvedStartingPortfolio(
            value=scenario.starting_portfolio,
            provenance=ValuationProvenance(source="repository_fixture"),
        )


def _scenario(name: str, starting_portfolio: float) -> WealthScenario:
    return WealthScenario(
        name=name,
        current_age=40,
        retirement_age=60,
        end_age=75,
        starting_portfolio=starting_portfolio,
        annual_spending=40_000,
        trials=100,
        seed=42,
    )


@pytest.mark.asyncio
async def test_repository_round_trips_revisions_and_comparisons(tmp_path) -> None:
    database = DatabaseManager(
        f"sqlite+aiosqlite:///{tmp_path / 'scenario-comparison.db'}"
    )
    try:
        await database.initialize()
        repository = SqlScenarioRevisionRepository(database)
        service = ScenarioComparisonService(
            repository=repository,
            portfolio_resolver=ExplicitPortfolioResolver(),
            execution_policy=PlannerExecutionPolicy(
                maximum_working_bytes=16 * 1024 * 1024,
                in_memory_path_bytes=8 * 1024 * 1024,
                maximum_temporary_bytes=32 * 1024 * 1024,
                batch_size=100,
            ),
        )
        baseline = await service.save_revision(
            _scenario("Baseline", 1_000_000)
        )
        alternative = await service.save_revision(
            _scenario("Alternative", 900_000)
        )
        comparison = await service.compare(
            baseline_revision_id=baseline.id,
            alternative_revision_ids=[alternative.id],
        )

        assert await repository.get_revision(baseline.id) == baseline
        assert await repository.get_comparison(comparison.id) == comparison
        assert await repository.next_revision_number("Baseline") == 2
        rows = await database.fetch_all(
            """
            SELECT manifest_json, result_json, created_at
            FROM scenario_comparisons
            """
        )
        assert len(rows) == 1
        assert "gross_returns" not in str(rows[0])
        assert "inflation_factors" not in str(rows[0])
        assert datetime.fromisoformat(str(rows[0]["created_at"])).tzinfo is timezone.utc

        corrupted_result = json.loads(comparison.model_dump_json())
        corrupted_result["baseline"]["metrics"]["success_probability"] = 0.123
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE scenario_comparisons
                    SET result_json = :result_json
                    WHERE id = :comparison_id
                    """
                ),
                {
                    "comparison_id": comparison.id,
                    "result_json": json.dumps(corrupted_result),
                },
            )
            await session.commit()
        with pytest.raises(ScenarioComparisonError) as comparison_error:
            await repository.get_comparison(comparison.id)
        assert (
            comparison_error.value.code
            is ScenarioComparisonErrorCode.PERSISTED_CONTENT_MISMATCH
        )

        corrupted_manifest = json.loads(baseline.manifest.model_dump_json())
        corrupted_manifest["scenario"]["annual_spending"] = 99_000
        async with database.session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE scenario_revisions
                    SET manifest_json = :manifest_json
                    WHERE id = :revision_id
                    """
                ),
                {
                    "revision_id": baseline.id,
                    "manifest_json": json.dumps(corrupted_manifest),
                },
            )
            await session.commit()
        with pytest.raises(ScenarioComparisonError) as revision_error:
            await repository.get_revision(baseline.id)
        assert (
            revision_error.value.code
            is ScenarioComparisonErrorCode.PERSISTED_CONTENT_MISMATCH
        )
    finally:
        await database.close()
