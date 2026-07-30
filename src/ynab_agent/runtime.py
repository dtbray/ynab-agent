"""Typed application resource scopes shared by interface adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from ynab_agent.config import Settings
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.spending_guardrails import SqlSpendingTierRepository
from ynab_agent.db.scenario_comparison import SqlScenarioRevisionRepository
from ynab_agent.db.wealth import SqlWealthRepository
from ynab_agent.services.planner_jobs import PlannerExecutionPolicy
from ynab_agent.services.scenario_comparison import (
    BoundedScenarioComparisonExecutor,
    ScenarioComparisonService,
    default_comparison_execution_policy,
)
from ynab_agent.services.spending_guardrails import SpendingGuardrailService
from ynab_agent.services.wealth import WealthService


DatabaseFactory = Callable[[str], DatabaseManager]


@asynccontextmanager
async def open_database(
    app_settings: Settings,
    *,
    initialize: bool = False,
    factory: DatabaseFactory = DatabaseManager,
) -> AsyncIterator[DatabaseManager]:
    """Construct and reliably dispose one database manager."""
    database = factory(app_settings.effective_database_url)
    try:
        if initialize:
            await database.initialize()
        yield database
    finally:
        await database.close()


@asynccontextmanager
async def open_wealth_service(
    app_settings: Settings,
    *,
    database_factory: DatabaseFactory = DatabaseManager,
) -> AsyncIterator[WealthService]:
    """Construct a wealth service for one interface operation."""
    async with open_database(app_settings, factory=database_factory) as database:
        yield WealthService(SqlWealthRepository(database))


@asynccontextmanager
async def open_spending_guardrail_service(
    app_settings: Settings,
    *,
    database_factory: DatabaseFactory = DatabaseManager,
) -> AsyncIterator[SpendingGuardrailService]:
    """Construct a tier-mapping service for one interface operation."""
    async with open_database(app_settings, factory=database_factory) as database:
        yield SpendingGuardrailService(
            SqlSpendingTierRepository(database)
        )


@asynccontextmanager
async def open_scenario_comparison_service(
    app_settings: Settings,
    *,
    database_factory: DatabaseFactory = DatabaseManager,
    execution_policy: PlannerExecutionPolicy | None = None,
    bounded_executor: BoundedScenarioComparisonExecutor | None = None,
) -> AsyncIterator[ScenarioComparisonService]:
    """Construct saved-scenario services over one scoped database."""
    async with open_database(app_settings, factory=database_factory) as database:
        yield ScenarioComparisonService(
            repository=SqlScenarioRevisionRepository(database),
            portfolio_resolver=WealthService(SqlWealthRepository(database)),
            execution_policy=(
                execution_policy or default_comparison_execution_policy()
            ),
            bounded_executor=bounded_executor,
        )
