"""Typed FastAPI dependencies and request-scoped resource ownership."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import hmac
from ipaddress import ip_address
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ynab_agent.config import Settings
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.runtime import (
    DatabaseFactory,
    open_spending_guardrail_service,
    open_scenario_comparison_service,
    open_wealth_service,
)
from ynab_agent.services.planner_jobs import (
    HistoricalDatasetRegistry,
    PlannerExecutionPolicy,
    PlannerJobDispatcher,
    PlannerJobRepository,
    PlannerJobService,
)
from ynab_agent.services.spending_guardrails import SpendingGuardrailService
from ynab_agent.services.scenario_comparison import (
    BoundedScenarioComparisonExecutor,
    ScenarioComparisonService,
)
from ynab_agent.services.wealth import WealthService

from .settings import HttpApiSettings


@dataclass(frozen=True)
class HttpApiRuntime:
    """Process-wide configuration installed by the application lifespan."""

    application_settings: Settings
    api_settings: HttpApiSettings
    planner_job_repository: PlannerJobRepository
    planner_job_dispatcher: PlannerJobDispatcher
    historical_datasets: HistoricalDatasetRegistry
    planner_execution_policy: PlannerExecutionPolicy
    scenario_comparison_executor: BoundedScenarioComparisonExecutor
    database_factory: DatabaseFactory = DatabaseManager


def get_http_runtime(request: Request) -> HttpApiRuntime:
    """Return lifespan-owned configuration for request dependencies."""
    return cast(HttpApiRuntime, request.app.state.http_runtime)


HttpRuntimeDep = Annotated[HttpApiRuntime, Depends(get_http_runtime)]


async def get_wealth_service(
    runtime: HttpRuntimeDep,
) -> AsyncIterator[WealthService]:
    """Yield a service whose database resources are closed after serialization."""
    async with open_wealth_service(
        runtime.application_settings,
        database_factory=runtime.database_factory,
    ) as service:
        yield service


WealthServiceDep = Annotated[
    WealthService,
    Depends(get_wealth_service, scope="function"),
]


async def get_spending_guardrail_service(
    runtime: HttpRuntimeDep,
) -> AsyncIterator[SpendingGuardrailService]:
    """Yield a tier-mapping service with function-scoped database ownership."""
    async with open_spending_guardrail_service(
        runtime.application_settings,
        database_factory=runtime.database_factory,
    ) as service:
        yield service


SpendingGuardrailServiceDep = Annotated[
    SpendingGuardrailService,
    Depends(get_spending_guardrail_service, scope="function"),
]


def get_planner_job_service(
    runtime: HttpRuntimeDep,
    wealth_service: WealthServiceDep,
) -> PlannerJobService:
    """Compose the job service from lifespan and request-scoped resources."""
    return PlannerJobService(
        repository=runtime.planner_job_repository,
        wealth_service=wealth_service,
        historical_datasets=runtime.historical_datasets,
        dispatcher=runtime.planner_job_dispatcher,
        execution_policy=runtime.planner_execution_policy,
    )


PlannerJobServiceDep = Annotated[
    PlannerJobService,
    Depends(get_planner_job_service),
]


async def get_scenario_comparison_service(
    runtime: HttpRuntimeDep,
) -> AsyncIterator[ScenarioComparisonService]:
    """Yield saved-scenario services with CPU work off the event loop."""
    async with open_scenario_comparison_service(
        runtime.application_settings,
        database_factory=runtime.database_factory,
        execution_policy=runtime.planner_execution_policy,
        bounded_executor=runtime.scenario_comparison_executor,
    ) as service:
        yield service


ScenarioComparisonServiceDep = Annotated[
    ScenarioComparisonService,
    Depends(get_scenario_comparison_service, scope="function"),
]


_bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Bearer key configured with YNAB_AGENT_HTTP_API_KEY.",
)
BearerCredentialsDep = Annotated[
    HTTPAuthorizationCredentials | None,
    Security(_bearer_scheme),
]


def require_api_access(
    request: Request,
    runtime: HttpRuntimeDep,
    credentials: BearerCredentialsDep,
) -> None:
    """Allow loopback development or require the configured bearer key."""
    configured_key = runtime.api_settings.api_key
    client_host = request.client.host if request.client is not None else None
    if configured_key is None:
        if (
            runtime.api_settings.allow_unauthenticated_loopback
            and _is_loopback(client_host)
        ):
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "HTTP API authentication is not configured for non-loopback access"
            ),
        )

    if (
        credentials is None
        or credentials.scheme.casefold() != "bearer"
        or not hmac.compare_digest(
            credentials.credentials,
            configured_key.get_secret_value(),
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False
