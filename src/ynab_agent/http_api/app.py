"""FastAPI composition root."""

from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import get_context

from fastapi import FastAPI

from ynab_agent.config import Settings
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.planner_jobs import SqlPlannerJobRepository
from ynab_agent.db.calibration import SqlCalibrationRepository
from ynab_agent.resources.historical import RegisteredHistoricalDatasets
from ynab_agent.runtime import DatabaseFactory, open_database
from ynab_agent.services.planner_jobs import (
    PlannerExecutionPolicy,
    PlannerJobRepository,
)
from ynab_agent.services.social_security import (
    SocialSecurityOptimizationExecutor,
)
from ynab_agent.workers.planner import PlannerJobWorker, PlannerRunner
from ynab_agent.workers.calibration import CalibrationWorker

from .dependencies import HttpApiRuntime
from .body_limits import PlannerRequestBodyLimitMiddleware
from .routers.allocation import router as allocation_router
from .routers.housing import router as housing_router
from .routers.planner_jobs import router as planner_jobs_router
from .routers.spending_guardrails import router as spending_guardrails_router
from .routers.scenario_comparison import router as scenario_comparison_router
from .routers.wealth import router as wealth_router
from .routers.calibration import router as calibration_router
from .settings import HttpApiSettings
from .social_security_executor import (
    BoundedSocialSecurityOptimizationExecutor,
)


def _application_version() -> str:
    try:
        return version("ynab-agent")
    except PackageNotFoundError:
        return "development"


def create_app(
    *,
    application_settings: Settings | None = None,
    api_settings: HttpApiSettings | None = None,
    database_factory: DatabaseFactory = DatabaseManager,
    planner_executor: Executor | None = None,
    planner_runner: PlannerRunner | None = None,
    social_security_optimizer: SocialSecurityOptimizationExecutor | None = None,
) -> FastAPI:
    """Create an HTTP app with resources resolved when its lifespan starts."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_application_settings = application_settings or Settings()
        resolved_api_settings = api_settings or HttpApiSettings()
        execution_policy = PlannerExecutionPolicy(
            maximum_working_bytes=(resolved_api_settings.planner_maximum_working_bytes),
            in_memory_path_bytes=(resolved_api_settings.planner_in_memory_path_bytes),
            maximum_temporary_bytes=(resolved_api_settings.planner_maximum_temporary_bytes),
            batch_size=resolved_api_settings.planner_batch_size,
            maximum_compute_units=(resolved_api_settings.planner_maximum_compute_units),
        )
        historical_datasets = RegisteredHistoricalDatasets.from_paths(
            resolved_api_settings.planner_historical_datasets
        )
        async with open_database(
            resolved_application_settings,
            factory=database_factory,
        ) as planner_database:
            planner_repository: PlannerJobRepository = SqlPlannerJobRepository(planner_database)
            calibration_repository = SqlCalibrationRepository(planner_database)
            shared_executor = planner_executor or ProcessPoolExecutor(
                max_workers=resolved_api_settings.planner_max_workers,
                mp_context=get_context("spawn"),
            )
            owns_executor = planner_executor is None
            worker = PlannerJobWorker(
                planner_repository,
                max_workers=resolved_api_settings.planner_max_workers,
                max_pending_jobs=(resolved_api_settings.planner_max_pending_jobs),
                maximum_total_working_bytes=(
                    resolved_api_settings.planner_maximum_total_working_bytes
                ),
                executor=shared_executor,
                **({"runner": planner_runner} if planner_runner is not None else {}),
            )
            runtime_installed = False
            calibration_worker_started = False
            calibration_worker = CalibrationWorker(
                calibration_repository,
                poll_interval_seconds=(resolved_api_settings.calibration_poll_interval_seconds),
                executor=shared_executor,
            )
            try:
                await worker.start()
                if hasattr(planner_database, "session_factory"):
                    await calibration_worker.start()
                    calibration_worker_started = True
                application.state.http_runtime = HttpApiRuntime(
                    application_settings=resolved_application_settings,
                    api_settings=resolved_api_settings,
                    planner_job_repository=planner_repository,
                    planner_job_dispatcher=worker,
                    historical_datasets=historical_datasets,
                    planner_execution_policy=execution_policy,
                    scenario_comparison_executor=worker,
                    social_security_optimizer=(
                        social_security_optimizer
                        or BoundedSocialSecurityOptimizationExecutor(
                            worker,
                            execution_policy,
                        )
                    ),
                    calibration_repository=calibration_repository,
                    database_factory=database_factory,
                )
                runtime_installed = True
                yield
            finally:
                if runtime_installed:
                    del application.state.http_runtime
                if calibration_worker_started:
                    await calibration_worker.stop()
                await worker.stop()
                if owns_executor:
                    shared_executor.shutdown(wait=False, cancel_futures=True)

    application = FastAPI(
        title="YNAB Agent",
        summary="Budget data and wealth-planning service boundaries.",
        version=_application_version(),
        lifespan=lifespan,
    )
    application.add_middleware(PlannerRequestBodyLimitMiddleware)
    application.include_router(wealth_router)
    application.include_router(allocation_router)
    application.include_router(housing_router)
    application.include_router(spending_guardrails_router)
    application.include_router(scenario_comparison_router)
    application.include_router(planner_jobs_router)
    application.include_router(calibration_router)
    return application


app = create_app()
