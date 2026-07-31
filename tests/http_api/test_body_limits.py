from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ynab_agent.http_api.body_limits import PlannerRequestBodyLimitMiddleware
from ynab_agent.http_api.dependencies import HttpApiRuntime
from ynab_agent.http_api.settings import HttpApiSettings


_PROFILE_ID = "00000000-0000-4000-8000-000000000097"
_CALIBRATION_BODY_PATHS = (
    "/wealth/calibration/profiles",
    f"/wealth/calibration/profiles/{_PROFILE_ID}/capture",
)


def _body_limit_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.http_runtime = HttpApiRuntime(
            application_settings=cast(Any, None),
            api_settings=HttpApiSettings(planner_max_request_body_bytes=1024),
            planner_job_repository=cast(Any, None),
            planner_job_dispatcher=cast(Any, None),
            historical_datasets=cast(Any, None),
            planner_execution_policy=cast(Any, None),
            scenario_comparison_executor=cast(Any, None),
            social_security_optimizer=cast(Any, None),
            calibration_repository=cast(Any, None),
            database_factory=cast(Any, None),
        )
        yield

    application = FastAPI(lifespan=lifespan)
    application.add_middleware(PlannerRequestBodyLimitMiddleware)

    for path in _CALIBRATION_BODY_PATHS:
        application.post(path)(lambda: {"accepted": True})

    return application


def test_calibration_content_length_bodies_are_bounded_before_parsing() -> None:
    with TestClient(
        _body_limit_app(),
        client=("127.0.0.1", 50000),
    ) as client:
        responses = [
            client.post(
                path,
                content=b"{" + b"x" * 1024 + b"}",
                headers={"Content-Type": "application/json"},
            )
            for path in _CALIBRATION_BODY_PATHS
        ]

    assert [response.status_code for response in responses] == [413, 413]
    assert all(
        response.json()["detail"]
        == {
            "code": "request_too_large",
            "message": "calibration request body exceeds the configured limit",
        }
        for response in responses
    )


def test_calibration_streamed_bodies_are_bounded_before_parsing() -> None:
    with TestClient(
        _body_limit_app(),
        client=("127.0.0.1", 50000),
    ) as client:
        responses = [
            client.post(
                path,
                content=iter((b"{", b"x" * 1024, b"}")),
                headers={"Content-Type": "application/json"},
            )
            for path in _CALIBRATION_BODY_PATHS
        ]

    assert [response.status_code for response in responses] == [413, 413]
    assert all(
        response.json()["detail"]
        == {
            "code": "request_too_large",
            "message": "calibration request body exceeds the configured limit",
        }
        for response in responses
    )
