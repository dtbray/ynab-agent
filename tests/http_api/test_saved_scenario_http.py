from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from ynab_agent.config import Settings
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.http_api.app import create_app
from ynab_agent.http_api.settings import HttpApiSettings
from ynab_agent.services.scenario_comparison import ScenarioComparison


def _settings(tmp_path: Path) -> Settings:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'scenario-http.db'}"

    async def initialize() -> None:
        database = DatabaseManager(database_url)
        try:
            await database.initialize()
        finally:
            await database.close()

    asyncio.run(initialize())
    return Settings(database_url=database_url)


def _scenario(name: str, starting_portfolio: float) -> dict[str, object]:
    return {
        "name": name,
        "current_age": 40,
        "retirement_age": 60,
        "end_age": 75,
        "starting_portfolio": starting_portfolio,
        "annual_spending": 40_000,
        "trials": 100,
        "seed": 42,
        "portfolio_allocation": {
            "market": {
                "us_equity": {
                    "expected_return": 0.08,
                    "volatility": 0.18,
                },
                "international_equity": {
                    "expected_return": 0.07,
                    "volatility": 0.20,
                },
                "bonds": {
                    "expected_return": 0.04,
                    "volatility": 0.07,
                },
                "cash": {
                    "expected_return": 0.02,
                    "volatility": 0.01,
                },
                "correlation": {
                    "values": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                },
            },
            "accounts": [
                {
                    "account_id": "portfolio",
                    "portfolio_weight": 1,
                    "target": {
                        "us_equity": 0.6,
                        "international_equity": 0.2,
                        "bonds": 0.15,
                        "cash": 0.05,
                    },
                }
            ],
        },
    }


def test_http_revision_and_comparison_representations_round_trip(
    tmp_path: Path,
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=_settings(tmp_path),
        planner_executor=executor,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            baseline_response = client.post(
                "/wealth/scenarios/revisions",
                json={"scenario": _scenario("Baseline", 1_000_000)},
            )
            alternative_response = client.post(
                "/wealth/scenarios/revisions",
                json={"scenario": _scenario("Alternative", 750_000)},
            )
            assert baseline_response.status_code == 201
            assert alternative_response.status_code == 201

            baseline = baseline_response.json()
            alternative = alternative_response.json()
            response = client.post(
                "/wealth/scenarios/comparisons",
                json={
                    "baseline_revision_id": baseline["id"],
                    "alternative_revision_ids": [alternative["id"]],
                    "named_stress": "equity_crash",
                },
            )
            assert response.status_code == 201, response.text
            comparison = ScenarioComparison.model_validate(response.json())
            assert comparison.baseline.revision_id == baseline["id"]
            assert comparison.alternatives[0].revision_id == alternative["id"]
            assert (
                comparison.manifest.common_paths.named_stress
                == "equity_crash"
            )
            assert comparison.alternatives[0].dominated_by_revision_ids == (
                baseline["id"],
            )

            stored = client.get(
                f"/wealth/scenarios/comparisons/{comparison.id}"
            )
            revision = client.get(
                f"/wealth/scenarios/revisions/{baseline['id']}"
            )
            assert stored.status_code == 200
            assert stored.json() == response.json()
            assert revision.status_code == 200
            assert revision.json() == baseline
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_scenario_post_bodies_are_bounded_before_json_parsing(
    tmp_path: Path,
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=_settings(tmp_path),
        api_settings=HttpApiSettings(planner_max_request_body_bytes=1024),
        planner_executor=executor,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            responses = [
                client.post(
                    path,
                    content=b"{" + b"x" * 1024 + b"}",
                    headers={"Content-Type": "application/json"},
                )
                for path in (
                    "/wealth/scenarios/revisions",
                    "/wealth/scenarios/comparisons",
                )
            ]

        assert [response.status_code for response in responses] == [413, 413]
        assert all(
            response.json()["detail"]
            == {
                "code": "request_too_large",
                "message": "scenario request body exceeds the configured limit",
            }
            for response in responses
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
