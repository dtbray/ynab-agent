from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
import time

from fastapi.testclient import TestClient

from ynab_agent.config import Settings
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.db.planner_jobs import SqlPlannerJobRepository
from ynab_agent.http_api.app import create_app
from ynab_agent.http_api.settings import HttpApiSettings
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.planning.outcomes import GoalKind
from ynab_agent.services.planner_jobs import (
    PlannerExecutionOutput,
    PlannerExecutionPolicy,
    PlannerEngineIdentity,
    PlannerGoalOutcome,
    PlannerJobPayload,
    PlannerSimulationResult,
    run_planner_job,
)


SCENARIO: dict[str, object] = {
    "name": "HTTP planner job",
    "current_age": 40,
    "retirement_age": 60,
    "end_age": 95,
    "starting_portfolio": 125_000,
    "annual_spending": 40_000,
    "trials": 100,
    "seed": 42,
}

STRATEGY_SCENARIO: dict[str, object] = {
    "name": "HTTP tax strategy",
    "current_age": 65,
    "retirement_age": 65,
    "end_age": 66,
    "starting_portfolio": 200_000,
    "annual_spending": 40_000,
    "trials": 100,
    "seed": 92,
    "tax_buckets": [
        {"tax_treatment": "tax_deferred", "starting_balance": 100_000},
        {"tax_treatment": "roth", "starting_balance": 100_000},
    ],
    "tax_assumptions": {
        "ordinary_income_tax_rate": 0.20,
        "long_term_capital_gains_tax_rate": 0.15,
        "apply_required_minimum_distributions": False,
        "withdrawal_order": ["tax_deferred", "roth"],
        "retirement_surplus_destination": "roth",
        "strategy": {
            "withdrawal_policy": "proportional",
            "proportional_withdrawal_fractions": [
                {"tax_treatment": "tax_deferred", "fraction": 0.5},
                {"tax_treatment": "roth", "fraction": 0.5},
            ],
        },
    },
}


def _runner_output(payload_json: str) -> str:
    payload = PlannerJobPayload.model_validate_json(payload_json)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "scenario": {"name": payload.scenario.name},
    }
    return PlannerExecutionOutput(
        result=PlannerSimulationResult(
            scenario=payload.scenario.name,
            starting_portfolio=payload.starting_portfolio,
            trials=payload.scenario.trials,
            seed=payload.scenario.seed,
            success_rate=0.91,
            success_rate_ci_95={"low": 0.84, "high": 0.96},
            depleted_trials=9,
            median_depletion_age=91,
            retirement_balance_real={
                "p10": 100_000,
                "p50": 200_000,
                "p90": 300_000,
            },
            ending_balance_real={
                "p10": 80_000,
                "p50": 180_000,
                "p90": 280_000,
            },
            annual_balance_real=[],
            funded_spending_ratio={"p10": 0.80, "p50": 0.95, "p90": 1.0},
            cumulative_shortfall_real={
                "p10": 0,
                "p50": 5_000,
                "p90": 25_000,
            },
            failure_duration_years={"p10": 1, "p50": 2, "p90": 4},
            longest_failure_streak_years={"p10": 1, "p50": 1, "p90": 3},
            recovered_trials=4,
            recovery_probability=0.4,
            goal_outcomes=[
                PlannerGoalOutcome(
                    name="planned_spending",
                    kind=GoalKind.PLANNED_SPENDING,
                    target_real=40_000,
                    attainment_probability=0.91,
                    attained_trials=91,
                    evaluation_basis=("all_planned_retirement_spending_funded"),
                )
            ],
            assumptions={"current_age": payload.scenario.current_age},
            engine={"schema_version": 1},
            reproducibility=manifest,
        ),
        manifest=manifest,
    ).model_dump_json()


def _database_settings(tmp_path: Path, name: str) -> Settings:
    database_url = f"sqlite+aiosqlite:///{tmp_path / name}"

    async def initialize() -> None:
        database = DatabaseManager(database_url)
        try:
            await database.initialize()
        finally:
            await database.close()

    asyncio.run(initialize())
    return Settings(database_url=database_url)


def _api_settings(**overrides: object) -> HttpApiSettings:
    values: dict[str, object] = {
        "planner_max_workers": 1,
        "planner_max_pending_jobs": 2,
        "planner_maximum_total_working_bytes": 2 * 1024 * 1024,
        "planner_maximum_working_bytes": 1024 * 1024,
        "planner_in_memory_path_bytes": 1024 * 1024,
        "planner_maximum_temporary_bytes": 2 * 1024 * 1024,
        "planner_batch_size": 100,
    }
    values.update(overrides)
    return HttpApiSettings.model_validate(values)


def _wait_for_terminal(
    client: TestClient,
    status_url: str,
) -> dict[str, object]:
    for _ in range(200):
        response = client.get(status_url)
        assert response.status_code == 200
        payload: dict[str, object] = response.json()
        if payload["state"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("planner job did not reach a terminal state")


def test_submit_is_idempotent_and_returns_typed_result_location(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(tmp_path, "http-planner.db")
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(),
        planner_executor=executor,
        planner_runner=_runner_output,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            first = client.post(
                "/planner/jobs",
                json={"scenario": SCENARIO},
            )
            assert first.status_code == 202
            accepted = first.json()
            assert first.headers["location"] == accepted["result_url"]
            assert accepted["duplicate"] is False
            assert accepted["state"] == "accepted"

            terminal = _wait_for_terminal(client, accepted["status_url"])
            assert terminal["state"] == "succeeded"
            result = client.get(accepted["result_url"])
            assert result.status_code == 200
            assert result.json()["job_id"] == accepted["job_id"]
            assert result.json()["result"]["success_rate"] == 0.91
            assert result.json()["result"]["funded_spending_ratio"]["p50"] == 0.95
            assert result.json()["result"]["recovery_probability"] == 0.4
            assert result.json()["result"]["goal_outcomes"][0]["kind"] == ("planned_spending")
            assert result.json()["manifest"] == {
                "schema_version": 1,
                "scenario": {"name": "HTTP planner job"},
            }

            duplicate = client.post(
                "/planner/jobs",
                json={"scenario": SCENARIO},
            )
            assert duplicate.status_code == 202
            assert duplicate.json()["duplicate"] is True
            assert duplicate.json()["job_id"] == accepted["job_id"]
            assert duplicate.headers["location"] == accepted["result_url"]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_http_job_preserves_healthcare_result_contract(
    tmp_path: Path,
) -> None:
    scenario = {
        "name": "HTTP healthcare",
        "current_age": 64,
        "retirement_age": 64,
        "end_age": 67,
        "starting_portfolio": 100_000,
        "annual_spending": 1,
        "accounts": [
            {
                "id": "home-reserve",
                "role": "taxable",
                "owner_person_id": "alex",
            },
            {
                "id": "cash",
                "role": "cash",
                "owner_person_id": "alex",
            },
        ],
        "tax_buckets": [
            {
                "account_id": "home-reserve",
                "owner_person_id": "alex",
                "tax_treatment": "taxable",
                "starting_balance": 0,
                "taxable_basis": 0,
            },
            {
                "account_id": "cash",
                "owner_person_id": "alex",
                "tax_treatment": "cash",
                "starting_balance": 100_000,
            },
        ],
        "tax_assumptions": {
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["taxable", "cash"],
            "retirement_surplus_destination": "cash",
        },
        "housing_plan": {
            "home": {
                "current_value": 5_000,
                "cost_basis": 5_000,
                "annual_appreciation_rate": 0,
                "maintenance_rate": 0,
                "property_tax_rate": 0,
                "insurance_rate": 0,
                "selling_cost_rate": 0,
            },
            "decision": {
                "kind": "sell",
                "event_age": 64,
                "proceeds_destination_account_id": "home-reserve",
            },
            "care": {"funding_account_id": "home-reserve"},
        },
        "return_mean": 0,
        "return_volatility": 0,
        "inflation_rate": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 18,
        "household": {
            "plan_start_date": "2026-01-02",
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1962-01-02",
                    "retirement_age_months": 64 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 90,
                    },
                }
            ],
        },
        "healthcare": {
            "medical_inflation_rate": 0,
            "ltc_funding_source": "home_equity",
            "home_equity_available_for_ltc_real": 5_000,
            "people": [
                {
                    "person_id": "alex",
                    "pre_medicare_aca_annual_premium_real": 2_000,
                    "long_term_care": {
                        "lifetime_incidence_probability": 1,
                        "minimum_onset_age": 64,
                        "maximum_onset_age": 64,
                        "mean_duration_years": 1,
                        "duration_standard_deviation_years": 0,
                        "maximum_duration_years": 1,
                        "annual_cost_real": 5_000,
                    },
                }
            ],
        },
    }
    application_settings = _database_settings(
        tmp_path,
        "http-healthcare.db",
    )
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(),
        planner_executor=executor,
        planner_runner=run_planner_job,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            accepted = client.post(
                "/planner/jobs",
                json={"scenario": scenario},
            )
            assert accepted.status_code == 202
            terminal = _wait_for_terminal(
                client,
                accepted.json()["status_url"],
            )
            assert terminal["state"] == "succeeded"
            response = client.get(accepted.json()["result_url"])

        assert response.status_code == 200
        payload = response.json()
        assert payload["result"]["healthcare_metrics"][
            "ltc_in_plan_incidence_probability"
        ] == 1
        assert payload["result"]["annual_healthcare_real"][0][
            "ltc_active_trials"
        ] == 100
        assert payload["result"]["annual_housing"][0][
            "care_funded_from_home_equity_nominal"
        ]["p50"] == 5_000
        assert payload["result"]["healthcare_metrics"][
            "lifetime_ltc_home_equity_used_real"
        ]["p50"] == 5_000
        assert payload["manifest"]["healthcare"]["assumptions"] == (
            WealthScenario.model_validate(scenario).healthcare.model_dump(
                mode="json"
            )
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_tax_strategy_route_reuses_durable_planner_admission(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(tmp_path, "http-tax-strategy.db")
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(),
        planner_executor=executor,
        planner_runner=_runner_output,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            accepted = client.post(
                "/wealth/tax/strategies/jobs",
                json={"scenario": STRATEGY_SCENARIO},
            )
            missing_strategy = client.post(
                "/wealth/tax/strategies/jobs",
                json={"scenario": SCENARIO},
            )

        assert accepted.status_code == 202
        assert accepted.headers["location"] == accepted.json()["result_url"]
        assert accepted.json()["status_url"].startswith("/planner/jobs/")
        assert missing_strategy.status_code == 422
        assert missing_strategy.json()["detail"] == (
            "scenario must configure tax_assumptions.strategy"
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_registered_history_is_resolved_without_accepting_http_paths(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "server-owned-history.csv"
    history_path.write_text(
        "year,nominal_return,inflation_rate\n2022,0.10,0.03\n2023,-0.05,0.04\n",
        encoding="utf-8",
    )
    application_settings = _database_settings(
        tmp_path,
        "http-history.db",
    )
    observed_payloads: list[str] = []

    def inspect_registered_dataset(payload_json: str) -> str:
        payload = PlannerJobPayload.model_validate_json(payload_json)
        assert payload.historical_dataset is not None
        assert payload.historical_dataset.dataset_id == "market-history"
        assert str(history_path) not in payload_json
        observed_payloads.append(payload_json)
        return _runner_output(payload_json)

    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(
            planner_historical_datasets={
                "market-history": history_path,
            }
        ),
        planner_executor=executor,
        planner_runner=inspect_registered_dataset,
    )
    historical_scenario = {
        **SCENARIO,
        "return_model": "historical_bootstrap",
        "historical_block_size": 1,
    }
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            accepted = client.post(
                "/planner/jobs",
                json={
                    "scenario": historical_scenario,
                    "historical_dataset_id": "market-history",
                },
            )
            assert accepted.status_code == 202
            terminal = _wait_for_terminal(
                client,
                accepted.json()["status_url"],
            )
            assert terminal["state"] == "succeeded"

            top_level_path = client.post(
                "/planner/jobs",
                json={
                    "scenario": SCENARIO,
                    "historical_path": str(history_path),
                },
            )
            nested_path = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "returns_path": str(history_path),
                    }
                },
            )
            unknown_dataset = client.post(
                "/planner/jobs",
                json={
                    "scenario": historical_scenario,
                    "historical_dataset_id": "unknown",
                },
            )

            assert top_level_path.status_code == 422
            assert nested_path.status_code == 422
            assert unknown_dataset.status_code == 422
            assert unknown_dataset.json()["detail"] == {
                "code": "historical_dataset_not_found",
                "message": "registered historical dataset was not found",
            }
            assert len(observed_payloads) == 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_app_lifespan_recovers_durable_accepted_job(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(
        tmp_path,
        "http-recovery.db",
    )

    async def seed_accepted_job() -> str:
        database = DatabaseManager(application_settings.effective_database_url)
        try:
            repository = SqlPlannerJobRepository(database)
            payload = PlannerJobPayload(
                engine_identity=PlannerEngineIdentity.current(),
                scenario=WealthScenario.model_validate(
                    {
                        **SCENARIO,
                        "name": "recovered by lifespan",
                    }
                ),
                starting_portfolio=125_000,
                valuation_provenance=ValuationProvenance(
                    source="explicit_scenario_input",
                    source_sha256="a" * 64,
                ),
                execution_policy=PlannerExecutionPolicy(
                    maximum_working_bytes=1024 * 1024,
                    in_memory_path_bytes=1024 * 1024,
                    maximum_temporary_bytes=2 * 1024 * 1024,
                    batch_size=100,
                ),
                required_working_bytes=100_000,
            )
            created = await repository.create_job(
                job_id="00000000-0000-4000-8000-000000000020",
                request_hash="8" * 64,
                payload=payload,
            )
            return created.job.id
        finally:
            await database.close()

    job_id = asyncio.run(seed_accepted_job())
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(),
        planner_executor=executor,
        planner_runner=_runner_output,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            terminal = _wait_for_terminal(
                client,
                f"/planner/jobs/{job_id}",
            )
            assert terminal["state"] == "succeeded"
            result = client.get(f"/planner/jobs/{job_id}/result")
            assert result.status_code == 200
            assert result.json()["result"]["scenario"] == ("recovered by lifespan")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_queue_saturation_is_429_and_pending_job_can_be_cancelled(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(
        tmp_path,
        "http-saturation.db",
    )
    running = Event()
    release = Event()

    def blocking_runner(payload_json: str) -> str:
        running.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test runner timed out")
        return _runner_output(payload_json)

    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(planner_max_pending_jobs=1),
        planner_executor=executor,
        planner_runner=blocking_runner,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            first = client.post(
                "/planner/jobs",
                json={"scenario": SCENARIO},
            )
            assert first.status_code == 202
            assert running.wait(timeout=5)

            second = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "name": "pending cancellation",
                    }
                },
            )
            assert second.status_code == 202

            saturated = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "name": "saturated request",
                    }
                },
            )
            assert saturated.status_code == 429
            assert saturated.headers["retry-after"] == "5"
            assert saturated.json()["detail"] == {
                "code": "queue_saturated",
                "message": "planner job capacity is currently saturated",
            }

            cancelled = client.delete(second.json()["status_url"])
            assert cancelled.status_code == 202
            assert cancelled.json()["state"] == "cancelled"
            result = client.get(second.json()["result_url"])
            assert result.status_code == 409
            assert result.json()["detail"]["code"] == "result_not_ready"
            release.set()
            terminal = _wait_for_terminal(
                client,
                first.json()["status_url"],
            )
            assert terminal["state"] == "succeeded"
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_resource_rejection_is_stable_and_happens_before_enqueue(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(
        tmp_path,
        "http-resource-limit.db",
    )
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(
            planner_maximum_working_bytes=1,
            planner_in_memory_path_bytes=1,
            planner_maximum_total_working_bytes=1024,
        ),
        planner_executor=executor,
        planner_runner=_runner_output,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            rejected = client.post(
                "/planner/jobs",
                json={"scenario": SCENARIO},
            )
            missing = client.get("/planner/jobs/00000000-0000-4000-8000-000000000099")

        assert rejected.status_code == 422
        assert rejected.json()["detail"] == {
            "code": "resource_limit",
            "message": ("requested scenario exceeds configured planner resource limits"),
        }
        assert missing.status_code == 404
        assert missing.json()["detail"] == {
            "code": "job_not_found",
            "message": "planner job was not found",
        }

        async def count_jobs() -> int:
            database = DatabaseManager(application_settings.effective_database_url)
            try:
                rows = await database.fetch_all("SELECT COUNT(*) AS job_count FROM planner_jobs")
                return int(str(rows[0]["job_count"]))
            finally:
                await database.close()

        assert asyncio.run(count_jobs()) == 0
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_submission_body_is_rejected_before_json_parsing_when_oversized(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(
        tmp_path,
        "http-request-body-limit.db",
    )
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(planner_max_request_body_bytes=1024),
        planner_executor=executor,
        planner_runner=_runner_output,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            response = client.post(
                "/planner/jobs",
                content=b"{" + (b"x" * 1024) + b"}",
                headers={"Content-Type": "application/json"},
            )
            streamed_response = client.post(
                "/planner/jobs",
                content=iter((b"{", b"x" * 1024, b"}")),
                headers={"Content-Type": "application/json"},
            )
            strategy_response = client.post(
                "/wealth/tax/strategies/jobs",
                content=b"{" + (b"x" * 1024) + b"}",
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 413
        assert response.json()["detail"] == {
            "code": "request_too_large",
            "message": "planner job request body exceeds the configured limit",
        }
        assert streamed_response.status_code == 413
        assert streamed_response.json() == response.json()
        assert strategy_response.status_code == 413
        assert strategy_response.json() == response.json()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_submission_collections_and_strings_are_bounded(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(
        tmp_path,
        "http-collection-limits.db",
    )
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(),
        planner_executor=executor,
        planner_runner=_runner_output,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            too_many_accounts = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "accounts": [
                            {"id": f"account-{index}", "role": "cash"} for index in range(101)
                        ],
                    }
                },
            )
            too_many_income_streams = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "income_streams": [
                            {
                                "name": f"income-{index}",
                                "start_age": 60,
                                "annual_amount": 1_000,
                            }
                            for index in range(65)
                        ],
                    }
                },
            )
            long_name = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "name": "x" * 201,
                    }
                },
            )
            long_account_id = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "accounts": [{"id": "x" * 129, "role": "cash"}],
                    }
                },
            )
            long_income_name = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "income_streams": [
                            {
                                "name": "x" * 201,
                                "start_age": 60,
                                "annual_amount": 1_000,
                            }
                        ],
                    }
                },
            )

        assert too_many_accounts.status_code == 422
        assert too_many_income_streams.status_code == 422
        assert long_name.status_code == 422
        assert long_account_id.status_code == 422
        assert long_income_name.status_code == 422
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_income_stream_compute_limit_rejects_before_enqueue(
    tmp_path: Path,
) -> None:
    application_settings = _database_settings(
        tmp_path,
        "http-compute-limit.db",
    )
    executor = ThreadPoolExecutor(max_workers=1)
    application = create_app(
        application_settings=application_settings,
        api_settings=_api_settings(planner_maximum_compute_units=10_000),
        planner_executor=executor,
        planner_runner=_runner_output,
    )
    try:
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ) as client:
            response = client.post(
                "/planner/jobs",
                json={
                    "scenario": {
                        **SCENARIO,
                        "income_streams": [
                            {
                                "name": f"income-{index}",
                                "start_age": 60,
                                "annual_amount": 1_000,
                            }
                            for index in range(4)
                        ],
                    }
                },
            )

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "resource_limit"
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_openapi_documents_job_operations_and_filtered_responses() -> None:
    schema = create_app().openapi()

    assert set(schema["paths"]["/planner/jobs"]) == {"post"}
    assert set(schema["paths"]["/planner/jobs/{job_id}"]) == {
        "get",
        "delete",
    }
    assert set(schema["paths"]["/planner/jobs/{job_id}/result"]) == {"get"}
    assert schema["paths"]["/planner/jobs"]["post"]["tags"] == ["planner-jobs"]
    assert schema["paths"]["/planner/jobs"]["post"]["security"] == [{"HTTPBearer": []}]
    result_fields = set(
        schema["components"]["schemas"]["PlannerSimulationResultRead"]["properties"]
    )
    assert {
        "funded_spending_ratio",
        "cumulative_shortfall_real",
        "failure_duration_years",
        "longest_failure_streak_years",
        "recovery_probability",
        "after_tax_ending_balance_real",
        "legacy_target_probability",
        "goal_outcomes",
    } <= result_fields
    status_fields = set(schema["components"]["schemas"]["PlannerJobStatusResponse"]["properties"])
    assert "payload" not in status_fields
    assert "result" not in status_fields
    assert status_fields == {
        "job_id",
        "request_hash",
        "state",
        "cancellation_requested",
        "error",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "result_url",
    }
