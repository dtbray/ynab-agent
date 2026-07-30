from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import cast
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from ynab_agent.config import Settings
from ynab_agent.runtime import DatabaseFactory
from ynab_agent.http_api.app import create_app
from ynab_agent.http_api.settings import HttpApiSettings


VALID_SCENARIO: dict[str, object] = {
    "name": "HTTP validation",
    "current_age": 40,
    "retirement_age": 60,
    "end_age": 95,
    "starting_portfolio": 125_000,
    "annual_spending": 40_000,
}


class FakeDatabase:
    def __init__(
        self,
        database_url: str,
        *,
        fail_reads: bool = False,
        fail_planner_reads: bool = False,
    ) -> None:
        self.database_url = database_url
        self.fail_reads = fail_reads
        self.fail_planner_reads = fail_planner_reads
        self.close_calls = 0

    async def fetch_all(
        self,
        sql: str,
        params: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        if self.fail_reads and "MAX(t.date)" in sql:
            raise RuntimeError("repository read failed")
        if self.fail_planner_reads and "FROM planner_jobs" in sql:
            raise RuntimeError("planner recovery failed")
        if "debt_interest_rates" in sql and "WHERE id IN" in sql:
            return [
                {
                    "id": "mortgage",
                    "name": "Home",
                    "type": "mortgage",
                    "on_budget": False,
                    "balance": -184_229_470,
                    "closed": False,
                    "deleted": False,
                    "last_reconciled_at": "2026-07-28T00:00:00+00:00",
                    "debt_interest_rates": '{"2021-10-01": 2750}',
                    "debt_minimum_payments": '{"2025-07-01": 1157200}',
                    "debt_escrow_amounts": '{"2025-07-01": 292340}',
                }
            ]
        if "MAX(t.date)" in sql:
            return [
                {
                    "id": "brokerage",
                    "name": "Brokerage",
                    "type": "otherAsset",
                    "on_budget": False,
                    "balance": 125_000_000,
                    "closed": False,
                    "deleted": False,
                    "last_reconciled_at": "2026-07-20T12:30:00+00:00",
                    "latest_transaction": "2026-07-28",
                }
            ]
        return []

    async def close(self) -> None:
        self.close_calls += 1


def _database_factory(
    created: list[FakeDatabase],
    *,
    fail_reads: bool = False,
    fail_planner_reads: bool = False,
) -> DatabaseFactory:
    def factory(database_url: str) -> FakeDatabase:
        database = FakeDatabase(
            database_url,
            fail_reads=fail_reads,
            fail_planner_reads=fail_planner_reads,
        )
        created.append(database)
        return database

    return cast(DatabaseFactory, cast(Callable[[str], object], factory))


def _settings() -> Settings:
    return Settings(database_url="sqlite+aiosqlite:///http-api-test.db")


def test_openapi_exposes_only_the_intended_service_surface() -> None:
    schema = create_app(application_settings=_settings()).openapi()

    assert set(schema["paths"]) == {
        "/wealth/accounts/freshness",
        "/wealth/accounts/{account_id}/mortgage-projection",
        "/wealth/scenarios/validate",
        "/wealth/spending-tiers/{category_id}",
        "/wealth/spending-tiers",
        "/wealth/spending-baseline",
        "/wealth/spending-guardrails/preview",
        "/wealth/scenarios/revisions",
        "/wealth/scenarios/revisions/{revision_id}",
        "/wealth/scenarios/comparisons",
        "/wealth/scenarios/comparisons/{comparison_id}",
        "/wealth/tax/calculate",
        "/planner/jobs",
        "/planner/jobs/{job_id}",
        "/planner/jobs/{job_id}/result",
    }
    assert set(schema["paths"]["/wealth/accounts/freshness"]) == {"get"}
    assert set(
        schema["paths"]["/wealth/accounts/{account_id}/mortgage-projection"]
    ) == {"get"}
    assert set(schema["paths"]["/wealth/scenarios/validate"]) == {"post"}
    assert set(schema["paths"]["/wealth/spending-tiers"]) == {"get"}
    assert set(
        schema["paths"]["/wealth/spending-tiers/{category_id}"]
    ) == {"put"}
    assert set(schema["paths"]["/wealth/spending-baseline"]) == {"get"}
    assert set(
        schema["paths"]["/wealth/spending-guardrails/preview"]
    ) == {"post"}
    assert set(schema["paths"]["/wealth/scenarios/revisions"]) == {"post"}
    assert set(
        schema["paths"]["/wealth/scenarios/revisions/{revision_id}"]
    ) == {"get"}
    assert set(schema["paths"]["/wealth/scenarios/comparisons"]) == {"post"}
    assert set(
        schema["paths"]["/wealth/scenarios/comparisons/{comparison_id}"]
    ) == {"get"}
    assert set(schema["paths"]["/wealth/tax/calculate"]) == {"post"}
    assert schema["paths"]["/wealth/accounts/freshness"]["get"]["tags"] == [
        "wealth"
    ]
    assert schema["paths"]["/wealth/scenarios/validate"]["post"]["tags"] == [
        "wealth"
    ]
    assert schema["paths"]["/wealth/scenarios/comparisons"]["post"]["tags"] == [
        "wealth-scenarios"
    ]
    assert schema["paths"]["/wealth/tax/calculate"]["post"]["tags"] == ["wealth"]
    assert schema["paths"][
        "/wealth/accounts/{account_id}/mortgage-projection"
    ]["get"]["tags"] == ["wealth"]

    freshness_fields = set(
        schema["components"]["schemas"]["AccountFreshnessRead"]["properties"]
    )
    assert freshness_fields == {
        "account_id",
        "name",
        "account_type",
        "is_tracking",
        "balance_milliunits",
        "last_reconciled_at",
        "latest_transaction",
        "stale_days",
    }
    assert "closed" not in freshness_fields
    assert "deleted" not in freshness_fields

    response_fields = set(
        schema["components"]["schemas"]["ScenarioValidationResponse"]["properties"]
    )
    assert response_fields == {
        "valid",
        "scenario_name",
        "starting_portfolio",
        "valuation_provenance",
    }
    assert schema["paths"]["/wealth/accounts/freshness"]["get"]["security"] == [
        {"HTTPBearer": []}
    ]
    assert schema["paths"]["/wealth/scenarios/validate"]["post"]["security"] == [
        {"HTTPBearer": []}
    ]
    assert schema["paths"]["/wealth/scenarios/revisions"]["post"]["security"] == [
        {"HTTPBearer": []}
    ]
    assert schema["paths"]["/wealth/tax/calculate"]["post"]["security"] == [
        {"HTTPBearer": []}
    ]
    assert schema["paths"][
        "/wealth/accounts/{account_id}/mortgage-projection"
    ]["get"]["security"] == [{"HTTPBearer": []}]


def test_application_version_comes_from_installed_package_metadata() -> None:
    with patch(
        "ynab_agent.http_api.app.version",
        return_value="9.8.7",
    ):
        application = create_app(application_settings=_settings())

    assert application.version == "9.8.7"


def test_tax_calculation_matches_cli_domain_contract() -> None:
    created: list[FakeDatabase] = []
    application = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(created),
    )

    with TestClient(
        application,
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.post(
            "/wealth/tax/calculate",
            json={
                "filing_status": "single",
                "taxpayer_birth_year": 1980,
                "ordinary_income": 100_000,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_income_tax"] == 16_090.50
    assert payload["marginal_ordinary_income_tax_rate"] == pytest.approx(0.2495)
    assert payload["policy_manifest"]["policy_id"] == "us_in_2026_v1"


def test_account_freshness_uses_service_and_closes_request_database() -> None:
    created: list[FakeDatabase] = []
    application = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(created),
    )

    with TestClient(
        application,
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.get("/wealth/accounts/freshness")

    assert response.status_code == 200
    assert response.json() == {
        "accounts": [
            {
                "account_id": "brokerage",
                "name": "Brokerage",
                "account_type": "otherAsset",
                "is_tracking": True,
                "balance_milliunits": 125_000_000,
                "last_reconciled_at": "2026-07-20T12:30:00Z",
                "latest_transaction": "2026-07-28",
                "stale_days": (date.today() - date(2026, 7, 28)).days,
            }
        ]
    }
    assert len(created) == 2
    assert all(database.close_calls == 1 for database in created)


def test_mortgage_projection_returns_planner_ready_cash_flows() -> None:
    created: list[FakeDatabase] = []
    application = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(created),
    )

    with TestClient(
        application,
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.get(
            "/wealth/accounts/mortgage/mortgage-projection",
            params={"current_age": 30, "as_of": "2026-07-29"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["estimated_payoff_date"] == "2050-12-01"
    assert payload["estimated_payoff_age"] == 55
    assert payload["annual_principal_and_interest"] == 10_378.32
    assert payload["annual_escrow"] == 3_508.08
    assert payload["suggested_expense_stream"] == {
        "name": "Mortgage principal and interest",
        "flow_type": "expense",
        "start_age": 30,
        "annual_amount": 10_378.32,
            "end_age": 54,
            "inflation_adjusted": True,
            "destination_tax_treatment": None,
    }
    assert payload["suggested_contribution_stream"] == {
        "name": "Redirected mortgage principal and interest",
        "flow_type": "contribution",
        "start_age": 55,
        "annual_amount": 10_378.32,
            "end_age": None,
            "inflation_adjusted": True,
            "destination_tax_treatment": None,
    }
    assert len(created) == 2
    assert all(database.close_calls == 1 for database in created)


def test_request_database_closes_when_service_raises() -> None:
    created: list[FakeDatabase] = []
    application = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(created, fail_reads=True),
    )

    with TestClient(
        application,
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/wealth/accounts/freshness")

    assert response.status_code == 500
    assert len(created) == 2
    assert all(database.close_calls == 1 for database in created)


def test_lifespan_closes_worker_database_when_recovery_fails() -> None:
    created: list[FakeDatabase] = []
    application = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(
            created,
            fail_planner_reads=True,
        ),
    )

    with pytest.raises(RuntimeError, match="planner recovery failed"):
        with TestClient(
            application,
            client=("127.0.0.1", 50000),
        ):
            pass

    assert len(created) == 1
    assert created[0].close_calls == 1


def test_scenario_validation_resolves_portfolio_without_running_simulation() -> None:
    created: list[FakeDatabase] = []
    application = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(created),
    )

    with TestClient(
        application,
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.post(
            "/wealth/scenarios/validate",
            json=VALID_SCENARIO,
        )

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "scenario_name": "HTTP validation",
        "starting_portfolio": 125_000.0,
        "valuation_provenance": {
            "source": "explicit_scenario_input",
            "as_of": None,
            "account_ids": [],
            "source_sha256": response.json()["valuation_provenance"][
                "source_sha256"
            ],
        },
    }
    assert (
        len(response.json()["valuation_provenance"]["source_sha256"]) == 64
    )
    assert len(created) == 2
    assert all(database.close_calls == 1 for database in created)


def test_scenario_validation_translates_cached_account_errors() -> None:
    created: list[FakeDatabase] = []
    application = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(created),
    )
    scenario = {
        **VALID_SCENARIO,
        "starting_portfolio": None,
        "accounts": [{"id": "missing-account", "role": "taxable"}],
    }

    with TestClient(
        application,
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.post(
            "/wealth/scenarios/validate",
            json=scenario,
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "scenario account IDs not found in the cache: missing-account"
    }
    assert len(created) == 2
    assert all(database.close_calls == 1 for database in created)


def test_non_loopback_access_fails_closed_until_bearer_auth_is_configured() -> None:
    unauthenticated_created: list[FakeDatabase] = []
    unauthenticated_app = create_app(
        application_settings=_settings(),
        database_factory=_database_factory(unauthenticated_created),
    )
    with TestClient(
        unauthenticated_app,
        client=("203.0.113.10", 50000),
    ) as client:
        response = client.get("/wealth/accounts/freshness")

    assert response.status_code == 503
    assert len(unauthenticated_created) == 1
    assert unauthenticated_created[0].close_calls == 1

    authenticated_created: list[FakeDatabase] = []
    authenticated_app = create_app(
        application_settings=_settings(),
        api_settings=HttpApiSettings(
            api_key=SecretStr("test-api-key"),
        ),
        database_factory=_database_factory(authenticated_created),
    )
    with TestClient(
        authenticated_app,
        client=("203.0.113.10", 50000),
    ) as client:
        missing = client.post(
            "/wealth/scenarios/validate",
            json=VALID_SCENARIO,
        )
        wrong = client.post(
            "/wealth/scenarios/validate",
            headers={"Authorization": "Bearer wrong"},
            json=VALID_SCENARIO,
        )
        accepted = client.post(
            "/wealth/scenarios/validate",
            headers={"Authorization": "Bearer test-api-key"},
            json=VALID_SCENARIO,
        )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert len(authenticated_created) == 2
    assert all(
        database.close_calls == 1 for database in authenticated_created
    )
