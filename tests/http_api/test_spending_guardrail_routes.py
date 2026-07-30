from __future__ import annotations

import pytest

from ynab_agent.http_api.routers.spending_guardrails import (
    preview_spending_guardrails,
)
from ynab_agent.http_api.schemas.spending_guardrails import (
    GuardrailPreviewRequest,
)


@pytest.mark.asyncio
async def test_http_preview_exposes_the_domain_audit_without_translation_drift() -> None:
    request = GuardrailPreviewRequest.model_validate(
        {
            "plan": {
                "baseline": {
                    "essential": 30_000,
                    "lifestyle": 20_000,
                    "discretionary": 10_000,
                },
                "essential_floor": 24_000,
                "policy": {
                    "kind": "withdrawal_rate_guardrails",
                    "lower_withdrawal_rate": 0.03,
                    "upper_withdrawal_rate": 0.05,
                },
            },
            "contexts": [
                {"age": 60, "opening_portfolio_real": 1_000_000},
                {"age": 61, "opening_portfolio_real": 2_000_000},
            ],
        }
    )

    response = await preview_spending_guardrails(request)

    assert response.summary.annual_path[0].action == "reduced"
    assert response.summary.annual_path[0].spending_delta.total == -6_000
    assert response.summary.annual_path[1].action == "restored"
