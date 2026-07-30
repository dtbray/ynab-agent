from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from ynab_agent.services.debt_reports import (
    DebtAccountSnapshot,
    DebtPlanRequest,
    DebtReportService,
    DebtStrategy,
    ScheduledDebtObligation,
    extract_first_milliunit_amount,
)


class FakeDebtReportRepository:
    def __init__(
        self,
        *,
        credit_cards: Sequence[DebtAccountSnapshot] = (),
        drag_accounts: Sequence[DebtAccountSnapshot] = (),
        plan_accounts: Sequence[DebtAccountSnapshot] = (),
        obligations: Sequence[ScheduledDebtObligation] = (),
    ) -> None:
        self.credit_cards = tuple(credit_cards)
        self.drag_accounts = tuple(drag_accounts)
        self.plan_accounts = tuple(plan_accounts)
        self.obligations = tuple(obligations)
        self.calls: list[tuple[str, object, str | None]] = []

    async def list_credit_cards(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]:
        self.calls.append(("credit_cards", "", plan_id))
        return self.credit_cards

    async def list_debt_drag_accounts(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]:
        self.calls.append(("drag_accounts", "", plan_id))
        return self.drag_accounts

    async def list_debt_plan_accounts(
        self,
        *,
        plan_id: str | None,
    ) -> Sequence[DebtAccountSnapshot]:
        self.calls.append(("plan_accounts", "", plan_id))
        return self.plan_accounts

    async def list_scheduled_debt_obligations(
        self,
        *,
        through: date,
        plan_id: str | None,
    ) -> Sequence[ScheduledDebtObligation]:
        self.calls.append(("obligations", through, plan_id))
        return self.obligations


def _account(
    name: str,
    *,
    balance: int,
    payment_available: int = 0,
    minimum_payment_data: str | None = None,
    goal_target: int = 0,
) -> DebtAccountSnapshot:
    return DebtAccountSnapshot(
        account=name,
        account_type="creditCard",
        balance_milliunits=balance,
        payment_available_milliunits=payment_available,
        goal_target_milliunits=goal_target,
        minimum_payment_data=minimum_payment_data,
    )


@pytest.mark.asyncio
async def test_credit_card_coverage_preserves_float_rules_and_sorting() -> None:
    repository = FakeDebtReportRepository(
        credit_cards=(
            _account("Covered", balance=-100_000, payment_available=100_000),
            _account("Largest Float", balance=-500_000, payment_available=100_000),
            _account("Positive", balance=50_000, payment_available=-25_000),
        )
    )

    rows = await DebtReportService(repository).credit_card_coverage(plan_id="plan-1")

    assert [row.account for row in rows] == ["Largest Float", "Covered", "Positive"]
    assert rows[0].payoff_needed_milliunits == 500_000
    assert rows[0].float_shortfall_milliunits == 400_000
    assert rows[1].float_shortfall_milliunits == 0
    assert rows[2].payoff_needed_milliunits == 0
    assert rows[2].float_shortfall_milliunits == 0
    assert repository.calls == [("credit_cards", "", "plan-1")]


@pytest.mark.asyncio
async def test_debt_drag_combines_coverage_scheduled_outflows_and_known_minimums() -> None:
    through = date(2026, 6, 1)
    repository = FakeDebtReportRepository(
        drag_accounts=(
            _account(
                "Visa",
                balance=-1_250_000,
                payment_available=1_000_000,
                minimum_payment_data='{"minimumPayment": {"amount": 50000}}',
            ),
            _account("Mortgage", balance=-200_000, minimum_payment_data="not json"),
        ),
        obligations=(
            ScheduledDebtObligation(
                group_name="Debt",
                category="Student Loan",
                payee="Servicer",
                amount_milliunits=-50_000,
                frequency="monthly",
            ),
        ),
    )

    summary = await DebtReportService(repository).debt_drag(
        through=through,
        plan_id="plan-1",
    )

    assert summary.payoff_needed_milliunits == 1_450_000
    assert summary.payment_available_milliunits == 1_000_000
    assert summary.float_shortfall_milliunits == 450_000
    assert summary.scheduled_obligations_milliunits == 50_000
    assert summary.known_card_minimums_milliunits == 50_000
    assert summary.monthly_debt_drag_milliunits == 100_000
    assert repository.calls == [
        ("drag_accounts", "", "plan-1"),
        ("obligations", through, "plan-1"),
    ]


@pytest.mark.asyncio
async def test_debt_drag_normalizes_scheduled_frequencies_to_monthly() -> None:
    repository = FakeDebtReportRepository(
        obligations=(
            ScheduledDebtObligation(
                group_name="Debt",
                category="Weekly Loan",
                payee="Weekly",
                amount_milliunits=-12_000,
                frequency="weekly",
            ),
            ScheduledDebtObligation(
                group_name="Debt",
                category="Monthly Loan",
                payee="Monthly",
                amount_milliunits=-24_000,
                frequency="monthly",
            ),
            ScheduledDebtObligation(
                group_name="Debt",
                category="Quarterly Loan",
                payee="Quarterly",
                amount_milliunits=-36_000,
                frequency="every3Months",
            ),
            ScheduledDebtObligation(
                group_name="Debt",
                category="Annual Loan",
                payee="Annual",
                amount_milliunits=-120_000,
                frequency="yearly",
            ),
        )
    )

    summary = await DebtReportService(repository).debt_drag(
        through=date(2026, 6, 1)
    )

    assert summary.scheduled_obligations_milliunits == 98_000
    assert summary.monthly_debt_drag_milliunits == 98_000


@pytest.mark.asyncio
async def test_debt_plan_preserves_enrichment_and_projection_outputs() -> None:
    repository = FakeDebtReportRepository(
        plan_accounts=(
            _account(
                "Visa",
                balance=-1_250_000,
                payment_available=1_000_000,
                minimum_payment_data='{"amount": 50000}',
            ),
        )
    )
    enrichment: dict[str, Any] = {
        "accounts": {"Visa": {"apr": 24, "minimum_payment": 50}},
        "pay_over_time": [
            {
                "name": "Installment",
                "balance": 300,
                "minimum_payment": 100,
                "monthly_fee": 5,
                "promo_end": "2026-12-01",
            }
        ],
    }

    rows = await DebtReportService(repository).debt_plan(
        DebtPlanRequest(
            start_month=date(2026, 1, 1),
            monthly_payment_milliunits=500_000,
            strategy=DebtStrategy.AVALANCHE,
            plan_id="plan-1",
        ),
        enrichment=enrichment,
    )

    by_name = {row.name: row for row in rows}
    assert by_name["Visa"].apr.as_tuple().exponent == -2
    assert by_name["Visa"].minimum_payment_milliunits == 50_000
    assert by_name["Visa"].payment_available_milliunits == 1_000_000
    assert by_name["Visa"].months_to_payoff == 4
    assert by_name["Visa"].interest_paid_milliunits == 54_397
    assert by_name["Installment"].kind == "pay_over_time"
    assert by_name["Installment"].fees_paid_milliunits == 20_000
    assert by_name["Installment"].promo_end == "2026-12-01"
    assert repository.calls == [("plan_accounts", "", "plan-1")]


@pytest.mark.asyncio
async def test_debt_plan_rejects_payment_below_combined_minimums() -> None:
    repository = FakeDebtReportRepository(
        plan_accounts=(
            _account("Visa", balance=-100_000, goal_target=60_000),
            _account("Mastercard", balance=-100_000, goal_target=50_000),
        )
    )

    with pytest.raises(ValueError, match="below minimums"):
        await DebtReportService(repository).debt_plan(
            DebtPlanRequest(
                start_month=date(2026, 1, 1),
                monthly_payment_milliunits=100_000,
            ),
            enrichment={"accounts": {}, "pay_over_time": []},
        )


@pytest.mark.asyncio
async def test_debt_plan_forwards_avalanche_snowball_and_explicit_priority() -> None:
    accounts = (
        _account("Large High", balance=-500_000),
        _account("Small Low", balance=-100_000),
    )
    base_accounts: dict[str, Any] = {
        "Large High": {"apr": 24, "minimum_payment": 10},
        "Small Low": {"apr": 0.01, "minimum_payment": 10},
    }

    payoff_months: dict[DebtStrategy, dict[str, int | None]] = {}
    for strategy in (DebtStrategy.AVALANCHE, DebtStrategy.SNOWBALL):
        rows = await DebtReportService(
            FakeDebtReportRepository(plan_accounts=accounts)
        ).debt_plan(
            DebtPlanRequest(
                start_month=date(2026, 1, 1),
                monthly_payment_milliunits=100_000,
                strategy=strategy,
            ),
            enrichment={"accounts": base_accounts, "pay_over_time": []},
        )
        payoff_months[strategy] = {
            row.name: row.months_to_payoff for row in rows
        }

    assert payoff_months[DebtStrategy.AVALANCHE] == {
        "Large High": 6,
        "Small Low": 7,
    }
    assert payoff_months[DebtStrategy.SNOWBALL] == {
        "Large High": 7,
        "Small Low": 2,
    }

    priority_accounts = {
        **base_accounts,
        "Large High": {**base_accounts["Large High"], "priority": 2},
        "Small Low": {**base_accounts["Small Low"], "priority": 1},
    }
    priority_rows = await DebtReportService(
        FakeDebtReportRepository(plan_accounts=accounts)
    ).debt_plan(
        DebtPlanRequest(
            start_month=date(2026, 1, 1),
            monthly_payment_milliunits=100_000,
            strategy=DebtStrategy.PRIORITY,
        ),
        enrichment={"accounts": priority_accounts, "pay_over_time": []},
    )
    priority_by_name = {row.name: row for row in priority_rows}
    assert priority_by_name["Small Low"].months_to_payoff == 2
    assert priority_by_name["Large High"].months_to_payoff == 7


def test_minimum_payment_parser_preserves_amount_key_precedence() -> None:
    assert (
        extract_first_milliunit_amount(
            '{"unrelated": 999999, "minimum_payment": {"amount": 42000}}'
        )
        == 42_000
    )
    assert extract_first_milliunit_amount('{"nested": [12.9]}') == 12
    assert extract_first_milliunit_amount("not json") is None
    assert extract_first_milliunit_amount(None) is None


def test_debt_plan_request_validates_bounds() -> None:
    with pytest.raises(ValueError, match="first day"):
        DebtPlanRequest(
            start_month=date(2026, 1, 2),
            monthly_payment_milliunits=1,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        DebtPlanRequest(
            start_month=date(2026, 1, 1),
            monthly_payment_milliunits=-1,
        )
    with pytest.raises(ValueError, match="at least one"):
        DebtPlanRequest(
            start_month=date(2026, 1, 1),
            monthly_payment_milliunits=1,
            max_months=0,
        )
