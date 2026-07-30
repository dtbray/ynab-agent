import asyncio
from collections.abc import Collection, Sequence
from datetime import date
import json

import pytest

from ynab_agent.planning.performance import (
    AccountCashFlow,
    AmbiguousPerformanceError,
    CashFlowReason,
    CashFlowTreatment,
    NonConvergentPerformanceError,
    calculate_performance,
)
from ynab_agent.services.wealth import (
    AccountFreshness,
    CashFlowCandidate,
    WealthAccount,
    WealthService,
)


class FakeWealthRepository:
    def __init__(
        self,
        *,
        accounts: Sequence[WealthAccount],
        candidates: Sequence[CashFlowCandidate] = (),
    ) -> None:
        self.accounts = tuple(accounts)
        self.candidates = tuple(candidates)
        self.requested_through: date | None = None

    async def get_accounts(
        self,
        account_ids: Collection[str],
    ) -> Sequence[WealthAccount]:
        selected = set(account_ids)
        return tuple(account for account in self.accounts if account.id in selected)

    async def list_account_freshness(
        self,
        *,
        tracking_only: bool,
    ) -> Sequence[AccountFreshness]:
        raise AssertionError("freshness is not part of performance calculation")

    async def get_cash_flow_candidates(
        self,
        account_ids: Collection[str],
        *,
        through: date,
    ) -> Sequence[CashFlowCandidate]:
        self.requested_through = through
        selected = set(account_ids)
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.account_id in selected and candidate.date <= through
        )


def _account(
    account_id: str,
    *,
    balance_milliunits: int = 1_210_000,
) -> WealthAccount:
    return WealthAccount(
        id=account_id,
        name=account_id.title(),
        type="otherAsset",
        on_budget=False,
        balance_milliunits=balance_milliunits,
        closed=False,
        deleted=False,
        last_reconciled_at=None,
    )


def _candidate(
    transaction_id: str,
    *,
    account_id: str = "brokerage",
    transaction_date: date,
    amount_milliunits: int,
    transfer_account_id: str | None = None,
    payee_name: str | None = None,
    memo: str | None = None,
    approved: bool | None = True,
    cleared: str | None = "cleared",
) -> CashFlowCandidate:
    return CashFlowCandidate(
        transaction_id=transaction_id,
        account_id=account_id,
        date=transaction_date,
        amount_milliunits=amount_milliunits,
        transfer_account_id=transfer_account_id,
        payee_name=payee_name,
        memo=memo,
        approved=approved,
        cleared=cleared,
    )


def test_service_returns_auditable_treatment_for_every_candidate() -> None:
    repository = FakeWealthRepository(
        accounts=[_account("brokerage"), _account("ira", balance_milliunits=0)],
        candidates=[
            _candidate(
                "starting",
                transaction_date=date(2020, 1, 1),
                amount_milliunits=1_000_000,
                payee_name=" Starting Balance ",
            ),
            _candidate(
                "external",
                transaction_date=date(2020, 6, 1),
                amount_milliunits=100_000,
                transfer_account_id="checking",
                payee_name="Transfer : Checking",
            ),
            _candidate(
                "internal",
                transaction_date=date(2020, 7, 1),
                amount_milliunits=-50_000,
                transfer_account_id="ira",
                payee_name="Transfer : IRA",
            ),
            _candidate(
                "dividend",
                transaction_date=date(2020, 8, 1),
                amount_milliunits=10_000,
                payee_name="Dividend",
                memo="reinvested",
            ),
            _candidate(
                "reconciliation",
                transaction_date=date(2020, 9, 1),
                amount_milliunits=25_000,
                payee_name="Reconciliation Balance Adjustment",
            ),
            _candidate(
                "market-value",
                transaction_date=date(2020, 10, 1),
                amount_milliunits=30_000,
                memo="Market value update",
            ),
            _candidate(
                "fee",
                transaction_date=date(2020, 11, 1),
                amount_milliunits=-5_000,
                payee_name="Management fee",
            ),
            _candidate(
                "other",
                transaction_date=date(2020, 12, 1),
                amount_milliunits=40_000,
                payee_name="Unclassified account activity",
            ),
        ],
    )

    result = asyncio.run(
        WealthService(repository).calculate_money_weighted_performance(
            account_ids=["brokerage", "ira"],
            valuation_date=date(2022, 1, 1),
            current_date=date(2026, 7, 29),
            ending_balance=1_210,
        )
    )

    evidence = {
        item.transaction_id: (item.treatment, item.reason)
        for item in result.cash_flow_classifications
    }
    assert evidence == {
        "starting": (CashFlowTreatment.INCLUDED, CashFlowReason.STARTING_BALANCE),
        "external": (CashFlowTreatment.INCLUDED, CashFlowReason.EXTERNAL_TRANSFER),
        "internal": (CashFlowTreatment.EXCLUDED, CashFlowReason.INTERNAL_TRANSFER),
        "dividend": (CashFlowTreatment.EXCLUDED, CashFlowReason.RETAINED_INCOME),
        "reconciliation": (
            CashFlowTreatment.EXCLUDED,
            CashFlowReason.RECONCILIATION_ADJUSTMENT,
        ),
        "market-value": (
            CashFlowTreatment.EXCLUDED,
            CashFlowReason.MARKET_VALUE_ADJUSTMENT,
        ),
        "fee": (CashFlowTreatment.EXCLUDED, CashFlowReason.FEE_OR_TAX),
        "other": (CashFlowTreatment.EXCLUDED, CashFlowReason.NON_TRANSFER_ACTIVITY),
    }
    assert result.included_transaction_count == 2
    assert result.excluded_transaction_count == 6
    assert result.contributions == 1_100
    assert result.withdrawals == 0
    assert result.valuation_source == "user_supplied_historical_balance"
    assert repository.requested_through == date(2022, 1, 1)

    payload = json.loads(json.dumps(result.as_dict(), default=str))
    assert payload["cash_flow_classifications"][1]["transaction_id"] == "external"
    assert payload["cash_flow_classifications"][1]["investor_cash_flow"] == -100


def test_external_withdrawal_is_positive_from_investor_perspective() -> None:
    repository = FakeWealthRepository(
        accounts=[_account("brokerage", balance_milliunits=900_000)],
        candidates=[
            _candidate(
                "starting",
                transaction_date=date(2020, 1, 1),
                amount_milliunits=1_000_000,
                payee_name="Starting Balance",
            ),
            _candidate(
                "withdrawal",
                transaction_date=date(2021, 1, 1),
                amount_milliunits=-200_000,
                transfer_account_id="checking",
            ),
        ],
    )

    result = asyncio.run(
        WealthService(repository).calculate_money_weighted_performance(
            account_ids=["brokerage"],
            valuation_date=date(2022, 1, 1),
            current_date=date(2022, 1, 1),
            ending_balance=None,
        )
    )

    withdrawal = next(
        item
        for item in result.cash_flow_classifications
        if item.transaction_id == "withdrawal"
    )
    assert withdrawal.reason is CashFlowReason.EXTERNAL_TRANSFER
    assert withdrawal.investor_cash_flow == 200
    assert result.contributions == 1_000
    assert result.withdrawals == 200
    assert result.valuation_source == "current_cached_account_balances"


@pytest.mark.parametrize(
    ("valuation_date", "current_date", "ending_balance", "message"),
    [
        (
            date(2025, 1, 1),
            date(2026, 1, 1),
            None,
            "historical --as-of requires --ending-balance",
        ),
        (
            date(2027, 1, 1),
            date(2026, 1, 1),
            1_000,
            "cannot be in the future",
        ),
    ],
)
def test_service_rejects_undated_or_future_terminal_valuations(
    valuation_date: date,
    current_date: date,
    ending_balance: float | None,
    message: str,
) -> None:
    repository = FakeWealthRepository(accounts=[_account("brokerage")])

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            WealthService(repository).calculate_money_weighted_performance(
                account_ids=["brokerage"],
                valuation_date=valuation_date,
                current_date=current_date,
                ending_balance=ending_balance,
            )
        )

    assert repository.requested_through is None


def test_calculator_rejects_cash_flows_with_ambiguous_xirr() -> None:
    with pytest.raises(AmbiguousPerformanceError, match="multiple valid XIRR"):
        calculate_performance(
            account_ids=["brokerage"],
            cash_flows=[
                AccountCashFlow(date=date(2020, 1, 1), amount=100),
                AccountCashFlow(date=date(2021, 1, 1), amount=-230),
                AccountCashFlow(date=date(2022, 1, 1), amount=132),
            ],
            ending_balance=0,
            valuation_date=date(2022, 1, 1),
        )


def test_nonconventional_cash_flows_without_a_root_are_not_called_ambiguous() -> None:
    with pytest.raises(NonConvergentPerformanceError, match="finite result"):
        calculate_performance(
            account_ids=["brokerage"],
            cash_flows=[
                AccountCashFlow(date=date(2020, 1, 1), amount=100),
                AccountCashFlow(date=date(2021, 1, 1), amount=-50),
                AccountCashFlow(date=date(2022, 1, 1), amount=10),
            ],
            ending_balance=0,
            valuation_date=date(2022, 1, 1),
        )


def test_calculator_reports_non_convergence_explicitly(monkeypatch) -> None:
    import pyxirr

    monkeypatch.setattr(pyxirr, "xirr", lambda *args, **kwargs: None)

    with pytest.raises(NonConvergentPerformanceError, match="finite result"):
        calculate_performance(
            account_ids=["brokerage"],
            cash_flows=[
                AccountCashFlow(date=date(2020, 1, 1), amount=1_000),
            ],
            ending_balance=1_100,
            valuation_date=date(2021, 1, 1),
        )
