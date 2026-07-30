import asyncio
from collections.abc import Collection, Sequence
from datetime import date, datetime, timedelta, timezone

import pytest

from ynab_agent.services.valuation_snapshots import (
    AccountValuationSnapshot,
    ValuationSnapshotSource,
)
from ynab_agent.services.wealth import (
    AccountFreshness,
    CashFlowCandidate,
    WealthAccount,
    WealthService,
)


class SnapshotWealthRepository:
    def __init__(
        self,
        *,
        accounts: Sequence[WealthAccount],
        candidates: Sequence[CashFlowCandidate] = (),
        snapshots: Sequence[AccountValuationSnapshot] = (),
    ) -> None:
        self.accounts = tuple(accounts)
        self.candidates = tuple(candidates)
        self.snapshots = tuple(snapshots)
        self.saved: tuple[AccountValuationSnapshot, ...] = ()
        self.snapshot_lookup: tuple[tuple[str, ...], date] | None = None

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
        assert tracking_only is True
        return tuple(
            AccountFreshness(account=account, latest_transaction=None)
            for account in self.accounts
        )

    async def get_cash_flow_candidates(
        self,
        account_ids: Collection[str],
        *,
        through: date,
    ) -> Sequence[CashFlowCandidate]:
        selected = set(account_ids)
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.account_id in selected and candidate.date <= through
        )

    async def upsert_valuation_snapshots(
        self,
        snapshots: Collection[AccountValuationSnapshot],
    ) -> int:
        self.saved = tuple(snapshots)
        return len(self.saved)

    async def get_reviewed_valuation_snapshots(
        self,
        account_ids: Collection[str],
        *,
        valuation_date: date,
    ) -> Sequence[AccountValuationSnapshot]:
        selected = tuple(dict.fromkeys(account_ids))
        self.snapshot_lookup = (selected, valuation_date)
        return tuple(
            snapshot
            for snapshot in self.snapshots
            if snapshot.account_id in selected
            and snapshot.valuation_date == valuation_date
            and snapshot.reviewed
        )


def _account(
    account_id: str,
    *,
    balance_milliunits: int,
    closed: bool = False,
) -> WealthAccount:
    return WealthAccount(
        id=account_id,
        name=account_id,
        type="otherAsset",
        on_budget=False,
        balance_milliunits=balance_milliunits,
        closed=closed,
        deleted=False,
        last_reconciled_at=None,
    )


def _starting_balance(account_id: str, amount_milliunits: int) -> CashFlowCandidate:
    return CashFlowCandidate(
        transaction_id=f"starting-{account_id}",
        account_id=account_id,
        date=date(2020, 1, 1),
        amount_milliunits=amount_milliunits,
        transfer_account_id=None,
        payee_name="Starting Balance",
        memo=None,
    )


def _reviewed_snapshot(
    account_id: str,
    *,
    valuation_date: date,
    balance_milliunits: int,
    observed_at: datetime,
    source: ValuationSnapshotSource = ValuationSnapshotSource.USER_REVIEWED,
) -> AccountValuationSnapshot:
    return AccountValuationSnapshot(
        account_id=account_id,
        valuation_date=valuation_date,
        balance_milliunits=balance_milliunits,
        source=source,
        observed_at=observed_at,
        reviewed=True,
    )


def test_capture_current_valuations_uses_utc_observation_date_and_stable_order() -> None:
    repository = SnapshotWealthRepository(
        accounts=[
            _account("zeta", balance_milliunits=200_000),
            _account("alpha", balance_milliunits=100_000),
            _account("closed", balance_milliunits=300_000, closed=True),
        ]
    )
    observed_at = datetime(
        2026,
        7,
        30,
        0,
        30,
        tzinfo=timezone(timedelta(hours=2)),
    )

    snapshots = asyncio.run(
        WealthService(repository).capture_current_valuations(
            observed_at=observed_at,
        )
    )

    assert [snapshot.account_id for snapshot in snapshots] == ["alpha", "zeta"]
    assert {snapshot.valuation_date for snapshot in snapshots} == {date(2026, 7, 29)}
    assert {snapshot.observed_at for snapshot in snapshots} == {
        datetime(2026, 7, 29, 22, 30, tzinfo=timezone.utc)
    }
    assert {snapshot.source for snapshot in snapshots} == {
        ValuationSnapshotSource.YNAB_SYNC
    }
    assert all(not snapshot.reviewed for snapshot in snapshots)
    assert repository.saved == snapshots


def test_capture_rejects_an_observation_without_timezone() -> None:
    repository = SnapshotWealthRepository(
        accounts=[_account("brokerage", balance_milliunits=100_000)]
    )

    with pytest.raises(ValueError, match="include a timezone"):
        asyncio.run(
            WealthService(repository).capture_current_valuations(
                observed_at=datetime(2026, 7, 29, 12, 0),
            )
        )


def test_historical_performance_uses_only_exact_reviewed_snapshots() -> None:
    valuation_date = date(2025, 12, 31)
    observed_at = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    repository = SnapshotWealthRepository(
        accounts=[
            _account("brokerage", balance_milliunits=999_000),
            _account("ira", balance_milliunits=999_000),
        ],
        candidates=[
            _starting_balance("brokerage", 100_000),
            _starting_balance("ira", 200_000),
        ],
        snapshots=[
            _reviewed_snapshot(
                "brokerage",
                valuation_date=valuation_date,
                balance_milliunits=150_000,
                observed_at=observed_at,
            ),
            _reviewed_snapshot(
                "ira",
                valuation_date=valuation_date,
                balance_milliunits=250_000,
                observed_at=observed_at,
                source=ValuationSnapshotSource.IMPORTED_REVIEWED,
            ),
        ],
    )

    result = asyncio.run(
        WealthService(repository).calculate_money_weighted_performance(
            account_ids=["brokerage", "ira"],
            valuation_date=valuation_date,
            current_date=date(2026, 7, 29),
            ending_balance=None,
        )
    )

    assert result.ending_balance == 400
    assert result.valuation_source == "exact_reviewed_account_snapshots"
    assert "imported_reviewed, user_reviewed" in result.valuation_note
    assert observed_at.isoformat() in result.valuation_note
    assert repository.snapshot_lookup == (("brokerage", "ira"), valuation_date)


def test_missing_historical_snapshot_names_accounts_and_override_skips_lookup() -> None:
    valuation_date = date(2025, 12, 31)
    repository = SnapshotWealthRepository(
        accounts=[
            _account("brokerage", balance_milliunits=999_000),
            _account("ira", balance_milliunits=999_000),
        ],
        candidates=[
            _starting_balance("brokerage", 100_000),
            _starting_balance("ira", 200_000),
        ],
        snapshots=[
            _reviewed_snapshot(
                "brokerage",
                valuation_date=valuation_date,
                balance_milliunits=150_000,
                observed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
        ],
    )

    with pytest.raises(ValueError, match=r"missing account IDs: ira"):
        asyncio.run(
            WealthService(repository).calculate_money_weighted_performance(
                account_ids=["brokerage", "ira"],
                valuation_date=valuation_date,
                current_date=date(2026, 7, 29),
                ending_balance=None,
            )
        )

    repository.snapshot_lookup = None
    result = asyncio.run(
        WealthService(repository).calculate_money_weighted_performance(
            account_ids=["brokerage", "ira"],
            valuation_date=valuation_date,
            current_date=date(2026, 7, 29),
            ending_balance=500,
        )
    )

    assert result.ending_balance == 500
    assert result.valuation_source == "user_supplied_historical_balance"
    assert repository.snapshot_lookup is None
