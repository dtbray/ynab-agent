"""Targeted read models for wealth-planning use cases."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import date, datetime, timezone

from sqlalchemy import text

from ynab_agent.db._coercion import as_int
from ynab_agent.db.manager import DatabaseManager
from ynab_agent.services.valuation_snapshots import (
    AccountValuationSnapshot,
    ValuationSnapshotConflictError,
    ValuationSnapshotSource,
)
from ynab_agent.services.wealth import (
    AccountFreshness,
    CashFlowCandidate,
    WealthAccount,
)


class SqlWealthRepository:
    """SQL-backed wealth read model with account-scoped queries."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    @staticmethod
    def _account(row: dict[str, object]) -> WealthAccount:
        return WealthAccount(
            id=str(row["id"]),
            name=str(row["name"] or ""),
            type=str(row["type"] or ""),
            on_budget=bool(row["on_budget"]),
            balance_milliunits=(
                as_int(row["balance"]) if row["balance"] is not None else 0
            ),
            closed=bool(row["closed"]),
            deleted=bool(row["deleted"]),
            last_reconciled_at=(
                str(row["last_reconciled_at"])
                if row["last_reconciled_at"] is not None
                else None
            ),
            debt_interest_rates=(
                str(row["debt_interest_rates"])
                if row.get("debt_interest_rates") is not None
                else None
            ),
            debt_minimum_payments=(
                str(row["debt_minimum_payments"])
                if row.get("debt_minimum_payments") is not None
                else None
            ),
            debt_escrow_amounts=(
                str(row["debt_escrow_amounts"])
                if row.get("debt_escrow_amounts") is not None
                else None
            ),
        )

    async def get_accounts(
        self,
        account_ids: Collection[str],
    ) -> Sequence[WealthAccount]:
        unique_ids = tuple(dict.fromkeys(account_ids))
        if not unique_ids:
            return ()
        placeholders = ", ".join(f":account_{index}" for index in range(len(unique_ids)))
        params = {
            f"account_{index}": account_id
            for index, account_id in enumerate(unique_ids)
        }
        rows = await self.database.fetch_all(
            f"""
            SELECT
              id, name, type, on_budget, balance, closed, deleted,
              last_reconciled_at, debt_interest_rates,
              debt_minimum_payments, debt_escrow_amounts
            FROM accounts
            WHERE id IN ({placeholders})
            """,
            params,
        )
        return tuple(self._account(row) for row in rows)

    async def list_account_freshness(
        self,
        *,
        tracking_only: bool,
    ) -> Sequence[AccountFreshness]:
        tracking_clause = "AND a.on_budget IS FALSE" if tracking_only else ""
        rows = await self.database.fetch_all(
            f"""
            SELECT
              a.id, a.name, a.type, a.on_budget, a.balance, a.closed, a.deleted,
              a.last_reconciled_at, a.debt_interest_rates,
              a.debt_minimum_payments, a.debt_escrow_amounts,
              MAX(t.date) AS latest_transaction
            FROM accounts a
            LEFT JOIN transactions t
              ON t.account_id = a.id
             AND t.deleted IS FALSE
            WHERE a.deleted IS FALSE
              AND a.closed IS FALSE
              {tracking_clause}
            GROUP BY
              a.id, a.name, a.type, a.on_budget, a.balance, a.closed, a.deleted,
              a.last_reconciled_at, a.debt_interest_rates,
              a.debt_minimum_payments, a.debt_escrow_amounts
            ORDER BY a.on_budget, a.name
            """
        )
        return tuple(
            AccountFreshness(
                account=self._account(row),
                latest_transaction=(
                    str(row["latest_transaction"])
                    if row["latest_transaction"] is not None
                    else None
                ),
            )
            for row in rows
        )

    async def get_cash_flow_candidates(
        self,
        account_ids: Collection[str],
        *,
        through: date,
    ) -> Sequence[CashFlowCandidate]:
        unique_ids = tuple(dict.fromkeys(account_ids))
        if not unique_ids:
            return ()
        placeholders = ", ".join(f":account_{index}" for index in range(len(unique_ids)))
        params: dict[str, object] = {
            f"account_{index}": account_id
            for index, account_id in enumerate(unique_ids)
        }
        params["through"] = through.isoformat()
        rows = await self.database.fetch_all(
            f"""
            SELECT
              id, account_id, date, amount, transfer_account_id, payee_name, memo,
              approved, cleared
            FROM transactions
            WHERE deleted IS FALSE
              AND account_id IN ({placeholders})
              AND date IS NOT NULL
              AND date <= :through
              AND amount IS NOT NULL
            ORDER BY date, id
            """,
            params,
        )
        return tuple(
            CashFlowCandidate(
                transaction_id=str(row["id"]),
                account_id=str(row["account_id"]),
                date=date.fromisoformat(str(row["date"])[:10]),
                amount_milliunits=as_int(row["amount"]),
                transfer_account_id=(
                    str(row["transfer_account_id"])
                    if row["transfer_account_id"] is not None
                    else None
                ),
                payee_name=(
                    str(row["payee_name"]) if row["payee_name"] is not None else None
                ),
                memo=str(row["memo"]) if row["memo"] is not None else None,
                approved=(
                    bool(row["approved"]) if row["approved"] is not None else None
                ),
                cleared=(
                    str(row["cleared"]) if row["cleared"] is not None else None
                ),
            )
            for row in rows
        )

    async def upsert_valuation_snapshots(
        self,
        snapshots: Collection[AccountValuationSnapshot],
    ) -> int:
        """Persist observations, allowing only newer values to replace a unique row."""
        normalized = sorted(
            snapshots,
            key=lambda snapshot: (
                snapshot.account_id,
                snapshot.valuation_date,
                snapshot.source.value,
                snapshot.observed_at,
                snapshot.reviewed,
            ),
        )
        changed = 0
        async with self.database.session_factory() as session:
            for snapshot in normalized:
                key = {
                    "account_id": snapshot.account_id,
                    "valuation_date": snapshot.valuation_date.isoformat(),
                    "source": snapshot.source.value,
                }
                existing_result = await session.execute(
                    text(
                        """
                        SELECT balance_milliunits, observed_at, reviewed
                        FROM account_valuation_snapshots
                        WHERE account_id = :account_id
                          AND valuation_date = :valuation_date
                          AND source = :source
                        """
                    ),
                    key,
                )
                existing = existing_result.mappings().one_or_none()
                observed_at = snapshot.observed_at.astimezone(timezone.utc)
                if existing is None:
                    await session.execute(
                        text(
                            """
                            INSERT INTO account_valuation_snapshots (
                              account_id, valuation_date, balance_milliunits, source,
                              observed_at, reviewed
                            )
                            VALUES (
                              :account_id, :valuation_date, :balance_milliunits, :source,
                              :observed_at, :reviewed
                            )
                            """
                        ),
                        {
                            **key,
                            "balance_milliunits": snapshot.balance_milliunits,
                            "observed_at": observed_at.isoformat(),
                            "reviewed": snapshot.reviewed,
                        },
                    )
                    changed += 1
                    continue

                existing_observed_at = _as_utc_datetime(existing["observed_at"])
                if observed_at < existing_observed_at:
                    continue
                if (
                    observed_at == existing_observed_at
                    and snapshot.balance_milliunits
                    != int(existing["balance_milliunits"])
                ):
                    raise ValuationSnapshotConflictError(
                        "conflicting valuation snapshots share account, date, source, "
                        f"and observed_at: {snapshot.account_id}, "
                        f"{snapshot.valuation_date.isoformat()}, {snapshot.source.value}, "
                        f"{observed_at.isoformat()}"
                    )

                reviewed = (
                    snapshot.reviewed
                    if observed_at > existing_observed_at
                    else bool(existing["reviewed"]) or snapshot.reviewed
                )
                if (
                    observed_at == existing_observed_at
                    and reviewed == bool(existing["reviewed"])
                ):
                    continue
                await session.execute(
                    text(
                        """
                        UPDATE account_valuation_snapshots
                        SET balance_milliunits = :balance_milliunits,
                            observed_at = :observed_at,
                            reviewed = :reviewed
                        WHERE account_id = :account_id
                          AND valuation_date = :valuation_date
                          AND source = :source
                        """
                    ),
                    {
                        **key,
                        "balance_milliunits": snapshot.balance_milliunits,
                        "observed_at": observed_at.isoformat(),
                        "reviewed": reviewed,
                    },
                )
                changed += 1
            await session.commit()
        return changed

    async def get_reviewed_valuation_snapshots(
        self,
        account_ids: Collection[str],
        *,
        valuation_date: date,
    ) -> Sequence[AccountValuationSnapshot]:
        """Return at most one reviewed observation per account for the exact date."""
        unique_ids = tuple(dict.fromkeys(account_ids))
        if not unique_ids:
            return ()
        placeholders = ", ".join(f":account_{index}" for index in range(len(unique_ids)))
        params: dict[str, object] = {
            f"account_{index}": account_id
            for index, account_id in enumerate(unique_ids)
        }
        params["valuation_date"] = valuation_date.isoformat()
        rows = await self.database.fetch_all(
            f"""
            SELECT
              account_id, valuation_date, balance_milliunits, source,
              observed_at, reviewed
            FROM account_valuation_snapshots
            WHERE account_id IN ({placeholders})
              AND valuation_date = :valuation_date
              AND reviewed IS TRUE
            ORDER BY account_id, observed_at DESC, source, balance_milliunits
            """,
            params,
        )
        by_account: dict[str, AccountValuationSnapshot] = {}
        for row in rows:
            account_id = str(row["account_id"])
            if account_id in by_account:
                continue
            by_account[account_id] = AccountValuationSnapshot(
                account_id=account_id,
                valuation_date=date.fromisoformat(str(row["valuation_date"])[:10]),
                balance_milliunits=as_int(row["balance_milliunits"]),
                source=ValuationSnapshotSource(str(row["source"])),
                observed_at=_as_utc_datetime(row["observed_at"]),
                reviewed=bool(row["reviewed"]),
            )
        return tuple(
            by_account[account_id]
            for account_id in unique_ids
            if account_id in by_account
        )


def _as_utc_datetime(value: object) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
