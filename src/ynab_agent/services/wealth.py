"""Wealth-planning orchestration independent of CLI and database details."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from typing import Protocol, runtime_checkable

from ynab_agent.planning.models import (
    AccountValuationInput,
    LIQUID_PORTFOLIO_ROLES,
    ValuationProvenance,
    WealthScenario,
)
from ynab_agent.planning.mortgage import (
    MortgageProjection,
    project_mortgage,
)
from ynab_agent.planning.performance import (
    AccountCashFlow,
    CashFlowClassification,
    CashFlowReason,
    CashFlowTreatment,
    PerformanceResult,
    calculate_performance,
)
from ynab_agent.services.valuation_snapshots import (
    AccountValuationSnapshot,
    ValuationSnapshotSource,
)


@dataclass(frozen=True)
class WealthAccount:
    id: str
    name: str
    type: str
    on_budget: bool
    balance_milliunits: int
    closed: bool
    deleted: bool
    last_reconciled_at: str | None
    debt_interest_rates: str | None = None
    debt_minimum_payments: str | None = None
    debt_escrow_amounts: str | None = None


@dataclass(frozen=True)
class AccountFreshness:
    account: WealthAccount
    latest_transaction: str | None


@dataclass(frozen=True)
class CashFlowCandidate:
    transaction_id: str
    account_id: str
    date: date
    amount_milliunits: int
    transfer_account_id: str | None
    payee_name: str | None
    memo: str | None
    approved: bool | None = None
    cleared: str | None = None


@dataclass(frozen=True)
class ResolvedStartingPortfolio:
    """A simulation-ready portfolio value and the inputs used to resolve it."""

    value: float
    provenance: ValuationProvenance


def _portfolio_input_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class WealthRepository(Protocol):
    async def get_accounts(
        self,
        account_ids: Collection[str],
    ) -> Sequence[WealthAccount]: ...

    async def list_account_freshness(
        self,
        *,
        tracking_only: bool,
    ) -> Sequence[AccountFreshness]: ...

    async def get_cash_flow_candidates(
        self,
        account_ids: Collection[str],
        *,
        through: date,
    ) -> Sequence[CashFlowCandidate]: ...


@runtime_checkable
class ValuationSnapshotRepository(Protocol):
    """Optional persistence seam for dated account valuations."""

    async def upsert_valuation_snapshots(
        self,
        snapshots: Collection[AccountValuationSnapshot],
    ) -> int: ...

    async def get_reviewed_valuation_snapshots(
        self,
        account_ids: Collection[str],
        *,
        valuation_date: date,
    ) -> Sequence[AccountValuationSnapshot]: ...


class WealthService:
    """Resolve YNAB read models into validated planning inputs and analytics."""

    def __init__(self, repository: WealthRepository) -> None:
        self.repository = repository

    async def list_account_freshness(
        self,
        *,
        tracking_only: bool,
    ) -> Sequence[AccountFreshness]:
        """Return account freshness through the application-service boundary."""
        return await self.repository.list_account_freshness(
            tracking_only=tracking_only
        )

    async def project_mortgage(
        self,
        account_id: str,
        *,
        as_of: date,
        current_age: int,
    ) -> MortgageProjection:
        """Derive a mortgage payoff estimate from cached YNAB loan terms."""
        accounts = await self.repository.get_accounts((account_id,))
        if not accounts:
            raise ValueError("mortgage account was not found in the cache")
        account = accounts[0]
        if account.closed or account.deleted:
            raise ValueError("mortgage account is closed or deleted")
        if account.type.lower() != "mortgage":
            raise ValueError("selected account is not a YNAB mortgage account")
        return project_mortgage(
            account_id=account.id,
            balance_milliunits=account.balance_milliunits,
            interest_rates=_dated_integer_history(
                account.debt_interest_rates,
                "interest rates",
            ),
            minimum_payments=_dated_integer_history(
                account.debt_minimum_payments,
                "minimum payments",
            ),
            escrow_amounts=_dated_integer_history(
                account.debt_escrow_amounts,
                "escrow amounts",
                required=False,
            ),
            as_of=as_of,
            current_age=current_age,
        )

    async def capture_current_valuations(
        self,
        *,
        observed_at: datetime,
    ) -> tuple[AccountValuationSnapshot, ...]:
        """Capture active tracking-account balances after a successful sync."""
        if not isinstance(self.repository, ValuationSnapshotRepository):
            raise RuntimeError("wealth repository does not support valuation snapshots")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("valuation observation time must include a timezone")

        observed_utc = observed_at.astimezone(timezone.utc)
        rows = await self.repository.list_account_freshness(tracking_only=True)
        snapshots = tuple(
            AccountValuationSnapshot(
                account_id=row.account.id,
                valuation_date=observed_utc.date(),
                balance_milliunits=row.account.balance_milliunits,
                source=ValuationSnapshotSource.YNAB_SYNC,
                observed_at=observed_utc,
                reviewed=False,
            )
            for row in sorted(rows, key=lambda item: item.account.id)
            if not row.account.closed and not row.account.deleted
        )
        await self.repository.upsert_valuation_snapshots(snapshots)
        return snapshots

    async def resolve_starting_portfolio_with_provenance(
        self,
        scenario: WealthScenario,
    ) -> ResolvedStartingPortfolio:
        """Resolve the portfolio and retain replay-safe input provenance."""
        if scenario.starting_portfolio is not None:
            value = float(scenario.starting_portfolio)
            return ResolvedStartingPortfolio(
                value=value,
                provenance=ValuationProvenance(
                    source="explicit_scenario_input",
                    source_sha256=_portfolio_input_sha256(
                        {"starting_portfolio": value}
                    ),
                ),
            )

        selected_roles = {
            account.id: account.role
            for account in scenario.accounts
            if account.role in LIQUID_PORTFOLIO_ROLES
        }
        selected_ids = tuple(sorted(selected_roles))
        accounts = await self.repository.get_accounts(selected_ids)
        by_id = {account.id: account for account in accounts}
        missing = sorted(set(selected_ids) - set(by_id))
        if missing:
            raise ValueError(f"scenario account IDs not found in the cache: {', '.join(missing)}")
        inactive = sorted(
            account_id
            for account_id, account in by_id.items()
            if account.closed or account.deleted
        )
        if inactive:
            raise ValueError(f"scenario accounts are closed or deleted: {', '.join(inactive)}")

        valuation_inputs = [
            {
                "account_id": account_id,
                "balance_milliunits": by_id[account_id].balance_milliunits,
            }
            for account_id in selected_ids
        ]
        negative_accounts = [
            account_id
            for account_id in selected_ids
            if by_id[account_id].balance_milliunits < 0
        ]
        if negative_accounts:
            raise ValueError(
                "selected liquid accounts have negative balances: "
                + ", ".join(negative_accounts)
            )
        account_values = tuple(
            AccountValuationInput(
                account_id=account_id,
                value=by_id[account_id].balance_milliunits / 1000,
            )
            for account_id in selected_ids
        )
        total_milliunits = sum(
            by_id[account_id].balance_milliunits for account_id in selected_ids
        )
        if total_milliunits < 0:
            raise ValueError("selected liquid account balances produce a negative portfolio")
        validate_linked_account_values(
            scenario,
            account_values,
            resolved_total=total_milliunits / 1000,
        )
        return ResolvedStartingPortfolio(
            value=total_milliunits / 1000,
            provenance=ValuationProvenance(
                source="cached_liquid_accounts",
                account_ids=selected_ids,
                account_values=account_values,
                source_sha256=_portfolio_input_sha256(
                    {"accounts": valuation_inputs}
                ),
            ),
        )

    async def resolve_starting_portfolio(self, scenario: WealthScenario) -> float:
        """Return only the value for compatibility with existing callers."""
        resolved = await self.resolve_starting_portfolio_with_provenance(scenario)
        return resolved.value

    async def calculate_money_weighted_performance(
        self,
        *,
        account_ids: Collection[str],
        valuation_date: date,
        current_date: date,
        ending_balance: float | None,
    ) -> PerformanceResult:
        selected_ids = tuple(dict.fromkeys(account_ids))
        if not selected_ids:
            raise ValueError("performance requires at least one tracking account ID")
        if valuation_date > current_date:
            raise ValueError("performance valuation date cannot be in the future")

        accounts = await self.repository.get_accounts(selected_ids)
        by_id = {account.id: account for account in accounts}
        missing = sorted(set(selected_ids) - set(by_id))
        if missing:
            raise ValueError(f"account IDs not found in the cache: {', '.join(missing)}")
        inactive = sorted(
            account_id
            for account_id, account in by_id.items()
            if account.closed or account.deleted
        )
        if inactive:
            raise ValueError(f"accounts are closed or deleted: {', '.join(inactive)}")
        on_budget = sorted(
            account_id for account_id, account in by_id.items() if account.on_budget
        )
        if on_budget:
            raise ValueError(
                "performance requires off-budget tracking accounts: "
                + ", ".join(on_budget)
            )

        snapshots: tuple[AccountValuationSnapshot, ...] = ()
        if ending_balance is not None:
            terminal_value = ending_balance
        elif valuation_date == current_date:
            terminal_value = sum(account.balance_milliunits for account in accounts) / 1000
        else:
            if not isinstance(self.repository, ValuationSnapshotRepository):
                raise ValueError(
                    "a historical --as-of requires --ending-balance or exact reviewed "
                    "account valuation snapshots; this repository has no snapshot support"
                )
            snapshots = tuple(
                await self.repository.get_reviewed_valuation_snapshots(
                    selected_ids,
                    valuation_date=valuation_date,
                )
            )
            invalid_snapshots = [
                snapshot
                for snapshot in snapshots
                if snapshot.account_id not in selected_ids
                or snapshot.valuation_date != valuation_date
                or not snapshot.reviewed
            ]
            if invalid_snapshots:
                raise RuntimeError(
                    "valuation repository returned a non-exact or unreviewed snapshot"
                )
            snapshots_by_id = {snapshot.account_id: snapshot for snapshot in snapshots}
            if len(snapshots_by_id) != len(snapshots):
                raise RuntimeError(
                    "valuation repository returned multiple snapshots for one account"
                )
            missing_snapshots = sorted(set(selected_ids) - set(snapshots_by_id))
            if missing_snapshots:
                raise ValueError(
                    "a historical --as-of requires --ending-balance or exact reviewed "
                    f"valuation snapshots for {valuation_date.isoformat()}; missing account "
                    f"IDs: {', '.join(missing_snapshots)}"
                )
            terminal_value = (
                sum(
                    snapshots_by_id[account_id].balance_milliunits
                    for account_id in selected_ids
                )
                / 1000
            )

        candidates = await self.repository.get_cash_flow_candidates(
            selected_ids,
            through=valuation_date,
        )
        selected_set = set(selected_ids)
        included_flows: list[AccountCashFlow] = []
        classifications: list[CashFlowClassification] = []
        for candidate in candidates:
            treatment, reason = _classify_cash_flow(
                candidate,
                selected_account_ids=selected_set,
            )
            account_amount = candidate.amount_milliunits / 1000
            investor_cash_flow = (
                -account_amount if treatment is CashFlowTreatment.INCLUDED else None
            )
            classifications.append(
                CashFlowClassification(
                    transaction_id=candidate.transaction_id,
                    account_id=candidate.account_id,
                    date=candidate.date,
                    account_amount=account_amount,
                    treatment=treatment,
                    reason=reason,
                    investor_cash_flow=investor_cash_flow,
                    transfer_account_id=candidate.transfer_account_id,
                    payee_name=candidate.payee_name,
                    memo=candidate.memo,
                    approved=candidate.approved,
                    cleared=candidate.cleared,
                )
            )
            if treatment is CashFlowTreatment.EXCLUDED:
                continue
            included_flows.append(
                AccountCashFlow(
                    date=candidate.date,
                    amount=account_amount,
                )
            )

        if snapshots:
            source_names = ", ".join(
                sorted({snapshot.source.value for snapshot in snapshots})
            )
            observation_times = ", ".join(
                sorted({snapshot.observed_at.isoformat() for snapshot in snapshots})
            )
            valuation_source = "exact_reviewed_account_snapshots"
            valuation_note = (
                f"The terminal valuation uses exact reviewed snapshots dated "
                f"{valuation_date.isoformat()} from {source_names}; observations: "
                f"{observation_times}."
            )
        elif ending_balance is None:
            valuation_source = "current_cached_account_balances"
            valuation_note = (
                "The terminal valuation is the current cached YNAB balance. "
                "Only starting balances and transfers across the selected portfolio "
                "boundary are treated as investor cash flows."
            )
        elif valuation_date < current_date:
            valuation_source = "user_supplied_historical_balance"
            valuation_note = (
                "The terminal valuation is the reviewed historical balance supplied "
                "by the user; YNAB does not provide a dated historical valuation."
            )
        else:
            valuation_source = "user_supplied_current_balance"
            valuation_note = (
                "The terminal valuation is the current balance supplied by the user."
            )

        return calculate_performance(
            account_ids=selected_ids,
            cash_flows=included_flows,
            ending_balance=terminal_value,
            valuation_date=valuation_date,
            cash_flow_classifications=classifications,
            valuation_source=valuation_source,
            valuation_note=valuation_note,
        )


def validate_linked_account_values(
    scenario: WealthScenario,
    account_values: tuple[AccountValuationInput, ...],
    *,
    resolved_total: float,
) -> None:
    """Fail closed when live account values disagree with replay inputs."""
    linked_buckets = {
        bucket.account_id: bucket
        for bucket in scenario.tax_buckets
        if bucket.account_id is not None
    }
    if not linked_buckets:
        return
    values = {
        account.account_id: account.value
        for account in account_values
    }
    if set(linked_buckets) != set(values):
        raise ValueError(
            "linked tax buckets must exactly cover resolved account values"
        )
    for account_id, bucket in linked_buckets.items():
        if abs(bucket.starting_balance - values[account_id]) > 0.01:
            raise ValueError(
                "linked tax bucket starting balance does not match resolved "
                f"account value: {account_id}"
            )
    allocation = scenario.portfolio_allocation
    if allocation is None:
        return
    for account in allocation.accounts:
        expected = resolved_total * account.portfolio_weight
        if abs(values[account.account_id] - expected) > 0.01:
            raise ValueError(
                "allocation account weight does not match resolved account "
                f"value: {account.account_id}"
            )


def _dated_integer_history(
    payload: str | None,
    label: str,
    *,
    required: bool = True,
) -> dict[str, int]:
    if payload is None:
        if required:
            raise ValueError(f"mortgage has no cached {label}")
        return {}
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"mortgage cached {label} are invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"mortgage cached {label} are invalid")
    try:
        return {str(day): int(value) for day, value in decoded.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mortgage cached {label} are invalid") from exc


def _classify_cash_flow(
    candidate: CashFlowCandidate,
    *,
    selected_account_ids: Collection[str],
) -> tuple[CashFlowTreatment, CashFlowReason]:
    """Classify one tracking-account transaction without guessing market performance."""
    if candidate.amount_milliunits == 0:
        return CashFlowTreatment.EXCLUDED, CashFlowReason.ZERO_AMOUNT

    payee = (candidate.payee_name or "").strip().casefold()
    memo = (candidate.memo or "").strip().casefold()
    description = f"{payee} {memo}"

    if payee == "starting balance":
        return CashFlowTreatment.INCLUDED, CashFlowReason.STARTING_BALANCE
    if candidate.transfer_account_id is not None:
        if candidate.transfer_account_id in selected_account_ids:
            return CashFlowTreatment.EXCLUDED, CashFlowReason.INTERNAL_TRANSFER
        return CashFlowTreatment.INCLUDED, CashFlowReason.EXTERNAL_TRANSFER
    if "reconcil" in description:
        return CashFlowTreatment.EXCLUDED, CashFlowReason.RECONCILIATION_ADJUSTMENT
    if any(
        marker in description
        for marker in ("market value", "market update", "balance update", "valuation")
    ):
        return CashFlowTreatment.EXCLUDED, CashFlowReason.MARKET_VALUE_ADJUSTMENT
    if any(
        marker in description
        for marker in ("dividend", "interest", "distribution", "capital gain", "reinvest")
    ):
        return CashFlowTreatment.EXCLUDED, CashFlowReason.RETAINED_INCOME
    if any(
        marker in description
        for marker in ("fee", "commission", "expense", "withholding", "tax")
    ):
        return CashFlowTreatment.EXCLUDED, CashFlowReason.FEE_OR_TAX
    if "adjust" in description:
        return CashFlowTreatment.EXCLUDED, CashFlowReason.OTHER_ADJUSTMENT
    return CashFlowTreatment.EXCLUDED, CashFlowReason.NON_TRANSFER_ACTIVITY
