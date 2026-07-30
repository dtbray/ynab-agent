"""Money-weighted performance calculations for irregular YNAB cash flows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
import math


@dataclass(frozen=True)
class AccountCashFlow:
    """One dated change to a tracked investment account."""

    date: date
    amount: float


class CashFlowTreatment(StrEnum):
    """Whether an account transaction participates in the XIRR calculation."""

    INCLUDED = "included"
    EXCLUDED = "excluded"


class CashFlowReason(StrEnum):
    """Auditable reason for including or excluding one account transaction."""

    STARTING_BALANCE = "starting_balance"
    EXTERNAL_TRANSFER = "external_transfer"
    INTERNAL_TRANSFER = "internal_transfer"
    RETAINED_INCOME = "retained_income"
    RECONCILIATION_ADJUSTMENT = "reconciliation_adjustment"
    MARKET_VALUE_ADJUSTMENT = "market_value_adjustment"
    FEE_OR_TAX = "fee_or_tax"
    OTHER_ADJUSTMENT = "other_adjustment"
    NON_TRANSFER_ACTIVITY = "non_transfer_activity"
    ZERO_AMOUNT = "zero_amount"


@dataclass(frozen=True)
class CashFlowClassification:
    """Source evidence and treatment for one candidate YNAB transaction."""

    transaction_id: str
    account_id: str
    date: date
    account_amount: float
    treatment: CashFlowTreatment
    reason: CashFlowReason
    investor_cash_flow: float | None
    transfer_account_id: str | None
    payee_name: str | None
    memo: str | None
    approved: bool | None
    cleared: str | None


class PerformanceCalculationError(ValueError):
    """Base class for performance inputs that cannot produce a truthful result."""


class AmbiguousPerformanceError(PerformanceCalculationError):
    """Raised when a cash-flow pattern can have more than one valid XIRR."""


class NonConvergentPerformanceError(PerformanceCalculationError):
    """Raised when no finite XIRR can be calculated."""


@dataclass(frozen=True)
class PerformanceResult:
    """Money-weighted performance across one or more investment accounts."""

    account_ids: tuple[str, ...]
    valuation_date: str
    first_cash_flow_date: str
    cash_flow_count: int
    included_transaction_count: int
    excluded_transaction_count: int
    cash_flow_policy: str
    valuation_source: str
    valuation_note: str
    cash_flow_classifications: tuple[CashFlowClassification, ...]
    contributions: float
    withdrawals: float
    ending_balance: float
    xirr: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_performance(
    *,
    account_ids: Iterable[str],
    cash_flows: Iterable[AccountCashFlow],
    ending_balance: float,
    valuation_date: date,
    excluded_transaction_count: int = 0,
    cash_flow_classifications: Iterable[CashFlowClassification] = (),
    valuation_source: str = "provided_ending_balance",
    valuation_note: str = (
        "Money-weighted return calculated from classified external cash flows "
        "and the supplied terminal valuation."
    ),
) -> PerformanceResult:
    """Calculate XIRR after converting YNAB account activity to investor cash flows."""
    try:
        from pyxirr import xirr
    except ImportError as exc:  # pragma: no cover - exercised without the planner extra
        raise RuntimeError(
            'performance calculations require the planner extra: '
            'pip install "ynab-agent[planner]"'
        ) from exc

    if not math.isfinite(ending_balance):
        raise PerformanceCalculationError("ending investment balance must be finite")
    if ending_balance < 0:
        raise PerformanceCalculationError("ending investment balance must not be negative")

    materialized_flows = tuple(cash_flows)
    classifications = tuple(cash_flow_classifications)
    by_date: dict[date, float] = defaultdict(float)
    contributions = 0.0
    withdrawals = 0.0
    for flow in materialized_flows:
        if not math.isfinite(flow.amount):
            raise PerformanceCalculationError("account cash-flow amounts must be finite")
        if flow.date > valuation_date:
            continue
        if flow.amount > 0:
            contributions += flow.amount
        elif flow.amount < 0:
            withdrawals += -flow.amount
        # Positive YNAB account activity is cash invested, and therefore a
        # negative cash flow from the investor's perspective.
        by_date[flow.date] -= flow.amount

    if not by_date:
        raise PerformanceCalculationError(
            "no defensible external cash flows were found on or before the valuation date"
        )
    by_date[valuation_date] += ending_balance
    dated_flows = sorted((flow_date, amount) for flow_date, amount in by_date.items())
    amounts = [amount for _, amount in dated_flows if amount != 0]
    if not any(amount < 0 for amount in amounts) or not any(amount > 0 for amount in amounts):
        raise PerformanceCalculationError(
            "XIRR requires at least one investment and one positive withdrawal "
            "or terminal valuation"
        )

    sign_changes = sum(
        left * right < 0
        for left, right in zip(amounts, amounts[1:], strict=False)
    )
    if sign_changes > 1 and _xirr_root_count(dated_flows) > 1:
        raise AmbiguousPerformanceError(
            "the cash-flow NPV profile contains multiple valid XIRR values; "
            "performance is ambiguous"
        )

    rate = xirr(
        [flow_date for flow_date, amount in dated_flows if amount != 0],
        amounts,
        silent=True,
    )
    if rate is None or not math.isfinite(float(rate)):
        raise NonConvergentPerformanceError(
            "XIRR did not converge to a finite result for these account cash flows"
        )

    ids = tuple(dict.fromkeys(account_ids))
    audited_excluded_count = sum(
        item.treatment is CashFlowTreatment.EXCLUDED for item in classifications
    )
    if classifications:
        excluded_transaction_count = audited_excluded_count
    return PerformanceResult(
        account_ids=ids,
        valuation_date=valuation_date.isoformat(),
        first_cash_flow_date=dated_flows[0][0].isoformat(),
        cash_flow_count=len(dated_flows),
        included_transaction_count=(
            sum(item.treatment is CashFlowTreatment.INCLUDED for item in classifications)
            if classifications
            else len(materialized_flows)
        ),
        excluded_transaction_count=excluded_transaction_count,
        cash_flow_policy="external_transfers_and_starting_balances",
        valuation_source=valuation_source,
        valuation_note=valuation_note,
        cash_flow_classifications=classifications,
        contributions=contributions,
        withdrawals=withdrawals,
        ending_balance=ending_balance,
        xirr=float(rate),
    )


def _xirr_root_count(dated_flows: list[tuple[date, float]]) -> int:
    """Count roots over a broad logarithmic rate domain for non-conventional flows."""
    from pyxirr import xnpv, zero_crossing_points

    # XIRR is only defined for rates above -100%. Sampling log(1 + rate)
    # provides useful resolution near zero without losing very large positive roots.
    lower = math.log(1e-6)
    upper = math.log(1_000_001)
    sample_count = 10_001
    step = (upper - lower) / (sample_count - 1)
    rates = [math.exp(lower + index * step) - 1 for index in range(sample_count)]
    values = xnpv(
        rates,
        [(flow_date, amount) for flow_date, amount in dated_flows if amount != 0],
    )
    finite_values = [
        value for value in values if value is not None and math.isfinite(value)
    ]
    return len(zero_crossing_points(finite_values))
