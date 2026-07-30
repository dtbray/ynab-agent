"""Public wealth API request and response schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ynab_agent.planning.models import (
    CashFlowStream,
    CashFlowType,
    ValuationProvenance,
    WealthScenario,
)
from ynab_agent.planning.mortgage import MortgageProjection
from ynab_agent.services.wealth import AccountFreshness, ResolvedStartingPortfolio


class AccountFreshnessRead(BaseModel):
    """Public account freshness without persistence-only state."""

    account_id: str
    name: str
    account_type: str
    is_tracking: bool
    balance_milliunits: int
    last_reconciled_at: datetime | None
    latest_transaction: date | None
    stale_days: int | None

    @classmethod
    def from_service(
        cls,
        freshness: AccountFreshness,
        *,
        today: date,
    ) -> AccountFreshnessRead:
        latest_transaction = (
            date.fromisoformat(freshness.latest_transaction[:10])
            if freshness.latest_transaction is not None
            else None
        )
        return cls(
            account_id=freshness.account.id,
            name=freshness.account.name,
            account_type=freshness.account.type,
            is_tracking=not freshness.account.on_budget,
            balance_milliunits=freshness.account.balance_milliunits,
            last_reconciled_at=(
                datetime.fromisoformat(freshness.account.last_reconciled_at)
                if freshness.account.last_reconciled_at is not None
                else None
            ),
            latest_transaction=latest_transaction,
            stale_days=(
                (today - latest_transaction).days
                if latest_transaction is not None
                else None
            ),
        )


class AccountFreshnessResponse(BaseModel):
    """Read-only freshness collection."""

    accounts: list[AccountFreshnessRead] = Field(default_factory=list)


class MortgageProjectionRead(BaseModel):
    """Public payoff estimate and planner-ready cash-flow suggestions."""

    account_id: str
    as_of: date
    current_balance: float
    annual_interest_rate: float
    monthly_payment: float
    monthly_principal_and_interest: float
    monthly_escrow: float
    annual_principal_and_interest: float
    annual_escrow: float
    remaining_months: int
    estimated_payoff_date: date
    estimated_payoff_age: int
    source_sha256: str
    suggested_expense_stream: CashFlowStream
    suggested_contribution_stream: CashFlowStream

    @classmethod
    def from_domain(
        cls,
        projection: MortgageProjection,
        *,
        current_age: int,
    ) -> MortgageProjectionRead:
        annual_principal_and_interest = round(
            projection.annual_principal_and_interest,
            2,
        )
        return cls(
            account_id=projection.account_id,
            as_of=projection.as_of,
            current_balance=projection.current_balance,
            annual_interest_rate=projection.annual_interest_rate,
            monthly_payment=projection.monthly_payment,
            monthly_principal_and_interest=(
                projection.monthly_principal_and_interest
            ),
            monthly_escrow=projection.monthly_escrow,
            annual_principal_and_interest=annual_principal_and_interest,
            annual_escrow=round(projection.annual_escrow, 2),
            remaining_months=projection.remaining_months,
            estimated_payoff_date=projection.estimated_payoff_date,
            estimated_payoff_age=projection.estimated_payoff_age,
            source_sha256=projection.source_sha256,
            suggested_expense_stream=CashFlowStream(
                name="Mortgage principal and interest",
                flow_type=CashFlowType.EXPENSE,
                start_age=current_age,
                end_age=projection.estimated_payoff_age - 1,
                annual_amount=annual_principal_and_interest,
            ),
            suggested_contribution_stream=CashFlowStream(
                name="Redirected mortgage principal and interest",
                flow_type=CashFlowType.CONTRIBUTION,
                start_age=projection.estimated_payoff_age,
                annual_amount=annual_principal_and_interest,
            ),
        )


class ScenarioValidationRequest(WealthScenario):
    """A wealth scenario accepted for structural and cached-account validation."""

    model_config = ConfigDict(extra="forbid")


class ValuationProvenanceRead(BaseModel):
    """Public provenance for the resolved scenario portfolio."""

    source: str
    as_of: date | None
    account_ids: tuple[str, ...]
    source_sha256: str | None

    @classmethod
    def from_domain(
        cls,
        provenance: ValuationProvenance,
    ) -> ValuationProvenanceRead:
        return cls.model_validate(provenance, from_attributes=True)


class ScenarioValidationResponse(BaseModel):
    """Successful validation and its resolved planning input."""

    valid: Literal[True] = True
    scenario_name: str
    starting_portfolio: float
    valuation_provenance: ValuationProvenanceRead

    @classmethod
    def from_service(
        cls,
        scenario: WealthScenario,
        resolved: ResolvedStartingPortfolio,
    ) -> ScenarioValidationResponse:
        return cls(
            scenario_name=scenario.name,
            starting_portfolio=resolved.value,
            valuation_provenance=ValuationProvenanceRead.from_domain(
                resolved.provenance
            ),
        )
