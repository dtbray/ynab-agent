"""Explicit, auditable housing state for wealth planning.

Housing is intentionally not a portfolio balance.  Cash enters the portfolio
only through a modeled sale, replacement transaction, or reverse mortgage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Protocol

from ynab_agent.planning.models import (
    HousingDecisionKind,
    HousingPlan,
    MortgageAssumptions,
    TaxAssumptions,
    WealthScenario,
)
from ynab_agent.planning.progressive_tax import calculate_income_tax_component_arrays

if TYPE_CHECKING:
    import numpy as np


HOUSING_MANIFEST_SCHEMA_VERSION = 3
HOUSING_ENGINE_VERSION = "annual_housing_v4"


def annual_mortgage_payment(mortgage: MortgageAssumptions) -> float:
    """Return the fixed annual payment for the explicit remaining term."""
    if mortgage.principal == 0:
        return 0.0
    if mortgage.annual_interest_rate == 0:
        return mortgage.principal / mortgage.remaining_years
    rate = mortgage.annual_interest_rate
    factor = (1 + rate) ** mortgage.remaining_years
    return mortgage.principal * rate * factor / (factor - 1)


@dataclass(frozen=True)
class HousingYear:
    """One deterministic nominal housing ledger row."""

    age: int
    action: str
    opening_home_value: float
    ending_home_value: float
    opening_mortgage_principal: float
    ending_mortgage_principal: float
    ending_reverse_mortgage_principal: float
    ending_home_equity: float
    maintenance: float
    property_tax: float
    insurance: float
    mortgage_payment: float
    rent: float
    care: float
    sale_price: float
    selling_cost: float
    reverse_mortgage_gross_proceeds: float
    reverse_mortgage_origination_cost: float
    forward_lien_payoff: float
    reverse_lien_payoff: float
    taxable_gain: float
    gain_exclusion: float
    federal_gain_tax: float
    state_gain_tax: float
    purchase_price: float
    replacement_cash_required: float
    liquid_deposit: float
    transaction_shortfall: float
    portfolio_spending: float
    portfolio_spending_excluding_care: float
    proceeds_destination_account_id: str | None
    gain_tax_accounting: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HousingDisposition:
    """Terminal home value available to an estate and its taxable gain."""

    pre_tax_value: float
    taxable_gain: float


@dataclass(frozen=True)
class HomeEquityCareFundingResult:
    """Per-trial exact-account result consumed by housing and future healthcare."""

    funded: np.ndarray
    unmet: np.ndarray
    realized_long_term_capital_gains: np.ndarray


class HomeEquityCareFundingPort(Protocol):
    """Fail-closed seam for funding care from one realized-equity account."""

    def fund_home_equity_care(
        self,
        trial_slice: slice,
        *,
        account_id: str,
        requested: float | np.ndarray,
    ) -> HomeEquityCareFundingResult:
        """Debit one exact account and return funded, unmet, and realized gain."""


@dataclass
class HousingState:
    """Mutable scalar property state shared across all economic trials."""

    plan: HousingPlan
    home_value: float
    cost_basis: float
    mortgage_principal: float
    mortgage_payment: float
    mortgage_years: int
    reverse_principal: float = 0.0
    rent_real: float = 0.0
    event_applied: bool = False

    @classmethod
    def from_plan(cls, plan: HousingPlan) -> HousingState:
        mortgage = plan.home.mortgage
        return cls(
            plan=plan,
            home_value=plan.home.current_value,
            cost_basis=plan.home.cost_basis,
            mortgage_principal=mortgage.principal,
            mortgage_payment=annual_mortgage_payment(mortgage),
            mortgage_years=mortgage.remaining_years,
        )

    def advance(
        self,
        *,
        age: int,
        tax_year: int,
        inflation_factor: float | np.ndarray,
        tax_assumptions: TaxAssumptions,
        default_costs_start_age: int,
        estimate_isolated_gain_tax: bool = False,
    ) -> HousingYear:
        """Advance one year and return portfolio cash-flow/action facts."""
        import numpy as np

        opening_home = self.home_value
        opening_mortgage = self.mortgage_principal
        action = "keep"
        sale_price = selling_cost = taxable_gain = exclusion = 0.0
        federal_gain_tax = state_gain_tax = purchase = liquid_deposit = 0.0
        reverse_gross = reverse_origination = 0.0
        forward_lien_payoff = reverse_lien_payoff = 0.0
        replacement_cash_required = 0.0
        transaction_spending = 0.0
        destination: str | None = None
        decision = self.plan.decision
        if decision.event_age == age and not self.event_applied:
            self.event_applied = True
            action = decision.kind.value
            destination = decision.proceeds_destination_account_id
            if decision.kind is HousingDecisionKind.REVERSE_MORTGAGE:
                principal = decision.reverse_mortgage_principal or 0.0
                maximum_principal = (
                    self.home_value * decision.reverse_mortgage_max_ltv
                )
                if principal > maximum_principal:
                    raise ValueError(
                        "reverse mortgage principal exceeds collateral LTV at event_age"
                    )
                origination = principal * decision.reverse_mortgage_origination_rate
                if principal - origination < self.mortgage_principal:
                    raise ValueError(
                        "reverse mortgage proceeds cannot pay off the forward mortgage"
                    )
                self.reverse_principal += principal
                reverse_gross = principal
                reverse_origination = origination
                forward_lien_payoff = self.mortgage_principal
                liquid_deposit = (
                    principal - origination - self.mortgage_principal
                )
                self.mortgage_principal = 0.0
                self.mortgage_payment = 0.0
                self.mortgage_years = 0
            else:
                sale_price = self.home_value
                selling_cost = sale_price * self.plan.home.selling_cost_rate
                raw_gain = max(0.0, sale_price - selling_cost - self.cost_basis)
                exclusion = min(raw_gain, self.plan.primary_residence_gain_exclusion)
                taxable_gain = raw_gain - exclusion
                if estimate_isolated_gain_tax:
                    federal_gain_tax, state_gain_tax = _isolated_gain_tax(
                        taxable_gain,
                        tax_year=tax_year,
                        assumptions=tax_assumptions,
                    )
                net = (
                    sale_price
                    - selling_cost
                    - self.mortgage_principal
                    - self.reverse_principal
                )
                forward_lien_payoff = self.mortgage_principal
                reverse_lien_payoff = self.reverse_principal
                self.home_value = 0.0
                self.cost_basis = 0.0
                self.mortgage_principal = 0.0
                self.mortgage_payment = 0.0
                self.mortgage_years = 0
                self.reverse_principal = 0.0
                if decision.kind in {
                    HousingDecisionKind.DOWNSIZE,
                    HousingDecisionKind.REPLACE,
                }:
                    purchase = decision.replacement_home_value or 0.0
                    replacement = decision.replacement_mortgage or MortgageAssumptions(
                        principal=0,
                        remaining_years=0,
                    )
                    cash_needed = max(0.0, purchase - replacement.principal)
                    replacement_cash_required = cash_needed
                    liquid_deposit = max(0.0, net - cash_needed)
                    transaction_spending = max(0.0, cash_needed - net)
                    self.home_value = purchase
                    self.cost_basis = purchase
                    self.mortgage_principal = replacement.principal
                    self.mortgage_payment = annual_mortgage_payment(replacement)
                    self.mortgage_years = replacement.remaining_years
                else:
                    liquid_deposit = max(0.0, net)
                    transaction_spending = max(0.0, -net)
                    if decision.kind is HousingDecisionKind.RENT:
                        self.rent_real = decision.annual_rent_real or 0.0

        costs_start = self.plan.costs_start_age
        if costs_start is None:
            costs_start = default_costs_start_age
        portfolio_funds_costs = age >= costs_start
        maintenance = self.home_value * self.plan.home.maintenance_rate
        property_tax = self.home_value * self.plan.home.property_tax_rate
        insurance = self.home_value * self.plan.home.insurance_rate
        payment = 0.0
        if self.mortgage_years > 0 and self.mortgage_principal > 0:
            interest = self.mortgage_principal * (
                self.plan.home.mortgage.annual_interest_rate
                if self.event_applied is False
                else (
                    decision.replacement_mortgage.annual_interest_rate
                    if decision.replacement_mortgage is not None
                    else self.plan.home.mortgage.annual_interest_rate
                )
            )
            payment = min(self.mortgage_payment, self.mortgage_principal + interest)
            self.mortgage_principal = max(0.0, self.mortgage_principal + interest - payment)
            self.mortgage_years -= 1
        inflation = np.asarray(inflation_factor, dtype=float)
        rent = (
            float(np.median(inflation)) * self.rent_real
            if portfolio_funds_costs
            else 0.0
        )
        care = 0.0
        if (
            self.plan.care is not None
            and self.plan.care.start_age is not None
            and self.plan.care.end_age is not None
            and self.plan.care.annual_cost_real is not None
            and self.plan.care.start_age <= age <= self.plan.care.end_age
        ):
            care = float(np.median(inflation)) * self.plan.care.annual_cost_real
        portfolio_spending_excluding_care = transaction_spending
        if portfolio_funds_costs:
            portfolio_spending_excluding_care += (
                maintenance + property_tax + insurance + payment + rent
            )
        portfolio_spending = portfolio_spending_excluding_care + care
        if self.home_value > 0:
            self.home_value *= 1 + self.plan.home.annual_appreciation_rate
        if self.reverse_principal > 0:
            self.reverse_principal *= 1 + decision.reverse_mortgage_interest_rate
        equity_after_reverse = (
            max(0.0, self.home_value - self.reverse_principal)
            if self.reverse_principal > 0
            else self.home_value
        )
        equity = equity_after_reverse - self.mortgage_principal
        return HousingYear(
            age=age,
            action=action,
            opening_home_value=opening_home,
            ending_home_value=self.home_value,
            opening_mortgage_principal=opening_mortgage,
            ending_mortgage_principal=self.mortgage_principal,
            ending_reverse_mortgage_principal=self.reverse_principal,
            ending_home_equity=equity,
            maintenance=maintenance,
            property_tax=property_tax,
            insurance=insurance,
            mortgage_payment=payment,
            rent=rent,
            care=care,
            sale_price=sale_price,
            selling_cost=selling_cost,
            reverse_mortgage_gross_proceeds=reverse_gross,
            reverse_mortgage_origination_cost=reverse_origination,
            forward_lien_payoff=forward_lien_payoff,
            reverse_lien_payoff=reverse_lien_payoff,
            taxable_gain=taxable_gain,
            gain_exclusion=exclusion,
            federal_gain_tax=federal_gain_tax,
            state_gain_tax=state_gain_tax,
            purchase_price=purchase,
            replacement_cash_required=replacement_cash_required,
            liquid_deposit=liquid_deposit,
            transaction_shortfall=transaction_spending,
            portfolio_spending=portfolio_spending,
            portfolio_spending_excluding_care=(
                portfolio_spending_excluding_care
            ),
            proceeds_destination_account_id=destination,
            gain_tax_accounting=(
                "isolated_projection_estimate"
                if estimate_isolated_gain_tax and taxable_gain > 0
                else (
                    "combined_annual_tax_engine"
                    if taxable_gain > 0
                    else "not_applicable"
                )
            ),
        )

    def disposition_value(
        self,
    ) -> HousingDisposition:
        """Return terminal estate proceeds and gain under nonrecourse reverse debt."""
        if self.home_value <= 0:
            return HousingDisposition(pre_tax_value=0.0, taxable_gain=0.0)
        selling_cost = self.home_value * self.plan.home.selling_cost_rate
        gain = max(0.0, self.home_value - selling_cost - self.cost_basis)
        taxable_gain = max(0.0, gain - self.plan.primary_residence_gain_exclusion)
        property_after_costs = self.home_value - selling_cost
        after_reverse = max(0.0, property_after_costs - self.reverse_principal)
        pre_tax = after_reverse - self.mortgage_principal
        return HousingDisposition(
            pre_tax_value=pre_tax,
            taxable_gain=taxable_gain,
        )


def _isolated_gain_tax(
    taxable_gain: float,
    *,
    tax_year: int,
    assumptions: TaxAssumptions,
) -> tuple[float, float]:
    if taxable_gain <= 0:
        return 0.0, 0.0
    if assumptions.progressive is None:
        return taxable_gain * assumptions.long_term_capital_gains_tax_rate, 0.0
    components = calculate_income_tax_component_arrays(
        assumptions.progressive,
        tax_year=tax_year,
        ordinary_income=0.0,
        long_term_capital_gains=taxable_gain,
        social_security_income=0.0,
    )
    return float(components.federal_income_tax), float(components.indiana_income_tax)


def housing_manifest(plan: HousingPlan) -> dict[str, object]:
    """Return the canonical, versioned housing assumptions manifest."""
    assumptions = plan.model_dump(mode="json")
    canonical = json.dumps(assumptions, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": HOUSING_MANIFEST_SCHEMA_VERSION,
        "engine_version": HOUSING_ENGINE_VERSION,
        "assumptions_sha256": hashlib.sha256(canonical).hexdigest(),
        "assumptions": assumptions,
        "timing": "decision_at_start_of_year_then_costs_then_year_end_appreciation",
        "portfolio_cashflow_timing": (
            "net liquidity is deposited into the exact account before "
            "allocation alignment and event-year portfolio returns"
        ),
        "proceeds_routing": "exact_linked_account_id_v1",
        "external_cash_basis_policy": (
            "external cash deposited to taxable accounts increases securities "
            "basis exactly once; home disposition gain remains separate"
        ),
        "liability_order": (
            "selling_or_origination_costs_then_forward_lien_then_reverse_lien_"
            "then_replacement_cash_then_net_deposit_or_transaction_shortfall"
        ),
        "care_funding_policy": (
            "an exact funding_account_id is debited first; only its unmet "
            "amount falls back to normal portfolio spending"
        ),
        "liquidity_invariant": (
            "home value is excluded from portfolio spending unless an explicit "
            "sale, replacement, rent, or reverse-mortgage event creates cash"
        ),
        "tax_scope": (
            "planning estimate using the versioned income-tax engine, primary-residence "
            "exclusion, explicit selling costs, and combined same-year modeled ordinary, "
            "Social Security, and long-term-gain income; not complete tax-preparation fidelity"
        ),
        "pre_cost_start_funding": "housing carrying costs are treated as externally funded",
        "retirement_boundary": (
            "liquidity events, portfolio-funded housing costs, and care begin "
            "no earlier than retirement_age"
        ),
        "reverse_mortgage_policy": (
            "gross nonrecourse principal is collateral-LTV bounded, pays origination "
            "and the forward lien first, stops forward payments, and cannot create "
            "negative terminal estate collateral"
        ),
        "replacement_property_policy": (
            "replacement homes inherit the configured appreciation, maintenance, "
            "property-tax, insurance, and selling-cost rates"
        ),
        "income_scope": (
            "combined gain tax includes income streams modeled by the retirement engine; "
            "unmodeled earned income and deductions remain outside scope"
        ),
        "terminal_tax_policy": (
            "home disposition gain stacks with tax-deferred, HSA, and taxable-account "
            "portfolio liquidation on one modeled return"
        ),
    }


def estimate_housing_state_bytes(trials: int, configured: bool) -> int:
    """Conservatively reserve annual audit, care, and terminal estate vectors."""
    return trials * 8 * 12 if configured else 0


def project_housing_plan(scenario: WealthScenario) -> dict[str, object]:
    """Project the deterministic housing ledger used by CLI and HTTP validation."""
    plan = scenario.housing_plan
    assumptions = scenario.tax_assumptions
    if plan is None or assumptions is None:
        raise ValueError("scenario must include a validated housing_plan and tax assumptions")
    state = HousingState.from_plan(plan)
    rows: list[dict[str, object]] = []
    for offset, age in enumerate(range(scenario.current_age, scenario.end_age)):
        tax_year = (
            assumptions.progressive.simulation_start_year + offset
            if assumptions.progressive is not None
            else 2026 + offset
        )
        row = state.advance(
            age=age,
            tax_year=tax_year,
            inflation_factor=(1 + scenario.inflation_rate) ** offset,
            tax_assumptions=assumptions,
            default_costs_start_age=scenario.retirement_age,
            estimate_isolated_gain_tax=True,
        )
        row_data = row.as_dict()
        destination_id = row.proceeds_destination_account_id
        destination_bucket = next(
            (
                bucket
                for bucket in scenario.tax_buckets
                if bucket.account_id == destination_id
            ),
            None,
        )
        row_data["destination_tax_treatment"] = (
            destination_bucket.tax_treatment.value
            if destination_bucket is not None
            else None
        )
        row_data["destination_owner_person_id"] = (
            destination_bucket.owner_person_id
            if destination_bucket is not None
            else None
        )
        rows.append(row_data)
    disposition = state.disposition_value()
    federal, state_tax = _isolated_gain_tax(
        disposition.taxable_gain,
        tax_year=(
            assumptions.progressive.simulation_start_year
            + scenario.end_age
            - scenario.current_age
            if assumptions.progressive is not None
            else 2026 + scenario.end_age - scenario.current_age
        ),
        assumptions=assumptions,
    )
    return {
        "manifest": housing_manifest(plan),
        "annual_housing": rows,
        "ending_disposition_value": disposition.pre_tax_value,
        "after_tax_ending_disposition_value": (
            disposition.pre_tax_value - federal - state_tax
        ),
    }
