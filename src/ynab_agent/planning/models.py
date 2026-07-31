"""Validated inputs for wealth-planning scenarios."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ynab_agent.planning.allocation import PortfolioAllocationPlan
from ynab_agent.planning.healthcare_models import HealthcareAssumptions
from ynab_agent.planning.spending_guardrails import (
    RetirementSpendingPlan,
)
from ynab_agent.planning.household_models import Household, age_months_on


MAX_SCENARIO_NAME_LENGTH = 200
MAX_ACCOUNT_ID_LENGTH = 128
MAX_SCENARIO_ACCOUNTS = 100
MAX_INCOME_STREAMS = 64
MAX_INCOME_STREAM_NAME_LENGTH = 200
MAX_CASH_FLOW_STREAMS = 64
MAX_CASH_FLOW_STREAM_NAME_LENGTH = 200
MAX_TAX_BUCKETS = 10
MAX_SPENDING_TIERS = 8
MAX_SPENDING_TIER_NAME_LENGTH = 100
MAX_IRMAA_LOOKBACK_YEARS = 2
MAX_ASSET_LOCATION_PREFERENCES = 16
MAX_ASSET_CLASS_NAME_LENGTH = 64
SUPPORTED_ASSET_LOCATION_CLASSES = {
    "us_equity",
    "international_equity",
    "bonds",
    "cash",
}


class AccountRole(StrEnum):
    """How a YNAB account participates in a wealth plan."""

    RETIREMENT = "retirement"
    TAXABLE = "taxable"
    CASH = "cash"
    REAL_ESTATE = "real_estate"
    DEBT = "debt"
    OTHER = "other"
    EXCLUDED = "excluded"


class ReturnModel(StrEnum):
    """How annual investment and inflation paths are generated."""

    LOGNORMAL = "lognormal"
    HISTORICAL_BOOTSTRAP = "historical_bootstrap"


class CashFlowType(StrEnum):
    """How a bounded recurring cash flow affects a retirement scenario."""

    CONTRIBUTION = "contribution"
    EXPENSE = "expense"


class TaxTreatment(StrEnum):
    """Tax character of one aggregated portfolio balance."""

    TAX_DEFERRED = "tax_deferred"
    ROTH = "roth"
    TAXABLE = "taxable"
    HSA = "hsa"
    CASH = "cash"


class HousingDecisionKind(StrEnum):
    """A deliberate event that can change an otherwise illiquid home."""

    KEEP = "keep"
    SELL = "sell"
    DOWNSIZE = "downsize"
    REPLACE = "replace"
    RENT = "rent"
    REVERSE_MORTGAGE = "reverse_mortgage"


class MortgageAssumptions(BaseModel):
    """A fixed-rate, fully amortizing mortgage balance and remaining term."""

    model_config = ConfigDict(allow_inf_nan=False)

    principal: float = Field(ge=0)
    annual_interest_rate: float = Field(default=0, ge=0, le=0.3)
    remaining_years: int = Field(default=0, ge=0, le=50)

    @model_validator(mode="after")
    def validate_term(self) -> MortgageAssumptions:
        if (self.principal > 0) != (self.remaining_years > 0):
            raise ValueError("mortgage principal and remaining_years must both be positive or zero")
        return self


class HomeAsset(BaseModel):
    """Explicit illiquid home valuation and bounded carrying assumptions."""

    model_config = ConfigDict(allow_inf_nan=False)

    current_value: float = Field(gt=0)
    cost_basis: float = Field(gt=0)
    annual_appreciation_rate: float = Field(default=0.025, ge=-0.2, le=0.3)
    maintenance_rate: float = Field(default=0.01, ge=0, le=0.2)
    property_tax_rate: float = Field(default=0.01, ge=0, le=0.2)
    insurance_rate: float = Field(default=0.005, ge=0, le=0.2)
    selling_cost_rate: float = Field(default=0.07, ge=0, le=0.25)
    mortgage: MortgageAssumptions = Field(
        default_factory=lambda: MortgageAssumptions(principal=0, remaining_years=0)
    )


class HousingDecision(BaseModel):
    """One explicit housing action, modeled at the start of ``event_age``."""

    model_config = ConfigDict(allow_inf_nan=False)

    kind: HousingDecisionKind = HousingDecisionKind.KEEP
    event_age: int | None = Field(default=None, ge=0, le=130)
    proceeds_destination_account_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ACCOUNT_ID_LENGTH,
    )
    replacement_home_value: float | None = Field(default=None, gt=0)
    replacement_mortgage: MortgageAssumptions | None = None
    annual_rent_real: float | None = Field(default=None, ge=0)
    reverse_mortgage_principal: float | None = Field(default=None, gt=0)
    reverse_mortgage_max_ltv: float = Field(default=0.8, gt=0, le=1)
    reverse_mortgage_interest_rate: float = Field(default=0.07, ge=0, le=0.3)
    reverse_mortgage_origination_rate: float = Field(default=0.03, ge=0, le=0.2)

    @model_validator(mode="after")
    def validate_action(self) -> HousingDecision:
        liquid = self.kind is not HousingDecisionKind.KEEP
        if liquid != (self.event_age is not None):
            raise ValueError("non-keep housing decisions require event_age; keep must omit it")
        if liquid != (self.proceeds_destination_account_id is not None):
            raise ValueError(
                "housing liquidity events require proceeds_destination_account_id; "
                "keep must omit it"
            )
        replacement = self.kind in {
            HousingDecisionKind.DOWNSIZE,
            HousingDecisionKind.REPLACE,
        }
        if replacement != (self.replacement_home_value is not None):
            raise ValueError("downsize/replace require replacement_home_value only")
        if not replacement and self.replacement_mortgage is not None:
            raise ValueError("replacement_mortgage requires downsize or replace")
        if (self.kind is HousingDecisionKind.RENT) != (self.annual_rent_real is not None):
            raise ValueError("rent requires annual_rent_real only")
        reverse = self.kind is HousingDecisionKind.REVERSE_MORTGAGE
        if reverse != (self.reverse_mortgage_principal is not None):
            raise ValueError("reverse_mortgage requires reverse_mortgage_principal only")
        return self


class CareFundingPlan(BaseModel):
    """Deterministic care costs or an exact-account reserve for healthcare LTC."""

    model_config = ConfigDict(allow_inf_nan=False)

    start_age: int | None = Field(default=None, ge=0, le=130)
    end_age: int | None = Field(default=None, ge=0, le=130)
    annual_cost_real: float | None = Field(default=None, gt=0)
    funding_account_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ACCOUNT_ID_LENGTH,
        description=(
            "Exact cash or taxable account containing realized home-equity "
            "proceeds. Funded care is debited here before any portfolio fallback."
        ),
    )

    @model_validator(mode="after")
    def validate_ages(self) -> CareFundingPlan:
        cost_fields = (self.start_age, self.end_age, self.annual_cost_real)
        if any(value is not None for value in cost_fields) and not all(
            value is not None for value in cost_fields
        ):
            raise ValueError(
                "deterministic care requires start_age, end_age, and annual_cost_real together"
            )
        if (
            self.start_age is not None
            and self.end_age is not None
            and self.end_age < self.start_age
        ):
            raise ValueError("care end_age must be at least start_age")
        if all(value is None for value in cost_fields) and self.funding_account_id is None:
            raise ValueError("a reserve-only care plan requires funding_account_id")
        return self


class HousingPlan(BaseModel):
    """An illiquid home plus a single deliberate disposition/financing event."""

    model_config = ConfigDict(allow_inf_nan=False)

    home: HomeAsset
    decision: HousingDecision = Field(default_factory=HousingDecision)
    costs_start_age: int | None = Field(default=None, ge=0, le=130)
    primary_residence_gain_exclusion: float = Field(default=250_000, ge=0, le=1_000_000)
    care: CareFundingPlan | None = None


class IncomeTaxTreatment(StrEnum):
    """How a retirement income stream participates in the tax model."""

    UNSPECIFIED = "unspecified"
    ORDINARY = "ordinary"
    SOCIAL_SECURITY = "social_security"
    TAX_FREE = "tax_free"


class TaxModel(StrEnum):
    """Annual income-tax calculation selected for account-aware planning."""

    EFFECTIVE_RATES = "effective_rates"
    PROGRESSIVE_US_INDIANA = "progressive_us_indiana"


class FederalFilingStatus(StrEnum):
    """Federal filing statuses supported by the versioned policy."""

    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married_filing_jointly"
    MARRIED_FILING_SEPARATELY = "married_filing_separately"
    HEAD_OF_HOUSEHOLD = "head_of_household"
    QUALIFYING_SURVIVING_SPOUSE = "qualifying_surviving_spouse"


class FutureTaxPolicyMode(StrEnum):
    """How a known-year policy is projected beyond its effective year."""

    INFLATION_INDEXED = "inflation_indexed"
    FIXED_NOMINAL = "fixed_nominal"


class WithdrawalPolicy(StrEnum):
    """How annual portfolio withdrawals are allocated across tax treatments."""

    ORDERED = "ordered"
    PROPORTIONAL = "proportional"


class TaxTreatmentFraction(BaseModel):
    """One target share of a proportional annual withdrawal."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    tax_treatment: TaxTreatment
    fraction: float = Field(gt=0, le=1)


class RothConversionStrategy(BaseModel):
    """Current-year bracket-fill rule for tax-deferred Roth conversions."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    policy_id: str = "roth_bracket_fill_v1"
    start_age: int = Field(ge=0, le=120)
    end_age: int = Field(ge=0, le=120)
    target_federal_ordinary_bracket_rate: float = Field(ge=0.10, le=0.35)
    max_annual_conversion_real: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_rule(self) -> RothConversionStrategy:
        if self.policy_id != "roth_bracket_fill_v1":
            raise ValueError("unsupported Roth conversion policy")
        if self.end_age < self.start_age:
            raise ValueError("Roth conversion end_age must be at least start_age")
        if self.target_federal_ordinary_bracket_rate not in {
            0.10,
            0.12,
            0.22,
            0.24,
            0.32,
            0.35,
        }:
            raise ValueError("Roth conversion target must be a bounded federal bracket rate")
        return self


class CapitalGainHarvestStrategy(BaseModel):
    """Current-year taxable-gain harvesting rule and realization cap."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    policy_id: str = "capital_gain_bracket_fill_v1"
    start_age: int = Field(ge=0, le=120)
    end_age: int = Field(ge=0, le=120)
    target_federal_long_term_capital_gains_rate: float = Field(ge=0, le=0.15)
    max_annual_gain_real: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_rule(self) -> CapitalGainHarvestStrategy:
        if self.policy_id != "capital_gain_bracket_fill_v1":
            raise ValueError("unsupported capital-gain harvest policy")
        if self.end_age < self.start_age:
            raise ValueError("capital-gain harvest end_age must be at least start_age")
        if self.target_federal_long_term_capital_gains_rate not in {0.0, 0.15}:
            raise ValueError("capital-gain harvest target must be 0% or 15%")
        return self


class AssetLocationPreference(BaseModel):
    """Typed advisory hook for a future multi-asset allocation engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_class: str = Field(min_length=1, max_length=MAX_ASSET_CLASS_NAME_LENGTH)
    preferred_tax_treatments: list[TaxTreatment] = Field(
        min_length=1,
        max_length=MAX_TAX_BUCKETS,
    )

    @model_validator(mode="after")
    def validate_treatments(self) -> AssetLocationPreference:
        if len(self.preferred_tax_treatments) != len(set(self.preferred_tax_treatments)):
            raise ValueError("asset-location tax treatments must be unique")
        return self


class TaxStrategyAssumptions(BaseModel):
    """Versioned, replayable annual tax-strategy decision rules."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    policy_id: str = "tax_strategy_v1"
    withdrawal_policy: WithdrawalPolicy = WithdrawalPolicy.ORDERED
    proportional_withdrawal_fractions: list[TaxTreatmentFraction] = Field(
        default_factory=list,
        max_length=MAX_TAX_BUCKETS,
    )
    roth_conversion: RothConversionStrategy | None = None
    capital_gain_harvest: CapitalGainHarvestStrategy | None = None
    asset_location_preferences: list[AssetLocationPreference] = Field(
        default_factory=list,
        max_length=MAX_ASSET_LOCATION_PREFERENCES,
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> TaxStrategyAssumptions:
        fractions = self.proportional_withdrawal_fractions
        treatments = [item.tax_treatment for item in fractions]
        if len(treatments) != len(set(treatments)):
            raise ValueError("proportional withdrawal tax treatments must be unique")
        if self.withdrawal_policy is WithdrawalPolicy.PROPORTIONAL:
            if not fractions or abs(sum(item.fraction for item in fractions) - 1) > 1e-9:
                raise ValueError("proportional withdrawal fractions must total 1")
        elif fractions:
            raise ValueError("ordered withdrawals must not define proportional fractions")
        asset_classes = [item.asset_class for item in self.asset_location_preferences]
        if len(asset_classes) != len(set(asset_classes)):
            raise ValueError("asset-location asset classes must be unique")
        if not set(asset_classes).issubset(SUPPORTED_ASSET_LOCATION_CLASSES):
            raise ValueError(
                "asset-location preferences require a supported asset class"
            )
        return self


class IrmaaLookbackMagi(BaseModel):
    """Known historical MAGI used before simulated lookback years exist."""

    model_config = ConfigDict(allow_inf_nan=False)

    tax_year: int = Field(ge=1900, le=2200)
    magi: float = Field(ge=0)
    filing_status: FederalFilingStatus | None = None


class ProgressiveTaxAssumptions(BaseModel):
    """Household and future-law inputs for the progressive tax policy."""

    model_config = ConfigDict(allow_inf_nan=False)

    policy_id: str = Field(default="us_in_2026_v1", min_length=1, max_length=64)
    filing_status: FederalFilingStatus
    simulation_start_year: int = Field(default=2026, ge=2026, le=2200)
    taxpayer_birth_year: int = Field(ge=1900, le=2200)
    spouse_birth_year: int | None = Field(default=None, ge=1900, le=2200)
    future_policy_mode: FutureTaxPolicyMode = FutureTaxPolicyMode.INFLATION_INDEXED
    bracket_inflation_rate: float = Field(default=0.025, ge=-0.05, le=0.15)
    healthcare_inflation_rate: float = Field(
        default=0.05,
        ge=-0.05,
        le=0.25,
        description=(
            "Reserved for a future indexed healthcare policy; the v1 ACA and "
            "IRMAA policy remains static and manifests that assumption"
        ),
    )
    taxpayer_blind: bool = False
    spouse_blind: bool = False
    dependent_count: int = Field(default=0, ge=0, le=20)
    dependent_child_count: int = Field(default=0, ge=0, le=20)
    first_time_adopted_child_count: int = Field(default=0, ge=0, le=20)
    indiana_resident: bool = True
    aca_household_size: int = Field(default=1, ge=1, le=20)
    aca_benchmark_annual_premium: float | None = Field(default=None, ge=0)
    irmaa_lookback_magi: list[IrmaaLookbackMagi] = Field(
        default_factory=list,
        max_length=MAX_IRMAA_LOOKBACK_YEARS,
    )
    married_filing_separately_lived_with_spouse: bool = False

    @model_validator(mode="after")
    def validate_household(self) -> ProgressiveTaxAssumptions:
        joint = self.filing_status is FederalFilingStatus.MARRIED_FILING_JOINTLY
        if joint != (self.spouse_birth_year is not None):
            raise ValueError(
                "married_filing_jointly requires spouse_birth_year and "
                "other filing statuses must omit it"
            )
        if self.spouse_blind and self.spouse_birth_year is None:
            raise ValueError("spouse_blind requires spouse_birth_year")
        if self.dependent_child_count > self.dependent_count:
            raise ValueError("dependent_child_count cannot exceed dependent_count")
        if self.first_time_adopted_child_count > self.dependent_child_count:
            raise ValueError("first_time_adopted_child_count cannot exceed dependent_child_count")
        years = [item.tax_year for item in self.irmaa_lookback_magi]
        if len(years) != len(set(years)):
            raise ValueError("IRMAA lookback tax years must be unique")
        return self


LIQUID_PORTFOLIO_ROLES = {
    AccountRole.RETIREMENT,
    AccountRole.TAXABLE,
    AccountRole.CASH,
}


class ScenarioAccount(BaseModel):
    """A YNAB tracking account and its planning role."""

    model_config = ConfigDict(allow_inf_nan=False)

    id: str = Field(min_length=1, max_length=MAX_ACCOUNT_ID_LENGTH)
    role: AccountRole
    owner_person_id: str | None = Field(default=None, min_length=1, max_length=64)


class IncomeStream(BaseModel):
    """An annual retirement income stream, expressed in today's dollars."""

    model_config = ConfigDict(allow_inf_nan=False)

    name: str = Field(
        min_length=1,
        max_length=MAX_INCOME_STREAM_NAME_LENGTH,
    )
    start_age: int = Field(ge=0, le=120)
    annual_amount: float = Field(ge=0)
    end_age: int | None = Field(default=None, ge=0, le=120)
    inflation_adjusted: bool = True
    tax_treatment: IncomeTaxTreatment = IncomeTaxTreatment.UNSPECIFIED

    @model_validator(mode="after")
    def validate_ages(self) -> IncomeStream:
        if self.end_age is not None and self.end_age < self.start_age:
            raise ValueError("income stream end_age must be at least start_age")
        return self


class CashFlowStream(BaseModel):
    """An age-bounded recurring contribution or expense in today's dollars."""

    model_config = ConfigDict(allow_inf_nan=False)

    name: str = Field(
        min_length=1,
        max_length=MAX_CASH_FLOW_STREAM_NAME_LENGTH,
    )
    flow_type: CashFlowType
    start_age: int = Field(ge=0, le=120)
    annual_amount: float = Field(ge=0)
    end_age: int | None = Field(default=None, ge=0, le=120)
    inflation_adjusted: bool = True
    destination_tax_treatment: TaxTreatment | None = None

    @model_validator(mode="after")
    def validate_ages(self) -> CashFlowStream:
        if self.end_age is not None and self.end_age < self.start_age:
            raise ValueError("cash flow stream end_age must be at least start_age")
        return self


class SpendingTier(BaseModel):
    """A named minimum annual real-spending goal below the full plan."""

    model_config = ConfigDict(allow_inf_nan=False)

    name: str = Field(
        min_length=1,
        max_length=MAX_SPENDING_TIER_NAME_LENGTH,
    )
    annual_amount: float = Field(gt=0)


class AccountValuationInput(BaseModel):
    """One immutable account value used to resolve a live portfolio."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    account_id: str = Field(min_length=1, max_length=MAX_ACCOUNT_ID_LENGTH)
    value: float = Field(ge=0)


class ValuationProvenance(BaseModel):
    """Origin of a portfolio value resolved before a simulation run."""

    model_config = ConfigDict(allow_inf_nan=False)

    source: str = Field(min_length=1, max_length=128)
    as_of: date | None = None
    account_ids: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_SCENARIO_ACCOUNTS,
    )
    account_values: tuple[AccountValuationInput, ...] = Field(
        default=(),
        max_length=MAX_SCENARIO_ACCOUNTS,
    )
    source_sha256: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_account_ids(self) -> ValuationProvenance:
        if len(self.account_ids) != len(set(self.account_ids)):
            raise ValueError("valuation provenance account IDs must be unique")
        if any(
            not account_id or len(account_id) > MAX_ACCOUNT_ID_LENGTH
            for account_id in self.account_ids
        ):
            raise ValueError("valuation provenance account IDs are invalid")
        value_ids = tuple(value.account_id for value in self.account_values)
        if len(value_ids) != len(set(value_ids)):
            raise ValueError(
                "valuation provenance account values must be unique"
            )
        if self.account_values and value_ids != self.account_ids:
            raise ValueError(
                "valuation provenance account values must match account IDs "
                "in stable order"
            )
        return self


class TaxBucket(BaseModel):
    """One bounded, aggregated balance with a single tax treatment."""

    model_config = ConfigDict(allow_inf_nan=False)

    tax_treatment: TaxTreatment
    starting_balance: float = Field(ge=0)
    taxable_basis: float | None = Field(default=None, ge=0)
    contribution_fraction: float = Field(default=0, ge=0, le=1)
    owner_person_id: str | None = Field(default=None, min_length=1, max_length=64)
    account_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ACCOUNT_ID_LENGTH,
    )

    @model_validator(mode="after")
    def validate_basis(self) -> TaxBucket:
        if self.tax_treatment is TaxTreatment.TAXABLE:
            if self.taxable_basis is None:
                raise ValueError("taxable buckets require taxable_basis")
        elif self.taxable_basis is not None:
            raise ValueError("taxable_basis is only valid for taxable buckets")
        return self


class TaxAssumptions(BaseModel):
    """Explicit effective-rate assumptions for account-aware withdrawals."""

    model_config = ConfigDict(allow_inf_nan=False)

    ordinary_income_tax_rate: float = Field(ge=0, lt=0.75)
    long_term_capital_gains_tax_rate: float = Field(ge=0, lt=0.75)
    tax_model: TaxModel = TaxModel.EFFECTIVE_RATES
    progressive: ProgressiveTaxAssumptions | None = None
    strategy: TaxStrategyAssumptions | None = None
    social_security_taxable_fraction: float = Field(default=0.85, ge=0, le=0.85)
    qualified_hsa_withdrawal_fraction: float = Field(default=1, ge=0, le=1)
    early_distribution_penalty_rate: float = Field(default=0.10, ge=0, lt=0.75)
    hsa_early_distribution_penalty_rate: float = Field(default=0.20, ge=0, lt=0.75)
    taxable_account_annual_tax_drag_rate: float = Field(default=0, ge=0, lt=0.25)
    rmd_start_age: int = Field(default=75, ge=70, le=90)
    apply_required_minimum_distributions: bool = True
    withdrawal_order: list[TaxTreatment] = Field(
        default_factory=lambda: [
            TaxTreatment.CASH,
            TaxTreatment.TAXABLE,
            TaxTreatment.TAX_DEFERRED,
            TaxTreatment.ROTH,
            TaxTreatment.HSA,
        ],
        min_length=1,
        max_length=MAX_TAX_BUCKETS,
    )
    retirement_surplus_destination: TaxTreatment = TaxTreatment.TAXABLE

    @model_validator(mode="after")
    def validate_withdrawal_order(self) -> TaxAssumptions:
        if len(self.withdrawal_order) != len(set(self.withdrawal_order)):
            raise ValueError("withdrawal_order tax treatments must be unique")
        if (self.tax_model is TaxModel.PROGRESSIVE_US_INDIANA) != (self.progressive is not None):
            raise ValueError(
                "progressive assumptions must be present exactly when "
                "tax_model is progressive_us_indiana"
            )
        return self


class WealthScenario(BaseModel):
    """Assumptions for a reproducible annual retirement simulation."""

    model_config = ConfigDict(allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=MAX_SCENARIO_NAME_LENGTH)
    current_age: int = Field(ge=0, le=100)
    retirement_age: int = Field(ge=0, le=120)
    end_age: int = Field(default=95, ge=1, le=130)
    accounts: list[ScenarioAccount] = Field(
        default_factory=list,
        max_length=MAX_SCENARIO_ACCOUNTS,
    )
    starting_portfolio: float | None = Field(default=None, ge=0)
    annual_contribution: float = Field(default=0, ge=0)
    contribution_inflation_adjusted: bool = True
    annual_spending: float = Field(gt=0)
    retirement_spending_plan: RetirementSpendingPlan | None = None
    portfolio_allocation: PortfolioAllocationPlan | None = None
    income_streams: list[IncomeStream] = Field(
        default_factory=list,
        max_length=MAX_INCOME_STREAMS,
    )
    cash_flow_streams: list[CashFlowStream] = Field(
        default_factory=list,
        max_length=MAX_CASH_FLOW_STREAMS,
    )
    tax_buckets: list[TaxBucket] = Field(
        default_factory=list,
        max_length=MAX_TAX_BUCKETS,
    )
    tax_assumptions: TaxAssumptions | None = None
    housing_plan: HousingPlan | None = None
    spending_tiers: list[SpendingTier] = Field(
        default_factory=list,
        max_length=MAX_SPENDING_TIERS,
    )
    legacy_target_real: float | None = Field(default=None, ge=0)
    household: Household | None = None
    healthcare: HealthcareAssumptions | None = None
    inflation_rate: float = Field(default=0.025, ge=-0.05, le=0.25)
    return_model: ReturnModel = ReturnModel.LOGNORMAL
    return_mean: float = Field(default=0.06, gt=-0.99, le=1.0)
    return_volatility: float = Field(default=0.16, ge=0, le=1.0)
    historical_block_size: int = Field(default=5, ge=1, le=50)
    annual_fee_rate: float = Field(default=0.002, ge=0, lt=0.25)
    withdrawal_tax_rate: float = Field(default=0, ge=0, lt=0.75)
    trials: int = Field(default=10_000, ge=100, le=100_000)
    seed: int = Field(default=20260728, ge=0, le=2**64 - 1)

    @model_validator(mode="after")
    def validate_scenario(self) -> WealthScenario:
        if self.retirement_age < self.current_age:
            raise ValueError("retirement_age must be at least current_age")
        if self.end_age <= self.retirement_age:
            raise ValueError("end_age must be greater than retirement_age")
        if self.starting_portfolio is None and not any(
            account.role in LIQUID_PORTFOLIO_ROLES for account in self.accounts
        ):
            raise ValueError(
                "provide starting_portfolio or at least one retirement, taxable, or cash account"
            )
        account_ids = [account.id for account in self.accounts]
        if len(account_ids) != len(set(account_ids)):
            raise ValueError("scenario account IDs must be unique")
        account_by_id = {
            account.id: account
            for account in self.accounts
        }
        linked_bucket_ids = [
            bucket.account_id
            for bucket in self.tax_buckets
            if bucket.account_id is not None
        ]
        if linked_bucket_ids and len(linked_bucket_ids) != len(
            self.tax_buckets
        ):
            raise ValueError(
                "tax bucket account linkage must be complete or omitted"
            )
        if len(linked_bucket_ids) != len(set(linked_bucket_ids)):
            raise ValueError("tax bucket account IDs must be unique")
        unknown_bucket_accounts = (
            set(linked_bucket_ids) - set(account_by_id)
        )
        if unknown_bucket_accounts:
            raise ValueError(
                "tax bucket account_id must identify a scenario account"
            )
        for bucket in self.tax_buckets:
            if bucket.account_id is None:
                continue
            account = account_by_id[bucket.account_id]
            if account.role not in LIQUID_PORTFOLIO_ROLES:
                raise ValueError(
                    "tax bucket account_id must identify a liquid scenario "
                    "account"
                )
            if bucket.owner_person_id != account.owner_person_id:
                raise ValueError(
                    "linked tax bucket and scenario account owners must match"
                )
        tier_names = [tier.name.casefold() for tier in self.spending_tiers]
        if len(tier_names) != len(set(tier_names)):
            raise ValueError("spending tier names must be unique")
        if {"planned_spending", "legacy_target"} & set(tier_names):
            raise ValueError("spending tier names cannot use reserved goal names")
        if any(
            tier.annual_amount > self.annual_spending
            for tier in self.spending_tiers
        ):
            raise ValueError(
                "spending tier annual amounts cannot exceed annual_spending"
            )
        if (
            self.retirement_spending_plan is not None
            and abs(
                self.retirement_spending_plan.baseline.total
                - self.annual_spending
            )
            > 0.01
        ):
            raise ValueError(
                "retirement spending tier baseline must equal annual_spending"
            )
        if self.household is not None:
            primary_age = age_months_on(
                self.household.people[0].birth_date,
                self.household.plan_start_date,
            ) // 12
            if primary_age != self.current_age:
                raise ValueError(
                    "current_age must equal the first household person's age "
                    "on plan_start_date"
                )
            person_ids = {person.id for person in self.household.people}
            unknown_owners = {
                bucket.owner_person_id
                for bucket in self.tax_buckets
                if bucket.owner_person_id is not None
                and bucket.owner_person_id not in person_ids
            }
            if unknown_owners:
                raise ValueError("tax bucket owner_person_id must identify a household person")
            unknown_account_owners = {
                account.owner_person_id
                for account in self.accounts
                if account.owner_person_id is not None
                and account.owner_person_id not in person_ids
            }
            if unknown_account_owners:
                raise ValueError(
                    "account owner_person_id must identify a household person"
                )
            progressive = (
                self.tax_assumptions.progressive
                if self.tax_assumptions is not None
                else None
            )
            if progressive is not None:
                people = self.household.people
                if len(people) == 2:
                    if (
                        progressive.filing_status
                        is not FederalFilingStatus.MARRIED_FILING_JOINTLY
                        or progressive.spouse_birth_year is None
                    ):
                        raise ValueError(
                            "two-person progressive households require "
                            "married-filing-jointly assumptions"
                        )
                    if (
                        progressive.taxpayer_birth_year != people[0].birth_date.year
                        or progressive.spouse_birth_year != people[1].birth_date.year
                    ):
                        raise ValueError(
                            "progressive taxpayer birth years must match household people"
                        )
                elif progressive.filing_status is FederalFilingStatus.MARRIED_FILING_JOINTLY:
                    raise ValueError(
                        "one-person progressive households cannot file jointly"
                    )
                elif progressive.taxpayer_birth_year != people[0].birth_date.year:
                    raise ValueError(
                        "progressive taxpayer birth year must match household person"
                    )
        elif any(
            account.owner_person_id is not None
            for account in self.accounts
        ) or any(
            bucket.owner_person_id is not None
            for bucket in self.tax_buckets
        ):
            raise ValueError("account ownership requires a household")
        if self.healthcare is not None:
            if self.household is None:
                raise ValueError("healthcare assumptions require a household")
            healthcare_ids = {
                person.person_id for person in self.healthcare.people
            }
            household_ids = {
                person.id for person in self.household.people
            }
            if healthcare_ids != household_ids:
                raise ValueError(
                    "healthcare assumptions must cover every household person "
                    "exactly once"
                )
            healthcare_has_ltc = any(
                person.long_term_care is not None
                for person in self.healthcare.people
            )
            deterministic_housing_care = (
                self.housing_plan is not None
                and self.housing_plan.care is not None
                and self.housing_plan.care.annual_cost_real is not None
            )
            if healthcare_has_ltc and deterministic_housing_care:
                raise ValueError(
                    "long-term-care costs must be modeled by healthcare or "
                    "housing care, not both"
                )
            if self.healthcare.ltc_funding_source == "home_equity":
                reserve = (
                    self.housing_plan.care
                    if self.housing_plan is not None
                    else None
                )
                if (
                    reserve is None
                    or reserve.funding_account_id is None
                    or reserve.annual_cost_real is not None
                ):
                    raise ValueError(
                        "home-equity LTC funding requires a reserve-only housing "
                        "care plan with funding_account_id"
                    )
        if self.portfolio_allocation is not None:
            configured_ids = {
                account.account_id
                for account in self.portfolio_allocation.accounts
            }
            liquid_ids = {
                account.id
                for account in self.accounts
                if account.role in LIQUID_PORTFOLIO_ROLES
            }
            if self.accounts and configured_ids != liquid_ids:
                raise ValueError(
                    "allocation account IDs must exactly cover liquid scenario "
                    "accounts"
                )
            if self.accounts and self.tax_buckets:
                if set(linked_bucket_ids) != configured_ids:
                    raise ValueError(
                        "tax-aware allocation requires one linked tax bucket "
                        "for every allocation account"
                    )
                if self.starting_portfolio is not None:
                    bucket_by_account = {
                        bucket.account_id: bucket
                        for bucket in self.tax_buckets
                    }
                    for allocation_account in (
                        self.portfolio_allocation.accounts
                    ):
                        bucket = bucket_by_account[
                            allocation_account.account_id
                        ]
                        expected = (
                            self.starting_portfolio
                            * allocation_account.portfolio_weight
                        )
                        if abs(bucket.starting_balance - expected) > 0.01:
                            raise ValueError(
                                "allocation account weights must match linked "
                                "tax bucket starting balances"
                            )
            tax_strategy = (
                self.tax_assumptions.strategy
                if self.tax_assumptions is not None
                else None
            )
            if (
                tax_strategy is not None
                and tax_strategy.asset_location_preferences
                and (
                    not self.accounts
                    or set(linked_bucket_ids) != configured_ids
                )
            ):
                raise ValueError(
                    "asset-location preferences require linked allocation "
                    "and tax accounts"
                )
            if any(
                point.age <= self.current_age
                for account in self.portfolio_allocation.accounts
                for point in account.glide_path
            ):
                raise ValueError(
                    "allocation glide-path ages must be greater than current_age"
                )
        self._validate_tax_model()
        self._validate_housing_plan()
        return self

    def _validate_housing_plan(self) -> None:
        plan = self.housing_plan
        if plan is None:
            return
        if not self.tax_buckets or self.tax_assumptions is None:
            raise ValueError("housing_plan requires explicit tax_buckets and tax_assumptions")
        event_age = plan.decision.event_age
        if event_age is not None and not self.current_age <= event_age < self.end_age:
            raise ValueError("housing event_age must be within the simulated ages")
        if event_age is not None and event_age < self.retirement_age:
            raise ValueError(
                "housing liquidity events before retirement_age are not supported"
            )
        if plan.costs_start_age is not None and not (
            self.current_age <= plan.costs_start_age < self.end_age
        ):
            raise ValueError("housing costs_start_age must be within the simulated ages")
        if (
            plan.costs_start_age is not None
            and plan.costs_start_age < self.retirement_age
        ):
            raise ValueError(
                "housing costs_start_age before retirement_age is not supported"
            )
        destination_id = plan.decision.proceeds_destination_account_id
        destination_bucket: TaxBucket | None = None
        if destination_id is not None:
            destination_buckets = [
                bucket
                for bucket in self.tax_buckets
                if bucket.account_id == destination_id
            ]
            if len(destination_buckets) != 1:
                raise ValueError(
                    "proceeds_destination_account_id must identify exactly one "
                    "linked tax bucket"
                )
            destination_bucket = destination_buckets[0]
            if destination_bucket.tax_treatment not in {
                TaxTreatment.CASH,
                TaxTreatment.TAXABLE,
            }:
                raise ValueError(
                    "housing proceeds destination account must be cash or taxable"
                )
        if plan.care is not None:
            if plan.care.start_age is not None and plan.care.end_age is not None:
                if not (
                    self.current_age
                    <= plan.care.start_age
                    <= plan.care.end_age
                    < self.end_age
                ):
                    raise ValueError("care ages must be within the simulated ages")
                if plan.care.start_age < self.retirement_age:
                    raise ValueError("care funding before retirement_age is not supported")
            funding_account_id = plan.care.funding_account_id
            if funding_account_id is not None:
                funding_start_age = plan.care.start_age
                if (
                    funding_start_age is None
                    and self.healthcare is not None
                    and self.healthcare.ltc_funding_source == "home_equity"
                ):
                    funding_start_age = min(
                        person.long_term_care.minimum_onset_age
                        for person in self.healthcare.people
                        if person.long_term_care is not None
                    )
                if (
                    event_age is None
                    or funding_start_age is None
                    or event_age > funding_start_age
                    or funding_account_id != destination_id
                    or destination_bucket is None
                ):
                    raise ValueError(
                        "home-equity care funding requires the exact proceeds "
                        "destination account and a liquidity event by care start_age"
                    )
        decision = plan.decision
        replacement = decision.replacement_mortgage
        if (
            replacement is not None
            and decision.replacement_home_value is not None
            and replacement.principal > decision.replacement_home_value
        ):
            raise ValueError(
                "replacement mortgage principal cannot exceed replacement home value"
            )
        if decision.kind is HousingDecisionKind.REVERSE_MORTGAGE:
            principal = decision.reverse_mortgage_principal or 0.0
            maximum_principal = (
                plan.home.current_value * decision.reverse_mortgage_max_ltv
            )
            if principal > maximum_principal:
                raise ValueError(
                    "reverse mortgage principal exceeds configured collateral LTV"
                )
            available_after_origination = principal * (
                1 - decision.reverse_mortgage_origination_rate
            )
            if available_after_origination < plan.home.mortgage.principal:
                raise ValueError(
                    "reverse mortgage proceeds must pay off the existing mortgage"
                )

    def _validate_tax_model(self) -> None:
        tax_aware = bool(self.tax_buckets) or self.tax_assumptions is not None
        if not tax_aware:
            return
        if not self.tax_buckets or self.tax_assumptions is None:
            raise ValueError("tax_buckets and tax_assumptions must be provided together")
        if self.starting_portfolio is None and not self.accounts:
            raise ValueError(
                "tax-aware scenarios require an explicit starting_portfolio "
                "or selected scenario accounts"
            )
        if self.withdrawal_tax_rate != 0:
            raise ValueError("tax-aware scenarios must set withdrawal_tax_rate to 0")
        bucket_keys = [
            (
                bucket.tax_treatment,
                bucket.owner_person_id,
                bucket.account_id,
            )
            for bucket in self.tax_buckets
        ]
        if len(bucket_keys) != len(set(bucket_keys)):
            raise ValueError(
                "tax_buckets must have unique tax-treatment, owner, and "
                "account identities"
            )
        treatments = [
            bucket.tax_treatment for bucket in self.tax_buckets
        ]
        bucket_total = sum(bucket.starting_balance for bucket in self.tax_buckets)
        if (
            self.starting_portfolio is not None
            and abs(bucket_total - self.starting_portfolio) > 0.01
        ):
            raise ValueError("tax bucket balances must equal starting_portfolio")
        contribution_total = sum(bucket.contribution_fraction for bucket in self.tax_buckets)
        uses_default_contribution_allocation = self.annual_contribution > 0 or any(
            cash_flow.flow_type is CashFlowType.CONTRIBUTION
            and cash_flow.destination_tax_treatment is None
            for cash_flow in self.cash_flow_streams
        )
        if uses_default_contribution_allocation and abs(contribution_total - 1) > 1e-9:
            raise ValueError(
                "tax bucket contribution fractions must total 1 "
                "when an unallocated contribution is present"
            )
        if (
            not uses_default_contribution_allocation
            and min(abs(contribution_total), abs(contribution_total - 1)) > 1e-9
        ):
            raise ValueError("tax bucket contribution fractions must total 0 or 1")
        treatment_set = set(treatments)
        if not treatment_set.issubset(self.tax_assumptions.withdrawal_order):
            raise ValueError("withdrawal_order must include every tax bucket treatment")
        if self.tax_assumptions.retirement_surplus_destination not in treatment_set:
            raise ValueError("retirement_surplus_destination requires a matching tax bucket")
        if self.retirement_age < 60 and TaxTreatment.ROTH in treatment_set:
            raise ValueError(
                "tax-aware Roth modeling currently requires retirement_age of at least 60"
            )
        for income_stream in self.income_streams:
            if income_stream.tax_treatment is IncomeTaxTreatment.UNSPECIFIED:
                raise ValueError("tax-aware scenarios require tax_treatment on every income stream")
        if self.tax_assumptions.progressive is not None:
            policy = self.tax_assumptions.progressive
            modeled_age = policy.simulation_start_year - policy.taxpayer_birth_year
            if modeled_age not in {self.current_age, self.current_age + 1}:
                raise ValueError(
                    "taxpayer_birth_year must align with current_age and simulation_start_year"
                )
        strategy = self.tax_assumptions.strategy
        if strategy is not None:
            if strategy.policy_id != "tax_strategy_v1":
                raise ValueError("unsupported tax strategy policy")
            if strategy.withdrawal_policy is WithdrawalPolicy.PROPORTIONAL:
                fractions = {
                    item.tax_treatment
                    for item in strategy.proportional_withdrawal_fractions
                }
                if fractions != treatment_set:
                    raise ValueError(
                        "proportional withdrawal fractions must cover every tax bucket"
                    )
            if (
                strategy.roth_conversion is not None
                or strategy.capital_gain_harvest is not None
            ) and self.tax_assumptions.progressive is None:
                raise ValueError(
                    "Roth conversion and gain-harvest strategies require progressive tax"
                )
            if strategy.roth_conversion is not None and not {
                TaxTreatment.TAX_DEFERRED,
                TaxTreatment.ROTH,
            }.issubset(treatment_set):
                raise ValueError(
                    "Roth conversion strategy requires tax_deferred and roth buckets"
                )
            if (
                strategy.roth_conversion is not None
                and (
                    strategy.roth_conversion.start_age < self.retirement_age
                    or strategy.roth_conversion.end_age >= self.end_age
                )
            ):
                raise ValueError(
                    "Roth conversion ages must fall within modeled retirement years"
                )
            if (
                strategy.capital_gain_harvest is not None
                and TaxTreatment.TAXABLE not in treatment_set
            ):
                raise ValueError("capital-gain harvest strategy requires a taxable bucket")
            if (
                strategy.capital_gain_harvest is not None
                and (
                    strategy.capital_gain_harvest.start_age < self.retirement_age
                    or strategy.capital_gain_harvest.end_age >= self.end_age
                )
            ):
                raise ValueError(
                    "capital-gain harvest ages must fall within modeled retirement years"
                )
            for preference in strategy.asset_location_preferences:
                if not set(preference.preferred_tax_treatments).issubset(treatment_set):
                    raise ValueError(
                        "asset-location preferences require matching tax buckets"
                    )
        if (
            self.tax_assumptions.ordinary_income_tax_rate
            + self.tax_assumptions.early_distribution_penalty_rate
            >= 1
        ):
            raise ValueError("ordinary tax and early-distribution penalty must total less than 1")
        if (
            self.tax_assumptions.ordinary_income_tax_rate
            + self.tax_assumptions.hsa_early_distribution_penalty_rate
            >= 1
        ):
            raise ValueError("ordinary tax and HSA penalty must total less than 1")
        for cash_flow in self.cash_flow_streams:
            if (
                cash_flow.flow_type is CashFlowType.CONTRIBUTION
                and cash_flow.destination_tax_treatment is not None
                and cash_flow.destination_tax_treatment not in treatment_set
            ):
                raise ValueError(
                    "contribution cash-flow destination requires a matching tax bucket"
                )
