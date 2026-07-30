"""Validated inputs for wealth-planning scenarios."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ynab_agent.planning.spending_guardrails import (
    RetirementSpendingPlan,
)


MAX_SCENARIO_NAME_LENGTH = 200
MAX_ACCOUNT_ID_LENGTH = 128
MAX_SCENARIO_ACCOUNTS = 100
MAX_INCOME_STREAMS = 64
MAX_INCOME_STREAM_NAME_LENGTH = 200
MAX_CASH_FLOW_STREAMS = 64
MAX_CASH_FLOW_STREAM_NAME_LENGTH = 200
MAX_TAX_BUCKETS = 5
MAX_SPENDING_TIERS = 8
MAX_SPENDING_TIER_NAME_LENGTH = 100
MAX_IRMAA_LOOKBACK_YEARS = 2


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


class IrmaaLookbackMagi(BaseModel):
    """Known historical MAGI used before simulated lookback years exist."""

    model_config = ConfigDict(allow_inf_nan=False)

    tax_year: int = Field(ge=1900, le=2200)
    magi: float = Field(ge=0)


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
        if any(year >= self.simulation_start_year for year in years):
            raise ValueError("IRMAA lookback tax years must precede simulation_start_year")
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


class ValuationProvenance(BaseModel):
    """Origin of a portfolio value resolved before a simulation run."""

    model_config = ConfigDict(allow_inf_nan=False)

    source: str = Field(min_length=1, max_length=128)
    as_of: date | None = None
    account_ids: tuple[str, ...] = Field(
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
        return self


class TaxBucket(BaseModel):
    """One bounded, aggregated balance with a single tax treatment."""

    model_config = ConfigDict(allow_inf_nan=False)

    tax_treatment: TaxTreatment
    starting_balance: float = Field(ge=0)
    taxable_basis: float | None = Field(default=None, ge=0)
    contribution_fraction: float = Field(default=0, ge=0, le=1)

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
    spending_tiers: list[SpendingTier] = Field(
        default_factory=list,
        max_length=MAX_SPENDING_TIERS,
    )
    legacy_target_real: float | None = Field(default=None, ge=0)
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
        self._validate_tax_model()
        return self

    def _validate_tax_model(self) -> None:
        tax_aware = bool(self.tax_buckets) or self.tax_assumptions is not None
        if not tax_aware:
            return
        if not self.tax_buckets or self.tax_assumptions is None:
            raise ValueError("tax_buckets and tax_assumptions must be provided together")
        if self.starting_portfolio is None:
            raise ValueError("tax-aware scenarios require an explicit starting_portfolio")
        if self.withdrawal_tax_rate != 0:
            raise ValueError("tax-aware scenarios must set withdrawal_tax_rate to 0")
        treatments = [bucket.tax_treatment for bucket in self.tax_buckets]
        if len(treatments) != len(set(treatments)):
            raise ValueError("tax_buckets must have unique tax treatments")
        bucket_total = sum(bucket.starting_balance for bucket in self.tax_buckets)
        if abs(bucket_total - self.starting_portfolio) > 0.01:
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
