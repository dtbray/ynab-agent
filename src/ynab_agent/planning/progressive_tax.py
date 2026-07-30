"""Versioned, auditable US federal and Indiana annual tax calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from typing import Any, Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ynab_agent.planning.models import (
    FederalFilingStatus,
    FutureTaxPolicyMode,
    IrmaaLookbackMagi,
    ProgressiveTaxAssumptions,
)


POLICY_RESOURCE = "tax_policy_us_in_2026_v1.json"
HEALTHCARE_POLICY_PROJECTION: Literal["static_2026_policy"] = (
    "static_2026_policy"
)


class TaxCalculationInput(BaseModel):
    """One household's annual tax facts, independent of an interface."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    policy_id: str = "us_in_2026_v1"
    tax_year: int = Field(default=2026, ge=2026, le=2200)
    filing_status: FederalFilingStatus
    taxpayer_birth_year: int = Field(ge=1900, le=2200)
    spouse_birth_year: int | None = Field(default=None, ge=1900, le=2200)
    ordinary_income: float = Field(default=0, ge=0)
    long_term_capital_gains: float = Field(default=0, ge=0)
    social_security_income: float = Field(default=0, ge=0)
    tax_exempt_interest: float = Field(default=0, ge=0)
    future_policy_mode: FutureTaxPolicyMode = FutureTaxPolicyMode.INFLATION_INDEXED
    bracket_inflation_rate: float = Field(default=0.025, ge=-0.05, le=0.15)
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
        max_length=2,
    )
    married_filing_separately_lived_with_spouse: bool = False

    @model_validator(mode="after")
    def validate_household(self) -> TaxCalculationInput:
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
        years = [value.tax_year for value in self.irmaa_lookback_magi]
        if len(years) != len(set(years)):
            raise ValueError("IRMAA lookback tax years must be unique")
        if any(year >= self.tax_year for year in years):
            raise ValueError("IRMAA lookback tax years must precede tax_year")
        return self


class TaxPolicyManifest(BaseModel):
    """Policy identity persisted with every calculation."""

    policy_id: str
    effective_year: int
    published_as_of: str
    resource_sha256: str
    future_policy_mode: FutureTaxPolicyMode
    bracket_inflation_rate: float
    healthcare_policy_projection: Literal["static_2026_policy"] = HEALTHCARE_POLICY_PROJECTION
    sources: list[dict[str, Any]]


class AnnualTaxResult(BaseModel):
    """Stable annual tax result with component and rate audit fields."""

    tax_year: int
    federal_ordinary_tax: float
    federal_long_term_capital_gains_tax: float
    federal_income_tax: float
    indiana_income_tax: float
    total_income_tax: float
    taxable_social_security: float
    federal_adjusted_gross_income: float
    federal_taxable_income: float
    federal_deduction: float
    aca_modified_adjusted_gross_income: float
    aca_fpl_percentage: float
    aca_expected_contribution: float | None
    aca_premium_tax_credit: float | None
    irmaa_lookback_tax_year: int
    irmaa_lookback_magi: float | None
    irmaa_tier: int | None
    irmaa_annual_surcharge: float | None
    effective_income_tax_rate: float
    marginal_ordinary_income_tax_rate: float
    marginal_long_term_capital_gains_tax_rate: float
    policy_manifest: TaxPolicyManifest


class _TaxComponents(TypedDict):
    federal_ordinary_tax: float
    federal_long_term_capital_gains_tax: float
    federal_income_tax: float
    indiana_income_tax: float
    total_income_tax: float
    taxable_social_security: float
    federal_adjusted_gross_income: float
    federal_taxable_income: float
    federal_deduction: float
    aca_modified_adjusted_gross_income: float
    aca_fpl_percentage: float
    aca_expected_contribution: float | None
    aca_premium_tax_credit: float | None
    irmaa_lookback_tax_year: int
    irmaa_lookback_magi: float | None
    irmaa_tier: int | None
    irmaa_annual_surcharge: float | None


@dataclass(frozen=True)
class IncomeTaxArrayResult:
    """Vectorized tax components retained for simulation audit output."""

    federal_income_tax: Any
    indiana_income_tax: Any
    total_income_tax: Any
    taxable_social_security: Any
    federal_deduction: Any
    federal_adjusted_gross_income: Any
    federal_taxable_income: Any
    federal_taxable_ordinary_income: Any


@lru_cache(maxsize=1)
def load_tax_policy() -> tuple[dict[str, Any], str]:
    """Load and fingerprint the bounded package policy."""
    raw = files("ynab_agent.resources").joinpath(POLICY_RESOURCE).read_bytes()
    policy = json.loads(raw)
    return policy, hashlib.sha256(raw).hexdigest()


def _inflation_factor(request: TaxCalculationInput, base_year: int) -> float:
    if request.future_policy_mode is FutureTaxPolicyMode.FIXED_NOMINAL:
        return 1.0
    return (1 + request.bracket_inflation_rate) ** (request.tax_year - base_year)


def _rounded_indexed(value: float, factor: float) -> float:
    # Federal indexed dollar parameters are generally published in $50 increments.
    return round(value * factor / 50) * 50


def _bracket_tax(
    taxable_income: float,
    brackets: list[list[float | None]],
    factor: float,
) -> float:
    tax = 0.0
    lower = 0.0
    for upper, raw_rate in brackets:
        rate = cast(float, raw_rate)
        indexed_upper = None if upper is None else _rounded_indexed(float(upper), factor)
        amount = max(
            0.0,
            taxable_income - lower
            if indexed_upper is None
            else min(taxable_income, indexed_upper) - lower,
        )
        tax += amount * rate
        if indexed_upper is None or taxable_income <= indexed_upper:
            break
        lower = indexed_upper
    return tax


def _taxable_social_security(
    request: TaxCalculationInput,
    policy: dict[str, Any],
) -> float:
    social_security = request.social_security_income
    if social_security == 0:
        return 0.0
    status_key = request.filing_status.value
    if (
        request.filing_status is FederalFilingStatus.MARRIED_FILING_SEPARATELY
        and request.married_filing_separately_lived_with_spouse
    ):
        status_key = "married_filing_separately_lived_together"
    first, second = cast(
        tuple[float, float],
        policy["federal"]["social_security"][status_key],
    )
    if first == second == 0:
        return 0.85 * social_security
    provisional = (
        request.ordinary_income
        + request.long_term_capital_gains
        + request.tax_exempt_interest
        + 0.5 * social_security
    )
    if provisional <= first:
        return 0.0
    if provisional <= second:
        return min(0.5 * social_security, 0.5 * (provisional - first))
    lower_band_taxable = min(0.5 * social_security, 0.5 * (second - first))
    return min(
        0.85 * social_security,
        0.85 * (provisional - second) + lower_band_taxable,
    )


def _age(request: TaxCalculationInput, birth_year: int) -> int:
    return request.tax_year - birth_year


def _federal_deduction(
    request: TaxCalculationInput,
    policy: dict[str, Any],
    *,
    federal_agi: float,
    factor: float,
) -> float:
    federal = policy["federal"]
    status = request.filing_status.value
    deduction = _rounded_indexed(federal["standard_deduction"][status], factor)
    married = request.filing_status in {
        FederalFilingStatus.MARRIED_FILING_JOINTLY,
        FederalFilingStatus.MARRIED_FILING_SEPARATELY,
        FederalFilingStatus.QUALIFYING_SURVIVING_SPOUSE,
    }
    additional = federal["additional_standard_deduction_aged_or_blind"][
        "married" if married else "unmarried"
    ]
    additional_count = int(_age(request, request.taxpayer_birth_year) >= 65)
    additional_count += int(request.taxpayer_blind)
    if request.spouse_birth_year is not None:
        additional_count += int(_age(request, request.spouse_birth_year) >= 65)
        additional_count += int(request.spouse_blind)
    deduction += additional_count * _rounded_indexed(additional, factor)

    enhanced = federal["enhanced_senior_deduction"]
    if (
        request.tax_year <= enhanced["expires_after_year"]
        and request.filing_status is not FederalFilingStatus.MARRIED_FILING_SEPARATELY
    ):
        eligible = int(_age(request, request.taxpayer_birth_year) >= 65)
        if request.spouse_birth_year is not None:
            eligible += int(_age(request, request.spouse_birth_year) >= 65)
        amount = eligible * enhanced["amount_per_eligible_person"]
        phaseout_start = enhanced["phaseout_start"][status]
        amount -= enhanced["phaseout_rate"] * max(0.0, federal_agi - phaseout_start)
        deduction += max(0.0, amount)
    return deduction


def _capital_gains_tax(
    taxable_ordinary: float,
    taxable_gains: float,
    thresholds: list[float],
    factor: float,
) -> float:
    zero_threshold = _rounded_indexed(thresholds[0], factor)
    twenty_threshold = _rounded_indexed(thresholds[1], factor)
    zero_amount = min(taxable_gains, max(0.0, zero_threshold - taxable_ordinary))
    remaining = taxable_gains - zero_amount
    fifteen_amount = min(
        remaining,
        max(0.0, twenty_threshold - max(taxable_ordinary, zero_threshold)),
    )
    return fifteen_amount * 0.15 + (remaining - fifteen_amount) * 0.20


def _indiana_tax(
    request: TaxCalculationInput,
    policy: dict[str, Any],
    *,
    federal_agi: float,
) -> float:
    if not request.indiana_resident:
        return 0.0
    indiana = policy["indiana"]
    rate = float(
        indiana["rate_by_year"].get(
            str(request.tax_year),
            indiana["rate_after_last_year"],
        )
    )
    filer_count = 2 if request.spouse_birth_year is not None else 1
    exemptions = indiana["personal_exemption"] * (filer_count + request.dependent_count)
    exemptions += request.dependent_child_count * indiana["dependent_child_additional_exemption"]
    exemptions += (
        request.first_time_adopted_child_count
        * indiana["first_time_adopted_child_additional_exemption"]
    )
    aged_or_blind_count = int(_age(request, request.taxpayer_birth_year) >= 65)
    aged_or_blind_count += int(request.taxpayer_blind)
    if request.spouse_birth_year is not None:
        aged_or_blind_count += int(_age(request, request.spouse_birth_year) >= 65)
        aged_or_blind_count += int(request.spouse_blind)
    exemptions += aged_or_blind_count * indiana["aged_or_blind_exemption"]
    aged_count = int(_age(request, request.taxpayer_birth_year) >= 65)
    if request.spouse_birth_year is not None:
        aged_count += int(_age(request, request.spouse_birth_year) >= 65)
    low_income_limit = (
        indiana["low_income_aged_agi_limit_married_separate"]
        if request.filing_status is FederalFilingStatus.MARRIED_FILING_SEPARATELY
        else indiana["low_income_aged_agi_limit"]
    )
    if federal_agi < low_income_limit:
        exemptions += aged_count * indiana["low_income_aged_exemption"]
    indiana_agi = request.ordinary_income + request.long_term_capital_gains
    return float(max(0.0, indiana_agi - exemptions) * rate)


def _fpl_amount(policy: dict[str, Any], household_size: int) -> float:
    fpl = policy["aca"]["contiguous_us_fpl"]
    if household_size <= 8:
        return float(fpl[str(household_size)])
    return float(fpl["8"] + (household_size - 8) * policy["aca"]["additional_person"])


def _aca_values(
    request: TaxCalculationInput,
    policy: dict[str, Any],
    *,
    federal_agi: float,
) -> tuple[float, float, float | None, float | None]:
    magi = (
        federal_agi
        + request.tax_exempt_interest
        + request.social_security_income
        - _taxable_social_security(request, policy)
    )
    fpl_percentage = magi / _fpl_amount(policy, request.aca_household_size) * 100
    benchmark = request.aca_benchmark_annual_premium
    aca = policy["aca"]
    if (
        benchmark is None
        or fpl_percentage < aca["minimum_fpl_percentage"]
        or fpl_percentage > aca["maximum_fpl_percentage"]
    ):
        return magi, fpl_percentage, None, None
    applicable = 0.0
    table = aca["applicable_percentage_table"]
    for index, (lower, upper, start_rate, end_rate) in enumerate(table):
        if lower <= fpl_percentage < upper or (index == len(table) - 1 and fpl_percentage == upper):
            position = 0.0 if upper == lower else (fpl_percentage - lower) / (upper - lower)
            applicable = start_rate + position * (end_rate - start_rate)
            break
    expected = magi * applicable
    return magi, fpl_percentage, expected, max(0.0, benchmark - expected)


def _irmaa_values(
    request: TaxCalculationInput,
    policy: dict[str, Any],
) -> tuple[int, float | None, int | None, float | None]:
    irmaa = policy["irmaa"]
    lookback_year = request.tax_year - irmaa["lookback_years"]
    magi = next(
        (value.magi for value in request.irmaa_lookback_magi if value.tax_year == lookback_year),
        None,
    )
    eligible_people = int(_age(request, request.taxpayer_birth_year) >= 65)
    if request.spouse_birth_year is not None:
        eligible_people += int(_age(request, request.spouse_birth_year) >= 65)
    if magi is None or eligible_people == 0:
        return lookback_year, magi, None, None
    if (
        request.filing_status is FederalFilingStatus.MARRIED_FILING_SEPARATELY
        and request.married_filing_separately_lived_with_spouse
    ):
        lower, upper = irmaa[
            "married_filing_separately_lived_together_thresholds"
        ]
        tier = 0 if magi <= lower else 4 if magi < upper else 5
    else:
        joint = request.filing_status is FederalFilingStatus.MARRIED_FILING_JOINTLY
        thresholds = irmaa["joint_thresholds" if joint else "individual_thresholds"]
        tier = sum(magi > threshold for threshold in thresholds[:-1])
        tier += int(magi >= thresholds[-1])
    per_person_surcharge = 12 * (
        irmaa["part_b_monthly_adjustments"][tier] + irmaa["part_d_monthly_adjustments"][tier]
    )
    return lookback_year, magi, tier, eligible_people * per_person_surcharge


def _calculate_components(
    request: TaxCalculationInput,
    policy: dict[str, Any],
) -> _TaxComponents:
    if request.policy_id != policy["policy_id"]:
        raise ValueError(f"unsupported tax policy: {request.policy_id}")
    factor = _inflation_factor(request, policy["effective_year"])
    taxable_social_security = _taxable_social_security(request, policy)
    federal_agi = (
        request.ordinary_income + request.long_term_capital_gains + taxable_social_security
    )
    deduction = _federal_deduction(
        request,
        policy,
        federal_agi=federal_agi,
        factor=factor,
    )
    taxable_income = max(0.0, federal_agi - deduction)
    taxable_ordinary = max(
        0.0,
        request.ordinary_income + taxable_social_security - deduction,
    )
    taxable_gains = min(
        request.long_term_capital_gains,
        max(0.0, taxable_income - taxable_ordinary),
    )
    status = request.filing_status.value
    ordinary_tax = _bracket_tax(
        taxable_ordinary,
        policy["federal"]["ordinary_brackets"][status],
        factor,
    )
    capital_gains_tax = _capital_gains_tax(
        taxable_ordinary,
        taxable_gains,
        policy["federal"]["capital_gains"][status],
        factor,
    )
    indiana_tax = _indiana_tax(
        request,
        policy,
        federal_agi=federal_agi,
    )
    aca_magi, fpl_percentage, expected, credit = _aca_values(
        request,
        policy,
        federal_agi=federal_agi,
    )
    lookback_year, lookback_magi, irmaa_tier, surcharge = _irmaa_values(
        request,
        policy,
    )
    return {
        "federal_ordinary_tax": ordinary_tax,
        "federal_long_term_capital_gains_tax": capital_gains_tax,
        "federal_income_tax": ordinary_tax + capital_gains_tax,
        "indiana_income_tax": indiana_tax,
        "total_income_tax": ordinary_tax + capital_gains_tax + indiana_tax,
        "taxable_social_security": taxable_social_security,
        "federal_adjusted_gross_income": federal_agi,
        "federal_taxable_income": taxable_income,
        "federal_deduction": deduction,
        "aca_modified_adjusted_gross_income": aca_magi,
        "aca_fpl_percentage": fpl_percentage,
        "aca_expected_contribution": expected,
        "aca_premium_tax_credit": credit,
        "irmaa_lookback_tax_year": lookback_year,
        "irmaa_lookback_magi": lookback_magi,
        "irmaa_tier": irmaa_tier,
        "irmaa_annual_surcharge": surcharge,
    }


def calculate_annual_tax(request: TaxCalculationInput) -> AnnualTaxResult:
    """Calculate a deterministic annual return and its $1 marginal audit."""
    policy, digest = load_tax_policy()
    components = _calculate_components(request, policy)
    ordinary_plus_one = request.model_copy(update={"ordinary_income": request.ordinary_income + 1})
    gains_plus_one = request.model_copy(
        update={"long_term_capital_gains": request.long_term_capital_gains + 1}
    )
    ordinary_marginal = (
        _calculate_components(ordinary_plus_one, policy)["total_income_tax"]
        - components["total_income_tax"]
    )
    gains_marginal = (
        _calculate_components(gains_plus_one, policy)["total_income_tax"]
        - components["total_income_tax"]
    )
    gross_income = (
        request.ordinary_income + request.long_term_capital_gains + request.social_security_income
    )
    manifest = TaxPolicyManifest(
        policy_id=policy["policy_id"],
        effective_year=policy["effective_year"],
        published_as_of=policy["published_as_of"],
        resource_sha256=digest,
        future_policy_mode=request.future_policy_mode,
        bracket_inflation_rate=request.bracket_inflation_rate,
        healthcare_policy_projection=HEALTHCARE_POLICY_PROJECTION,
        sources=policy["sources"],
    )
    return AnnualTaxResult(
        tax_year=request.tax_year,
        **components,
        effective_income_tax_rate=(
            components["total_income_tax"] / gross_income if gross_income > 0 else 0.0
        ),
        marginal_ordinary_income_tax_rate=ordinary_marginal,
        marginal_long_term_capital_gains_tax_rate=gains_marginal,
        policy_manifest=manifest,
    )


def calculate_income_tax_arrays(
    assumptions: ProgressiveTaxAssumptions,
    *,
    tax_year: int,
    ordinary_income: Any,
    long_term_capital_gains: Any,
    social_security_income: Any,
) -> Any:
    """Vectorized federal and Indiana tax for bounded simulation batches."""
    return calculate_income_tax_component_arrays(
        assumptions,
        tax_year=tax_year,
        ordinary_income=ordinary_income,
        long_term_capital_gains=long_term_capital_gains,
        social_security_income=social_security_income,
    ).total_income_tax


def calculate_income_tax_component_arrays(
    assumptions: ProgressiveTaxAssumptions,
    *,
    tax_year: int,
    ordinary_income: Any,
    long_term_capital_gains: Any,
    social_security_income: Any,
) -> IncomeTaxArrayResult:
    """Return vectorized federal and Indiana components for simulation audits."""
    import numpy as np

    policy, _ = load_tax_policy()
    if assumptions.policy_id != policy["policy_id"]:
        raise ValueError(f"unsupported tax policy: {assumptions.policy_id}")
    ordinary = np.asarray(ordinary_income, dtype=float)
    gains = np.asarray(long_term_capital_gains, dtype=float)
    social_security = np.asarray(social_security_income, dtype=float)
    factor = (
        1.0
        if assumptions.future_policy_mode is FutureTaxPolicyMode.FIXED_NOMINAL
        else (1 + assumptions.bracket_inflation_rate) ** (tax_year - policy["effective_year"])
    )
    status = assumptions.filing_status.value
    social_key = status
    if (
        assumptions.filing_status is FederalFilingStatus.MARRIED_FILING_SEPARATELY
        and assumptions.married_filing_separately_lived_with_spouse
    ):
        social_key = "married_filing_separately_lived_together"
    first, second = policy["federal"]["social_security"][social_key]
    provisional = ordinary + gains + 0.5 * social_security
    if first == second == 0:
        taxable_social_security = 0.85 * social_security
    else:
        taxable_social_security = np.where(
            provisional <= first,
            0.0,
            np.where(
                provisional <= second,
                np.minimum(0.5 * social_security, 0.5 * (provisional - first)),
                np.minimum(
                    0.85 * social_security,
                    0.85 * (provisional - second)
                    + np.minimum(
                        0.5 * social_security,
                        0.5 * (second - first),
                    ),
                ),
            ),
        )
    federal_agi = ordinary + gains + taxable_social_security
    federal = policy["federal"]
    deduction = np.full_like(
        federal_agi,
        _rounded_indexed(federal["standard_deduction"][status], factor),
    )
    married = assumptions.filing_status in {
        FederalFilingStatus.MARRIED_FILING_JOINTLY,
        FederalFilingStatus.MARRIED_FILING_SEPARATELY,
        FederalFilingStatus.QUALIFYING_SURVIVING_SPOUSE,
    }
    additional = federal["additional_standard_deduction_aged_or_blind"][
        "married" if married else "unmarried"
    ]
    additional_count = int(tax_year - assumptions.taxpayer_birth_year >= 65)
    additional_count += int(assumptions.taxpayer_blind)
    if assumptions.spouse_birth_year is not None:
        additional_count += int(tax_year - assumptions.spouse_birth_year >= 65)
        additional_count += int(assumptions.spouse_blind)
    deduction += additional_count * _rounded_indexed(additional, factor)
    enhanced = federal["enhanced_senior_deduction"]
    if (
        tax_year <= enhanced["expires_after_year"]
        and assumptions.filing_status is not FederalFilingStatus.MARRIED_FILING_SEPARATELY
    ):
        eligible = int(tax_year - assumptions.taxpayer_birth_year >= 65)
        if assumptions.spouse_birth_year is not None:
            eligible += int(tax_year - assumptions.spouse_birth_year >= 65)
        enhanced_amount = eligible * enhanced["amount_per_eligible_person"] - enhanced[
            "phaseout_rate"
        ] * np.maximum(
            0.0,
            federal_agi - enhanced["phaseout_start"][status],
        )
        deduction = deduction + np.maximum(0.0, enhanced_amount)
    taxable_income = np.maximum(0.0, federal_agi - deduction)
    taxable_ordinary = np.maximum(
        0.0,
        ordinary + taxable_social_security - deduction,
    )
    taxable_gains = np.minimum(
        gains,
        np.maximum(0.0, taxable_income - taxable_ordinary),
    )
    ordinary_tax = np.zeros_like(taxable_income)
    lower = 0.0
    for raw_upper, rate in federal["ordinary_brackets"][status]:
        upper = None if raw_upper is None else _rounded_indexed(raw_upper, factor)
        amount = np.maximum(
            0.0,
            taxable_ordinary - lower
            if upper is None
            else np.minimum(taxable_ordinary, upper) - lower,
        )
        ordinary_tax += amount * rate
        if upper is None:
            break
        lower = upper
    zero_threshold, twenty_threshold = (
        _rounded_indexed(value, factor) for value in federal["capital_gains"][status]
    )
    zero_gains = np.minimum(
        taxable_gains,
        np.maximum(0.0, zero_threshold - taxable_ordinary),
    )
    remaining_gains = taxable_gains - zero_gains
    fifteen_gains = np.minimum(
        remaining_gains,
        np.maximum(
            0.0,
            twenty_threshold - np.maximum(taxable_ordinary, zero_threshold),
        ),
    )
    capital_gains_tax = fifteen_gains * 0.15 + (remaining_gains - fifteen_gains) * 0.20
    federal_income_tax = ordinary_tax + capital_gains_tax
    if not assumptions.indiana_resident:
        indiana_tax = np.zeros_like(federal_income_tax)
        return IncomeTaxArrayResult(
            federal_income_tax=federal_income_tax,
            indiana_income_tax=indiana_tax,
            total_income_tax=federal_income_tax,
            taxable_social_security=taxable_social_security,
            federal_deduction=deduction,
            federal_adjusted_gross_income=federal_agi,
            federal_taxable_income=taxable_income,
            federal_taxable_ordinary_income=taxable_ordinary,
        )
    indiana = policy["indiana"]
    rate = indiana["rate_by_year"].get(
        str(tax_year),
        indiana["rate_after_last_year"],
    )
    filer_count = 2 if assumptions.spouse_birth_year is not None else 1
    exemptions = indiana["personal_exemption"] * (filer_count + assumptions.dependent_count)
    exemptions += (
        assumptions.dependent_child_count * indiana["dependent_child_additional_exemption"]
    )
    exemptions += (
        assumptions.first_time_adopted_child_count
        * indiana["first_time_adopted_child_additional_exemption"]
    )
    aged_or_blind = int(tax_year - assumptions.taxpayer_birth_year >= 65)
    aged_or_blind += int(assumptions.taxpayer_blind)
    aged = int(tax_year - assumptions.taxpayer_birth_year >= 65)
    if assumptions.spouse_birth_year is not None:
        aged_or_blind += int(tax_year - assumptions.spouse_birth_year >= 65)
        aged_or_blind += int(assumptions.spouse_blind)
        aged += int(tax_year - assumptions.spouse_birth_year >= 65)
    exemptions += aged_or_blind * indiana["aged_or_blind_exemption"]
    low_income_limit = (
        indiana["low_income_aged_agi_limit_married_separate"]
        if assumptions.filing_status is FederalFilingStatus.MARRIED_FILING_SEPARATELY
        else indiana["low_income_aged_agi_limit"]
    )
    exemptions = exemptions + np.where(
        federal_agi < low_income_limit,
        aged * indiana["low_income_aged_exemption"],
        0,
    )
    indiana_tax = np.maximum(0.0, ordinary + gains - exemptions) * rate
    return IncomeTaxArrayResult(
        federal_income_tax=federal_income_tax,
        indiana_income_tax=indiana_tax,
        total_income_tax=federal_income_tax + indiana_tax,
        taxable_social_security=taxable_social_security,
        federal_deduction=deduction,
        federal_adjusted_gross_income=federal_agi,
        federal_taxable_income=taxable_income,
        federal_taxable_ordinary_income=taxable_ordinary,
    )


def federal_ordinary_bracket_ceiling(
    assumptions: ProgressiveTaxAssumptions,
    *,
    tax_year: int,
    rate: float,
) -> float:
    """Return the indexed upper taxable-income edge for a bounded bracket."""
    policy, _ = load_tax_policy()
    factor = (
        1.0
        if assumptions.future_policy_mode is FutureTaxPolicyMode.FIXED_NOMINAL
        else (1 + assumptions.bracket_inflation_rate) ** (
            tax_year - policy["effective_year"]
        )
    )
    for upper, bracket_rate in policy["federal"]["ordinary_brackets"][
        assumptions.filing_status.value
    ]:
        if float(bracket_rate) == rate:
            if upper is None:
                raise ValueError("the top ordinary bracket has no fill ceiling")
            return _rounded_indexed(float(upper), factor)
    raise ValueError("unsupported federal ordinary bracket rate")


def federal_capital_gain_bracket_ceiling(
    assumptions: ProgressiveTaxAssumptions,
    *,
    tax_year: int,
    rate: float,
) -> float:
    """Return the indexed upper taxable-income edge for the 0% or 15% band."""
    policy, _ = load_tax_policy()
    factor = (
        1.0
        if assumptions.future_policy_mode is FutureTaxPolicyMode.FIXED_NOMINAL
        else (1 + assumptions.bracket_inflation_rate) ** (
            tax_year - policy["effective_year"]
        )
    )
    thresholds = policy["federal"]["capital_gains"][assumptions.filing_status.value]
    index = 0 if rate == 0 else 1 if rate == 0.15 else None
    if index is None:
        raise ValueError("capital-gain bracket rate must be 0% or 15%")
    return _rounded_indexed(float(thresholds[index]), factor)


def calculate_irmaa_surcharge_arrays(
    assumptions: ProgressiveTaxAssumptions,
    *,
    tax_year: int,
    lookback_magi: Any | None,
) -> Any:
    """Return current-year Part B and D IRMAA surcharges from prior MAGI."""
    import numpy as np

    if lookback_magi is None:
        return np.asarray(0.0)
    policy, _ = load_tax_policy()
    irmaa = policy["irmaa"]
    magi = np.asarray(lookback_magi, dtype=float)
    eligible_people = int(tax_year - assumptions.taxpayer_birth_year >= 65)
    if assumptions.spouse_birth_year is not None:
        eligible_people += int(tax_year - assumptions.spouse_birth_year >= 65)
    if eligible_people == 0:
        return np.zeros_like(magi)
    if (
        assumptions.filing_status is FederalFilingStatus.MARRIED_FILING_SEPARATELY
        and assumptions.married_filing_separately_lived_with_spouse
    ):
        lower, upper = irmaa[
            "married_filing_separately_lived_together_thresholds"
        ]
        tier = np.where(magi <= lower, 0, np.where(magi < upper, 4, 5))
    else:
        thresholds = irmaa[
            "joint_thresholds"
            if assumptions.filing_status is FederalFilingStatus.MARRIED_FILING_JOINTLY
            else "individual_thresholds"
        ]
        tier = np.zeros_like(magi, dtype=int)
        for threshold in thresholds[:-1]:
            tier += magi > threshold
        tier += magi >= thresholds[-1]
    part_b = np.asarray(irmaa["part_b_monthly_adjustments"], dtype=float)
    part_d = np.asarray(irmaa["part_d_monthly_adjustments"], dtype=float)
    return 12 * eligible_people * (part_b[tier] + part_d[tier])


def rmd_start_age(birth_year: int) -> int:
    """Return the statutory RMD start age for a birth year."""
    policy, _ = load_tax_policy()
    for row in policy["rmd"]["start_age_by_birth_year"]:
        latest = row["latest_birth_year"]
        if latest is None or birth_year <= latest:
            return int(row["age"])
    raise RuntimeError("RMD policy has no terminal birth-year rule")


def rmd_divisor(age: int) -> float:
    """Return the bounded Uniform Lifetime divisor for an age."""
    policy, _ = load_tax_policy()
    divisors = policy["rmd"]["uniform_lifetime_divisors"]
    return float(divisors.get(str(min(max(age, 72), 120)), 2.0))
