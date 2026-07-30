"""Tax-engine port for household income and filing-status transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from ynab_agent.planning.models import (
    FederalFilingStatus,
    TaxAssumptions,
    TaxModel,
)
from ynab_agent.planning.progressive_tax import (
    IncomeTaxArrayResult,
    calculate_income_tax_component_arrays,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class HouseholdIncomeTaxInput:
    """Vectorized annual household income passed to a tax implementation."""

    ordinary_income: np.ndarray
    social_security_income: np.ndarray
    joint_filing: np.ndarray
    tax_year: int
    long_term_capital_gains: np.ndarray


class HouseholdTaxEngine(Protocol):
    """Vectorized household tax calculation selected by scenario policy."""

    engine_id: str

    def income_tax(
        self,
        tax_input: HouseholdIncomeTaxInput,
        *,
        assumptions: TaxAssumptions,
    ) -> np.ndarray: ...


class EffectiveRateHouseholdTaxEngine:
    """Existing effective-rate behavior behind the household tax port."""

    engine_id = "effective_rate_household_v1"

    def income_tax(
        self,
        tax_input: HouseholdIncomeTaxInput,
        *,
        assumptions: TaxAssumptions,
    ) -> np.ndarray:
        # Filing status is carried into the port even though a single effective
        # rate does not use it. A progressive #91 adapter can consume it directly.
        return assumptions.ordinary_income_tax_rate * (
            tax_input.ordinary_income
            + tax_input.social_security_income
            * assumptions.social_security_taxable_fraction
        )


def calculate_household_progressive_tax_components(
    assumptions: TaxAssumptions,
    tax_input: HouseholdIncomeTaxInput,
) -> IncomeTaxArrayResult:
    """Apply MFJ tax while both spouses live and single tax after survivor transition."""
    import numpy as np

    progressive = assumptions.progressive
    if progressive is None or assumptions.tax_model is not TaxModel.PROGRESSIVE_US_INDIANA:
        raise ValueError("progressive household tax requires progressive assumptions")
    joint = np.asarray(tax_input.joint_filing, dtype=bool)
    if np.any(joint) and (
        progressive.filing_status is not FederalFilingStatus.MARRIED_FILING_JOINTLY
        or progressive.spouse_birth_year is None
    ):
        raise ValueError("joint household trials require married-filing-jointly assumptions")
    single = progressive.model_copy(
        update={
            "filing_status": FederalFilingStatus.SINGLE,
            "spouse_birth_year": None,
            "spouse_blind": False,
            "aca_household_size": 1,
        }
    )
    joint_result = calculate_income_tax_component_arrays(
        progressive,
        tax_year=tax_input.tax_year,
        ordinary_income=tax_input.ordinary_income,
        long_term_capital_gains=tax_input.long_term_capital_gains,
        social_security_income=tax_input.social_security_income,
    )
    single_result = calculate_income_tax_component_arrays(
        single,
        tax_year=tax_input.tax_year,
        ordinary_income=tax_input.ordinary_income,
        long_term_capital_gains=tax_input.long_term_capital_gains,
        social_security_income=tax_input.social_security_income,
    )
    return IncomeTaxArrayResult(
        federal_income_tax=np.where(
            joint, joint_result.federal_income_tax, single_result.federal_income_tax
        ),
        indiana_income_tax=np.where(
            joint, joint_result.indiana_income_tax, single_result.indiana_income_tax
        ),
        total_income_tax=np.where(
            joint, joint_result.total_income_tax, single_result.total_income_tax
        ),
        taxable_social_security=np.where(
            joint,
            joint_result.taxable_social_security,
            single_result.taxable_social_security,
        ),
        federal_deduction=np.where(
            joint, joint_result.federal_deduction, single_result.federal_deduction
        ),
        federal_adjusted_gross_income=np.where(
            joint,
            joint_result.federal_adjusted_gross_income,
            single_result.federal_adjusted_gross_income,
        ),
        federal_taxable_income=np.where(
            joint,
            joint_result.federal_taxable_income,
            single_result.federal_taxable_income,
        ),
        federal_taxable_ordinary_income=np.where(
            joint,
            joint_result.federal_taxable_ordinary_income,
            single_result.federal_taxable_ordinary_income,
        ),
    )


class ProgressiveHouseholdTaxEngine:
    """Progressive US/Indiana tax with per-trial filing-status transitions."""

    engine_id = "progressive_us_indiana_household_v1"

    def income_tax(
        self,
        tax_input: HouseholdIncomeTaxInput,
        *,
        assumptions: TaxAssumptions,
    ) -> np.ndarray:
        return cast(
            "np.ndarray",
            calculate_household_progressive_tax_components(
                assumptions,
                tax_input,
            ).total_income_tax,
        )


def household_tax_engine_for(
    assumptions: TaxAssumptions,
    requested: HouseholdTaxEngine | None = None,
) -> HouseholdTaxEngine:
    """Select policy-compatible tax math without progressive flat-rate fallback."""
    if assumptions.tax_model is TaxModel.PROGRESSIVE_US_INDIANA:
        if requested is not None and not isinstance(
            requested, ProgressiveHouseholdTaxEngine
        ):
            raise ValueError(
                "progressive scenarios require the progressive household tax engine"
            )
        return requested or ProgressiveHouseholdTaxEngine()
    return requested or EffectiveRateHouseholdTaxEngine()
