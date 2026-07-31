"""Validated healthcare, Medicare, and long-term-care assumptions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_HEALTHCARE_PEOPLE = 2


class LongTermCareAssumptions(BaseModel):
    """One person's bounded, probabilistic long-term-care risk."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    lifetime_incidence_probability: float = Field(ge=0, le=1)
    minimum_onset_age: int = Field(ge=18, le=120)
    maximum_onset_age: int = Field(ge=18, le=120)
    mean_duration_years: float = Field(gt=0, le=20)
    duration_standard_deviation_years: float = Field(ge=0, le=10)
    maximum_duration_years: int = Field(ge=1, le=20)
    annual_cost_real: float = Field(ge=0)
    annual_cost_log_volatility: float = Field(default=0, ge=0, le=2)
    insurance_annual_benefit_real: float = Field(default=0, ge=0)
    insurance_benefit_years: int = Field(default=0, ge=0, le=20)

    @model_validator(mode="after")
    def validate_bounds(self) -> LongTermCareAssumptions:
        if self.maximum_onset_age < self.minimum_onset_age:
            raise ValueError(
                "maximum_onset_age must be at least minimum_onset_age"
            )
        if self.mean_duration_years > self.maximum_duration_years:
            raise ValueError(
                "mean_duration_years cannot exceed maximum_duration_years"
            )
        if self.insurance_annual_benefit_real > 0:
            if self.insurance_benefit_years == 0:
                raise ValueError(
                    "positive LTC insurance benefit requires benefit years"
                )
        elif self.insurance_benefit_years != 0:
            raise ValueError(
                "LTC insurance benefit years require a positive annual benefit"
            )
        return self


class PersonHealthcareAssumptions(BaseModel):
    """Coverage and care assumptions for one household member."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    person_id: str = Field(min_length=1, max_length=64)
    medicare_start_age: int = Field(default=65, ge=18, le=100)
    pre_medicare_aca_annual_premium_real: float = Field(
        default=0,
        ge=0,
        description=(
            "Annual ACA or other pre-Medicare premium in plan-start dollars, "
            "net of any expected premium tax credit"
        ),
    )
    pre_medicare_annual_out_of_pocket_real: float = Field(default=0, ge=0)
    medicare_annual_premium_real: float = Field(
        default=0,
        ge=0,
        description="Base Medicare and supplement premiums, excluding IRMAA",
    )
    medicare_annual_out_of_pocket_real: float = Field(default=0, ge=0)
    long_term_care: LongTermCareAssumptions | None = None


class HealthcareAssumptions(BaseModel):
    """Household healthcare policy and separately inflated care costs."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    policy_id: str = "healthcare_ltc_v1"
    medical_inflation_rate: float = Field(default=0.05, ge=-0.05, le=0.25)
    people: list[PersonHealthcareAssumptions] = Field(
        min_length=1,
        max_length=MAX_HEALTHCARE_PEOPLE,
    )
    ltc_funding_source: Literal["portfolio", "home_equity"] = "portfolio"
    home_equity_available_for_ltc_real: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_policy(self) -> HealthcareAssumptions:
        if self.policy_id != "healthcare_ltc_v1":
            raise ValueError("unsupported healthcare policy")
        person_ids = [person.person_id for person in self.people]
        if len(person_ids) != len(set(person_ids)):
            raise ValueError("healthcare person IDs must be unique")
        has_ltc = any(person.long_term_care is not None for person in self.people)
        if self.ltc_funding_source == "home_equity":
            if not has_ltc:
                raise ValueError("home-equity LTC funding requires long-term-care assumptions")
            if self.home_equity_available_for_ltc_real <= 0:
                raise ValueError(
                    "home-equity LTC funding requires a positive real funding limit"
                )
        elif self.home_equity_available_for_ltc_real != 0:
            raise ValueError(
                "home_equity_available_for_ltc_real requires "
                "ltc_funding_source='home_equity'"
            )
        return self
