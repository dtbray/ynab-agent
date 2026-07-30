"""Bounded household, work, pension, and longevity planning inputs."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_HOUSEHOLD_PEOPLE = 2
MAX_PERSON_ID_LENGTH = 64
MAX_TIMELINE_ENTRIES = 16
MIN_SOCIAL_SECURITY_CLAIM_AGE_MONTHS = 62 * 12
MAX_SOCIAL_SECURITY_CLAIM_AGE_MONTHS = 70 * 12
MIN_SURVIVOR_CLAIM_AGE_MONTHS = 60 * 12


class LongevityMode(StrEnum):
    """How a person's final attained age is selected."""

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"


class LongevityAssumption(BaseModel):
    """Bounded scenario longevity; not an actuarial mortality forecast."""

    model_config = ConfigDict(allow_inf_nan=False)

    mode: LongevityMode = LongevityMode.DETERMINISTIC
    death_age: int | None = Field(default=95, ge=1, le=130)
    mean_death_age: float | None = Field(default=None, ge=1, le=130)
    standard_deviation_years: float | None = Field(default=None, gt=0, le=30)
    minimum_death_age: int = Field(default=60, ge=1, le=130)
    maximum_death_age: int = Field(default=110, ge=1, le=130)

    @model_validator(mode="after")
    def validate_mode(self) -> LongevityAssumption:
        if self.minimum_death_age > self.maximum_death_age:
            raise ValueError("minimum_death_age cannot exceed maximum_death_age")
        if self.mode is LongevityMode.DETERMINISTIC:
            if self.death_age is None:
                raise ValueError("deterministic longevity requires death_age")
            if self.mean_death_age is not None or self.standard_deviation_years is not None:
                raise ValueError(
                    "deterministic longevity cannot include probabilistic parameters"
                )
        else:
            if self.mean_death_age is None or self.standard_deviation_years is None:
                raise ValueError(
                    "probabilistic longevity requires mean_death_age and "
                    "standard_deviation_years"
                )
        return self


class WorkTimeline(BaseModel):
    """Annual covered work earnings in today's dollars."""

    model_config = ConfigDict(allow_inf_nan=False)

    start_age: int = Field(ge=0, le=120)
    end_age: int = Field(ge=0, le=120)
    annual_covered_earnings: float = Field(ge=0)
    inflation_adjusted: bool = True

    @model_validator(mode="after")
    def validate_ages(self) -> WorkTimeline:
        if self.end_age < self.start_age:
            raise ValueError("work timeline end_age must be at least start_age")
        return self


class PensionTimeline(BaseModel):
    """Annual pension income in today's dollars."""

    model_config = ConfigDict(allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=100)
    start_age: int = Field(ge=0, le=120)
    end_age: int | None = Field(default=None, ge=0, le=130)
    annual_amount: float = Field(ge=0)
    inflation_adjusted: bool = True
    tax_free: bool = False
    survivor_fraction: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_ages(self) -> PensionTimeline:
        if self.end_age is not None and self.end_age < self.start_age:
            raise ValueError("pension timeline end_age must be at least start_age")
        return self


class Person(BaseModel):
    """One member of a retirement-planning household."""

    model_config = ConfigDict(allow_inf_nan=False)

    id: str = Field(min_length=1, max_length=MAX_PERSON_ID_LENGTH)
    name: str = Field(min_length=1, max_length=100)
    birth_date: date
    retirement_age_months: int = Field(ge=0, le=120 * 12)
    primary_insurance_amount_monthly: float = Field(default=0, ge=0)
    family_maximum_monthly: float | None = Field(default=None, gt=0)
    social_security_claim_age_months: int | None = Field(
        default=None,
        ge=MIN_SOCIAL_SECURITY_CLAIM_AGE_MONTHS,
        le=MAX_SOCIAL_SECURITY_CLAIM_AGE_MONTHS,
    )
    survivor_claim_age_months: int = Field(
        default=MIN_SURVIVOR_CLAIM_AGE_MONTHS,
        ge=MIN_SURVIVOR_CLAIM_AGE_MONTHS,
        le=MAX_SOCIAL_SECURITY_CLAIM_AGE_MONTHS,
    )
    work: list[WorkTimeline] = Field(
        default_factory=list,
        max_length=MAX_TIMELINE_ENTRIES,
    )
    pensions: list[PensionTimeline] = Field(
        default_factory=list,
        max_length=MAX_TIMELINE_ENTRIES,
    )
    longevity: LongevityAssumption = Field(default_factory=LongevityAssumption)

    @model_validator(mode="after")
    def validate_social_security(self) -> Person:
        if (
            self.family_maximum_monthly is not None
            and self.primary_insurance_amount_monthly == 0
        ):
            raise ValueError(
                "family_maximum_monthly requires a positive primary insurance amount"
            )
        return self


class Household(BaseModel):
    """One- or two-person household whose assets remain in the household."""

    model_config = ConfigDict(allow_inf_nan=False)

    plan_start_date: date
    people: list[Person] = Field(min_length=1, max_length=MAX_HOUSEHOLD_PEOPLE)
    survivor_spending_fraction: float = Field(default=0.75, gt=0, le=1)
    longevity_correlation: float = Field(default=0, ge=-0.95, le=0.95)
    earnings_test_policy_year: int = Field(default=2026, ge=2026, le=2026)

    @model_validator(mode="after")
    def validate_people(self) -> Household:
        person_ids = [person.id for person in self.people]
        if len(person_ids) != len(set(person_ids)):
            raise ValueError("household person IDs must be unique")
        names = [person.name.casefold() for person in self.people]
        if len(names) != len(set(names)):
            raise ValueError("household person names must be unique")
        return self


def age_months_on(birth_date: date, on_date: date) -> int:
    """Return completed calendar months of age on a date."""
    months = (on_date.year - birth_date.year) * 12 + on_date.month - birth_date.month
    if on_date.day < birth_date.day:
        months -= 1
    return months
