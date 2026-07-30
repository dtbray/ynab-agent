"""Auditable Social Security claiming rules for bounded household projections.

The formulas and frozen earnings-test values are drawn from official SSA sources
listed by :func:`social_security_policy_manifest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

from ynab_agent.planning.household_models import Person, age_months_on

EARNINGS_TEST_POLICY_YEAR = 2026
EARNINGS_TEST_UNDER_FRA_LIMIT = 24_480.0
EARNINGS_TEST_FRA_YEAR_LIMIT = 65_160.0
FAMILY_MAXIMUM_BEND_POINTS_2026 = (1_643.0, 2_371.0, 3_093.0)


def full_retirement_age_months(birth_date: date) -> int:
    """Return retirement full retirement age, including SSA's January 1 rule."""
    birth_year = birth_date.year - (1 if birth_date.month == birth_date.day == 1 else 0)
    if birth_year <= 1937:
        return 65 * 12
    if birth_year <= 1942:
        return 65 * 12 + (birth_year - 1937) * 2
    if birth_year <= 1954:
        return 66 * 12
    if birth_year <= 1959:
        return 66 * 12 + (birth_year - 1954) * 2
    return 67 * 12


def survivor_full_retirement_age_months(birth_date: date) -> int:
    """Return survivor FRA, which differs from retirement FRA before 1962."""
    birth_year = birth_date.year - (1 if birth_date.month == birth_date.day == 1 else 0)
    if birth_year <= 1939:
        return 65 * 12
    if birth_year <= 1944:
        return 65 * 12 + (birth_year - 1939) * 2
    if birth_year <= 1956:
        return 66 * 12
    if birth_year <= 1961:
        return 66 * 12 + (birth_year - 1956) * 2
    return 67 * 12


def retired_worker_factor(claim_age_months: int, fra_months: int) -> float:
    """Return the SSA early-retirement or delayed-retirement multiplier."""
    delta = claim_age_months - fra_months
    if delta < 0:
        early = -delta
        first_36 = min(early, 36)
        beyond_36 = max(0, early - 36)
        return 1 - first_36 * (5 / 9 / 100) - beyond_36 * (5 / 12 / 100)
    delayed = min(delta, 70 * 12 - fra_months)
    return 1 + delayed * (2 / 3 / 100)


def spouse_factor(claim_age_months: int, fra_months: int) -> float:
    """Return the early-claim multiplier for a spouse's half-PIA amount."""
    early = max(0, fra_months - claim_age_months)
    first_36 = min(early, 36)
    beyond_36 = max(0, early - 36)
    return 1 - first_36 * (25 / 36 / 100) - beyond_36 * (5 / 12 / 100)


def survivor_factor(claim_age_months: int, fra_months: int) -> float:
    """Return the SSA 71.5%-to-100% survivor-age factor."""
    if claim_age_months >= fra_months:
        return 1.0
    eligible_span = max(1, fra_months - 60 * 12)
    elapsed = max(0, claim_age_months - 60 * 12)
    return 0.715 + 0.285 * min(1.0, elapsed / eligible_span)


def family_maximum_monthly(primary_insurance_amount_monthly: float) -> float:
    """Return the 2026 OASI family maximum from a worker's monthly PIA."""
    first, second, third = FAMILY_MAXIMUM_BEND_POINTS_2026
    pia = primary_insurance_amount_monthly
    return (
        min(pia, first) * 1.50
        + min(max(0.0, pia - first), second - first) * 2.72
        + min(max(0.0, pia - second), third - second) * 1.34
        + max(0.0, pia - third) * 1.75
    )


def annual_work_earnings(
    person: Person,
    age_years: int,
    *,
    inflation_factor: float = 1.0,
) -> float:
    return sum(
        timeline.annual_covered_earnings
        * (inflation_factor if timeline.inflation_adjusted else 1.0)
        for timeline in person.work
        if timeline.start_age <= age_years <= timeline.end_age
    )


def earnings_test_withholding(
    *,
    annual_benefit: float,
    annual_covered_earnings: float,
    age_months: int,
    fra_months: int,
) -> float:
    """Return annual benefits withheld under the frozen 2026 earnings test."""
    if age_months >= fra_months:
        return 0.0
    reaches_fra_this_year = age_months + 12 >= fra_months
    if reaches_fra_this_year:
        excess = max(0.0, annual_covered_earnings - EARNINGS_TEST_FRA_YEAR_LIMIT)
        return min(annual_benefit, excess / 3)
    excess = max(0.0, annual_covered_earnings - EARNINGS_TEST_UNDER_FRA_LIMIT)
    return min(annual_benefit, excess / 2)


def earnings_test_withheld_months(
    *,
    annual_benefit: float,
    annual_withheld: float,
) -> int:
    """Approximate months with a full or partial work deduction."""
    if annual_benefit <= 0 or annual_withheld <= 0:
        return 0
    monthly_benefit = annual_benefit / 12
    return min(12, math.ceil(annual_withheld / monthly_benefit - 1e-9))


@dataclass(frozen=True)
class EntitlementCreditMonths:
    """Adjustment-for-reduction credits tracked independently by record."""

    retirement: int = 0
    spousal: int = 0
    survivor: int = 0


@dataclass(frozen=True)
class EntitlementAmounts:
    """Annual values split across retirement, spouse, and survivor records."""

    retirement: float = 0.0
    spousal: float = 0.0
    survivor: float = 0.0

    @property
    def total(self) -> float:
        return self.retirement + self.spousal + self.survivor


@dataclass(frozen=True)
class PersonSocialSecurity:
    own: float
    spousal: float
    survivor: float
    earnings_test_withheld: float
    earnings_test_credit_months_applied: int
    earnings_test_credit_annual: float
    credit_months_by_entitlement: EntitlementCreditMonths
    credit_annual_by_entitlement: EntitlementAmounts
    entitlement_onset_months: EntitlementCreditMonths

    @property
    def paid(self) -> float:
        return max(0.0, self.own + self.spousal + self.survivor - self.earnings_test_withheld)


def person_social_security(
    person: Person,
    *,
    spouse: Person | None,
    age_months: int,
    spouse_alive: bool,
    person_alive: bool,
    family_maximum_override_monthly: float | None = None,
    earnings_test_credit_months: int = 0,
    entitlement_credit_months: EntitlementCreditMonths | None = None,
    spouse_death_age_months: int | None = None,
    spouse_retirement_credit_months_at_death: int = 0,
    annual_covered_earnings: float | None = None,
    work_inflation_factor: float = 1.0,
    apply_earnings_test: bool = True,
) -> PersonSocialSecurity:
    """Calculate one annual benefit before household family-maximum allocation."""
    zero_credits = EntitlementCreditMonths()
    zero_amounts = EntitlementAmounts()
    if not person_alive:
        return PersonSocialSecurity(
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            zero_credits,
            zero_amounts,
            zero_credits,
        )
    credits = entitlement_credit_months or EntitlementCreditMonths(
        retirement=earnings_test_credit_months,
    )
    claim_age = person.social_security_claim_age_months
    fra = full_retirement_age_months(person.birth_date)
    retirement_credit_applied = 0
    effective_claim_age = claim_age
    if claim_age is not None and age_months >= fra:
        retirement_credit_applied = min(
            max(0, credits.retirement),
            max(0, fra - claim_age),
        )
        effective_claim_age = claim_age + retirement_credit_applied
    own = 0.0
    own_without_credit = 0.0
    if (
        person.primary_insurance_amount_monthly > 0
        and claim_age is not None
        and age_months >= claim_age
    ):
        if effective_claim_age is None:  # pragma: no cover - narrowed above
            raise RuntimeError("effective claim age is missing")
        own = (
            person.primary_insurance_amount_monthly
            * 12
            * retired_worker_factor(effective_claim_age, fra)
        )
        own_without_credit = (
            person.primary_insurance_amount_monthly
            * 12
            * retired_worker_factor(claim_age, fra)
        )

    spousal = 0.0
    spousal_without_credit = 0.0
    spousal_credit_applied = 0
    spousal_onset = 0
    survivor = 0.0
    survivor_without_credit = 0.0
    survivor_credit_applied = 0
    survivor_onset = 0
    if spouse is not None and spouse.primary_insurance_amount_monthly > 0:
        spouse_claim = spouse.social_security_claim_age_months
        spouse_age_offset = age_months_on(
            spouse.birth_date,
            person.birth_date,
        )
        person_request_age = claim_age or 62 * 12
        if spouse_alive and spouse_claim is not None:
            spousal_onset = max(
                person_request_age,
                spouse_claim - spouse_age_offset,
            )
            if age_months >= spousal_onset:
                if age_months >= fra:
                    spousal_credit_applied = min(
                        max(0, credits.spousal),
                        max(0, fra - spousal_onset),
                    )
                effective_spousal_onset = (
                    spousal_onset + spousal_credit_applied
                )
                spousal_excess = max(
                    0.0,
                    spouse.primary_insurance_amount_monthly * 6
                    - person.primary_insurance_amount_monthly * 12,
                )
                spousal = spousal_excess * spouse_factor(
                    effective_spousal_onset,
                    fra,
                )
                spousal_without_credit = spousal_excess * spouse_factor(
                    spousal_onset,
                    fra,
                )
                spouse_worker = (
                    spouse.primary_insurance_amount_monthly
                    * 12
                    * retired_worker_factor(
                        spouse_claim,
                        full_retirement_age_months(spouse.birth_date),
                    )
                )
                family_maximum = (
                    family_maximum_override_monthly
                    if family_maximum_override_monthly is not None
                    else family_maximum_monthly(
                        spouse.primary_insurance_amount_monthly
                    )
                )
                spousal = min(
                    spousal,
                    max(0.0, family_maximum * 12 - spouse_worker),
                )
                spousal_without_credit = min(
                    spousal_without_credit,
                    max(0.0, family_maximum * 12 - spouse_worker),
                )
        elif not spouse_alive:
            person_age_at_death = (
                spouse_death_age_months - spouse_age_offset
                if spouse_death_age_months is not None
                else person.survivor_claim_age_months
            )
            survivor_onset = max(
                person.survivor_claim_age_months,
                person_age_at_death,
            )
            if age_months >= survivor_onset:
                survivor_fra = survivor_full_retirement_age_months(
                    person.birth_date
                )
                survivor_credit_applied = min(
                    max(0, credits.survivor),
                    max(0, survivor_fra - survivor_onset),
                )
                effective_survivor_onset = (
                    survivor_onset + survivor_credit_applied
                )
                deceased_fra = full_retirement_age_months(
                    spouse.birth_date
                )
                filed_before_death = (
                    spouse_claim is not None
                    and (
                        spouse_death_age_months is None
                        or spouse_claim <= spouse_death_age_months
                    )
                )
                if filed_before_death:
                    if spouse_claim is None:  # pragma: no cover - narrowed above
                        raise RuntimeError("deceased filing age is missing")
                    deceased_entitlement_age = (
                        spouse_claim
                        + spouse_retirement_credit_months_at_death
                    )
                else:
                    death_age = (
                        spouse_death_age_months
                        if spouse_death_age_months is not None
                        else deceased_fra
                    )
                    deceased_entitlement_age = max(
                        deceased_fra,
                        min(death_age, 70 * 12),
                    )
                deceased_pia = (
                    spouse.primary_insurance_amount_monthly * 12
                )
                deceased_worker = (
                    deceased_pia
                    * retired_worker_factor(
                        deceased_entitlement_age,
                        deceased_fra,
                    )
                )
                survivor_reduction = survivor_factor(
                    effective_survivor_onset,
                    survivor_fra,
                )
                unrestricted_base = max(deceased_pia, deceased_worker)
                unrestricted = unrestricted_base * survivor_reduction
                survivor_without_credit_unrestricted = (
                    unrestricted_base
                    * survivor_factor(survivor_onset, survivor_fra)
                )
                if deceased_worker < deceased_pia:
                    rib_lim = max(0.825 * deceased_pia, deceased_worker)
                    unrestricted = min(unrestricted, rib_lim)
                    survivor_without_credit_unrestricted = min(
                        survivor_without_credit_unrestricted,
                        rib_lim,
                    )
                survivor = max(0.0, unrestricted - own)
                survivor_without_credit = max(
                    0.0,
                    survivor_without_credit_unrestricted - own_without_credit,
                )

    annual_work = (
        annual_covered_earnings
        if annual_covered_earnings is not None
        else annual_work_earnings(
            person,
            age_months // 12,
            inflation_factor=work_inflation_factor,
        )
    )
    withheld = (
        earnings_test_withholding(
            annual_benefit=own + spousal + survivor,
            annual_covered_earnings=annual_work,
            age_months=age_months,
            fra_months=fra,
        )
        if apply_earnings_test
        else 0.0
    )
    credit_by_type = EntitlementCreditMonths(
        retirement=retirement_credit_applied,
        spousal=spousal_credit_applied,
        survivor=survivor_credit_applied,
    )
    credit_annual_by_type = EntitlementAmounts(
        retirement=max(0.0, own - own_without_credit),
        spousal=max(0.0, spousal - spousal_without_credit),
        survivor=max(0.0, survivor - survivor_without_credit),
    )
    return PersonSocialSecurity(
        own,
        spousal,
        survivor,
        withheld,
        max(
            retirement_credit_applied,
            spousal_credit_applied,
            survivor_credit_applied,
        ),
        credit_annual_by_type.total,
        credit_by_type,
        credit_annual_by_type,
        EntitlementCreditMonths(
            retirement=claim_age or 0,
            spousal=spousal_onset,
            survivor=survivor_onset,
        ),
    )


def social_security_policy_manifest() -> dict[str, object]:
    """Return the exact policy snapshot and primary sources used by the engine."""
    return {
        "policy_id": "ssa_retirement_household_2026_v3",
        "earnings_test_policy_year": EARNINGS_TEST_POLICY_YEAR,
        "earnings_test_under_fra_limit": EARNINGS_TEST_UNDER_FRA_LIMIT,
        "earnings_test_fra_year_limit": EARNINGS_TEST_FRA_YEAR_LIMIT,
        "family_maximum_bend_points": FAMILY_MAXIMUM_BEND_POINTS_2026,
        "claim_age_months": {"minimum": 62 * 12, "maximum": 70 * 12},
        "sources": [
            "https://www.ssa.gov/benefits/retirement/planner/applying2.html",
            "https://www.ssa.gov/oact/progdata/nra.html",
            "https://www.ssa.gov/policy/docs/statcomps/supplement/2025/apnc.html",
            "https://www.ssa.gov/OP_Home/handbook/handbook.07/handbook-0724.html",
            "https://www.ssa.gov/blog/en/posts/2024-07-11.html",
            "https://www.ssa.gov/benefits/retirement/planner/claiming.html",
            "https://www.ssa.gov/survivor/amount",
            "https://www.ssa.gov/cola/factsheets/2026.html",
            "https://www.ssa.gov/faqs/en/questions/KA-02107.html",
            "https://www.ssa.gov/oact/COLA/familymax.html",
            "https://secure.ssa.gov/poms.nsf/lnx/0302501021",
        ],
        "limitations": [
            "annual projection applies the earnings test to annual covered earnings",
            (
                "annual withholding is charged from the year's first entitled "
                "month to approximate full or partial work-deduction months"
            ),
            "family maximum uses 2026 bend points unless explicitly overridden",
            "government-pension offsets and disabled/dependent benefits are out of scope",
        ],
    }
