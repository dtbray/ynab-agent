from __future__ import annotations

import pytest

from ynab_agent.planning.progressive_tax import (
    TaxCalculationInput,
    calculate_annual_tax,
    calculate_income_tax_component_arrays,
    rmd_start_age,
)
from ynab_agent.planning.models import ProgressiveTaxAssumptions, WealthScenario
from ynab_agent.planning.simulation import simulate


def test_single_ordinary_income_matches_independent_2026_golden() -> None:
    result = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="single",
            taxpayer_birth_year=1980,
            ordinary_income=100_000,
        )
    )

    # 2026 federal: $83,900 taxable through the 22% bracket.
    assert result.federal_income_tax == pytest.approx(13_170)
    # Indiana: ($100,000 - $1,000 exemption) * 2.95%.
    assert result.indiana_income_tax == pytest.approx(2_920.50)
    assert result.total_income_tax == pytest.approx(16_090.50)
    assert result.effective_income_tax_rate == pytest.approx(0.160905)
    assert result.marginal_ordinary_income_tax_rate == pytest.approx(0.2495)


def test_social_security_and_senior_deductions_match_golden() -> None:
    result = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="married_filing_jointly",
            taxpayer_birth_year=1959,
            spouse_birth_year=1959,
            ordinary_income=40_000,
            social_security_income=30_000,
        )
    )

    assert result.taxable_social_security == pytest.approx(15_350)
    assert result.federal_deduction == pytest.approx(47_500)
    assert result.federal_income_tax == pytest.approx(785)
    assert result.indiana_income_tax == pytest.approx(1_062)
    assert result.total_income_tax == pytest.approx(1_847)


def test_senior_deductions_respect_qss_and_mfs_filing_rules() -> None:
    qualifying_survivor = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="qualifying_surviving_spouse",
            taxpayer_birth_year=1950,
        )
    )
    married_separate = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="married_filing_separately",
            taxpayer_birth_year=1950,
        )
    )

    # QSS uses the married aged amount: $32,200 + $1,650 + $6,000.
    assert qualifying_survivor.federal_deduction == pytest.approx(39_850)
    # The enhanced senior deduction requires a married taxpayer to file jointly.
    assert married_separate.federal_deduction == pytest.approx(17_750)


@pytest.mark.parametrize(
    ("filing_status", "expected_deduction"),
    [
        ("qualifying_surviving_spouse", 39_850),
        ("married_filing_separately", 17_750),
    ],
)
def test_vectorized_senior_deductions_match_scalar_filing_rules(
    filing_status: str,
    expected_deduction: float,
) -> None:
    assumptions = ProgressiveTaxAssumptions.model_validate(
        {
            "filing_status": filing_status,
            "taxpayer_birth_year": 1950,
        }
    )

    result = calculate_income_tax_component_arrays(
        assumptions,
        tax_year=2026,
        ordinary_income=[0],
        long_term_capital_gains=[0],
        social_security_income=[0],
    )

    assert result.federal_deduction[0] == pytest.approx(expected_deduction)


def test_long_term_gain_stacking_matches_independent_golden() -> None:
    result = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="single",
            taxpayer_birth_year=1980,
            ordinary_income=50_000,
            long_term_capital_gains=50_000,
        )
    )

    assert result.federal_ordinary_tax == pytest.approx(3_820)
    assert result.federal_long_term_capital_gains_tax == pytest.approx(5_167.50)
    assert result.federal_income_tax == pytest.approx(8_987.50)
    assert result.total_income_tax == pytest.approx(11_908)


def test_aca_and_irmaa_hooks_are_explicit_and_auditable() -> None:
    result = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="single",
            taxpayer_birth_year=1958,
            ordinary_income=30_000,
            aca_household_size=1,
            aca_benchmark_annual_premium=12_000,
            irmaa_lookback_magi=[{"tax_year": 2024, "magi": 150_000}],
        )
    )

    assert result.aca_fpl_percentage == pytest.approx(191.6932907)
    assert result.aca_expected_contribution is not None
    assert result.aca_premium_tax_credit is not None
    assert result.irmaa_lookback_tax_year == 2024
    assert result.irmaa_lookback_magi == 150_000
    assert result.irmaa_tier == 2
    assert result.irmaa_annual_surcharge == pytest.approx(12 * (202.90 + 37.50))


def test_irmaa_top_threshold_is_inclusive_and_charged_per_eligible_person() -> None:
    single = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="single",
            taxpayer_birth_year=1950,
            irmaa_lookback_magi=[{"tax_year": 2024, "magi": 500_000}],
        )
    )
    joint = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="married_filing_jointly",
            taxpayer_birth_year=1950,
            spouse_birth_year=1950,
            irmaa_lookback_magi=[{"tax_year": 2024, "magi": 750_000}],
        )
    )

    top_surcharge = 12 * (487 + 91)
    assert single.irmaa_tier == 5
    assert single.irmaa_annual_surcharge == pytest.approx(top_surcharge)
    assert joint.irmaa_tier == 5
    assert joint.irmaa_annual_surcharge == pytest.approx(2 * top_surcharge)


@pytest.mark.parametrize(
    ("magi", "expected_tier", "expected_monthly_adjustment"),
    [
        (109_000, 0, 0),
        (109_001, 4, 446.30 + 83.30),
        (391_000, 5, 487 + 91),
    ],
)
def test_irmaa_married_filing_separately_lived_together_uses_special_schedule(
    magi: float,
    expected_tier: int,
    expected_monthly_adjustment: float,
) -> None:
    result = calculate_annual_tax(
        TaxCalculationInput(
            filing_status="married_filing_separately",
            married_filing_separately_lived_with_spouse=True,
            taxpayer_birth_year=1950,
            irmaa_lookback_magi=[{"tax_year": 2024, "magi": magi}],
        )
    )

    assert result.irmaa_tier == expected_tier
    assert result.irmaa_annual_surcharge == pytest.approx(
        12 * expected_monthly_adjustment
    )


@pytest.mark.parametrize(
    ("birth_year", "start_age"),
    [(1948, 70), (1949, 72), (1950, 72), (1951, 73), (1959, 73), (1960, 75)],
)
def test_rmd_birth_year_schedule(birth_year: int, start_age: int) -> None:
    assert rmd_start_age(birth_year) == start_age


def test_future_policy_mode_is_manifested_and_changes_indexed_thresholds() -> None:
    indexed = calculate_annual_tax(
        TaxCalculationInput(
            tax_year=2030,
            filing_status="single",
            taxpayer_birth_year=1980,
            ordinary_income=100_000,
            future_policy_mode="inflation_indexed",
        )
    )
    fixed = calculate_annual_tax(
        TaxCalculationInput(
            tax_year=2030,
            filing_status="single",
            taxpayer_birth_year=1980,
            ordinary_income=100_000,
            future_policy_mode="fixed_nominal",
        )
    )

    assert indexed.total_income_tax < fixed.total_income_tax
    assert indexed.policy_manifest.future_policy_mode == "inflation_indexed"
    assert fixed.policy_manifest.future_policy_mode == "fixed_nominal"
    assert indexed.policy_manifest.healthcare_policy_projection == "static_2026_policy"
    assert len(indexed.policy_manifest.resource_sha256) == 64
    assert indexed.policy_manifest.sources


def test_simulation_selects_progressive_engine_and_manifests_policy() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "progressive",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 62,
            "starting_portfolio": 100_000,
            "annual_spending": 20_000,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "seed": 91,
            "tax_buckets": [
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 100_000,
                }
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "single",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1966,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["tax_deferred"],
                "retirement_surplus_destination": "tax_deferred",
            },
        }
    )

    result = simulate(scenario, 100_000)

    assert result.assumptions["tax_model"] == "progressive_us_indiana_v1"
    manifest = result.reproducibility["tax_policy"]
    assert isinstance(manifest, dict)
    assert manifest["policy_id"] == "us_in_2026_v1"
    assert len(str(manifest["resource_sha256"])) == 64
    assert result.lifetime_tax_real is not None
    assert result.lifetime_tax_real["p50"] > 0
    assert len(result.annual_tax_audit) == 2
    assert result.annual_tax_audit[0]["tax_year"] == 2026
    assert "marginal_ordinary_income_tax_rate" in result.annual_tax_audit[0]
    assert "federal_income_tax_nominal" in result.annual_tax_audit[0]
    assert "indiana_income_tax_nominal" in result.annual_tax_audit[0]
    assert "taxable_social_security_nominal" in result.annual_tax_audit[0]
    assert "federal_deduction_nominal" in result.annual_tax_audit[0]
    assert "realized_long_term_capital_gains_nominal" in result.annual_tax_audit[0]
    healthcare_projection = manifest["healthcare_policy_projection"]
    assert isinstance(healthcare_projection, dict)
    assert healthcare_projection == {
        "mode": "static_2026_policy",
        "healthcare_inflation_rate_applied": False,
        "supplied_healthcare_inflation_rate": 0.05,
        "static_policy_fields": [
            "aca_federal_poverty_levels",
            "aca_applicable_percentages",
            "irmaa_thresholds",
            "irmaa_surcharges",
        ],
    }


def test_progressive_withdrawal_converges_to_need_for_huge_balance() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "need-scaled withdrawal",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 61,
            "starting_portfolio": 1_000_000_000_000,
            "annual_spending": 1,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "tax_buckets": [
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 1_000_000_000_000,
                }
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "single",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1966,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["tax_deferred"],
                "retirement_surplus_destination": "tax_deferred",
            },
        }
    )

    result = simulate(scenario, 1_000_000_000_000)

    withdrawn = 1_000_000_000_000 - result.ending_balance_real["p50"]
    assert withdrawn == pytest.approx(1, abs=0.005)


def test_progressive_rmd_age_comes_from_tax_year_and_birth_year() -> None:
    values: dict[str, object] = {
        "name": "tax-year RMD age",
        "current_age": 72,
        "retirement_age": 72,
        "end_age": 73,
        "starting_portfolio": 26_500,
        "annual_spending": 1,
        "inflation_rate": 0,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "income_streams": [
            {
                "name": "pension",
                "start_age": 72,
                "annual_amount": 50_000,
                "tax_treatment": "ordinary",
            }
        ],
        "tax_buckets": [
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 26_500,
            },
            {
                "tax_treatment": "taxable",
                "starting_balance": 0,
                "taxable_basis": 0,
            },
        ],
        "tax_assumptions": {
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1953,
            },
            "withdrawal_order": ["taxable", "tax_deferred"],
            "retirement_surplus_destination": "taxable",
        },
    }
    with_rmd = simulate(WealthScenario.model_validate(values), 26_500)
    without_rmd_scenario = WealthScenario.model_validate(values)
    if without_rmd_scenario.tax_assumptions is None:  # pragma: no cover
        raise AssertionError("test scenario must be tax aware")
    without_rmd = simulate(
        without_rmd_scenario.model_copy(
            update={
                "tax_assumptions": without_rmd_scenario.tax_assumptions.model_copy(
                    update={"apply_required_minimum_distributions": False}
                )
            }
        ),
        26_500,
    )

    assert with_rmd.lifetime_tax_real is not None
    assert without_rmd.lifetime_tax_real is not None
    assert with_rmd.lifetime_tax_real["p50"] > without_rmd.lifetime_tax_real["p50"]


def test_qualified_hsa_fraction_applies_to_progressive_estate_value() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "qualified HSA estate",
            "current_age": 65,
            "retirement_age": 65,
            "end_age": 66,
            "starting_portfolio": 100_000,
            "annual_spending": 1,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "tax_buckets": [
                {
                    "tax_treatment": "hsa",
                    "starting_balance": 100_000,
                }
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "single",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1961,
                },
                "qualified_hsa_withdrawal_fraction": 1,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["hsa"],
                "retirement_surplus_destination": "hsa",
            },
        }
    )

    result = simulate(scenario, 100_000)

    assert result.after_tax_ending_balance_real == pytest.approx(
        {"p10": 99_999, "p50": 99_999, "p90": 99_999}
    )
