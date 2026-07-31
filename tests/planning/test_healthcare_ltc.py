from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ynab_agent.planning.healthcare import (
    estimate_healthcare_state_bytes,
    prepare_healthcare_state,
)
from ynab_agent.planning.household import prepare_household_state
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.models import FederalFilingStatus
from ynab_agent.planning.paths import RunPolicy, SimulationPaths
from ynab_agent.planning.progressive_tax import (
    calculate_irmaa_surcharge_arrays,
)
from ynab_agent.planning.simulation import simulate
from ynab_agent.services.planner_jobs import (
    PlannerExecutionPolicy,
    PlannerSimulationResult,
)


def _paths(scenario: WealthScenario) -> SimulationPaths:
    years = scenario.end_age - scenario.current_age
    return SimulationPaths(
        gross_returns=np.ones((years, scenario.trials)),
        inflation_factors=np.ones(years + 1),
    )


def _scenario(**updates: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "household healthcare",
        "current_age": 64,
        "retirement_age": 64,
        "end_age": 69,
        "starting_portfolio": 500_000,
        "annual_spending": 1,
        "return_mean": 0,
        "return_volatility": 0,
        "inflation_rate": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 820,
        "household": {
            "plan_start_date": "2026-01-02",
            "survivor_spending_fraction": 0.5,
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1962-01-02",
                    "retirement_age_months": 64 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 66,
                    },
                },
                {
                    "id": "blair",
                    "name": "Blair",
                    "birth_date": "1960-01-02",
                    "retirement_age_months": 66 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 90,
                    },
                },
            ],
        },
        "healthcare": {
            "medical_inflation_rate": 0,
            "people": [
                {
                    "person_id": "alex",
                    "pre_medicare_aca_annual_premium_real": 1_000,
                    "pre_medicare_annual_out_of_pocket_real": 100,
                    "medicare_annual_premium_real": 2_000,
                    "medicare_annual_out_of_pocket_real": 200,
                },
                {
                    "person_id": "blair",
                    "medicare_annual_premium_real": 3_000,
                    "medicare_annual_out_of_pocket_real": 300,
                },
            ],
        },
    }
    values.update(updates)
    return WealthScenario.model_validate(values)


def _exact_home_equity_ltc_scenario() -> WealthScenario:
    values = _scenario().model_dump(mode="json")
    values.update(
        {
            "name": "exact home-equity LTC",
            "end_age": 67,
            "starting_portfolio": 100,
            "accounts": [
                {
                    "id": "alex-home-reserve",
                    "role": "taxable",
                    "owner_person_id": "alex",
                },
                {
                    "id": "blair-taxable",
                    "role": "taxable",
                    "owner_person_id": "blair",
                },
            ],
            "tax_buckets": [
                {
                    "account_id": "alex-home-reserve",
                    "owner_person_id": "alex",
                    "tax_treatment": "taxable",
                    "starting_balance": 0,
                    "taxable_basis": 0,
                },
                {
                    "account_id": "blair-taxable",
                    "owner_person_id": "blair",
                    "tax_treatment": "taxable",
                    "starting_balance": 100,
                    "taxable_basis": 100,
                },
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["taxable"],
                "retirement_surplus_destination": "taxable",
            },
            "housing_plan": {
                "home": {
                    "current_value": 15_000,
                    "cost_basis": 15_000,
                    "annual_appreciation_rate": 0,
                    "maintenance_rate": 0,
                    "property_tax_rate": 0,
                    "insurance_rate": 0,
                    "selling_cost_rate": 0,
                },
                "decision": {
                    "kind": "sell",
                    "event_age": 64,
                    "proceeds_destination_account_id": "alex-home-reserve",
                },
                "care": {
                    "funding_account_id": "alex-home-reserve",
                },
            },
            "healthcare": {
                "medical_inflation_rate": 0,
                "ltc_funding_source": "home_equity",
                "home_equity_available_for_ltc_real": 50_000,
                "people": [
                    {
                        "person_id": "alex",
                        "long_term_care": {
                            "lifetime_incidence_probability": 1,
                            "minimum_onset_age": 64,
                            "maximum_onset_age": 64,
                            "mean_duration_years": 2,
                            "duration_standard_deviation_years": 0,
                            "maximum_duration_years": 2,
                            "annual_cost_real": 20_000,
                        },
                    },
                    {"person_id": "blair"},
                ],
            },
        }
    )
    return WealthScenario.model_validate(values)


def test_healthcare_requires_exact_household_coverage() -> None:
    with pytest.raises(
        ValidationError,
        match="must cover every household person exactly once",
    ):
        _scenario(
            healthcare={
                "people": [
                    {
                        "person_id": "alex",
                        "medicare_annual_premium_real": 1_000,
                    }
                ]
            }
        )


def test_planner_preflight_reserves_healthcare_state() -> None:
    scenario = _scenario()
    without_healthcare = scenario.model_copy(update={"healthcare": None})
    policy = PlannerExecutionPolicy(
        maximum_working_bytes=64 * 1024 * 1024,
        in_memory_path_bytes=64 * 1024 * 1024,
        maximum_temporary_bytes=64 * 1024 * 1024,
        batch_size=100,
    )

    with_healthcare = policy.required_working_bytes(
        scenario,
        paired_historical_inflation=False,
    )
    without = policy.required_working_bytes(
        without_healthcare,
        paired_historical_inflation=False,
    )

    assert with_healthcare - without == estimate_healthcare_state_bytes(
        scenario
    )
    retained_matrices = (
        (scenario.end_age - scenario.current_age)
        * scenario.trials
        * 8
        * 8
    )
    assert estimate_healthcare_state_bytes(scenario) > retained_matrices


def test_mixed_age_coverage_stops_at_death_without_survivor_scaling() -> None:
    scenario = _scenario()
    paths = _paths(scenario)
    household = prepare_household_state(
        scenario,
        inflation_factors=paths.inflation_factors,
    )
    state = prepare_healthcare_state(
        scenario,
        household_state=household,
        inflation_factors=paths.inflation_factors,
    )

    assert state is not None
    assert state.pre_medicare_premium[:, 0].tolist() == [
        1_000,
        0,
        0,
        0,
        0,
    ]
    assert state.medicare_premium[:, 0].tolist() == [
        3_000,
        5_000,
        3_000,
        3_000,
        3_000,
    ]
    assert state.out_of_pocket[:, 0].tolist() == [
        400,
        500,
        300,
        300,
        300,
    ]


def test_ltc_is_seeded_truncated_at_death_and_applies_insurance_once() -> None:
    scenario = _scenario(
        healthcare={
            "medical_inflation_rate": 0,
            "people": [
                {
                    "person_id": "alex",
                    "long_term_care": {
                        "lifetime_incidence_probability": 1,
                        "minimum_onset_age": 64,
                        "maximum_onset_age": 64,
                        "mean_duration_years": 5,
                        "duration_standard_deviation_years": 0,
                        "maximum_duration_years": 5,
                        "annual_cost_real": 20_000,
                        "insurance_annual_benefit_real": 5_000,
                        "insurance_benefit_years": 2,
                    },
                },
                {"person_id": "blair"},
            ],
        }
    )
    paths = _paths(scenario)
    household = prepare_household_state(
        scenario,
        inflation_factors=paths.inflation_factors,
    )
    first = prepare_healthcare_state(
        scenario,
        household_state=household,
        inflation_factors=paths.inflation_factors,
    )
    second = prepare_healthcare_state(
        scenario,
        household_state=household,
        inflation_factors=paths.inflation_factors,
    )

    assert first is not None and second is not None
    assert np.array_equal(first.ltc_gross_cost, second.ltc_gross_cost)
    assert first.ltc_gross_cost[:, 0].tolist() == [
        20_000,
        20_000,
        0,
        0,
        0,
    ]
    assert first.ltc_insurance_benefit[:, 0].tolist() == [
        5_000,
        5_000,
        0,
        0,
        0,
    ]
    assert first.ltc_home_equity_used[:, 0].tolist() == [
        0,
        0,
        0,
        0,
        0,
    ]
    assert first.ltc_net_cost[:, 0].tolist() == [
        15_000,
        15_000,
        0,
        0,
        0,
    ]
    assert np.all(first.ltc_selected)
    assert np.all(first.ltc_active_in_plan)


def test_synthetic_home_equity_pool_without_exact_reserve_fails_closed() -> None:
    with pytest.raises(
        ValidationError,
        match="requires ltc_funding_source='home_equity'",
    ):
        _scenario(
            healthcare={
                "medical_inflation_rate": 0,
                "home_equity_available_for_ltc_real": 10_000,
                "people": [
                    {
                        "person_id": "alex",
                        "long_term_care": {
                            "lifetime_incidence_probability": 1,
                            "minimum_onset_age": 65,
                            "maximum_onset_age": 65,
                            "mean_duration_years": 1,
                            "duration_standard_deviation_years": 0,
                            "maximum_duration_years": 1,
                            "annual_cost_real": 30_000,
                        },
                    },
                    {"person_id": "blair"},
                ],
            }
        )


def test_exact_home_equity_ltc_partial_exhaustion_is_batched_and_account_keyed() -> None:
    scenario = _exact_home_equity_ltc_scenario()

    full = simulate(scenario, 100, run_policy=RunPolicy(batch_size=100))
    batched = simulate(scenario, 100, run_policy=RunPolicy(batch_size=17))

    assert full.ending_balance_real == batched.ending_balance_real
    assert full.annual_housing == batched.annual_housing
    assert full.annual_healthcare_real == batched.annual_healthcare_real
    assert full.healthcare_metrics == batched.healthcare_metrics
    first, second = full.annual_housing[:2]
    assert first["proceeds_destination_account_id"] == "alex-home-reserve"
    assert first["destination_owner_person_id"] == "alex"
    assert first["care_funded_from_home_equity_nominal"]["p50"] == 15_000
    assert first["care_portfolio_fallback_nominal"]["p50"] == 5_000
    assert second["care_funded_from_home_equity_nominal"]["p50"] == 0
    assert second["care_portfolio_fallback_nominal"]["p50"] == 20_000
    assert full.healthcare_metrics is not None
    assert full.healthcare_metrics[
        "lifetime_ltc_home_equity_used_real"
    ]["p50"] == 15_000
    assert full.healthcare_metrics["lifetime_ltc_net_cost_real"]["p50"] == 25_000
    persisted = PlannerSimulationResult.model_validate(full.as_dict())
    assert persisted.annual_housing == full.annual_housing
    assert persisted.annual_healthcare_real == full.annual_healthcare_real


def test_simulation_reports_ltc_incidence_funding_and_shortfall_manifest() -> None:
    scenario = _scenario(
        starting_portfolio=0,
        end_age=67,
        healthcare={
            "medical_inflation_rate": 0,
            "people": [
                {
                    "person_id": "alex",
                    "long_term_care": {
                        "lifetime_incidence_probability": 1,
                        "minimum_onset_age": 64,
                        "maximum_onset_age": 64,
                        "mean_duration_years": 2,
                        "duration_standard_deviation_years": 0,
                        "maximum_duration_years": 2,
                        "annual_cost_real": 20_000,
                        "insurance_annual_benefit_real": 5_000,
                        "insurance_benefit_years": 1,
                    },
                },
                {"person_id": "blair"},
            ],
        },
    )
    result = simulate(
        scenario,
        0,
        prepared_paths=_paths(scenario),
    )

    assert result.healthcare_metrics is not None
    assert result.healthcare_metrics[
        "ltc_lifetime_selection_probability"
    ] == 1
    assert result.healthcare_metrics[
        "ltc_in_plan_incidence_probability"
    ] == 1
    assert result.healthcare_metrics["trials_with_in_plan_ltc"] == 100
    assert result.healthcare_metrics[
        "lifetime_ltc_home_equity_used_real"
    ] == pytest.approx({"p10": 0, "p50": 0, "p90": 0})
    assert result.healthcare_metrics[
        "lifetime_ltc_net_cost_real"
    ] == pytest.approx({"p10": 35_000, "p50": 35_000, "p90": 35_000})
    assert result.healthcare_metrics["ltc_shortfall_probability"] == 1
    assert result.healthcare_metrics[
        "ltc_shortfall_severity_real"
    ] == pytest.approx({"p10": 35_000, "p50": 35_000, "p90": 35_000})
    assert result.ending_balance_real["p50"] == 0
    manifest = result.reproducibility["healthcare"]
    assert isinstance(manifest, dict)
    assert manifest["assumptions"] == scenario.healthcare.model_dump(mode="json")
    assert "excluded from annual_spending" in manifest[
        "annual_spending_interaction"
    ]
    persisted = PlannerSimulationResult.model_validate(result.as_dict())
    assert persisted.healthcare_metrics == result.healthcare_metrics
    assert persisted.annual_healthcare_real == result.annual_healthcare_real

    compact = simulate(
        scenario,
        0,
        prepared_paths=_paths(scenario),
        include_annual_path=False,
    )
    assert compact.annual_healthcare_real == []
    assert compact.healthcare_metrics == result.healthcare_metrics


def test_ltc_lifetime_selection_is_distinct_from_in_plan_incidence() -> None:
    scenario = _scenario(
        healthcare={
            "medical_inflation_rate": 0,
            "people": [
                {
                    "person_id": "alex",
                    "long_term_care": {
                        "lifetime_incidence_probability": 1,
                        "minimum_onset_age": 100,
                        "maximum_onset_age": 100,
                        "mean_duration_years": 1,
                        "duration_standard_deviation_years": 0,
                        "maximum_duration_years": 1,
                        "annual_cost_real": 20_000,
                    },
                },
                {"person_id": "blair"},
            ],
        }
    )

    result = simulate(
        scenario,
        500_000,
        prepared_paths=_paths(scenario),
    )

    assert result.healthcare_metrics[
        "ltc_lifetime_selection_probability"
    ] == 1
    assert result.healthcare_metrics[
        "ltc_in_plan_incidence_probability"
    ] == 0
    assert result.healthcare_metrics["trials_with_in_plan_ltc"] == 0


def test_prepared_healthcare_state_must_match_scenario() -> None:
    scenario = _scenario()
    paths = _paths(scenario)
    household = prepare_household_state(
        scenario,
        inflation_factors=paths.inflation_factors,
    )
    state = prepare_healthcare_state(
        scenario,
        household_state=household,
        inflation_factors=paths.inflation_factors,
    )
    changed = scenario.model_copy(
        update={
            "healthcare": scenario.healthcare.model_copy(
                update={"medical_inflation_rate": 0.04}
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="does not match scenario assumptions",
    ):
        simulate(
            changed,
            500_000,
            prepared_paths=_paths(changed),
            prepared_healthcare_state=state,
        )
    with pytest.raises(
        ValueError,
        match="requires healthcare assumptions",
    ):
        simulate(
            scenario.model_copy(update={"healthcare": None}),
            500_000,
            prepared_paths=paths,
            prepared_healthcare_state=state,
        )
    different_inflation_paths = SimulationPaths(
        gross_returns=paths.gross_returns,
        inflation_factors=np.asarray(
            [1, 1.1, 1.2, 1.3, 1.4, 1.5]
        ),
    )
    with pytest.raises(
        ValueError,
        match="does not match inflation paths",
    ):
        simulate(
            scenario,
            500_000,
            prepared_paths=different_inflation_paths,
            prepared_healthcare_state=state,
        )


def test_prepared_household_state_rejects_foreign_death_timeline() -> None:
    scenario = _scenario()
    paths = _paths(scenario)
    household_state = prepare_household_state(
        scenario,
        inflation_factors=paths.inflation_factors,
    )
    payload = scenario.model_dump(mode="json")
    payload["household"]["people"][0]["longevity"]["death_age"] = 90
    changed = WealthScenario.model_validate(payload)

    with pytest.raises(
        ValueError,
        match="prepared household state does not match scenario assumptions",
    ):
        simulate(
            changed,
            500_000,
            prepared_paths=_paths(changed),
            prepared_household_state=household_state,
        )


def test_exact_irmaa_history_overrides_same_year_home_gain_magi() -> None:
    scenario = WealthScenario.model_validate(
        {
            "name": "exact IRMAA history with home gain",
            "current_age": 64,
            "retirement_age": 64,
            "end_age": 67,
            "starting_portfolio": 1,
            "annual_spending": 1,
            "accounts": [{"id": "cash", "role": "cash"}],
            "tax_buckets": [
                {
                    "account_id": "cash",
                    "tax_treatment": "cash",
                    "starting_balance": 1,
                }
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["cash"],
                "retirement_surplus_destination": "cash",
                "progressive": {
                    "filing_status": "single",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1962,
                    "indiana_resident": False,
                    "irmaa_lookback_magi": [
                        {"tax_year": 2026, "magi": 250_000}
                    ],
                },
            },
            "housing_plan": {
                "home": {
                    "current_value": 500_000,
                    "cost_basis": 100_000,
                    "annual_appreciation_rate": 0,
                    "maintenance_rate": 0,
                    "property_tax_rate": 0,
                    "insurance_rate": 0,
                    "selling_cost_rate": 0,
                },
                "decision": {
                    "kind": "sell",
                    "event_age": 64,
                    "proceeds_destination_account_id": "cash",
                },
            },
            "household": {
                "plan_start_date": "2026-01-02",
                "people": [
                    {
                        "id": "alex",
                        "name": "Alex",
                        "birth_date": "1962-01-02",
                        "retirement_age_months": 64 * 12,
                        "longevity": {
                            "mode": "deterministic",
                            "death_age": 90,
                        },
                    }
                ],
            },
            "healthcare": {
                "medical_inflation_rate": 0,
                "people": [
                    {
                        "person_id": "alex",
                        "medicare_start_age": 66,
                    }
                ],
            },
            "return_mean": 0,
            "return_volatility": 0,
            "inflation_rate": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "seed": 92,
        }
    )

    result = simulate(scenario, 1, prepared_paths=_paths(scenario))

    assert result.annual_tax_audit[0][
        "realized_long_term_capital_gains_nominal"
    ]["p50"] == 150_000
    assert result.lifetime_irmaa_surcharge_real["p50"] == pytest.approx(
        6_355.20
    )
    assert result.annual_tax_audit[2]["irmaa_lookback_source"] == (
        "supplied_exact_history"
    )
    assert result.annual_healthcare_real[2]["irmaa_lookback_source"] == (
        "supplied_exact_history"
    )
    healthcare_manifest = result.reproducibility["healthcare"]
    tax_manifest = result.reproducibility["tax_policy"]
    assert healthcare_manifest["irmaa_lookback_sources"] == {
        "precedence": "supplied_exact_history_then_simulated_magi",
        "supplied_exact_history_tax_years": [2026],
    }
    assert tax_manifest["irmaa_hook"]["selection_precedence"].startswith(
        "exact_tax_year_minus_two_supplied_history_over_simulated"
    )

    missing_values = scenario.model_dump(mode="json")
    missing_values["tax_assumptions"]["progressive"]["irmaa_lookback_magi"] = []
    missing_values["healthcare"]["people"][0]["medicare_start_age"] = 64
    missing = WealthScenario.model_validate(missing_values)
    with pytest.raises(ValueError, match="missing exact IRMAA"):
        simulate(missing, 1, prepared_paths=_paths(missing))


def test_irmaa_uses_current_enrollment_but_lookback_filing_status() -> None:
    base: dict[str, object] = {
        "name": "IRMAA survivor",
        "current_age": 65,
        "retirement_age": 65,
        "end_age": 70,
        "starting_portfolio": 1_000_000,
        "annual_spending": 1,
        "income_streams": [
            {
                "name": "lookback MAGI",
                "start_age": 65,
                "end_age": 68,
                "annual_amount": 250_000,
                "inflation_adjusted": False,
                "tax_treatment": "ordinary",
            }
        ],
        "return_mean": 0,
        "return_volatility": 0,
        "inflation_rate": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 91,
        "household": {
            "plan_start_date": "2026-01-02",
            "people": [
                {
                    "id": "alex",
                    "name": "Alex",
                    "birth_date": "1961-01-02",
                    "retirement_age_months": 65 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 66,
                    },
                },
                {
                    "id": "blair",
                    "name": "Blair",
                    "birth_date": "1961-01-02",
                    "retirement_age_months": 65 * 12,
                    "longevity": {
                        "mode": "deterministic",
                        "death_age": 90,
                    },
                },
            ],
        },
        "healthcare": {
            "medical_inflation_rate": 0,
            "people": [
                {"person_id": "alex"},
                {
                    "person_id": "blair",
                    "medicare_start_age": 67,
                },
            ],
        },
        "tax_buckets": [
            {
                "tax_treatment": "cash",
                "starting_balance": 1_000_000,
            }
        ],
        "tax_assumptions": {
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["cash"],
            "retirement_surplus_destination": "cash",
            "progressive": {
                "filing_status": "married_filing_jointly",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1961,
                "spouse_birth_year": 1961,
                "indiana_resident": False,
                "irmaa_lookback_magi": [
                    {
                        "tax_year": 2024,
                        "magi": 250_000,
                        "filing_status": "married_filing_jointly",
                    },
                    {
                        "tax_year": 2025,
                        "magi": 250_000,
                        "filing_status": "married_filing_jointly",
                    },
                ],
            },
        },
    }
    scenario = WealthScenario.model_validate(base)
    expected_joint = float(
        calculate_irmaa_surcharge_arrays(
            scenario.tax_assumptions.progressive,  # type: ignore[union-attr]
            tax_year=2026,
            lookback_magi=250_000,
            eligible_people=1,
            lookback_filing_status=(
                FederalFilingStatus.MARRIED_FILING_JOINTLY
            ),
        )
    )
    expected_joint_survivor = float(
        calculate_irmaa_surcharge_arrays(
            scenario.tax_assumptions.progressive,  # type: ignore[union-attr]
            tax_year=2027,
            lookback_magi=250_000,
            eligible_people=1,
            lookback_filing_status=(
                FederalFilingStatus.MARRIED_FILING_JOINTLY
            ),
        )
    )
    expected_single_survivor = float(
        calculate_irmaa_surcharge_arrays(
            scenario.tax_assumptions.progressive,  # type: ignore[union-attr]
            tax_year=2029,
            lookback_magi=250_000,
            eligible_people=1,
            lookback_filing_status=FederalFilingStatus.SINGLE,
        )
    )

    result = simulate(
        scenario,
        1_000_000,
        prepared_paths=_paths(scenario),
    )

    assert expected_joint > 0
    assert expected_single_survivor > expected_joint_survivor
    assert result.annual_tax_audit[0][
        "irmaa_annual_surcharge_nominal"
    ]["p50"] == pytest.approx(expected_joint)
    assert result.annual_tax_audit[1][
        "irmaa_annual_surcharge_nominal"
    ]["p50"] == 0
    assert result.annual_tax_audit[2][
        "irmaa_annual_surcharge_nominal"
    ]["p50"] == pytest.approx(expected_joint_survivor)
    assert result.annual_tax_audit[3][
        "irmaa_annual_surcharge_nominal"
    ]["p50"] == pytest.approx(expected_single_survivor)
    assert result.lifetime_irmaa_surcharge_real["p50"] == pytest.approx(
        expected_joint
        + expected_joint_survivor
        + expected_single_survivor
        + result.annual_tax_audit[4][
            "irmaa_annual_surcharge_nominal"
        ]["p50"]
    )
