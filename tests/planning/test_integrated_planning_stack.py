from __future__ import annotations

from ynab_agent.planning.allocation import estimate_allocation_state_bytes
from ynab_agent.planning.household import estimate_household_state_bytes
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.outcomes import estimate_outcome_state_bytes
from ynab_agent.planning.paths import PathSpec, estimate_path_resources
from ynab_agent.planning.simulation import (
    REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION,
    SIMULATION_ENGINE_VERSION,
    SIMULATION_RESULT_SCHEMA_VERSION,
    simulate,
)
from ynab_agent.planning.spending_guardrails import (
    estimate_guardrail_state_bytes,
)
from ynab_agent.planning.taxes import (
    PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET,
    estimate_tax_state_bytes,
)
from ynab_agent.planning.tax_strategies import (
    TAX_STRATEGY_EVALUATIONS_PER_YEAR,
)
from ynab_agent.services.planner_jobs import (
    PLANNER_JOB_REQUEST_SCHEMA_VERSION,
    PlannerExecutionPolicy,
)
from ynab_agent.services.scenario_comparison import (
    SCENARIO_COMPARISON_SCHEMA_VERSION,
)


def _integrated_scenario() -> WealthScenario:
    return WealthScenario.model_validate(
        {
            "name": "integrated household allocation",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 63,
            "accounts": [
                {
                    "id": f"{owner}-{treatment}",
                    "role": (
                        "taxable"
                        if treatment == "taxable"
                        else "retirement"
                    ),
                    "owner_person_id": owner,
                }
                for owner in ("alex", "blair")
                for treatment in ("tax_deferred", "roth", "taxable")
            ],
            "starting_portfolio": 400_000,
            "annual_spending": 40_000,
            "retirement_spending_plan": {
                "baseline": {
                    "essential": 25_000,
                    "lifestyle": 10_000,
                    "discretionary": 5_000,
                    "one_time": 0,
                },
                "essential_floor": 20_000,
                "policy": {"kind": "fixed_real"},
            },
            "portfolio_allocation": {
                "market": {
                    "us_equity": {
                        "expected_return": 0.08,
                        "volatility": 0.18,
                    },
                    "international_equity": {
                        "expected_return": 0.07,
                        "volatility": 0.20,
                    },
                    "bonds": {
                        "expected_return": 0.04,
                        "volatility": 0.07,
                    },
                    "cash": {
                        "expected_return": 0.025,
                        "volatility": 0.01,
                    },
                    "correlation": {
                        "values": [
                            [1, 0.60, 0.20, 0.05],
                            [0.60, 1, 0.15, 0.05],
                            [0.20, 0.15, 1, 0.20],
                            [0.05, 0.05, 0.20, 1],
                        ]
                    },
                },
                "accounts": [
                    {
                        "account_id": f"{owner}-{treatment}",
                        "portfolio_weight": balance / 400_000,
                        "annual_fee_rate": (
                            0.001
                            if treatment == "taxable"
                            else 0.002
                        ),
                        "target": (
                            {
                                "us_equity": 0.20,
                                "international_equity": 0.10,
                                "bonds": 0.65,
                                "cash": 0.05,
                            }
                            if treatment == "tax_deferred"
                            else {
                                "us_equity": 0.70,
                                "international_equity": 0.20,
                                "bonds": 0.05,
                                "cash": 0.05,
                            }
                        ),
                        **(
                            {
                                "glide_path": [
                                    {
                                        "age": 62,
                                        "weights": {
                                            "us_equity": 0.60,
                                            "international_equity": 0.20,
                                            "bonds": 0.15,
                                            "cash": 0.05,
                                        },
                                    }
                                ]
                            }
                            if treatment == "taxable"
                            else {}
                        ),
                    }
                    for owner in ("alex", "blair")
                    for treatment, balance in (
                        ("tax_deferred", 100_000),
                        ("roth", 50_000),
                        ("taxable", 50_000),
                    )
                ],
                "rebalancing": {
                    "frequency_years": 1,
                    "drift_threshold": 0.05,
                },
            },
            "household": {
                "plan_start_date": "2026-01-02",
                "survivor_spending_fraction": 0.75,
                "people": [
                    {
                        "id": "alex",
                        "name": "Alex",
                        "birth_date": "1966-01-02",
                        "retirement_age_months": 60 * 12,
                        "primary_insurance_amount_monthly": 2_500,
                        "social_security_claim_age_months": 62 * 12,
                        "longevity": {
                            "mode": "deterministic",
                            "death_age": 61,
                        },
                    },
                    {
                        "id": "blair",
                        "name": "Blair",
                        "birth_date": "1966-01-02",
                        "retirement_age_months": 60 * 12,
                        "primary_insurance_amount_monthly": 1_500,
                        "social_security_claim_age_months": 62 * 12,
                        "survivor_claim_age_months": 60 * 12,
                        "longevity": {
                            "mode": "deterministic",
                            "death_age": 95,
                        },
                    },
                ],
            },
            "tax_buckets": [
                {
                    "tax_treatment": treatment,
                    "owner_person_id": owner,
                    "account_id": f"{owner}-{treatment}",
                    "starting_balance": balance,
                    **(
                        {"taxable_basis": balance / 2}
                        if treatment == "taxable"
                        else {}
                    ),
                }
                for owner in ("alex", "blair")
                for treatment, balance in (
                    ("tax_deferred", 100_000),
                    ("roth", 50_000),
                    ("taxable", 50_000),
                )
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "married_filing_jointly",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1966,
                    "spouse_birth_year": 1966,
                    "aca_household_size": 2,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": [
                    "taxable",
                    "tax_deferred",
                    "roth",
                ],
                "retirement_surplus_destination": "roth",
                "strategy": {
                    "withdrawal_policy": "proportional",
                    "proportional_withdrawal_fractions": [
                        {
                            "tax_treatment": "taxable",
                            "fraction": 0.4,
                        },
                        {
                            "tax_treatment": "tax_deferred",
                            "fraction": 0.3,
                        },
                        {
                            "tax_treatment": "roth",
                            "fraction": 0.3,
                        },
                    ],
                    "roth_conversion": {
                        "start_age": 60,
                        "end_age": 61,
                        "target_federal_ordinary_bracket_rate": 0.10,
                        "max_annual_conversion_real": 10_000,
                    },
                    "capital_gain_harvest": {
                        "start_age": 60,
                        "end_age": 61,
                        "target_federal_long_term_capital_gains_rate": 0,
                        "max_annual_gain_real": 10_000,
                    },
                    "asset_location_preferences": [
                        {
                            "asset_class": "bonds",
                            "preferred_tax_treatments": ["tax_deferred"],
                        },
                        {
                            "asset_class": "us_equity",
                            "preferred_tax_treatments": ["taxable", "roth"],
                        },
                    ],
                },
            },
            "inflation_rate": 0.02,
            "return_mean": 0.06,
            "return_volatility": 0.16,
            "annual_fee_rate": 0,
            "trials": 100,
            "seed": 95,
        }
    )


def test_combined_household_owner_strategy_and_multi_asset_replays() -> None:
    scenario = _integrated_scenario()

    result = simulate(scenario, 400_000)
    replay = simulate(scenario, 400_000)

    assert result.annual_tax_strategy_actions == (
        replay.annual_tax_strategy_actions
    )
    assert result.annual_allocation_real == replay.annual_allocation_real
    assert result.household_cash_flow_audit == (
        replay.household_cash_flow_audit
    )
    assert len(result.annual_allocation_real) == 3
    assert len(result.annual_tax_strategy_actions) == 3
    assert result.annual_tax_strategy_actions[0][
        "roth_conversion_nominal"
    ]["p50"] > 0
    assert result.annual_tax_strategy_actions[0][
        "harvested_long_term_capital_gains_nominal"
    ]["p50"] > 0
    assert {
        row["owner_person_id"]
        for row in result.annual_tax_audit[0]["owner_flows"]
    } == {"alex", "blair"}
    survivor_row = next(
        row
        for row in result.household_cash_flow_audit
        if row["year"] == 2027 and row["person_id"] == "blair"
    )
    assert survivor_row["filing_status"] == {
        "married_filing_jointly_probability": 0.0,
        "single_probability": 1.0,
        "no_filer_probability": 0.0,
    }
    social_security_row = next(
        row
        for row in result.household_cash_flow_audit
        if row["year"] == 2028 and row["person_id"] == "blair"
    )
    assert social_security_row["social_security_paid_real"]["p50"] > 0

    assert SIMULATION_RESULT_SCHEMA_VERSION == 11
    assert REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION == 10
    assert SIMULATION_ENGINE_VERSION == "wealth_simulation_v12"
    assert PLANNER_JOB_REQUEST_SCHEMA_VERSION == 8
    assert SCENARIO_COMPARISON_SCHEMA_VERSION == 8
    assert result.reproducibility["household"] is not None
    assert result.reproducibility["portfolio_allocation"] is not None
    assert result.reproducibility["multi_asset_paths"] is not None
    tax_policy = result.reproducibility["tax_policy"]
    assert tax_policy["strategy"]["strategy"] == (
        scenario.tax_assumptions.strategy.model_dump(mode="json")
    )


def test_combined_resource_preflight_is_the_exact_aggregate() -> None:
    scenario = _integrated_scenario()
    policy = PlannerExecutionPolicy(
        maximum_working_bytes=128 * 1024 * 1024,
        in_memory_path_bytes=128 * 1024 * 1024,
        maximum_temporary_bytes=128 * 1024 * 1024,
        batch_size=37,
        maximum_compute_units=100_000_000,
    )
    path_estimate = estimate_path_resources(
        PathSpec.from_scenario(
            scenario,
            paired_historical_inflation=False,
        ),
        batch_size=policy.batch_size,
    )
    components = {
        "paths": policy.to_run_policy().required_working_bytes(
            path_estimate
        ),
        "tax": estimate_tax_state_bytes(scenario),
        "outcomes": estimate_outcome_state_bytes(scenario),
        "guardrails": estimate_guardrail_state_bytes(scenario.trials),
        "household": estimate_household_state_bytes(scenario),
        "allocation": estimate_allocation_state_bytes(
            scenario.portfolio_allocation,
            scenario.trials,
            batch_size=policy.batch_size,
        ),
    }

    assert all(value > 0 for value in components.values())
    assert policy.required_working_bytes(
        scenario,
        paired_historical_inflation=False,
    ) == sum(components.values())

    bucket_visits = len(scenario.tax_buckets)
    strategy = scenario.tax_assumptions.strategy
    action_count = int(strategy.roth_conversion is not None) + int(
        strategy.capital_gain_harvest is not None
    )
    projection_evaluations = (
        bucket_visits * 2 * PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET
        + 4
    )
    action_projections = (
        action_count * TAX_STRATEGY_EVALUATIONS_PER_YEAR // 2
    ) + 1
    work_per_trial_year = (
        1
        + bucket_visits
        + 1
        + len(scenario.household.people) * 5
        + len(scenario.portfolio_allocation.accounts) * 4
        + bucket_visits
        + bucket_visits
        * 2
        * PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET
        + 4
        + action_projections * projection_evaluations
    )
    expected_compute_units = (
        (scenario.end_age - scenario.current_age)
        * scenario.trials
        * work_per_trial_year
    )

    assert PlannerExecutionPolicy._compute_units(scenario) == (
        expected_compute_units
    )
