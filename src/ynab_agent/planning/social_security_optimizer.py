"""Bounded household Social Security claiming-strategy optimizer."""

from __future__ import annotations

from itertools import product

from pydantic import BaseModel, ConfigDict, Field

from ynab_agent.planning.household_models import (
    MAX_SOCIAL_SECURITY_CLAIM_AGE_MONTHS,
    MIN_SOCIAL_SECURITY_CLAIM_AGE_MONTHS,
)
from ynab_agent.planning.models import (
    ReturnModel,
    ValuationProvenance,
    WealthScenario,
)
from ynab_agent.planning.paths import PreparedExperiment, RunPolicy
from ynab_agent.planning.simulation import (
    SIMULATION_ENGINE_VERSION,
    canonical_scenario_sha256,
    prepare_simulation_paths,
    simulate,
)
from ynab_agent.planning.social_security import social_security_policy_manifest
from ynab_agent.planning.progressive_tax import load_tax_policy
from ynab_agent.planning.tax_engine import (
    HouseholdTaxEngine,
    household_tax_engine_for,
)


MAX_CLAIMING_STRATEGIES = 81
DEFAULT_MAXIMUM_SOCIAL_SECURITY_COMPUTE_UNITS = 50_000_000


class ClaimingStrategyEvaluation(BaseModel):
    """Household portfolio outcome for one pair of claiming ages."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    claim_age_by_person: dict[str, int]
    success_rate: float = Field(ge=0, le=1)
    funded_spending_ratio_p50: float = Field(ge=0, le=1)
    tax_adjusted_ending_portfolio_p50: float = Field(ge=0)
    lifetime_tax_real_p50: float | None = Field(default=None, ge=0)


class SocialSecurityOptimizationResult(BaseModel):
    """Ranked, bounded claiming comparison using household portfolio outcomes."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    objective: str
    recommended: ClaimingStrategyEvaluation
    evaluations: list[ClaimingStrategyEvaluation] = Field(
        min_length=1,
        max_length=MAX_CLAIMING_STRATEGIES,
    )
    strategy_count: int = Field(ge=1, le=MAX_CLAIMING_STRATEGIES)
    compute_units: int = Field(gt=0)
    manifest: dict[str, object]


def estimate_social_security_compute_units(
    scenario: WealthScenario,
    *,
    strategy_count: int,
) -> int:
    """Return the bounded trial-year-strategy work estimate."""
    return (
        (scenario.end_age - scenario.current_age)
        * scenario.trials
        * strategy_count
    )


def optimize_social_security(
    scenario: WealthScenario,
    starting_portfolio: float,
    *,
    valuation_provenance: ValuationProvenance,
    candidate_ages: tuple[int, ...] = tuple(range(62, 71)),
    run_policy: RunPolicy | None = None,
    maximum_compute_units: int = DEFAULT_MAXIMUM_SOCIAL_SECURITY_COMPUTE_UNITS,
    tax_engine: HouseholdTaxEngine | None = None,
    aggregate_compute_units: int | None = None,
) -> SocialSecurityOptimizationResult:
    """Compare claiming ages by household funded spending and portfolio outcomes."""
    if scenario.household is None:
        raise ValueError("Social Security optimization requires a household")
    if not scenario.tax_buckets or scenario.tax_assumptions is None:
        raise ValueError(
            "Social Security optimization requires explicit tax buckets and "
            "tax assumptions"
        )
    if scenario.return_model is ReturnModel.HISTORICAL_BOOTSTRAP:
        raise ValueError(
            "historical optimization requires a prepared historical experiment"
        )
    if not candidate_ages or len(candidate_ages) > 9:
        raise ValueError("candidate_ages must contain between 1 and 9 ages")
    if len(candidate_ages) != len(set(candidate_ages)):
        raise ValueError("candidate_ages must be unique")
    if any(age < 62 or age > 70 for age in candidate_ages):
        raise ValueError("candidate claiming ages must be between 62 and 70")

    if not any(
        person.primary_insurance_amount_monthly > 0
        for person in scenario.household.people
    ):
        raise ValueError("at least one household person requires a positive PIA")
    claimable_indexes = list(range(len(scenario.household.people)))
    strategy_values = list(product(candidate_ages, repeat=len(claimable_indexes)))
    if len(strategy_values) > MAX_CLAIMING_STRATEGIES:
        raise ValueError("claiming comparison exceeds the 81-strategy limit")
    compute_units = (
        aggregate_compute_units
        if aggregate_compute_units is not None
        else estimate_social_security_compute_units(
            scenario,
            strategy_count=len(strategy_values),
        )
    )
    if compute_units <= 0:
        raise ValueError("claiming comparison compute estimate must be positive")
    if compute_units > maximum_compute_units:
        raise ValueError(
            "claiming comparison exceeds the compute limit; reduce trials, "
            "horizon, or candidate ages"
        )

    paths = prepare_simulation_paths(scenario, run_policy=run_policy)
    if not isinstance(paths, PreparedExperiment):  # pragma: no cover - return contract
        raise RuntimeError("optimizer requires a prepared experiment")
    evaluations: list[ClaimingStrategyEvaluation] = []
    try:
        for values in strategy_values:
            people = list(scenario.household.people)
            claim_age_by_person: dict[str, int] = {}
            for index, age in zip(claimable_indexes, values, strict=True):
                months = age * 12
                if not (
                    MIN_SOCIAL_SECURITY_CLAIM_AGE_MONTHS
                    <= months
                    <= MAX_SOCIAL_SECURITY_CLAIM_AGE_MONTHS
                ):  # pragma: no cover - candidate validation
                    raise RuntimeError("candidate claim age escaped validated bounds")
                people[index] = people[index].model_copy(
                    update={"social_security_claim_age_months": months}
                )
                claim_age_by_person[people[index].id] = age
            candidate_household = scenario.household.model_copy(
                update={"people": people}
            )
            candidate = scenario.model_copy(
                update={"household": candidate_household}
            )
            result = simulate(
                candidate,
                starting_portfolio,
                prepared_paths=paths,
                include_annual_path=False,
                tax_engine=tax_engine,
                valuation_provenance=valuation_provenance,
            )
            ending = result.after_tax_ending_balance_real
            if ending is None:  # pragma: no cover - explicit tax input invariant
                raise RuntimeError("optimizer did not receive an after-tax outcome")
            evaluations.append(
                ClaimingStrategyEvaluation(
                    claim_age_by_person=claim_age_by_person,
                    success_rate=result.success_rate,
                    funded_spending_ratio_p50=(
                        result.funded_spending_ratio["p50"]
                    ),
                    tax_adjusted_ending_portfolio_p50=ending["p50"],
                    lifetime_tax_real_p50=(
                        result.lifetime_tax_real["p50"]
                        if result.lifetime_tax_real is not None
                        else None
                    ),
                )
            )
    finally:
        paths.close()

    evaluations.sort(
        key=lambda item: (
            item.success_rate,
            item.funded_spending_ratio_p50,
            item.tax_adjusted_ending_portfolio_p50,
        ),
        reverse=True,
    )
    progressive_policy = (
        load_tax_policy()
        if scenario.tax_assumptions.progressive is not None
        else None
    )
    return SocialSecurityOptimizationResult(
        objective=(
            "lexicographic household success rate, median funded spending, "
            "then median after-tax ending portfolio; never cumulative benefits"
        ),
        recommended=evaluations[0],
        evaluations=evaluations,
        strategy_count=len(evaluations),
        compute_units=compute_units,
        manifest={
            "schema_version": 2,
            "simulation_engine_version": SIMULATION_ENGINE_VERSION,
            "scenario_sha256": canonical_scenario_sha256(scenario),
            "starting_portfolio": starting_portfolio,
            "valuation_provenance": valuation_provenance.model_dump(mode="json"),
            "seed": scenario.seed,
            "strategy_limit": MAX_CLAIMING_STRATEGIES,
            "candidate_ages": list(candidate_ages),
            "compute_units": compute_units,
            "maximum_compute_units": maximum_compute_units,
            "market_path_policy": "one prepared experiment reused by every strategy",
            "longevity_policy": "same seed and model for every strategy",
            "ranking": (
                "success_rate,funded_spending_p50,"
                "tax_adjusted_ending_portfolio_p50"
            ),
            "tax_engine_port": household_tax_engine_for(
                scenario.tax_assumptions,
                tax_engine,
            ).engine_id,
            "progressive_tax_policy": (
                {
                    "policy_id": progressive_policy[0]["policy_id"],
                    "resource_sha256": progressive_policy[1],
                }
                if progressive_policy is not None
                else None
            ),
            "social_security": social_security_policy_manifest(),
        },
    )
