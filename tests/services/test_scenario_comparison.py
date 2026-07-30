from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.planning.spending_guardrails import (
    FixedRealSpendingPolicy,
    RetirementSpendingPlan,
    SpendingTierAmounts,
    WithdrawalRateGuardrailPolicy,
)
from ynab_agent.services.planner_jobs import PlannerExecutionPolicy
from ynab_agent.services.scenario_comparison import (
    ChangeKind,
    ScenarioComparison,
    ScenarioComparisonError,
    ScenarioComparisonErrorCode,
    ScenarioComparisonService,
    ScenarioRevision,
)
from ynab_agent.services.wealth import ResolvedStartingPortfolio


class MemoryScenarioRepository:
    def __init__(self) -> None:
        self.revisions: dict[str, ScenarioRevision] = {}
        self.comparisons: dict[str, ScenarioComparison] = {}

    async def next_revision_number(self, scenario_name: str) -> int:
        return (
            max(
                (
                    revision.revision
                    for revision in self.revisions.values()
                    if revision.scenario_name == scenario_name
                ),
                default=0,
            )
            + 1
        )

    async def create_revision(self, revision: ScenarioRevision) -> None:
        self.revisions[revision.id] = revision

    async def get_revision(self, revision_id: str) -> ScenarioRevision | None:
        return self.revisions.get(revision_id)

    async def create_comparison(self, comparison: ScenarioComparison) -> None:
        self.comparisons[comparison.id] = comparison

    async def get_comparison(
        self,
        comparison_id: str,
    ) -> ScenarioComparison | None:
        return self.comparisons.get(comparison_id)


class ExplicitPortfolioResolver:
    async def resolve_starting_portfolio_with_provenance(
        self,
        scenario: WealthScenario,
    ) -> ResolvedStartingPortfolio:
        assert scenario.starting_portfolio is not None
        return ResolvedStartingPortfolio(
            value=scenario.starting_portfolio,
            provenance=ValuationProvenance(
                source="deterministic_fixture",
                as_of=datetime(2026, 7, 30, tzinfo=timezone.utc).date(),
            ),
        )


def _scenario(
    *,
    name: str,
    starting_portfolio: float = 1_000_000,
    annual_spending: float = 40_000,
    seed: int = 42,
) -> WealthScenario:
    return WealthScenario(
        name=name,
        current_age=40,
        retirement_age=60,
        end_age=75,
        starting_portfolio=starting_portfolio,
        annual_contribution=20_000,
        annual_spending=annual_spending,
        return_mean=0.05,
        return_volatility=0.10,
        trials=100,
        seed=seed,
    )


def _service(repository: MemoryScenarioRepository) -> ScenarioComparisonService:
    return ScenarioComparisonService(
        repository=repository,
        portfolio_resolver=ExplicitPortfolioResolver(),
        execution_policy=PlannerExecutionPolicy(
            maximum_working_bytes=16 * 1024 * 1024,
            in_memory_path_bytes=8 * 1024 * 1024,
            maximum_temporary_bytes=32 * 1024 * 1024,
            batch_size=100,
        ),
    )


@pytest.mark.asyncio
async def test_saved_revisions_are_numbered_resolved_and_content_addressed() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    scenario = _scenario(name="Baseline")

    first = await service.save_revision(scenario)
    second = await service.save_revision(
        scenario.model_copy(update={"annual_spending": 42_000})
    )

    assert (first.revision, second.revision) == (1, 2)
    assert first.manifest.scenario.starting_portfolio == 1_000_000
    assert first.manifest.valuation_provenance.source == "deterministic_fixture"
    assert len(first.manifest_sha256) == 64
    assert first.manifest_sha256 != second.manifest_sha256
    assert await service.get_revision(first.id) == first


@pytest.mark.asyncio
async def test_common_path_comparison_returns_stable_deltas_and_dominance() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    baseline = await service.save_revision(_scenario(name="Baseline"))
    lower_portfolio = await service.save_revision(
        _scenario(name="Lower portfolio", starting_portfolio=500_000)
    )

    comparison = await service.compare(
        baseline_revision_id=baseline.id,
        alternative_revision_ids=[lower_portfolio.id],
    )

    alternative = comparison.alternatives[0]
    assert alternative.delta.success_probability <= 0
    assert comparison.baseline.metrics.funded_spending_real_p50 is not None
    assert alternative.delta.estate_value_real_p10 <= 0
    assert alternative.dominated_by_revision_ids == (baseline.id,)
    assert [(change.path, change.kind) for change in alternative.input_changes] == [
        ("starting_portfolio", ChangeKind.INPUT)
    ]
    assert comparison.manifest.common_paths.seed == 42
    assert comparison.manifest.revision_manifest_sha256 == {
        baseline.id: baseline.manifest_sha256,
        lower_portfolio.id: lower_portfolio.manifest_sha256,
    }
    assert (
        comparison.manifest.engine_identity.simulation_engine_version
        == "wealth_simulation_v8"
    )
    assert comparison.manifest.schema_version == 4
    assert comparison.manifest.policy_version == "retirement_outcomes_v3"
    assert comparison.manifest.engine_identity.result_schema_version == 7
    assert comparison.manifest.engine_identity.manifest_schema_version == 6
    assert await service.get_comparison(comparison.id) == comparison


@pytest.mark.asyncio
async def test_progressive_tax_comparison_retains_tax_outcomes_and_attribution() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    base_values = _scenario(name="Indiana taxes").model_dump(mode="json")
    base_values.update(
        {
            "current_age": 70,
            "retirement_age": 70,
            "end_age": 72,
            "annual_contribution": 0,
            "tax_buckets": [
                {
                    "tax_treatment": "tax_deferred",
                    "starting_balance": 1_000_000,
                }
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "single",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1956,
                    "indiana_resident": True,
                },
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["tax_deferred"],
                "retirement_surplus_destination": "tax_deferred",
            },
        }
    )
    baseline = await service.save_revision(
        WealthScenario.model_validate(base_values)
    )
    alternative_values = dict(base_values)
    alternative_values["name"] = "Federal taxes only"
    alternative_values["tax_assumptions"] = {
        **base_values["tax_assumptions"],
        "progressive": {
            **base_values["tax_assumptions"]["progressive"],
            "indiana_resident": False,
        },
    }
    alternative = await service.save_revision(
        WealthScenario.model_validate(alternative_values)
    )

    comparison = await service.compare(
        baseline_revision_id=baseline.id,
        alternative_revision_ids=[alternative.id],
    )

    result = comparison.alternatives[0]
    assert comparison.baseline.metrics.lifetime_tax_real_p50 is not None
    assert result.metrics.after_tax_estate_value_real_p50 is not None
    assert result.delta.lifetime_tax_real_p50 is not None
    assert any(
        change.path.endswith("progressive.indiana_resident")
        and change.kind is ChangeKind.TAX_POLICY
        for change in result.input_changes
    )


@pytest.mark.asyncio
async def test_tax_strategy_comparison_exposes_irmaa_costs_and_tradeoffs() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    base_values = {
        "name": "No conversions",
        "current_age": 65,
        "retirement_age": 65,
        "end_age": 70,
        "starting_portfolio": 1_000_000,
        "annual_spending": 40_000,
        "inflation_rate": 0,
        "return_mean": 0,
        "return_volatility": 0,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 92,
        "tax_buckets": [
            {"tax_treatment": "tax_deferred", "starting_balance": 900_000},
            {"tax_treatment": "roth", "starting_balance": 100_000},
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
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred", "roth"],
            "retirement_surplus_destination": "roth",
        },
    }
    baseline = await service.save_revision(WealthScenario.model_validate(base_values))
    strategy_values = {
        **base_values,
        "name": "Bracket-fill conversion",
        "tax_assumptions": {
            **base_values["tax_assumptions"],
            "strategy": {
                "withdrawal_policy": "ordered",
                "roth_conversion": {
                    "start_age": 65,
                    "end_age": 65,
                    "target_federal_ordinary_bracket_rate": 0.24,
                    "max_annual_conversion_real": 300_000,
                },
            },
        },
    }
    strategy = await service.save_revision(
        WealthScenario.model_validate(strategy_values)
    )

    comparison = await service.compare(
        baseline_revision_id=baseline.id,
        alternative_revision_ids=[strategy.id],
    )

    alternative = comparison.alternatives[0]
    assert alternative.delta.lifetime_tax_real_p50 is not None
    assert alternative.delta.lifetime_irmaa_surcharge_real_p50 is not None
    assert alternative.delta.lifetime_irmaa_surcharge_real_p50 > 0
    assert alternative.delta.irmaa_exposure_probability == pytest.approx(1)
    assert {
        tradeoff.metric: tradeoff.direction.value
        for tradeoff in alternative.tradeoffs
    }["lifetime_irmaa_surcharge_real_p50"] == "cost"
    assert any(
        change.path.startswith("tax_assumptions.strategy")
        and change.kind is ChangeKind.TAX_POLICY
        for change in alternative.input_changes
    )


@pytest.mark.asyncio
async def test_spending_change_is_attributed_and_exposes_tradeoffs() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    baseline = await service.save_revision(_scenario(name="Baseline"))
    higher_spending = await service.save_revision(
        _scenario(name="Spend more", annual_spending=50_000)
    )

    comparison = await service.compare(
        baseline_revision_id=baseline.id,
        alternative_revision_ids=[higher_spending.id],
    )

    alternative = comparison.alternatives[0]
    assert alternative.input_changes[0].path == "annual_spending"
    assert alternative.input_changes[0].kind is ChangeKind.SPENDING_POLICY
    assert alternative.delta.requested_annual_spending_real == 10_000
    directions = {
        tradeoff.metric: tradeoff.direction.value
        for tradeoff in alternative.tradeoffs
    }
    assert directions["requested_annual_spending_real"] == "benefit"


@pytest.mark.asyncio
async def test_guardrail_concessions_are_explicit_comparison_tradeoffs() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    baseline_amounts = SpendingTierAmounts(
        essential=25_000,
        lifestyle=10_000,
        discretionary=5_000,
    )
    fixed_plan = RetirementSpendingPlan(
        baseline=baseline_amounts,
        essential_floor=20_000,
        policy=FixedRealSpendingPolicy(),
    )
    guarded_plan = fixed_plan.model_copy(
        update={
            "policy": WithdrawalRateGuardrailPolicy(
                lower_withdrawal_rate=0.10,
                upper_withdrawal_rate=0.20,
                reduction_rate=0.25,
                restoration_rate=0.25,
            )
        }
    )
    scenario = WealthScenario(
        name="Fixed",
        current_age=60,
        retirement_age=60,
        end_age=64,
        starting_portfolio=200_000,
        annual_spending=40_000,
        retirement_spending_plan=fixed_plan,
        return_mean=0,
        return_volatility=0,
        inflation_rate=0,
        annual_fee_rate=0,
        trials=100,
        seed=42,
    )
    baseline = await service.save_revision(scenario)
    guarded = await service.save_revision(
        scenario.model_copy(
            update={
                "name": "Guarded",
                "retirement_spending_plan": guarded_plan,
            }
        )
    )

    comparison = await service.compare(
        baseline_revision_id=baseline.id,
        alternative_revision_ids=[guarded.id],
    )

    alternative = comparison.alternatives[0]
    assert alternative.metrics.guardrail_reduction_real_p50 is not None
    assert alternative.metrics.guardrail_reduction_real_p50 > 0
    assert alternative.delta.funded_spending_real_p50 is not None
    assert {
        tradeoff.metric: tradeoff.direction.value
        for tradeoff in alternative.tradeoffs
    }["guardrail_reduction_real_p50"] == "cost"
    assert any(
        change.kind is ChangeKind.SPENDING_POLICY
        for change in alternative.input_changes
    )


@pytest.mark.asyncio
async def test_comparison_rejects_different_economic_path_assumptions() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    baseline = await service.save_revision(_scenario(name="Baseline"))
    different_seed = await service.save_revision(
        _scenario(name="Different seed", seed=43)
    )

    with pytest.raises(ScenarioComparisonError) as error:
        await service.compare(
            baseline_revision_id=baseline.id,
            alternative_revision_ids=[different_seed.id],
        )

    assert error.value.code is ScenarioComparisonErrorCode.INCOMPATIBLE_PATHS


@pytest.mark.asyncio
async def test_comparison_rejects_duplicate_revision_ids() -> None:
    repository = MemoryScenarioRepository()
    service = _service(repository)
    baseline = await service.save_revision(_scenario(name="Baseline"))

    with pytest.raises(ScenarioComparisonError) as error:
        await service.compare(
            baseline_revision_id=baseline.id,
            alternative_revision_ids=[baseline.id],
        )

    assert error.value.code is ScenarioComparisonErrorCode.DUPLICATE_ALTERNATIVE


@pytest.mark.asyncio
async def test_comparison_aggregates_compute_without_summing_sequential_memory() -> None:
    repository = MemoryScenarioRepository()
    policy = PlannerExecutionPolicy(
        maximum_working_bytes=16 * 1024 * 1024,
        in_memory_path_bytes=8 * 1024 * 1024,
        maximum_temporary_bytes=32 * 1024 * 1024,
        batch_size=100,
        maximum_compute_units=5_000,
    )
    scenarios = [
        _scenario(name="Baseline"),
        _scenario(name="Alternative", starting_portfolio=900_000),
    ]
    individual_requirements = [
        policy.required_working_bytes(
            scenario,
            paired_historical_inflation=False,
        )
        for scenario in scenarios
    ]
    assert max(individual_requirements) < sum(individual_requirements)
    roomy_policy = policy.model_copy(
        update={"maximum_compute_units": 20_000}
    )
    assert roomy_policy.required_comparison_working_bytes(
        scenarios,
        paired_historical_inflation=False,
    ) == max(individual_requirements)
    with pytest.raises(ValueError, match="comparison compute work"):
        policy.required_comparison_working_bytes(
            scenarios,
            paired_historical_inflation=False,
        )

    service = ScenarioComparisonService(
        repository=repository,
        portfolio_resolver=ExplicitPortfolioResolver(),
        execution_policy=policy,
    )
    revisions = [
        await service.save_revision(scenario) for scenario in scenarios
    ]
    with pytest.raises(ScenarioComparisonError) as error:
        await service.compare(
            baseline_revision_id=revisions[0].id,
            alternative_revision_ids=[revisions[1].id],
        )
    assert error.value.code is ScenarioComparisonErrorCode.RESOURCE_LIMIT
