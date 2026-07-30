"""Wealth-planning models and simulations."""

from ynab_agent.planning.household_models import (
    Household,
    LongevityAssumption,
    PensionTimeline,
    Person,
    WorkTimeline,
)
from ynab_agent.planning.models import (
    FederalFilingStatus,
    FutureTaxPolicyMode,
    IncomeTaxTreatment,
    ProgressiveTaxAssumptions,
    RothConversionStrategy,
    ReturnModel,
    CapitalGainHarvestStrategy,
    AssetLocationPreference,
    TaxAssumptions,
    TaxBucket,
    TaxModel,
    TaxTreatment,
    TaxStrategyAssumptions,
    TaxTreatmentFraction,
    WealthScenario,
    WithdrawalPolicy,
)
from ynab_agent.planning.progressive_tax import (
    AnnualTaxResult,
    TaxCalculationInput,
    TaxPolicyManifest,
    calculate_annual_tax,
)
from ynab_agent.planning.paths import SimulationPaths
from ynab_agent.planning.simulation import (
    SimulationResult,
    prepare_simulation_paths,
    simulate,
)
from ynab_agent.planning.solver import SolveResult, SolveVariable, solve_scenario
from ynab_agent.planning.social_security_optimizer import (
    SocialSecurityOptimizationResult,
    optimize_social_security,
)

__all__ = [
    "ReturnModel",
    "FederalFilingStatus",
    "FutureTaxPolicyMode",
    "IncomeTaxTreatment",
    "ProgressiveTaxAssumptions",
    "RothConversionStrategy",
    "CapitalGainHarvestStrategy",
    "AssetLocationPreference",
    "Household",
    "LongevityAssumption",
    "PensionTimeline",
    "Person",
    "SimulationResult",
    "SimulationPaths",
    "SolveResult",
    "SolveVariable",
    "TaxAssumptions",
    "TaxBucket",
    "TaxCalculationInput",
    "TaxModel",
    "TaxPolicyManifest",
    "TaxTreatment",
    "TaxStrategyAssumptions",
    "TaxTreatmentFraction",
    "AnnualTaxResult",
    "SocialSecurityOptimizationResult",
    "WealthScenario",
    "WithdrawalPolicy",
    "WorkTimeline",
    "optimize_social_security",
    "prepare_simulation_paths",
    "calculate_annual_tax",
    "simulate",
    "solve_scenario",
]
