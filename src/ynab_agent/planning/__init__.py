"""Wealth-planning models and simulations."""

from ynab_agent.planning.models import (
    FederalFilingStatus,
    FutureTaxPolicyMode,
    IncomeTaxTreatment,
    ProgressiveTaxAssumptions,
    ReturnModel,
    TaxAssumptions,
    TaxBucket,
    TaxModel,
    TaxTreatment,
    WealthScenario,
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

__all__ = [
    "ReturnModel",
    "FederalFilingStatus",
    "FutureTaxPolicyMode",
    "IncomeTaxTreatment",
    "ProgressiveTaxAssumptions",
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
    "AnnualTaxResult",
    "WealthScenario",
    "prepare_simulation_paths",
    "calculate_annual_tax",
    "simulate",
    "solve_scenario",
]
