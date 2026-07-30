"""Deterministic current-year tax-strategy decisions for wealth simulations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ynab_agent.planning.models import (
    ProgressiveTaxAssumptions,
    TaxStrategyAssumptions,
)
from ynab_agent.planning.progressive_tax import (
    federal_capital_gain_bracket_ceiling,
    federal_ordinary_bracket_ceiling,
)

if TYPE_CHECKING:
    import numpy as np


TAX_STRATEGY_SCHEMA_VERSION = 1
TAX_STRATEGY_DECISION_POLICY = "current_year_joint_cash_need_v2"
TAX_STRATEGY_BISECTION_STEPS = 48
TAX_STRATEGY_EVALUATIONS_PER_YEAR = 2 * TAX_STRATEGY_BISECTION_STEPS


@dataclass(frozen=True)
class TaxStrategyDecision:
    """Per-trial actions selected without access to future return paths."""

    roth_conversion: np.ndarray
    harvested_long_term_capital_gains: np.ndarray


def decide_tax_strategy(
    strategy: TaxStrategyAssumptions,
    progressive: ProgressiveTaxAssumptions,
    *,
    age: int,
    tax_year: int,
    tax_deferred_balance: np.ndarray,
    taxable_balance: np.ndarray,
    taxable_basis: np.ndarray,
    nominal_conversion_cap: float | np.ndarray,
    nominal_harvest_cap: float | np.ndarray,
    project_final_taxable_position: Callable[
        [np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ],
) -> TaxStrategyDecision:
    """Choose actions that fit after projected current-year withdrawals."""
    import numpy as np

    shape = tax_deferred_balance.shape
    conversion = np.zeros(shape, dtype=float)
    conversion_rule = strategy.roth_conversion
    conversion_active = (
        conversion_rule is not None
        and conversion_rule.start_age <= age <= conversion_rule.end_age
    )
    if conversion_active and conversion_rule is not None:
        ceiling = federal_ordinary_bracket_ceiling(
            progressive,
            tax_year=tax_year,
            rate=conversion_rule.target_federal_ordinary_bracket_rate,
        )
        high = np.minimum(tax_deferred_balance, nominal_conversion_cap)
        low = np.zeros(shape, dtype=float)
        zero_gains = np.zeros(shape, dtype=float)
        for _ in range(TAX_STRATEGY_BISECTION_STEPS):
            candidate = (low + high) / 2
            final_taxable_ordinary, _ = project_final_taxable_position(
                candidate,
                zero_gains,
            )
            fits = final_taxable_ordinary <= ceiling
            low = np.where(fits, candidate, low)
            high = np.where(fits, high, candidate)
        conversion = low

    harvest = np.zeros(shape, dtype=float)
    harvest_rule = strategy.capital_gain_harvest
    if (
        harvest_rule is not None
        and harvest_rule.start_age <= age <= harvest_rule.end_age
    ):
        ceiling = federal_capital_gain_bracket_ceiling(
            progressive,
            tax_year=tax_year,
            rate=harvest_rule.target_federal_long_term_capital_gains_rate,
        )
        embedded_gain = np.maximum(0.0, taxable_balance - taxable_basis)
        high = np.minimum(embedded_gain, nominal_harvest_cap)
        low = np.zeros(shape, dtype=float)
        ordinary_limit: np.ndarray | None = None
        if conversion_active and conversion_rule is not None:
            ordinary_ceiling = federal_ordinary_bracket_ceiling(
                progressive,
                tax_year=tax_year,
                rate=conversion_rule.target_federal_ordinary_bracket_rate,
            )
            baseline_ordinary, _ = project_final_taxable_position(
                conversion,
                np.zeros(shape, dtype=float),
            )
            # If forced income already exceeds the target, harvesting may
            # proceed only while it does not make that ordinary position worse
            # (for example by causing more Social Security to become taxable).
            ordinary_limit = np.maximum(ordinary_ceiling, baseline_ordinary)
        for _ in range(TAX_STRATEGY_BISECTION_STEPS):
            candidate = (low + high) / 2
            final_taxable_ordinary, final_taxable_income = (
                project_final_taxable_position(
                    conversion,
                    candidate,
                )
            )
            fits = final_taxable_income <= ceiling
            if ordinary_limit is not None:
                fits &= final_taxable_ordinary <= ordinary_limit
            low = np.where(fits, candidate, low)
            high = np.where(fits, high, candidate)
        harvest = low

    return TaxStrategyDecision(
        roth_conversion=conversion,
        harvested_long_term_capital_gains=harvest,
    )


def tax_strategy_manifest(
    strategy: TaxStrategyAssumptions | None,
) -> dict[str, object] | None:
    """Return the complete versioned strategy recipe persisted with a run."""
    if strategy is None:
        return None
    return {
        "schema_version": TAX_STRATEGY_SCHEMA_VERSION,
        "decision_policy": TAX_STRATEGY_DECISION_POLICY,
        "decision_timing": "after_current_year_return_joint_with_current_year_withdrawals",
        "future_path_information_used": False,
        "strategy": strategy.model_dump(mode="json"),
        "asset_location_execution": "typed_advisory_hook",
    }
