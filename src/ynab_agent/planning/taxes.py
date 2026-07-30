"""Bounded account-aware tax mechanics for retirement simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ynab_agent.planning.models import (
    TaxAssumptions,
    TaxModel,
    TaxTreatment,
    WealthScenario,
)
from ynab_agent.planning.progressive_tax import (
    calculate_income_tax_component_arrays,
    calculate_income_tax_arrays,
    rmd_divisor,
    rmd_start_age,
)

if TYPE_CHECKING:
    import numpy as np


# IRS Publication 590-B, Table III (Uniform Lifetime), current through 2025.
_RMD_DIVISORS = {
    72: 27.4,
    73: 26.5,
    74: 25.5,
    75: 24.6,
    76: 23.7,
    77: 22.9,
    78: 22.0,
    79: 21.1,
    80: 20.2,
    81: 19.4,
    82: 18.5,
    83: 17.7,
    84: 16.8,
    85: 16.0,
    86: 15.2,
    87: 14.4,
    88: 13.7,
    89: 12.9,
    90: 12.2,
    91: 11.5,
    92: 10.8,
    93: 10.1,
    94: 9.5,
    95: 8.9,
    96: 8.4,
    97: 7.8,
    98: 7.3,
    99: 6.8,
    100: 6.4,
    101: 6.0,
    102: 5.6,
    103: 5.2,
    104: 4.9,
    105: 4.6,
    106: 4.3,
    107: 4.1,
    108: 3.9,
    109: 3.7,
    110: 3.5,
    111: 3.4,
    112: 3.3,
    113: 3.1,
    114: 3.0,
    115: 2.9,
    116: 2.8,
    117: 2.7,
    118: 2.5,
    119: 2.3,
    120: 2.0,
}

PROGRESSIVE_WITHDRAWAL_MAX_BRACKET_STEPS = 64
PROGRESSIVE_WITHDRAWAL_MAX_BISECTION_STEPS = 64
PROGRESSIVE_TAX_EVALUATIONS_PER_BUCKET = (
    PROGRESSIVE_WITHDRAWAL_MAX_BRACKET_STEPS
    + PROGRESSIVE_WITHDRAWAL_MAX_BISECTION_STEPS
    + 2
)


@dataclass
class TaxAwarePortfolio:
    """Per-trial balances and basis for a bounded set of tax treatments."""

    balances: dict[TaxTreatment, np.ndarray]
    taxable_basis: np.ndarray | None
    cumulative_tax_real: np.ndarray
    annual_tax_nominal: np.ndarray
    annual_effective_rate: np.ndarray
    annual_marginal_ordinary_rate: np.ndarray
    annual_marginal_ltcg_rate: np.ndarray
    annual_federal_tax_nominal: np.ndarray
    annual_state_tax_nominal: np.ndarray
    annual_taxable_social_security_nominal: np.ndarray
    annual_federal_deduction_nominal: np.ndarray
    annual_realized_long_term_capital_gains_nominal: np.ndarray
    annual_early_distribution_penalty_nominal: np.ndarray

    @classmethod
    def from_scenario(
        cls,
        scenario: WealthScenario,
        *,
        starting_portfolio: float,
    ) -> TaxAwarePortfolio:
        import numpy as np

        if not scenario.tax_buckets or scenario.tax_assumptions is None:
            raise ValueError("tax-aware portfolio requires tax buckets and assumptions")
        bucket_total = sum(bucket.starting_balance for bucket in scenario.tax_buckets)
        if abs(bucket_total - starting_portfolio) > 0.01:
            raise ValueError("tax bucket balances do not match resolved starting portfolio")
        balances = {
            bucket.tax_treatment: np.full(
                scenario.trials,
                bucket.starting_balance,
                dtype=float,
            )
            for bucket in scenario.tax_buckets
        }
        taxable_bucket = next(
            (
                bucket
                for bucket in scenario.tax_buckets
                if bucket.tax_treatment is TaxTreatment.TAXABLE
            ),
            None,
        )
        taxable_basis = None
        if taxable_bucket is not None:
            if taxable_bucket.taxable_basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable bucket is missing its basis")
            taxable_basis = np.full(
                scenario.trials,
                taxable_bucket.taxable_basis,
                dtype=float,
            )
        return cls(
            balances=balances,
            taxable_basis=taxable_basis,
            cumulative_tax_real=np.zeros(scenario.trials, dtype=float),
            annual_tax_nominal=np.zeros(scenario.trials, dtype=float),
            annual_effective_rate=np.zeros(scenario.trials, dtype=float),
            annual_marginal_ordinary_rate=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            annual_marginal_ltcg_rate=np.zeros(scenario.trials, dtype=float),
            annual_federal_tax_nominal=np.zeros(scenario.trials, dtype=float),
            annual_state_tax_nominal=np.zeros(scenario.trials, dtype=float),
            annual_taxable_social_security_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            annual_federal_deduction_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            annual_realized_long_term_capital_gains_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            annual_early_distribution_penalty_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
        )

    def total(self, trial_slice: slice) -> np.ndarray:
        import numpy as np

        return np.sum(
            [balance[trial_slice] for balance in self.balances.values()],
            axis=0,
        )

    def opening_tax_deferred(self, trial_slice: slice) -> np.ndarray:
        import numpy as np

        balance = self.balances.get(TaxTreatment.TAX_DEFERRED)
        if balance is None:
            return np.zeros_like(self.cumulative_tax_real[trial_slice])
        return balance[trial_slice].copy()

    def after_tax_estate_value(
        self,
        trial_slice: slice,
        *,
        assumptions: TaxAssumptions,
        tax_year: int,
    ) -> np.ndarray:
        """Return liquidation value after account-character taxes."""
        import numpy as np

        if assumptions.progressive is not None:
            total = self.total(trial_slice)
            ordinary = np.zeros_like(total)
            tax_deferred = self.balances.get(TaxTreatment.TAX_DEFERRED)
            if tax_deferred is not None:
                ordinary += tax_deferred[trial_slice]
            hsa = self.balances.get(TaxTreatment.HSA)
            if hsa is not None:
                ordinary += hsa[trial_slice] * (1 - assumptions.qualified_hsa_withdrawal_fraction)
            gains = np.zeros_like(total)
            taxable = self.balances.get(TaxTreatment.TAXABLE)
            if taxable is not None:
                if self.taxable_basis is None:  # pragma: no cover - invariant
                    raise RuntimeError("taxable balance is missing its basis")
                gains = np.maximum(
                    0.0,
                    taxable[trial_slice] - self.taxable_basis[trial_slice],
                )
            liquidation_tax = calculate_income_tax_arrays(
                assumptions.progressive,
                tax_year=tax_year,
                ordinary_income=ordinary,
                long_term_capital_gains=gains,
                social_security_income=np.zeros_like(total),
            )
            return cast("np.ndarray", total - liquidation_tax)

        estate = np.zeros_like(self.cumulative_tax_real[trial_slice])
        for treatment, all_balances in self.balances.items():
            balance = all_balances[trial_slice]
            if treatment is TaxTreatment.TAX_DEFERRED:
                estate += balance * (1 - assumptions.ordinary_income_tax_rate)
                continue
            if treatment is TaxTreatment.HSA:
                unqualified = 1 - assumptions.qualified_hsa_withdrawal_fraction
                estate += balance * (1 - unqualified * assumptions.ordinary_income_tax_rate)
                continue
            if treatment is TaxTreatment.TAXABLE:
                if self.taxable_basis is None:  # pragma: no cover - model invariant
                    raise RuntimeError("taxable balance is missing its basis")
                gains = np.maximum(
                    0.0,
                    balance - self.taxable_basis[trial_slice],
                )
                estate += balance - gains * assumptions.long_term_capital_gains_tax_rate
                continue
            estate += balance
        return estate

    def apply_growth(
        self,
        trial_slice: slice,
        gross_return: np.ndarray,
        *,
        inflation_factor: float | np.ndarray,
        assumptions: TaxAssumptions,
    ) -> None:
        for balance in self.balances.values():
            balance[trial_slice] *= gross_return
        taxable = self.balances.get(TaxTreatment.TAXABLE)
        if taxable is None or assumptions.taxable_account_annual_tax_drag_rate == 0:
            return
        tax_drag = taxable[trial_slice] * assumptions.taxable_account_annual_tax_drag_rate
        taxable[trial_slice] -= tax_drag
        self.cumulative_tax_real[trial_slice] += tax_drag / inflation_factor

    def add_contribution(
        self,
        trial_slice: slice,
        amount: float | np.ndarray,
        *,
        scenario: WealthScenario,
        destination: TaxTreatment | None,
    ) -> None:
        if destination is not None:
            self._deposit(trial_slice, destination, amount)
            return
        for bucket in scenario.tax_buckets:
            if bucket.contribution_fraction == 0:
                continue
            self._deposit(
                trial_slice,
                bucket.tax_treatment,
                amount * bucket.contribution_fraction,
            )

    def fund_retirement_spending(
        self,
        trial_slice: slice,
        *,
        age: int,
        tax_year: int,
        spending: float | np.ndarray,
        ordinary_income: float | np.ndarray,
        social_security_income: float | np.ndarray,
        tax_free_income: float | np.ndarray,
        opening_tax_deferred: np.ndarray,
        inflation_factor: float | np.ndarray,
        assumptions: TaxAssumptions,
    ) -> np.ndarray:
        """Fund one retirement year and return unmet spending by trial."""
        import numpy as np

        ordinary = np.asarray(ordinary_income, dtype=float)
        social_security = np.asarray(social_security_income, dtype=float)
        tax_free = np.asarray(tax_free_income, dtype=float)
        required_minimum = self._required_minimum_distribution(
            trial_slice,
            age=age,
            tax_year=tax_year,
            opening_balance=opening_tax_deferred,
            assumptions=assumptions,
        )
        ordinary = ordinary + required_minimum
        if assumptions.tax_model is TaxModel.PROGRESSIVE_US_INDIANA:
            return self._fund_progressive_spending(
                trial_slice,
                age=age,
                tax_year=tax_year,
                spending=spending,
                ordinary_income=ordinary,
                social_security_income=social_security,
                tax_free_income=tax_free,
                inflation_factor=inflation_factor,
                assumptions=assumptions,
            )
        income_tax = assumptions.ordinary_income_tax_rate * (
            ordinary + social_security * assumptions.social_security_taxable_fraction
        )
        self.cumulative_tax_real[trial_slice] += income_tax / inflation_factor
        available_cash = ordinary + social_security + tax_free - income_tax
        remaining = np.maximum(0.0, spending - available_cash)
        surplus = np.maximum(0.0, available_cash - spending)
        if np.any(surplus):
            self._deposit(
                trial_slice,
                assumptions.retirement_surplus_destination,
                surplus,
            )

        for treatment in assumptions.withdrawal_order:
            if not np.any(remaining > 0.005):
                break
            tax_rate = self._withdrawal_tax_rate(
                treatment,
                age=age,
                assumptions=assumptions,
            )
            gross, tax = self._withdraw_for_net_need(
                trial_slice,
                treatment=treatment,
                remaining=remaining,
                tax_rate=tax_rate,
                capital_gains_rate=assumptions.long_term_capital_gains_tax_rate,
            )
            remaining = np.maximum(0.0, remaining - (gross - tax))
            self.cumulative_tax_real[trial_slice] += tax / inflation_factor
        return cast("np.ndarray", remaining)

    def _fund_progressive_spending(
        self,
        trial_slice: slice,
        *,
        age: int,
        tax_year: int,
        spending: float | np.ndarray,
        ordinary_income: np.ndarray,
        social_security_income: np.ndarray,
        tax_free_income: np.ndarray,
        inflation_factor: float | np.ndarray,
        assumptions: TaxAssumptions,
    ) -> np.ndarray:
        """Fund spending against the incremental liability of each withdrawal."""
        import numpy as np

        progressive = assumptions.progressive
        if progressive is None:  # pragma: no cover - model invariant
            raise RuntimeError("progressive tax assumptions are missing")
        shape = self.cumulative_tax_real[trial_slice].shape
        ordinary = np.broadcast_to(ordinary_income, shape).astype(float, copy=True)
        social_security = np.broadcast_to(
            social_security_income,
            shape,
        ).astype(float, copy=True)
        tax_free = np.broadcast_to(tax_free_income, shape).astype(float, copy=False)
        zero_gains = np.zeros(shape, dtype=float)
        starting_components = calculate_income_tax_component_arrays(
            progressive,
            tax_year=tax_year,
            ordinary_income=ordinary,
            long_term_capital_gains=zero_gains,
            social_security_income=social_security,
        )
        income_tax = starting_components.total_income_tax
        self.cumulative_tax_real[trial_slice] += income_tax / inflation_factor
        available_cash = ordinary + social_security + tax_free - income_tax
        remaining = np.maximum(0.0, spending - available_cash)
        surplus = np.maximum(0.0, available_cash - spending)
        if np.any(surplus):
            self._deposit(
                trial_slice,
                assumptions.retirement_surplus_destination,
                surplus,
            )

        gains = zero_gains
        current_tax = income_tax
        annual_penalty = np.zeros(shape, dtype=float)
        for treatment in assumptions.withdrawal_order:
            if not np.any(remaining > 0.005):
                break
            balance = self.balances.get(treatment)
            if balance is None:
                continue
            available = balance[trial_slice]
            ordinary_fraction: float | np.ndarray = 0.0
            gain_fraction: float | np.ndarray = 0.0
            penalty_rate = 0.0
            basis_ratio: np.ndarray | None = None
            if treatment is TaxTreatment.TAX_DEFERRED:
                ordinary_fraction = 1.0
                if age < 60:
                    penalty_rate = assumptions.early_distribution_penalty_rate
            elif treatment is TaxTreatment.HSA:
                ordinary_fraction = 1 - assumptions.qualified_hsa_withdrawal_fraction
                if age < 65:
                    penalty_rate = (
                        assumptions.hsa_early_distribution_penalty_rate * ordinary_fraction
                    )
            elif treatment is TaxTreatment.TAXABLE:
                if self.taxable_basis is None:  # pragma: no cover - invariant
                    raise RuntimeError("taxable balance is missing its basis")
                basis_ratio = np.divide(
                    self.taxable_basis[trial_slice],
                    available,
                    out=np.zeros_like(available),
                    where=available > 0,
                )
                gain_fraction = np.maximum(0.0, 1 - basis_ratio)

            low = np.zeros_like(remaining)
            high = np.minimum(
                available,
                np.maximum(remaining, 0.01),
            )
            after_tax = current_tax
            net = np.zeros_like(remaining)
            for _ in range(PROGRESSIVE_WITHDRAWAL_MAX_BRACKET_STEPS):
                after_tax = calculate_income_tax_arrays(
                    progressive,
                    tax_year=tax_year,
                    ordinary_income=ordinary + high * ordinary_fraction,
                    long_term_capital_gains=gains + high * gain_fraction,
                    social_security_income=social_security,
                )
                net = high - (after_tax - current_tax) - high * penalty_rate
                needs_larger_bracket = (net < remaining) & (high < available)
                if not np.any(needs_larger_bracket):
                    break
                low = np.where(needs_larger_bracket, high, low)
                high = np.where(
                    needs_larger_bracket,
                    np.minimum(available, np.maximum(0.01, high * 2)),
                    high,
                )
            unresolved = (net < remaining) & (high < available)
            if np.any(unresolved):
                high = np.where(unresolved, available, high)
                after_tax = calculate_income_tax_arrays(
                    progressive,
                    tax_year=tax_year,
                    ordinary_income=ordinary + high * ordinary_fraction,
                    long_term_capital_gains=gains + high * gain_fraction,
                    social_security_income=social_security,
                )
                net = high - (after_tax - current_tax) - high * penalty_rate
            fundable = net >= remaining
            low = np.where(fundable, low, available)
            high = np.where(fundable, high, available)
            for _ in range(PROGRESSIVE_WITHDRAWAL_MAX_BISECTION_STEPS):
                active = fundable & ((high - low) > 0.005)
                if not np.any(active):
                    break
                gross = (low + high) / 2
                after_tax = calculate_income_tax_arrays(
                    progressive,
                    tax_year=tax_year,
                    ordinary_income=ordinary + gross * ordinary_fraction,
                    long_term_capital_gains=gains + gross * gain_fraction,
                    social_security_income=social_security,
                )
                net = gross - (after_tax - current_tax) - gross * penalty_rate
                insufficient = active & (net < remaining)
                sufficient = active & ~insufficient
                low = np.where(insufficient, gross, low)
                high = np.where(sufficient, gross, high)
            gross = np.where(fundable, high, available)
            final_components = calculate_income_tax_component_arrays(
                progressive,
                tax_year=tax_year,
                ordinary_income=ordinary + gross * ordinary_fraction,
                long_term_capital_gains=gains + gross * gain_fraction,
                social_security_income=social_security,
            )
            after_tax = final_components.total_income_tax
            incremental_tax = after_tax - current_tax
            penalty = gross * penalty_rate
            net = gross - incremental_tax - penalty
            balance[trial_slice] -= gross
            if basis_ratio is not None:
                taxable_basis = self.taxable_basis
                if taxable_basis is None:  # pragma: no cover - invariant
                    raise RuntimeError("taxable balance is missing its basis")
                taxable_basis[trial_slice] -= np.minimum(
                    taxable_basis[trial_slice],
                    gross * basis_ratio,
                )
            ordinary = ordinary + gross * ordinary_fraction
            gains = gains + gross * gain_fraction
            current_tax = after_tax
            remaining = np.maximum(0.0, remaining - net)
            self.cumulative_tax_real[trial_slice] += (incremental_tax + penalty) / inflation_factor
            annual_penalty += penalty
        final_components = calculate_income_tax_component_arrays(
            progressive,
            tax_year=tax_year,
            ordinary_income=ordinary,
            long_term_capital_gains=gains,
            social_security_income=social_security,
        )
        current_tax = final_components.total_income_tax
        total_income = ordinary + gains + social_security
        self.annual_tax_nominal[trial_slice] = current_tax + annual_penalty
        self.annual_effective_rate[trial_slice] = np.divide(
            current_tax + annual_penalty,
            total_income,
            out=np.zeros_like(current_tax),
            where=total_income > 0,
        )
        ordinary_plus_one = calculate_income_tax_arrays(
            progressive,
            tax_year=tax_year,
            ordinary_income=ordinary + 1,
            long_term_capital_gains=gains,
            social_security_income=social_security,
        )
        gains_plus_one = calculate_income_tax_arrays(
            progressive,
            tax_year=tax_year,
            ordinary_income=ordinary,
            long_term_capital_gains=gains + 1,
            social_security_income=social_security,
        )
        self.annual_marginal_ordinary_rate[trial_slice] = ordinary_plus_one - current_tax
        self.annual_marginal_ltcg_rate[trial_slice] = gains_plus_one - current_tax
        self.annual_federal_tax_nominal[trial_slice] = final_components.federal_income_tax
        self.annual_state_tax_nominal[trial_slice] = final_components.indiana_income_tax
        self.annual_taxable_social_security_nominal[trial_slice] = (
            final_components.taxable_social_security
        )
        self.annual_federal_deduction_nominal[trial_slice] = final_components.federal_deduction
        self.annual_realized_long_term_capital_gains_nominal[trial_slice] = gains
        self.annual_early_distribution_penalty_nominal[trial_slice] = annual_penalty
        return cast("np.ndarray", remaining)

    def _deposit(
        self,
        trial_slice: slice,
        treatment: TaxTreatment,
        amount: float | np.ndarray,
    ) -> None:
        balance = self.balances[treatment]
        balance[trial_slice] += amount
        if treatment is TaxTreatment.TAXABLE:
            if self.taxable_basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable balance is missing its basis")
            self.taxable_basis[trial_slice] += amount

    def _required_minimum_distribution(
        self,
        trial_slice: slice,
        *,
        age: int,
        tax_year: int,
        opening_balance: np.ndarray,
        assumptions: TaxAssumptions,
    ) -> np.ndarray:
        import numpy as np

        tax_deferred = self.balances.get(TaxTreatment.TAX_DEFERRED)
        progressive = assumptions.progressive
        rmd_age = tax_year - progressive.taxpayer_birth_year if progressive is not None else age
        progressive_start_age = (
            rmd_start_age(progressive.taxpayer_birth_year)
            if progressive is not None
            else assumptions.rmd_start_age
        )
        if (
            tax_deferred is None
            or not assumptions.apply_required_minimum_distributions
            or rmd_age < progressive_start_age
        ):
            return np.zeros_like(opening_balance)
        divisor = (
            rmd_divisor(rmd_age)
            if progressive is not None
            else _RMD_DIVISORS.get(min(rmd_age, 120), 2.0)
        )
        required = np.minimum(
            tax_deferred[trial_slice],
            opening_balance / divisor,
        )
        tax_deferred[trial_slice] -= required
        return cast("np.ndarray", required)

    def _withdrawal_tax_rate(
        self,
        treatment: TaxTreatment,
        *,
        age: int,
        assumptions: TaxAssumptions,
    ) -> float:
        if treatment is TaxTreatment.TAX_DEFERRED:
            penalty = assumptions.early_distribution_penalty_rate if age < 60 else 0
            return assumptions.ordinary_income_tax_rate + penalty
        if treatment is TaxTreatment.HSA:
            unqualified = 1 - assumptions.qualified_hsa_withdrawal_fraction
            penalty = assumptions.hsa_early_distribution_penalty_rate if age < 65 else 0
            return unqualified * (assumptions.ordinary_income_tax_rate + penalty)
        return 0.0

    def _withdraw_for_net_need(
        self,
        trial_slice: slice,
        *,
        treatment: TaxTreatment,
        remaining: np.ndarray,
        tax_rate: float,
        capital_gains_rate: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        balance = self.balances.get(treatment)
        if balance is None:
            zeros = np.zeros_like(remaining)
            return zeros, zeros
        available = balance[trial_slice]
        effective_rate: float | np.ndarray = tax_rate
        basis_ratio: np.ndarray | None = None
        if treatment is TaxTreatment.TAXABLE:
            if self.taxable_basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable balance is missing its basis")
            basis = self.taxable_basis[trial_slice]
            basis_ratio = np.divide(
                basis,
                available,
                out=np.zeros_like(available),
                where=available > 0,
            )
            gain_ratio = np.maximum(0.0, 1 - basis_ratio)
            effective_rate = gain_ratio * capital_gains_rate
        gross_needed = np.divide(
            remaining,
            1 - effective_rate,
            out=np.zeros_like(remaining),
            where=(1 - effective_rate) > 0,
        )
        gross = np.minimum(available, gross_needed)
        tax = gross * effective_rate
        balance[trial_slice] -= gross
        if treatment is TaxTreatment.TAXABLE and basis_ratio is not None:
            taxable_basis = self.taxable_basis
            if taxable_basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable balance is missing its basis")
            basis_reduction = np.minimum(
                taxable_basis[trial_slice],
                gross * basis_ratio,
            )
            taxable_basis[trial_slice] -= basis_reduction
        return gross, tax


def estimate_tax_state_bytes(scenario: WealthScenario) -> int:
    """Conservative resident-memory estimate for tax-aware trial state."""
    if not scenario.tax_buckets:
        return 0
    # Portfolio totals/rates plus the component arrays retained for annual audits.
    array_count = len(scenario.tax_buckets) + 11
    if any(bucket.tax_treatment is TaxTreatment.TAXABLE for bucket in scenario.tax_buckets):
        array_count += 1
    if (
        scenario.tax_assumptions is not None
        and scenario.tax_assumptions.tax_model is TaxModel.PROGRESSIVE_US_INDIANA
    ):
        # Bisection and bracket calculations reuse these bounded batch arrays.
        array_count += 14
    return scenario.trials * array_count * 8
