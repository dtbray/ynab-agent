"""Bounded account-aware tax mechanics for retirement simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ynab_agent.planning.housing import HomeEquityCareFundingResult
from ynab_agent.planning.models import (
    FederalFilingStatus,
    TaxAssumptions,
    TaxModel,
    TaxTreatment,
    WealthScenario,
    WithdrawalPolicy,
)
from ynab_agent.planning.progressive_tax import (
    IncomeTaxArrayResult,
    calculate_income_tax_arrays,
    rmd_divisor,
    rmd_start_age,
)
from ynab_agent.planning.tax_engine import (
    HouseholdIncomeTaxInput,
    HouseholdTaxEngine,
    calculate_household_progressive_tax_components,
    household_tax_engine_for,
)
from ynab_agent.planning.tax_strategies import decide_tax_strategy

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


@dataclass(frozen=True)
class TaxBucketKey:
    """Stable owner-aware identity for one tax-character balance."""

    tax_treatment: TaxTreatment
    owner_person_id: str | None
    account_id: str | None = None


@dataclass
class TaxAwarePortfolio:
    """Per-trial balances and basis for a bounded set of tax treatments."""

    balances: dict[TaxBucketKey, np.ndarray]
    taxable_basis: dict[TaxBucketKey, np.ndarray]
    annual_rmd_nominal: dict[TaxBucketKey, np.ndarray]
    annual_withdrawal_nominal: dict[TaxBucketKey, np.ndarray]
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
    annual_modified_adjusted_gross_income_nominal: np.ndarray
    annual_irmaa_surcharge_nominal: np.ndarray
    annual_roth_conversion_nominal: np.ndarray
    annual_harvested_long_term_capital_gains_nominal: np.ndarray
    cumulative_irmaa_surcharge_real: np.ndarray
    irmaa_exposed: np.ndarray

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
            TaxBucketKey(
                bucket.tax_treatment,
                bucket.owner_person_id,
                bucket.account_id,
            ): np.full(
                scenario.trials,
                bucket.starting_balance,
                dtype=float,
            )
            for bucket in scenario.tax_buckets
        }
        taxable_basis: dict[TaxBucketKey, np.ndarray] = {}
        for bucket in scenario.tax_buckets:
            if bucket.tax_treatment is not TaxTreatment.TAXABLE:
                continue
            if bucket.taxable_basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable bucket is missing its basis")
            key = TaxBucketKey(
                bucket.tax_treatment,
                bucket.owner_person_id,
                bucket.account_id,
            )
            taxable_basis[key] = np.full(
                scenario.trials,
                bucket.taxable_basis,
                dtype=float,
            )
        return cls(
            balances=balances,
            taxable_basis=taxable_basis,
            annual_rmd_nominal={
                key: np.zeros(scenario.trials, dtype=float)
                for key in balances
            },
            annual_withdrawal_nominal={
                key: np.zeros(scenario.trials, dtype=float)
                for key in balances
            },
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
            annual_modified_adjusted_gross_income_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            annual_irmaa_surcharge_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            annual_roth_conversion_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            annual_harvested_long_term_capital_gains_nominal=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            cumulative_irmaa_surcharge_real=np.zeros(
                scenario.trials,
                dtype=float,
            ),
            irmaa_exposed=np.zeros(scenario.trials, dtype=bool),
        )

    def reset_annual_audit(self) -> None:
        """Clear tax and strategy audit arrays before evaluating one year."""
        for values in (
            self.annual_tax_nominal,
            self.annual_effective_rate,
            self.annual_marginal_ordinary_rate,
            self.annual_marginal_ltcg_rate,
            self.annual_federal_tax_nominal,
            self.annual_state_tax_nominal,
            self.annual_taxable_social_security_nominal,
            self.annual_federal_deduction_nominal,
            self.annual_realized_long_term_capital_gains_nominal,
            self.annual_early_distribution_penalty_nominal,
            self.annual_modified_adjusted_gross_income_nominal,
            self.annual_irmaa_surcharge_nominal,
            self.annual_roth_conversion_nominal,
            self.annual_harvested_long_term_capital_gains_nominal,
        ):
            values.fill(0)
        for values in self.annual_rmd_nominal.values():
            values.fill(0)
        for values in self.annual_withdrawal_nominal.values():
            values.fill(0)

    def total(self, trial_slice: slice) -> np.ndarray:
        import numpy as np

        return np.sum(
            [balance[trial_slice] for balance in self.balances.values()],
            axis=0,
        )

    def account_balances(
        self,
        trial_slice: slice,
    ) -> dict[str, np.ndarray]:
        """Return linked balances by stable scenario account identity."""
        import numpy as np

        account_ids = {
            key.account_id
            for key in self.balances
            if key.account_id is not None
        }
        return {
            account_id: np.sum(
                [
                    balance[trial_slice]
                    for key, balance in self.balances.items()
                    if key.account_id == account_id
                ],
                axis=0,
            )
            for account_id in account_ids
        }

    def _keys_for_treatment(
        self,
        treatment: TaxTreatment,
    ) -> tuple[TaxBucketKey, ...]:
        return tuple(
            key
            for key in self.balances
            if key.tax_treatment is treatment
        )

    def opening_tax_deferred(
        self,
        trial_slice: slice,
    ) -> dict[TaxBucketKey, np.ndarray]:
        return {
            key: self.balances[key][trial_slice].copy()
            for key in self._keys_for_treatment(TaxTreatment.TAX_DEFERRED)
        }

    def reset_annual_ownership_audit(self, trial_slice: slice) -> None:
        """Clear owner-level annual flows for one bounded trial batch."""
        for values in self.annual_rmd_nominal.values():
            values[trial_slice] = 0.0
        for values in self.annual_withdrawal_nominal.values():
            values[trial_slice] = 0.0

    def after_tax_estate_value(
        self,
        trial_slice: slice,
        *,
        assumptions: TaxAssumptions,
        tax_year: int,
        joint_filing: np.ndarray | None = None,
        external_disposition_value: float | np.ndarray = 0.0,
        external_long_term_capital_gains: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Return one combined liquidation value after account-character taxes."""
        import numpy as np

        if assumptions.progressive is not None:
            total = self.total(trial_slice)
            ordinary = np.zeros_like(total)
            for key in self._keys_for_treatment(TaxTreatment.TAX_DEFERRED):
                ordinary += self.balances[key][trial_slice]
            for key in self._keys_for_treatment(TaxTreatment.HSA):
                ordinary += self.balances[key][trial_slice] * (
                    1 - assumptions.qualified_hsa_withdrawal_fraction
                )
            aggregate_gains = np.zeros_like(total)
            for key in self._keys_for_treatment(TaxTreatment.TAXABLE):
                taxable = self.balances[key]
                basis = self.taxable_basis.get(key)
                if basis is None:  # pragma: no cover - invariant
                    raise RuntimeError("taxable balance is missing its basis")
                aggregate_gains += (
                    taxable[trial_slice] - basis[trial_slice]
                )
            gains = np.maximum(0.0, aggregate_gains) + np.broadcast_to(
                np.asarray(
                    external_long_term_capital_gains,
                    dtype=float,
                ),
                total.shape,
            )
            liquidation_tax = (
                calculate_income_tax_arrays(
                    assumptions.progressive,
                    tax_year=tax_year,
                    ordinary_income=ordinary,
                    long_term_capital_gains=gains,
                    social_security_income=np.zeros_like(total),
                )
                if joint_filing is None
                else calculate_household_progressive_tax_components(
                    assumptions,
                    HouseholdIncomeTaxInput(
                        ordinary_income=ordinary,
                        social_security_income=np.zeros_like(total),
                        joint_filing=joint_filing,
                        tax_year=tax_year,
                        long_term_capital_gains=gains,
                    ),
                ).total_income_tax
            )
            return cast(
                "np.ndarray",
                total + external_disposition_value - liquidation_tax,
            )

        estate = np.zeros_like(self.cumulative_tax_real[trial_slice])
        estate += (
            external_disposition_value
            - np.asarray(external_long_term_capital_gains, dtype=float)
            * assumptions.long_term_capital_gains_tax_rate
        )
        for key, all_balances in self.balances.items():
            treatment = key.tax_treatment
            balance = all_balances[trial_slice]
            if treatment is TaxTreatment.TAX_DEFERRED:
                estate += balance * (1 - assumptions.ordinary_income_tax_rate)
                continue
            if treatment is TaxTreatment.HSA:
                unqualified = 1 - assumptions.qualified_hsa_withdrawal_fraction
                estate += balance * (1 - unqualified * assumptions.ordinary_income_tax_rate)
                continue
            if treatment is TaxTreatment.TAXABLE:
                basis = self.taxable_basis.get(key)
                if basis is None:  # pragma: no cover - model invariant
                    raise RuntimeError("taxable balance is missing its basis")
                gains = np.maximum(
                    0.0,
                    balance - basis[trial_slice],
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
        account_gross_returns: dict[str, np.ndarray] | None = None,
        inflation_factor: float | np.ndarray,
        assumptions: TaxAssumptions,
    ) -> None:
        for key, balance in self.balances.items():
            selected_return = gross_return
            if (
                key.account_id is not None
                and account_gross_returns is not None
            ):
                if key.account_id not in account_gross_returns:
                    raise ValueError(
                        "linked tax bucket is missing its account return"
                    )
                selected_return = account_gross_returns[key.account_id]
            balance[trial_slice] *= selected_return
        taxable_keys = self._keys_for_treatment(TaxTreatment.TAXABLE)
        if not taxable_keys or assumptions.taxable_account_annual_tax_drag_rate == 0:
            return
        for key in taxable_keys:
            taxable = self.balances[key]
            tax_drag = (
                taxable[trial_slice]
                * assumptions.taxable_account_annual_tax_drag_rate
            )
            taxable[trial_slice] -= tax_drag
            self.cumulative_tax_real[trial_slice] += (
                tax_drag / inflation_factor
            )

    def add_contribution(
        self,
        trial_slice: slice,
        amount: float | np.ndarray,
        *,
        scenario: WealthScenario,
        destination: TaxTreatment | None,
    ) -> None:
        if destination is not None:
            self._deposit_treatment(trial_slice, destination, amount)
            return
        for bucket in scenario.tax_buckets:
            if bucket.contribution_fraction == 0:
                continue
            self._deposit_key(
                trial_slice,
                TaxBucketKey(
                    bucket.tax_treatment,
                    bucket.owner_person_id,
                    bucket.account_id,
                ),
                amount * bucket.contribution_fraction,
            )

    def _key_for_account(self, account_id: str) -> TaxBucketKey:
        keys = tuple(
            key
            for key in self.balances
            if key.account_id == account_id
        )
        if len(keys) != 1:
            raise ValueError(
                "account_id must identify exactly one linked tax bucket"
            )
        return keys[0]

    def deposit_external_cash_to_account(
        self,
        trial_slice: slice,
        *,
        account_id: str,
        amount: float | np.ndarray,
    ) -> TaxBucketKey:
        """Deposit external proceeds into one exact cash or taxable account."""
        key = self._key_for_account(account_id)
        if key.tax_treatment not in {
            TaxTreatment.CASH,
            TaxTreatment.TAXABLE,
        }:
            raise ValueError(
                "external cash destination account must be cash or taxable"
            )
        self._deposit_key(trial_slice, key, amount)
        return key

    def account_balance_and_basis(
        self,
        trial_slice: slice,
        *,
        account_id: str,
    ) -> tuple[TaxBucketKey, np.ndarray, np.ndarray | None]:
        """Return copied exact-account state for deterministic audit output."""
        key = self._key_for_account(account_id)
        basis = self.taxable_basis.get(key)
        return (
            key,
            self.balances[key][trial_slice].copy(),
            basis[trial_slice].copy() if basis is not None else None,
        )

    def fund_home_equity_care(
        self,
        trial_slice: slice,
        *,
        account_id: str,
        requested: float | np.ndarray,
    ) -> HomeEquityCareFundingResult:
        """Debit care from one exact realized-equity account, fail closed."""
        import numpy as np

        key = self._key_for_account(account_id)
        if key.tax_treatment not in {
            TaxTreatment.CASH,
            TaxTreatment.TAXABLE,
        }:
            raise ValueError(
                "home-equity care funding account must be cash or taxable"
            )
        balance = self.balances[key][trial_slice]
        request = np.broadcast_to(
            np.asarray(requested, dtype=float),
            balance.shape,
        )
        funded = np.minimum(balance, request)
        realized_gains = np.zeros_like(funded)
        if key.tax_treatment is TaxTreatment.TAXABLE:
            basis = self.taxable_basis.get(key)
            if basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable balance is missing its basis")
            basis_slice = basis[trial_slice]
            basis_ratio = np.divide(
                basis_slice,
                balance,
                out=np.zeros_like(balance),
                where=balance > 0,
            )
            basis_reduction = np.minimum(
                basis_slice,
                funded * basis_ratio,
            )
            realized_gains = np.maximum(0.0, funded - basis_reduction)
            basis_slice -= basis_reduction
        balance -= funded
        return HomeEquityCareFundingResult(
            funded=funded,
            unmet=np.maximum(0.0, request - funded),
            realized_long_term_capital_gains=realized_gains,
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
        opening_tax_deferred: dict[TaxBucketKey, np.ndarray],
        inflation_factor: float | np.ndarray,
        assumptions: TaxAssumptions,
        external_long_term_capital_gains: float | np.ndarray = 0.0,
        joint_filing: np.ndarray | None = None,
        tax_engine: HouseholdTaxEngine | None = None,
        owner_birth_years: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Fund one retirement year and return unmet spending by trial."""
        import numpy as np

        ordinary = np.asarray(ordinary_income, dtype=float)
        social_security = np.asarray(social_security_income, dtype=float)
        tax_free = np.asarray(tax_free_income, dtype=float)
        owner_birth_year_map = owner_birth_years or {}
        self.reset_annual_ownership_audit(trial_slice)
        required_minimum = self._required_minimum_distribution(
            trial_slice,
            age=age,
            tax_year=tax_year,
            opening_balance=opening_tax_deferred,
            assumptions=assumptions,
            owner_birth_years=owner_birth_year_map,
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
                external_long_term_capital_gains=(
                    external_long_term_capital_gains
                ),
                inflation_factor=inflation_factor,
                assumptions=assumptions,
                joint_filing=joint_filing,
                tax_engine=tax_engine,
                owner_birth_years=owner_birth_year_map,
            )
        selected_tax_engine = household_tax_engine_for(assumptions, tax_engine)
        shape = self.cumulative_tax_real[trial_slice].shape
        ordinary = np.broadcast_to(ordinary, shape).astype(float, copy=True)
        social_security = np.broadcast_to(
            social_security, shape
        ).astype(float, copy=True)
        income_tax = selected_tax_engine.income_tax(
            HouseholdIncomeTaxInput(
                ordinary_income=ordinary,
                social_security_income=social_security,
                joint_filing=(
                    np.zeros_like(ordinary, dtype=bool)
                    if joint_filing is None
                    else joint_filing
                ),
                tax_year=tax_year,
                long_term_capital_gains=np.broadcast_to(
                    np.asarray(
                        external_long_term_capital_gains,
                        dtype=float,
                    ),
                    ordinary.shape,
                ),
            ),
            assumptions=assumptions,
        )
        self.cumulative_tax_real[trial_slice] += income_tax / inflation_factor
        available_cash = ordinary + social_security + tax_free - income_tax
        remaining = np.maximum(0.0, spending - available_cash)
        surplus = np.maximum(0.0, available_cash - spending)
        if np.any(surplus):
            self._deposit_treatment(
                trial_slice,
                assumptions.retirement_surplus_destination,
                surplus,
            )

        requests: list[tuple[TaxTreatment, np.ndarray | None]]
        strategy = assumptions.strategy
        if (
            strategy is not None
            and strategy.withdrawal_policy is WithdrawalPolicy.PROPORTIONAL
        ):
            initial_need = remaining.copy()
            requests = [
                (item.tax_treatment, initial_need * item.fraction)
                for item in strategy.proportional_withdrawal_fractions
            ]
            requests.extend(
                (treatment, None)
                for treatment in assumptions.withdrawal_order
            )
        else:
            requests = [
                (treatment, None)
                for treatment in assumptions.withdrawal_order
            ]
        for treatment, target in requests:
            if not np.any(remaining > 0.005):
                break
            requested = (
                remaining.copy()
                if target is None
                else np.minimum(remaining, target)
            )
            if not np.any(requested > 0.005):
                continue
            for key in self._keys_for_treatment(treatment):
                if not np.any(requested > 0.005):
                    break
                owner_age = (
                    tax_year - owner_birth_year_map[key.owner_person_id]
                    if key.owner_person_id is not None
                    and key.owner_person_id in owner_birth_year_map
                    else age
                )
                tax_rate = self._withdrawal_tax_rate(
                    treatment,
                    age=owner_age,
                    assumptions=assumptions,
                )
                gross, tax = self._withdraw_for_net_need(
                    trial_slice,
                    key=key,
                    remaining=requested,
                    tax_rate=tax_rate,
                    capital_gains_rate=(
                        assumptions.long_term_capital_gains_tax_rate
                    ),
                )
                remaining = np.maximum(0.0, remaining - (gross - tax))
                requested = np.maximum(0.0, requested - (gross - tax))
                self.cumulative_tax_real[trial_slice] += (
                    tax / inflation_factor
                )
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
        external_long_term_capital_gains: float | np.ndarray,
        inflation_factor: float | np.ndarray,
        assumptions: TaxAssumptions,
        joint_filing: np.ndarray | None,
        tax_engine: HouseholdTaxEngine | None,
        owner_birth_years: dict[str, int],
    ) -> np.ndarray:
        """Jointly choose bounded tax actions and fund the current year."""
        import numpy as np

        progressive = assumptions.progressive
        if progressive is None:  # pragma: no cover - model invariant
            raise RuntimeError("progressive tax assumptions are missing")
        shape = self.cumulative_tax_real[trial_slice].shape
        zeros = np.zeros(shape, dtype=float)
        base_gains = np.broadcast_to(
            np.asarray(
                external_long_term_capital_gains,
                dtype=float,
            ),
            shape,
        ).astype(float, copy=True)
        ordinary = np.broadcast_to(
            ordinary_income,
            shape,
        ).astype(float, copy=True)
        strategy = assumptions.strategy
        if strategy is None:
            return self._execute_progressive_spending(
                trial_slice,
                age=age,
                tax_year=tax_year,
                spending=spending,
                ordinary_income=ordinary,
                social_security_income=social_security_income,
                tax_free_income=tax_free_income,
                inflation_factor=inflation_factor,
                assumptions=assumptions,
                joint_filing=joint_filing,
                tax_engine=tax_engine,
                owner_birth_years=owner_birth_years,
                initial_gains=base_gains,
                noncash_ordinary_income=zeros,
            )

        balance_snapshots = {
            key: values[trial_slice].copy()
            for key, values in self.balances.items()
        }
        basis_snapshots = {
            key: values[trial_slice].copy()
            for key, values in self.taxable_basis.items()
        }
        cumulative_tax_snapshot = (
            self.cumulative_tax_real[trial_slice].copy()
        )
        audit_names = (
            "annual_tax_nominal",
            "annual_effective_rate",
            "annual_marginal_ordinary_rate",
            "annual_marginal_ltcg_rate",
            "annual_federal_tax_nominal",
            "annual_state_tax_nominal",
            "annual_taxable_social_security_nominal",
            "annual_federal_deduction_nominal",
            "annual_realized_long_term_capital_gains_nominal",
            "annual_early_distribution_penalty_nominal",
            "annual_modified_adjusted_gross_income_nominal",
        )
        audit_snapshots = {
            name: getattr(self, name)[trial_slice].copy()
            for name in audit_names
        }
        withdrawal_snapshots = {
            key: values[trial_slice].copy()
            for key, values in self.annual_withdrawal_nominal.items()
        }

        def restore_state() -> None:
            for key, values in balance_snapshots.items():
                self.balances[key][trial_slice] = values
            for key, values in basis_snapshots.items():
                self.taxable_basis[key][trial_slice] = values
            self.cumulative_tax_real[trial_slice] = (
                cumulative_tax_snapshot
            )
            for name, values in audit_snapshots.items():
                getattr(self, name)[trial_slice] = values
            for key, values in withdrawal_snapshots.items():
                self.annual_withdrawal_nominal[key][trial_slice] = values

        def aggregate(treatment: TaxTreatment) -> np.ndarray:
            keys = self._keys_for_treatment(treatment)
            if not keys:
                return np.zeros(shape, dtype=float)
            return np.sum(
                [self.balances[key][trial_slice] for key in keys],
                axis=0,
            )

        def aggregate_taxable_basis() -> np.ndarray:
            keys = self._keys_for_treatment(TaxTreatment.TAXABLE)
            if not keys:
                return np.zeros(shape, dtype=float)
            return np.sum(
                [self.taxable_basis[key][trial_slice] for key in keys],
                axis=0,
            )

        roth_keys = self._keys_for_treatment(TaxTreatment.ROTH)

        def conversion_destination(
            source: TaxBucketKey,
        ) -> TaxBucketKey | None:
            return next(
                (
                    key
                    for key in roth_keys
                    if key.owner_person_id == source.owner_person_id
                    and (
                        key.account_id == source.account_id
                        or source.account_id is None
                    )
                ),
                next(
                    (
                        key
                        for key in roth_keys
                        if key.owner_person_id == source.owner_person_id
                    ),
                    None,
                ),
            )

        def convertible_tax_deferred_balance() -> np.ndarray:
            keys = [
                key
                for key in self._keys_for_treatment(
                    TaxTreatment.TAX_DEFERRED
                )
                if conversion_destination(key) is not None
            ]
            if not keys:
                return np.zeros(shape, dtype=float)
            return np.sum(
                [self.balances[key][trial_slice] for key in keys],
                axis=0,
            )

        def apply_actions(
            conversion: np.ndarray,
            harvest: np.ndarray,
        ) -> None:
            conversion_remaining = conversion.copy()
            deferred_keys = self._keys_for_treatment(
                TaxTreatment.TAX_DEFERRED
            )
            for source in deferred_keys:
                destination = conversion_destination(source)
                if destination is None:
                    continue
                source_balance = self.balances[source][trial_slice]
                moved = np.minimum(source_balance, conversion_remaining)
                source_balance -= moved
                self.balances[destination][trial_slice] += moved
                conversion_remaining -= moved

            harvest_remaining = harvest.copy()
            for key in self._keys_for_treatment(TaxTreatment.TAXABLE):
                balance = self.balances[key][trial_slice]
                basis = self.taxable_basis[key][trial_slice]
                harvested = np.minimum(
                    np.maximum(0.0, balance - basis),
                    harvest_remaining,
                )
                basis += harvested
                harvest_remaining -= harvested

        def project_final_taxable_position(
            conversion: np.ndarray,
            harvest: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            restore_state()
            apply_actions(conversion, harvest)
            self._execute_progressive_spending(
                trial_slice,
                age=age,
                tax_year=tax_year,
                spending=spending,
                ordinary_income=ordinary + conversion,
                social_security_income=social_security_income,
                tax_free_income=tax_free_income,
                inflation_factor=inflation_factor,
                assumptions=assumptions,
                joint_filing=joint_filing,
                tax_engine=tax_engine,
                owner_birth_years=owner_birth_years,
                initial_gains=base_gains + harvest,
                noncash_ordinary_income=conversion,
            )
            adjusted_gross_income = (
                self.annual_modified_adjusted_gross_income_nominal[
                    trial_slice
                ]
            )
            deduction = self.annual_federal_deduction_nominal[
                trial_slice
            ]
            realized_gains = (
                self.annual_realized_long_term_capital_gains_nominal[
                    trial_slice
                ]
            )
            taxable_income = np.maximum(
                0.0,
                adjusted_gross_income - deduction,
            )
            taxable_ordinary = np.maximum(
                0.0,
                taxable_income - realized_gains,
            )
            restore_state()
            return taxable_ordinary.copy(), taxable_income.copy()

        conversion_cap: float | np.ndarray = 0.0
        if strategy.roth_conversion is not None:
            conversion_cap = (
                strategy.roth_conversion.max_annual_conversion_real
                * inflation_factor
            )
        harvest_cap: float | np.ndarray = 0.0
        if strategy.capital_gain_harvest is not None:
            harvest_cap = (
                strategy.capital_gain_harvest.max_annual_gain_real
                * inflation_factor
            )
        decision = decide_tax_strategy(
            strategy,
            progressive,
            age=age,
            tax_year=tax_year,
            tax_deferred_balance=convertible_tax_deferred_balance(),
            taxable_balance=aggregate(TaxTreatment.TAXABLE),
            taxable_basis=aggregate_taxable_basis(),
            nominal_conversion_cap=conversion_cap,
            nominal_harvest_cap=harvest_cap,
            project_final_taxable_position=(
                project_final_taxable_position
            ),
        )
        restore_state()
        apply_actions(
            decision.roth_conversion,
            decision.harvested_long_term_capital_gains,
        )
        self.annual_roth_conversion_nominal[trial_slice] = (
            decision.roth_conversion
        )
        self.annual_harvested_long_term_capital_gains_nominal[
            trial_slice
        ] = decision.harvested_long_term_capital_gains
        return self._execute_progressive_spending(
            trial_slice,
            age=age,
            tax_year=tax_year,
            spending=spending,
            ordinary_income=ordinary + decision.roth_conversion,
            social_security_income=social_security_income,
            tax_free_income=tax_free_income,
            inflation_factor=inflation_factor,
            assumptions=assumptions,
            joint_filing=joint_filing,
            tax_engine=tax_engine,
            owner_birth_years=owner_birth_years,
            initial_gains=(
                base_gains
                + decision.harvested_long_term_capital_gains
            ),
            noncash_ordinary_income=decision.roth_conversion,
        )

    def _execute_progressive_spending(
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
        joint_filing: np.ndarray | None,
        tax_engine: HouseholdTaxEngine | None,
        owner_birth_years: dict[str, int],
        initial_gains: np.ndarray,
        noncash_ordinary_income: np.ndarray,
    ) -> np.ndarray:
        """Execute one progressive-tax year against the current owner state."""
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
        gains = initial_gains.copy()
        filing = (
            np.full(
                shape,
                progressive.filing_status
                is FederalFilingStatus.MARRIED_FILING_JOINTLY,
                dtype=bool,
            )
            if joint_filing is None
            else np.broadcast_to(joint_filing, shape)
        )
        selected_tax_engine = household_tax_engine_for(assumptions, tax_engine)

        def tax_input(
            ordinary_value: np.ndarray,
            gains_value: np.ndarray,
        ) -> HouseholdIncomeTaxInput:
            return HouseholdIncomeTaxInput(
                ordinary_income=ordinary_value,
                social_security_income=social_security,
                joint_filing=filing,
                tax_year=tax_year,
                long_term_capital_gains=gains_value,
            )

        def tax_components(
            ordinary_value: np.ndarray,
            gains_value: np.ndarray,
        ) -> IncomeTaxArrayResult:
            return calculate_household_progressive_tax_components(
                assumptions,
                tax_input(ordinary_value, gains_value),
            )

        def total_tax(
            ordinary_value: np.ndarray,
            gains_value: np.ndarray,
        ) -> np.ndarray:
            return selected_tax_engine.income_tax(
                tax_input(ordinary_value, gains_value),
                assumptions=assumptions,
            )

        starting_components = tax_components(ordinary, gains)
        income_tax = starting_components.total_income_tax
        self.cumulative_tax_real[trial_slice] += income_tax / inflation_factor
        available_cash = (
            ordinary
            - noncash_ordinary_income
            + social_security
            + tax_free
            - income_tax
        )
        remaining = np.maximum(0.0, spending - available_cash)
        surplus = np.maximum(0.0, available_cash - spending)
        if np.any(surplus):
            self._deposit_treatment(
                trial_slice,
                assumptions.retirement_surplus_destination,
                surplus,
            )

        current_tax = income_tax
        annual_penalty = np.zeros(shape, dtype=float)
        requests: list[tuple[TaxTreatment, np.ndarray | None]]
        strategy = assumptions.strategy
        if (
            strategy is not None
            and strategy.withdrawal_policy is WithdrawalPolicy.PROPORTIONAL
        ):
            initial_need = remaining.copy()
            requests = [
                (item.tax_treatment, initial_need * item.fraction)
                for item in strategy.proportional_withdrawal_fractions
            ]
            requests.extend(
                (treatment, None)
                for treatment in assumptions.withdrawal_order
            )
        else:
            requests = [
                (treatment, None)
                for treatment in assumptions.withdrawal_order
            ]
        for treatment, target in requests:
            if not np.any(remaining > 0.005):
                break
            requested = (
                remaining.copy()
                if target is None
                else np.minimum(remaining, target)
            )
            if not np.any(requested > 0.005):
                continue
            for key in self._keys_for_treatment(treatment):
                if not np.any(requested > 0.005):
                    break
                balance = self.balances[key]
                available = balance[trial_slice]
                owner_age = (
                    tax_year - owner_birth_years[key.owner_person_id]
                    if key.owner_person_id is not None
                    and key.owner_person_id in owner_birth_years
                    else age
                )
                ordinary_fraction: float | np.ndarray = 0.0
                gain_fraction: float | np.ndarray = 0.0
                penalty_rate = 0.0
                basis_ratio: np.ndarray | None = None
                if treatment is TaxTreatment.TAX_DEFERRED:
                    ordinary_fraction = 1.0
                    if owner_age < 60:
                        penalty_rate = assumptions.early_distribution_penalty_rate
                elif treatment is TaxTreatment.HSA:
                    ordinary_fraction = (
                        1 - assumptions.qualified_hsa_withdrawal_fraction
                    )
                    if owner_age < 65:
                        penalty_rate = (
                            assumptions.hsa_early_distribution_penalty_rate
                            * ordinary_fraction
                        )
                elif treatment is TaxTreatment.TAXABLE:
                    taxable_basis = self.taxable_basis.get(key)
                    if taxable_basis is None:  # pragma: no cover - invariant
                        raise RuntimeError("taxable balance is missing its basis")
                    basis_ratio = np.divide(
                        taxable_basis[trial_slice],
                        available,
                        out=np.zeros_like(available),
                        where=available > 0,
                    )
                    gain_fraction = np.maximum(0.0, 1 - basis_ratio)

                low = np.zeros_like(remaining)
                high = np.minimum(
                    available,
                    np.maximum(requested, 0.01),
                )
                after_tax = current_tax
                net = np.zeros_like(remaining)
                for _ in range(PROGRESSIVE_WITHDRAWAL_MAX_BRACKET_STEPS):
                    after_tax = total_tax(
                        ordinary + high * ordinary_fraction,
                        gains + high * gain_fraction,
                    )
                    net = (
                        high
                        - (after_tax - current_tax)
                        - high * penalty_rate
                    )
                    needs_larger_bracket = (
                        (net < requested) & (high < available)
                    )
                    if not np.any(needs_larger_bracket):
                        break
                    low = np.where(needs_larger_bracket, high, low)
                    high = np.where(
                        needs_larger_bracket,
                        np.minimum(
                            available,
                            np.maximum(0.01, high * 2),
                        ),
                        high,
                    )
                unresolved = (net < requested) & (high < available)
                if np.any(unresolved):
                    high = np.where(unresolved, available, high)
                    after_tax = total_tax(
                        ordinary + high * ordinary_fraction,
                        gains + high * gain_fraction,
                    )
                    net = (
                        high
                        - (after_tax - current_tax)
                        - high * penalty_rate
                    )
                fundable = net >= requested
                low = np.where(fundable, low, available)
                high = np.where(fundable, high, available)
                for _ in range(PROGRESSIVE_WITHDRAWAL_MAX_BISECTION_STEPS):
                    active = fundable & ((high - low) > 0.005)
                    if not np.any(active):
                        break
                    gross = (low + high) / 2
                    after_tax = total_tax(
                        ordinary + gross * ordinary_fraction,
                        gains + gross * gain_fraction,
                    )
                    net = (
                        gross
                        - (after_tax - current_tax)
                        - gross * penalty_rate
                    )
                    insufficient = active & (net < requested)
                    sufficient = active & ~insufficient
                    low = np.where(insufficient, gross, low)
                    high = np.where(sufficient, gross, high)
                gross = np.where(fundable, high, available)
                final_components = tax_components(
                    ordinary + gross * ordinary_fraction,
                    gains + gross * gain_fraction,
                )
                after_tax = final_components.total_income_tax
                incremental_tax = after_tax - current_tax
                penalty = gross * penalty_rate
                net = gross - incremental_tax - penalty
                balance[trial_slice] -= gross
                self.annual_withdrawal_nominal[key][trial_slice] += gross
                if basis_ratio is not None:
                    taxable_basis = self.taxable_basis.get(key)
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
                requested = np.maximum(0.0, requested - net)
                self.cumulative_tax_real[trial_slice] += (
                    (incremental_tax + penalty) / inflation_factor
                )
                annual_penalty += penalty
        final_components = tax_components(ordinary, gains)
        current_tax = final_components.total_income_tax
        total_income = ordinary + gains + social_security
        self.annual_tax_nominal[trial_slice] = current_tax + annual_penalty
        self.annual_effective_rate[trial_slice] = np.divide(
            current_tax + annual_penalty,
            total_income,
            out=np.zeros_like(current_tax),
            where=total_income > 0,
        )
        ordinary_plus_one = total_tax(ordinary + 1, gains)
        gains_plus_one = total_tax(ordinary, gains + 1)
        self.annual_marginal_ordinary_rate[trial_slice] = ordinary_plus_one - current_tax
        self.annual_marginal_ltcg_rate[trial_slice] = gains_plus_one - current_tax
        self.annual_federal_tax_nominal[trial_slice] = final_components.federal_income_tax
        self.annual_state_tax_nominal[trial_slice] = final_components.indiana_income_tax
        self.annual_taxable_social_security_nominal[trial_slice] = (
            final_components.taxable_social_security
        )
        self.annual_federal_deduction_nominal[trial_slice] = final_components.federal_deduction
        self.annual_realized_long_term_capital_gains_nominal[trial_slice] = gains
        self.annual_modified_adjusted_gross_income_nominal[trial_slice] = (
            final_components.federal_adjusted_gross_income
        )
        self.annual_early_distribution_penalty_nominal[trial_slice] = annual_penalty
        return cast("np.ndarray", remaining)

    def _deposit_key(
        self,
        trial_slice: slice,
        key: TaxBucketKey,
        amount: float | np.ndarray,
    ) -> None:
        balance = self.balances[key]
        balance[trial_slice] += amount
        if key.tax_treatment is TaxTreatment.TAXABLE:
            taxable_basis = self.taxable_basis.get(key)
            if taxable_basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable balance is missing its basis")
            taxable_basis[trial_slice] += amount

    def _deposit_treatment(
        self,
        trial_slice: slice,
        treatment: TaxTreatment,
        amount: float | np.ndarray,
    ) -> None:
        import numpy as np

        keys = self._keys_for_treatment(treatment)
        if not keys:
            raise ValueError(
                f"deposit destination has no {treatment.value} tax bucket"
            )
        if len(keys) == 1:
            self._deposit_key(trial_slice, keys[0], amount)
            return
        balances = np.stack(
            [self.balances[key][trial_slice] for key in keys]
        )
        totals = np.sum(balances, axis=0)
        for index, key in enumerate(keys):
            fraction = np.divide(
                balances[index],
                totals,
                out=np.full_like(totals, 1 / len(keys)),
                where=totals > 0,
            )
            self._deposit_key(trial_slice, key, amount * fraction)

    def _required_minimum_distribution(
        self,
        trial_slice: slice,
        *,
        age: int,
        tax_year: int,
        opening_balance: dict[TaxBucketKey, np.ndarray],
        assumptions: TaxAssumptions,
        owner_birth_years: dict[str, int],
    ) -> np.ndarray:
        import numpy as np

        aggregate = np.zeros_like(self.cumulative_tax_real[trial_slice])
        if not assumptions.apply_required_minimum_distributions:
            return aggregate
        progressive = assumptions.progressive
        for key in self._keys_for_treatment(TaxTreatment.TAX_DEFERRED):
            birth_year = (
                owner_birth_years.get(key.owner_person_id)
                if key.owner_person_id is not None
                else None
            )
            if birth_year is None and progressive is not None:
                birth_year = progressive.taxpayer_birth_year
            owner_age = tax_year - birth_year if birth_year is not None else age
            start_age = (
                rmd_start_age(birth_year)
                if progressive is not None and birth_year is not None
                else assumptions.rmd_start_age
            )
            if owner_age < start_age:
                continue
            divisor = (
                rmd_divisor(owner_age)
                if progressive is not None
                else _RMD_DIVISORS.get(min(owner_age, 120), 2.0)
            )
            balance = self.balances[key]
            required = np.minimum(
                balance[trial_slice],
                opening_balance[key] / divisor,
            )
            balance[trial_slice] -= required
            self.annual_rmd_nominal[key][trial_slice] += required
            aggregate += required
        return aggregate

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
        key: TaxBucketKey,
        remaining: np.ndarray,
        tax_rate: float,
        capital_gains_rate: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        treatment = key.tax_treatment
        balance = self.balances[key]
        available = balance[trial_slice]
        effective_rate: float | np.ndarray = tax_rate
        basis_ratio: np.ndarray | None = None
        if treatment is TaxTreatment.TAXABLE:
            basis = self.taxable_basis.get(key)
            if basis is None:  # pragma: no cover - model invariant
                raise RuntimeError("taxable balance is missing its basis")
            basis_ratio = np.divide(
                basis[trial_slice],
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
        self.annual_withdrawal_nominal[key][trial_slice] += gross
        if treatment is TaxTreatment.TAXABLE and basis_ratio is not None:
            taxable_basis = self.taxable_basis.get(key)
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
    bucket_count = len(scenario.tax_buckets)
    taxable_count = sum(
        bucket.tax_treatment is TaxTreatment.TAXABLE
        for bucket in scenario.tax_buckets
    )
    # Every bucket retains a balance plus annual owner-level RMD and withdrawal
    # audit arrays. Taxable buckets retain their own basis; fixed tax, IRMAA,
    # and strategy audit state contributes another seventeen arrays.
    array_count = 3 * bucket_count + taxable_count + 17
    if (
        scenario.tax_assumptions is not None
        and scenario.tax_assumptions.tax_model is TaxModel.PROGRESSIVE_US_INDIANA
    ):
        # Bisections, external-gain state, starting/projected owner balances,
        # withdrawal ledgers, and starting/projected taxable bases.
        array_count += 15 + 3 * bucket_count + 2 * taxable_count
    if (
        scenario.tax_assumptions is not None
        and scenario.tax_assumptions.strategy is not None
    ):
        # Joint conversion/harvest low, high, and candidate projection state.
        array_count += 14
    return scenario.trials * array_count * 8
