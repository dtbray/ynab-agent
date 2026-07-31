"""Bounded multi-asset assumptions, allocation strategies, and runtime state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import numpy as np


ASSET_CLASS_COUNT = 4
ASSET_CLASS_NAMES = (
    "us_equity",
    "international_equity",
    "bonds",
    "cash",
)
MAX_ALLOCATION_ACCOUNTS = 16
MAX_GLIDE_PATH_POINTS = 16
_WEIGHT_TOLERANCE = 1e-9
_PSD_TOLERANCE = 1e-10


class AssetWeights(BaseModel):
    """Explicit weights for the four supported liquid asset classes."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    us_equity: float = Field(default=0, ge=0, le=1)
    international_equity: float = Field(default=0, ge=0, le=1)
    bonds: float = Field(default=0, ge=0, le=1)
    cash: float = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_total(self) -> AssetWeights:
        if abs(sum(self.as_tuple()) - 1) > _WEIGHT_TOLERANCE:
            raise ValueError("asset weights must sum to 1")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.us_equity,
            self.international_equity,
            self.bonds,
            self.cash,
        )


class AssetReturnAssumption(BaseModel):
    """Arithmetic expected return and annual volatility."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    expected_return: float = Field(gt=-0.99, le=1)
    volatility: float = Field(ge=0, le=1)


CorrelationRow = tuple[float, float, float, float]
CorrelationValues = tuple[
    CorrelationRow,
    CorrelationRow,
    CorrelationRow,
    CorrelationRow,
]


class CorrelationMatrix(BaseModel):
    """Symmetric positive-semidefinite correlation matrix in asset order."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    values: CorrelationValues

    @model_validator(mode="after")
    def validate_matrix(self) -> CorrelationMatrix:
        matrix = self.values
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                if not -1 <= value <= 1:
                    raise ValueError(
                        "correlation values must be between -1 and 1"
                    )
                if row_index == column_index and abs(value - 1) > 1e-12:
                    raise ValueError(
                        "correlation matrix diagonal values must equal 1"
                    )
                if abs(value - matrix[column_index][row_index]) > 1e-12:
                    raise ValueError(
                        "correlation matrix must be symmetric"
                    )
        if not _is_positive_semidefinite(matrix):
            raise ValueError(
                "correlation matrix must be positive semidefinite"
            )
        return self


class AssetMarketAssumptions(BaseModel):
    """Return assumptions independent of allocation and account strategy."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    us_equity: AssetReturnAssumption
    international_equity: AssetReturnAssumption
    bonds: AssetReturnAssumption
    cash: AssetReturnAssumption
    correlation: CorrelationMatrix

    @model_validator(mode="after")
    def validate_lognormal_transform(self) -> AssetMarketAssumptions:
        try:
            covariance = self.lognormal_covariance_matrix()
        except ValueError as exc:
            raise ValueError(
                f"simple-return correlations cannot be represented by "
                f"lognormal returns: {exc}"
            ) from exc
        if not _is_positive_semidefinite(covariance):
            raise ValueError(
                "simple-return correlations transform to a non-positive-"
                "semidefinite lognormal covariance matrix"
            )
        return self

    def assumptions(self) -> tuple[AssetReturnAssumption, ...]:
        return (
            self.us_equity,
            self.international_equity,
            self.bonds,
            self.cash,
        )

    def covariance_matrix(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        """Return the annual simple-return covariance implied by inputs."""
        volatility = tuple(
            assumption.volatility
            for assumption in self.assumptions()
        )
        return tuple(
            tuple(
                self.correlation.values[row][column]
                * volatility[row]
                * volatility[column]
                for column in range(ASSET_CLASS_COUNT)
            )
            for row in range(ASSET_CLASS_COUNT)
        )

    def lognormal_parameters(
        self,
    ) -> tuple[tuple[float, float], ...]:
        """Return log-space mean and volatility for each gross return."""
        parameters: list[tuple[float, float]] = []
        for assumption in self.assumptions():
            gross_mean = 1 + assumption.expected_return
            variance_ratio = (
                assumption.volatility / gross_mean
            ) ** 2
            log_variance = math.log1p(variance_ratio)
            parameters.append(
                (
                    math.log(gross_mean) - log_variance / 2,
                    math.sqrt(log_variance),
                )
            )
        return tuple(parameters)

    def lognormal_covariance_matrix(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        """Transform requested simple-return correlations into log space."""
        assumptions = self.assumptions()
        parameters = self.lognormal_parameters()
        rows: list[tuple[float, ...]] = []
        for row in range(ASSET_CLASS_COUNT):
            values: list[float] = []
            for column in range(ASSET_CLASS_COUNT):
                if row == column:
                    values.append(parameters[row][1] ** 2)
                    continue
                simple_covariance = (
                    self.correlation.values[row][column]
                    * assumptions[row].volatility
                    * assumptions[column].volatility
                )
                gross_mean_product = (
                    (1 + assumptions[row].expected_return)
                    * (1 + assumptions[column].expected_return)
                )
                log_argument = 1 + simple_covariance / gross_mean_product
                if log_argument <= 0:
                    raise ValueError(
                        "a requested correlation implies non-positive "
                        f"lognormal covariance argument at [{row},{column}]"
                    )
                values.append(math.log(log_argument))
            rows.append(tuple(values))
        return tuple(rows)


class GlidePathPoint(BaseModel):
    """Target allocation that applies at one age."""

    model_config = ConfigDict(frozen=True)

    age: int = Field(ge=0, le=130)
    weights: AssetWeights


class InvestmentAccountAllocation(BaseModel):
    """Starting account share, fees, target allocation, and glide path."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    account_id: str = Field(min_length=1, max_length=128)
    portfolio_weight: float = Field(ge=0, le=1)
    annual_fee_rate: float = Field(default=0, ge=0, lt=0.25)
    target: AssetWeights
    glide_path: tuple[GlidePathPoint, ...] = Field(
        default=(),
        max_length=MAX_GLIDE_PATH_POINTS,
    )

    @model_validator(mode="after")
    def validate_glide_path(self) -> InvestmentAccountAllocation:
        ages = [point.age for point in self.glide_path]
        if any(
            current <= previous
            for previous, current in zip(ages, ages[1:])
        ):
            raise ValueError(
                "account glide-path ages must be strictly increasing"
            )
        return self


class RebalancingPolicy(BaseModel):
    """Calendar and optional drift threshold for account rebalancing."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    frequency_years: int | None = Field(default=1, ge=1, le=20)
    drift_threshold: float | None = Field(default=None, gt=0, le=1)


class PortfolioAllocationPlan(BaseModel):
    """Complete multi-asset market and account strategy assumptions."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    market: AssetMarketAssumptions
    accounts: tuple[InvestmentAccountAllocation, ...] = Field(
        min_length=1,
        max_length=MAX_ALLOCATION_ACCOUNTS,
    )
    rebalancing: RebalancingPolicy = Field(
        default_factory=RebalancingPolicy
    )
    asset_location_strategy: Literal[
        "account_glide_path_linear_v1"
    ] = "account_glide_path_linear_v1"

    @model_validator(mode="after")
    def validate_accounts(self) -> PortfolioAllocationPlan:
        account_ids = [account.account_id for account in self.accounts]
        if len(account_ids) != len(set(account_ids)):
            raise ValueError("allocation account IDs must be unique")
        if (
            abs(
                sum(
                    account.portfolio_weight
                    for account in self.accounts
                )
                - 1
            )
            > _WEIGHT_TOLERANCE
        ):
            raise ValueError("allocation account portfolio weights must sum to 1")
        return self


class PortfolioAllocationManifest(BaseModel):
    """Versioned material allocation and path assumptions."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 2
    plan_sha256: str = Field(min_length=64, max_length=64)
    strategy_identity: str = Field(min_length=1, max_length=100)
    plan: PortfolioAllocationPlan
    covariance_matrix: tuple[tuple[float, ...], ...]
    lognormal_covariance_matrix: tuple[tuple[float, ...], ...]


class PortfolioAllocationValidation(BaseModel):
    """Shared CLI/HTTP representation of a validated allocation."""

    model_config = ConfigDict(frozen=True)

    valid: Literal[True] = True
    manifest: PortfolioAllocationManifest


class AssetLocationStrategy(Protocol):
    """Hook for tax- or account-aware target allocation strategies."""

    @property
    def identity(self) -> str: ...

    def target_weights(
        self,
        account: InvestmentAccountAllocation,
        *,
        current_age: int,
        age: int,
    ) -> AssetWeights: ...


class GlidePathAssetLocationStrategy:
    """Default linear account glide-path strategy."""

    identity = "account_glide_path_linear_v1"

    def target_weights(
        self,
        account: InvestmentAccountAllocation,
        *,
        current_age: int,
        age: int,
    ) -> AssetWeights:
        points = (
            GlidePathPoint(age=current_age, weights=account.target),
            *account.glide_path,
        )
        if age <= points[0].age:
            return points[0].weights
        for start, end in zip(points, points[1:]):
            if age > end.age:
                continue
            span = end.age - start.age
            if span <= 0:
                return end.weights
            fraction = (age - start.age) / span
            return _interpolate_weights(
                start.weights,
                end.weights,
                fraction,
            )
        return points[-1].weights


def _preference_adjusted_targets(
    plan: PortfolioAllocationPlan,
    base_targets: np.ndarray,
    *,
    capacities: np.ndarray,
    account_tax_treatments: dict[str, str],
    preferences: Sequence[tuple[str, tuple[str, ...]]],
) -> np.ndarray:
    """Place assets by tax preference while preserving portfolio totals."""
    import numpy as np

    account_ids = tuple(account.account_id for account in plan.accounts)
    if set(account_tax_treatments) != set(account_ids):
        raise ValueError(
            "asset-location preferences require tax treatment for every "
            "allocation account"
        )
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim not in (1, 2) or capacities.shape[0] != len(
        plan.accounts
    ):
        raise ValueError(
            "asset-location capacities must contain every allocation account"
        )
    if np.any(~np.isfinite(capacities)) or np.any(capacities < 0):
        raise ValueError(
            "asset-location capacities must be finite and nonnegative"
        )
    vectorized = capacities.ndim == 2
    working_capacities = (
        capacities
        if vectorized
        else capacities[:, None]
    )
    asset_totals = np.einsum(
        "ab,ac->cb",
        working_capacities,
        base_targets,
    )
    remaining_capacity = working_capacities.copy()
    located = np.zeros(
        (
            len(plan.accounts),
            ASSET_CLASS_COUNT,
            working_capacities.shape[1],
        ),
        dtype=float,
    )
    preference_by_asset = {
        asset: preferred
        for asset, preferred in preferences
    }
    preferred_assets = [
        ASSET_CLASS_NAMES.index(asset)
        for asset, _ in preferences
    ]
    asset_order = (
        *preferred_assets,
        *(
            index
            for index in range(ASSET_CLASS_COUNT)
            if index not in preferred_assets
        ),
    )
    for order_index, asset_index in enumerate(asset_order):
        remaining_asset = asset_totals[asset_index].copy()
        preferred = preference_by_asset.get(
            ASSET_CLASS_NAMES[asset_index],
            (),
        )
        preference_rank = {
            treatment: index
            for index, treatment in enumerate(preferred)
        }
        account_order = sorted(
            range(len(plan.accounts)),
            key=lambda index: (
                preference_rank.get(
                    account_tax_treatments[account_ids[index]],
                    len(preferred) + 1,
                ),
                -base_targets[index, asset_index],
                index,
            ),
        )
        if order_index == len(asset_order) - 1:
            for account_index in account_order:
                amount = remaining_capacity[account_index]
                located[account_index, asset_index] = amount
                remaining_asset -= amount
            if np.any(np.abs(remaining_asset) > 1e-9):
                raise RuntimeError(
                    "asset-location strategy did not conserve portfolio assets"
                )
            continue
        for account_index in account_order:
            amount = np.minimum(
                remaining_asset,
                remaining_capacity[account_index],
            )
            located[account_index, asset_index] = amount
            remaining_capacity[account_index] -= amount
            remaining_asset -= amount
            if np.all(remaining_asset <= 1e-12):
                break
        if np.any(remaining_asset > 1e-9):  # pragma: no cover - invariant
            raise RuntimeError(
                "asset-location strategy could not place portfolio assets"
            )
    np.divide(
        located,
        working_capacities[:, None, :],
        out=located,
        where=working_capacities[:, None, :] > 0,
    )
    result = located if vectorized else located[:, :, 0]
    return result


@dataclass(frozen=True)
class AllocationYearDecision:
    """Vectorized effective return and auditable strategy effects."""

    portfolio_gross_return: np.ndarray
    account_gross_returns: dict[str, np.ndarray]
    fee_real: np.ndarray
    turnover_real: np.ndarray
    rebalanced: np.ndarray


class PortfolioAllocationState:
    """Per-trial account and asset weights retained between simulation years."""

    def __init__(
        self,
        plan: PortfolioAllocationPlan,
        *,
        trials: int,
        current_age: int,
        global_annual_fee_rate: float,
        strategy: AssetLocationStrategy | None = None,
        account_tax_treatments: dict[str, str] | None = None,
        asset_location_preferences: Sequence[
            tuple[str, tuple[str, ...]]
        ] = (),
    ) -> None:
        import numpy as np

        self.plan = plan
        self.current_age = current_age
        self.strategy = strategy or GlidePathAssetLocationStrategy()
        self.account_tax_treatments = account_tax_treatments or {}
        self.asset_location_preferences = tuple(
            asset_location_preferences
        )
        self.strategy_identity = (
            f"{self.strategy.identity}+tax_preference_priority_v1"
            if self.asset_location_preferences
            else self.strategy.identity
        )
        self.global_annual_fee_rate = global_annual_fee_rate
        self.weights = np.empty(
            (len(plan.accounts), ASSET_CLASS_COUNT, trials),
            dtype=float,
        )
        initial_targets = self._target_matrix(current_age)
        for index, account in enumerate(plan.accounts):
            initial = initial_targets[index]
            self.weights[index] = (
                account.portfolio_weight * initial[:, None]
            )
        self.account_gross_returns = np.ones(
            (len(plan.accounts), trials),
            dtype=float,
        )

    def _target_matrix(
        self,
        age: int,
        *,
        capacities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return deterministic per-account targets for one modeled age."""
        import numpy as np

        base_targets = np.asarray(
            [
                self.strategy.target_weights(
                    account,
                    current_age=self.current_age,
                    age=age,
                ).as_tuple()
                for account in self.plan.accounts
            ],
            dtype=float,
        )
        if not self.asset_location_preferences:
            if capacities is None or capacities.ndim == 1:
                return base_targets
            return np.broadcast_to(
                base_targets[:, :, None],
                (
                    len(self.plan.accounts),
                    ASSET_CLASS_COUNT,
                    capacities.shape[1],
                ),
            )
        location_capacities = (
            capacities
            if capacities is not None
            else np.asarray(
                [
                    account.portfolio_weight
                    for account in self.plan.accounts
                ],
                dtype=float,
            )
        )
        return _preference_adjusted_targets(
            self.plan,
            base_targets,
            capacities=location_capacities,
            account_tax_treatments=self.account_tax_treatments,
            preferences=self.asset_location_preferences,
        )

    def align_account_balances(
        self,
        trial_slice: slice,
        account_balances: dict[str, np.ndarray],
        *,
        age: int,
    ) -> None:
        """Align allocation shares to authoritative linked account values."""
        import numpy as np

        expected_ids = {
            account.account_id
            for account in self.plan.accounts
        }
        if set(account_balances) != expected_ids:
            raise ValueError(
                "linked account balances do not match allocation accounts"
            )
        first_balance = account_balances[self.plan.accounts[0].account_id]
        total = np.zeros_like(first_balance)
        for account in self.plan.accounts:
            total += account_balances[account.account_id]
        capacities = np.zeros(
            (len(self.plan.accounts), total.size),
            dtype=float,
        )
        for index, account in enumerate(self.plan.accounts):
            np.divide(
                account_balances[account.account_id],
                total,
                out=capacities[index],
                where=total > 0,
            )
        targets = self._target_matrix(
            age,
            capacities=capacities,
        )
        for index, account in enumerate(self.plan.accounts):
            weights = self.weights[index, :, trial_slice]
            existing_total = np.sum(weights, axis=0)
            within_account = np.divide(
                weights,
                existing_total[None, :],
                out=targets[index].copy(),
                where=existing_total[None, :] > 0,
            )
            within_account *= capacities[index, None, :]
            weights[:] = within_account

    def apply_year(
        self,
        trial_slice: slice,
        *,
        asset_gross_returns: np.ndarray,
        opening_portfolio_nominal: np.ndarray,
        inflation_factor: float | np.ndarray,
        age: int,
        year_index: int,
    ) -> AllocationYearDecision:
        """Apply asset returns, account fees, and scheduled rebalancing."""
        import numpy as np

        expected_shape = (
            ASSET_CLASS_COUNT,
            opening_portfolio_nominal.size,
        )
        if asset_gross_returns.shape != expected_shape:
            raise ValueError(
                "multi-asset return batch must contain four aligned assets"
            )
        weights = self.weights[:, :, trial_slice]
        opening_account_weights = np.sum(weights, axis=1)
        pre_fee = weights * asset_gross_returns[None, :, :]
        account_fee_rates = np.asarray(
            [
                1
                - (1 - self.global_annual_fee_rate)
                * (1 - account.annual_fee_rate)
                for account in self.plan.accounts
            ],
            dtype=float,
        )[:, None, None]
        fee_fraction = pre_fee * account_fee_rates
        net = pre_fee - fee_fraction
        account_net = np.sum(net, axis=1)
        account_gross_returns = np.divide(
            account_net,
            opening_account_weights,
            out=np.ones_like(account_net),
            where=opening_account_weights > 0,
        )
        self.account_gross_returns[:, trial_slice] = (
            account_gross_returns
        )
        portfolio_gross_return = np.sum(net, axis=(0, 1))
        safe_gross = np.where(portfolio_gross_return > 0, portfolio_gross_return, 1)
        weights[:] = net / safe_gross[None, None, :]

        inflation = np.asarray(inflation_factor, dtype=float)
        fee_real = (
            opening_portfolio_nominal
            * np.sum(fee_fraction, axis=(0, 1))
            / inflation
        )
        turnover_fraction = np.zeros(
            opening_portfolio_nominal.shape,
            dtype=float,
        )
        rebalanced = np.zeros(
            opening_portfolio_nominal.shape,
            dtype=bool,
        )
        frequency = self.plan.rebalancing.frequency_years
        scheduled = (
            frequency is not None and (year_index + 1) % frequency == 0
        )
        if scheduled:
            threshold = self.plan.rebalancing.drift_threshold
            live_capacities = np.sum(weights, axis=1)
            target_matrix = self._target_matrix(
                age + 1,
                capacities=live_capacities,
            )
            for account_index, account in enumerate(self.plan.accounts):
                account_total = live_capacities[account_index]
                target = target_matrix[account_index]
                current = np.divide(
                    weights[account_index],
                    account_total[None, :],
                    out=np.zeros_like(weights[account_index]),
                    where=account_total[None, :] > 0,
                )
                should_rebalance = np.ones(
                    account_total.shape,
                    dtype=bool,
                )
                if threshold is not None:
                    should_rebalance = (
                        np.max(np.abs(current - target), axis=0)
                        >= threshold
                    )
                proposed = account_total[None, :] * target
                change = np.where(
                    should_rebalance[None, :],
                    proposed - weights[account_index],
                    0,
                )
                turnover_fraction += 0.5 * np.sum(
                    np.abs(change),
                    axis=0,
                )
                weights[account_index] += change
                rebalanced |= should_rebalance

        turnover_real = (
            opening_portfolio_nominal
            * portfolio_gross_return
            * turnover_fraction
            / inflation
        )
        return AllocationYearDecision(
            portfolio_gross_return=portfolio_gross_return,
            account_gross_returns={
                account.account_id: account_gross_returns[index]
                for index, account in enumerate(self.plan.accounts)
            },
            fee_real=fee_real,
            turnover_real=turnover_real,
            rebalanced=rebalanced,
        )

    def asset_weights(self) -> dict[str, np.ndarray]:
        """Return current aggregate portfolio weights by public asset name."""
        aggregate = self.weights.sum(axis=0)
        return {
            "us_equity": aggregate[0],
            "international_equity": aggregate[1],
            "bonds": aggregate[2],
            "cash": aggregate[3],
        }

    def account_weights(self) -> dict[str, np.ndarray]:
        """Return current portfolio share for every configured account."""
        return {
            account.account_id: self.weights[index].sum(axis=0)
            for index, account in enumerate(self.plan.accounts)
        }


def estimate_allocation_state_bytes(
    plan: PortfolioAllocationPlan | None,
    trials: int,
    *,
    batch_size: int | None = None,
) -> int:
    """Conservatively estimate persistent and annual allocation arrays."""
    if trials < 0:
        raise ValueError("trials must not be negative")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if plan is None:
        return 0
    effective_batch = min(batch_size or trials, trials)
    components = len(plan.accounts) * ASSET_CLASS_COUNT
    # Persistent account/asset weights and annual audit vectors are trial-wide.
    # The annual observer then materializes either four aggregate asset-weight
    # arrays or one account-weight array per account, while percentile work
    # retains one more trial vector. During each batch, pre-fee, fee, and net
    # component tensors coexist with normalization and rebalancing work arrays.
    # Count every boolean as eight bytes to retain a conservative upper bound.
    account_count = len(plan.accounts)
    annual_audit_arrays = 3
    observer_peak_arrays = max(
        ASSET_CLASS_COUNT + 1,
        account_count + 1,
    )
    persistent_bytes = trials * (
        components
        + account_count
        + annual_audit_arrays
        + observer_peak_arrays
    ) * 8
    transient_component_bytes = effective_batch * components * 4 * 8
    # Linked account balances are released before apply_year. Its peak retains
    # opening weights, account net values, account returns, live rebalancing
    # capacities, and the location solver's remaining capacities.
    transient_account_bytes = effective_batch * account_count * 5 * 8
    transient_vector_bytes = effective_batch * (
        ASSET_CLASS_COUNT * 4 + 12
    ) * 8
    return (
        persistent_bytes
        + transient_component_bytes
        + transient_account_bytes
        + transient_vector_bytes
    )


def portfolio_allocation_manifest(
    plan: PortfolioAllocationPlan,
    *,
    strategy_identity: str | None = None,
) -> PortfolioAllocationManifest:
    """Fingerprint every material market, account, fee, and strategy input."""
    identity = strategy_identity or plan.asset_location_strategy
    canonical = json.dumps(
        {
            "plan": plan.model_dump(mode="json"),
            "strategy_identity": identity,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return PortfolioAllocationManifest(
        plan_sha256=sha256(canonical).hexdigest(),
        strategy_identity=identity,
        plan=plan,
        covariance_matrix=plan.market.covariance_matrix(),
        lognormal_covariance_matrix=(
            plan.market.lognormal_covariance_matrix()
        ),
    )


def validate_portfolio_allocation(
    plan: PortfolioAllocationPlan,
) -> PortfolioAllocationValidation:
    """Return the canonical representation shared by CLI and HTTP."""
    return PortfolioAllocationValidation(
        manifest=portfolio_allocation_manifest(plan)
    )


def _interpolate_weights(
    start: AssetWeights,
    end: AssetWeights,
    fraction: float,
) -> AssetWeights:
    values = tuple(
        start_value + (end_value - start_value) * fraction
        for start_value, end_value in zip(
            start.as_tuple(),
            end.as_tuple(),
        )
    )
    return AssetWeights(
        us_equity=values[0],
        international_equity=values[1],
        bonds=values[2],
        cash=values[3],
    )


def _is_positive_semidefinite(
    matrix: Sequence[Sequence[float]],
) -> bool:
    """Check PSD with a zero-pivot-aware LDLᵀ factorization."""
    dimension = len(matrix)
    lower = [
        [0.0 for _ in range(dimension)]
        for _ in range(dimension)
    ]
    diagonal = [0.0 for _ in range(dimension)]
    for row in range(dimension):
        lower[row][row] = 1.0
        for column in range(row):
            residual = matrix[row][column] - sum(
                lower[row][index]
                * diagonal[index]
                * lower[column][index]
                for index in range(column)
            )
            if abs(diagonal[column]) <= _PSD_TOLERANCE:
                if abs(residual) > _PSD_TOLERANCE:
                    return False
                lower[row][column] = 0
            else:
                lower[row][column] = residual / diagonal[column]
        pivot = matrix[row][row] - sum(
            lower[row][index] ** 2 * diagonal[index]
            for index in range(row)
        )
        if pivot < -_PSD_TOLERANCE:
            return False
        diagonal[row] = 0.0 if abs(pivot) <= _PSD_TOLERANCE else pivot
    return True
