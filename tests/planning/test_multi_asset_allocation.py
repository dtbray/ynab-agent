from __future__ import annotations

import numpy as np
from pydantic import ValidationError
import pytest

from ynab_agent.planning.allocation import (
    ASSET_CLASS_COUNT,
    MAX_ALLOCATION_ACCOUNTS,
    AssetMarketAssumptions,
    CorrelationMatrix,
    PortfolioAllocationPlan,
    PortfolioAllocationState,
    estimate_allocation_state_bytes,
    portfolio_allocation_manifest,
)
from ynab_agent.planning.historical import load_historical_series
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.paths import (
    MultiAssetSimulationPaths,
    PathSpec,
    estimate_path_resources,
    multi_asset_path_manifest,
)
from ynab_agent.planning.simulation import (
    prepare_simulation_paths,
    simulate,
)
from ynab_agent.planning.stress import (
    NamedStressName,
    resolve_named_stress,
)


IDENTITY_CORRELATION = {
    "values": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
}


def _market(
    *,
    correlation: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
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
        "correlation": correlation or IDENTITY_CORRELATION,
    }


def _plan(
    *,
    target: dict[str, float] | None = None,
    fee: float = 0,
    glide_path: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "market": _market(),
        "accounts": [
            {
                "account_id": "portfolio",
                "portfolio_weight": 1,
                "annual_fee_rate": fee,
                "target": target
                or {
                    "us_equity": 0.6,
                    "international_equity": 0.2,
                    "bonds": 0.15,
                    "cash": 0.05,
                },
                "glide_path": glide_path or [],
            }
        ],
        "rebalancing": {"frequency_years": 1},
    }


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "multi asset",
        "current_age": 40,
        "retirement_age": 42,
        "end_age": 43,
        "starting_portfolio": 100_000,
        "annual_contribution": 0,
        "annual_spending": 20_000,
        "portfolio_allocation": _plan(),
        "inflation_rate": 0,
        "return_mean": 0.06,
        "return_volatility": 0.16,
        "annual_fee_rate": 0,
        "trials": 100,
        "seed": 7,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def test_invalid_correlation_matrices_fail_clearly() -> None:
    with pytest.raises(ValidationError, match="symmetric"):
        CorrelationMatrix.model_validate(
            {
                "values": [
                    [1, 0.5, 0, 0],
                    [0.4, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            }
        )


def test_non_psd_transformed_lognormal_covariance_fails_clearly() -> None:
    with pytest.raises(
        ValidationError,
        match="non-positive-semidefinite lognormal covariance",
    ):
        AssetMarketAssumptions.model_validate(
            {
                "us_equity": {
                    "expected_return": -0.6917861797,
                    "volatility": 0.5546448976,
                },
                "international_equity": {
                    "expected_return": -0.4138669461,
                    "volatility": 0.6853916245,
                },
                "bonds": {
                    "expected_return": -0.1982005973,
                    "volatility": 0.8939813867,
                },
                "cash": {
                    "expected_return": 0.1984547502,
                    "volatility": 0.5389845444,
                },
                "correlation": {
                    "values": [
                        [1, 0.2507119975, -0.0558998340, 0.2497460286],
                        [0.2507119975, 1, -0.5283046963, 0.0195972074],
                        [-0.0558998340, -0.5283046963, 1, 0.3476604267],
                        [0.2497460286, 0.0195972074, 0.3476604267, 1],
                    ]
                },
            }
        )

    with pytest.raises(ValidationError, match="positive semidefinite"):
        CorrelationMatrix.model_validate(
            {
                "values": [
                    [1, 0.9, 0.9, 0],
                    [0.9, 1, -0.9, 0],
                    [0.9, -0.9, 1, 0],
                    [0, 0, 0, 1],
                ]
            }
        )


def test_covariance_and_manifest_capture_every_material_assumption() -> None:
    market = AssetMarketAssumptions.model_validate(_market())
    plan = PortfolioAllocationPlan.model_validate(_plan(fee=0.004))

    covariance = market.covariance_matrix()
    manifest = portfolio_allocation_manifest(plan)

    assert covariance[0][0] == pytest.approx(0.18**2)
    assert covariance[0][1] == 0
    assert manifest.lognormal_covariance_matrix[0][0] > 0
    assert manifest.plan.accounts[0].annual_fee_rate == 0.004
    assert manifest.plan.rebalancing.frequency_years == 1
    assert len(manifest.plan_sha256) == 64


def test_correlated_paths_are_seeded_and_resource_bounded() -> None:
    scenario = _scenario(trials=1_000)

    first = prepare_simulation_paths(scenario)
    second = prepare_simulation_paths(scenario)
    try:
        assert first.asset_gross_returns is not None
        assert second.asset_gross_returns is not None
        assert first.asset_gross_returns.shape == (4, 3, 1_000)
        np.testing.assert_array_equal(
            first.asset_gross_returns,
            second.asset_gross_returns,
        )
        assert first.generator == "numpy_correlated_lognormal_v2"
        spec = PathSpec.from_scenario(
            scenario,
            paired_historical_inflation=False,
        )
        estimate = estimate_path_resources(spec)
        assert estimate.retained_path_bytes == (
            4 * 3 * 1_000 * 8 + 4 * 8
        )
    finally:
        first.close()
        second.close()


def test_generated_simple_return_correlations_match_high_volatility_inputs() -> None:
    correlation = [
        [1, 0.6, 0.35, 0.1],
        [0.6, 1, 0.3, 0.1],
        [0.35, 0.3, 1, 0.2],
        [0.1, 0.1, 0.2, 1],
    ]
    plan = _plan()
    market = plan["market"]
    assert isinstance(market, dict)
    for asset, volatility in (
        ("us_equity", 0.8),
        ("international_equity", 0.7),
        ("bonds", 0.5),
        ("cash", 0.2),
    ):
        assumption = market[asset]
        assert isinstance(assumption, dict)
        assumption["volatility"] = volatility
    market["correlation"] = {"values": correlation}
    scenario = _scenario(
        trials=100_000,
        end_age=41,
        retirement_age=40,
        portfolio_allocation=plan,
    )

    paths = prepare_simulation_paths(scenario)
    try:
        assert paths.asset_gross_returns is not None
        achieved = np.corrcoef(paths.asset_gross_returns[:, 0, :])
    finally:
        paths.close()

    np.testing.assert_allclose(
        achieved,
        np.asarray(correlation),
        atol=0.02,
    )


def test_allocation_strategies_share_common_market_paths() -> None:
    first = _scenario()
    second = _scenario(
        annual_fee_rate=0.01,
        portfolio_allocation=_plan(
            target={
                "us_equity": 0.2,
                "international_equity": 0.1,
                "bonds": 0.6,
                "cash": 0.1,
            }
        )
    )
    paths = prepare_simulation_paths(first)
    try:
        first_result = simulate(first, 100_000, prepared_paths=paths)
        second_result = simulate(second, 100_000, prepared_paths=paths)
    finally:
        paths.close()

    assert first_result.ending_balance_real != (
        second_result.ending_balance_real
    )
    assert (
        first_result.reproducibility["portfolio_allocation"]
        != second_result.reproducibility["portfolio_allocation"]
    )


def test_fees_rebalancing_and_glide_path_are_visible_in_annual_audit() -> None:
    scenario = _scenario(
        portfolio_allocation=_plan(
            target={
                "us_equity": 1,
                "international_equity": 0,
                "bonds": 0,
                "cash": 0,
            },
            fee=0.01,
            glide_path=[
                {
                    "age": 41,
                    "weights": {
                        "us_equity": 0.5,
                        "international_equity": 0,
                        "bonds": 0.5,
                        "cash": 0,
                    },
                }
            ],
        ),
    )
    asset_returns = np.ones((4, 3, 100), dtype=float)
    asset_returns[0, 0] = 2
    asset_returns[0, 1] = 2
    paths = MultiAssetSimulationPaths(
        gross_returns=asset_returns[0],
        inflation_factors=np.ones(4),
        asset_gross_returns=asset_returns,
        source_name="named_stress_equity_shock",
        source_metadata=(("fixture", "fee-rebalance"),),
    )

    result = simulate(scenario, 100_000, prepared_paths=paths)

    first_year = result.annual_allocation_real[0]
    assert first_year["rebalanced_trials"] == 100
    assert first_year["fee_real"]["p50"] == pytest.approx(2_000)
    assert first_year["turnover_real"]["p50"] == pytest.approx(99_000)
    assert first_year["asset_weights"]["us_equity"]["p50"] == 0.5
    assert first_year["asset_weights"]["bonds"]["p50"] == 0.5
    assert first_year["account_gross_returns"]["portfolio"][
        "p50"
    ] == pytest.approx(1.98)
    assert result.annual_allocation_real[1][
        "account_gross_returns"
    ]["portfolio"]["p50"] == pytest.approx(1.485)
    manifest = result.reproducibility["portfolio_allocation"]
    assert manifest["plan"]["accounts"][0]["glide_path"][0]["age"] == 41
    assert result.reproducibility["multi_asset_paths"] == {
        "schema_version": 1,
        "source_name": "named_stress_equity_shock",
        "source_sha256": paths.source_sha256,
        "source_metadata": {"fixture": "fee-rebalance"},
    }


def test_custom_paths_hash_every_array_and_metadata_and_verify_supplied_hash() -> None:
    assets = np.ones((4, 2, 3), dtype=float)
    inflation = np.ones(3, dtype=float)
    first = MultiAssetSimulationPaths(
        gross_returns=assets[0],
        inflation_factors=inflation,
        asset_gross_returns=assets,
        source_name="custom",
        source_metadata=(("version", "1"),),
    )
    changed_assets = np.array(assets, copy=True)
    changed_assets[3, 1, 2] = 1.01
    changed = MultiAssetSimulationPaths(
        gross_returns=changed_assets[0],
        inflation_factors=inflation,
        asset_gross_returns=changed_assets,
        source_name="custom",
        source_metadata=(("version", "1"),),
    )
    changed_metadata = MultiAssetSimulationPaths(
        gross_returns=np.array(assets[0], copy=True),
        inflation_factors=np.array(inflation, copy=True),
        asset_gross_returns=np.array(assets, copy=True),
        source_name="custom",
        source_metadata=(("version", "2"),),
    )
    changed_inflation_values = np.array(inflation, copy=True)
    changed_inflation_values[1] = 1.02
    changed_inflation = MultiAssetSimulationPaths(
        gross_returns=np.array(assets[0], copy=True),
        inflation_factors=changed_inflation_values,
        asset_gross_returns=np.array(assets, copy=True),
        source_name="custom",
        source_metadata=(("version", "1"),),
    )

    assert first.source_sha256 != changed.source_sha256
    assert first.source_sha256 != changed_metadata.source_sha256
    assert first.source_sha256 != changed_inflation.source_sha256
    assert first.asset_gross_returns.flags.writeable is False
    verified = MultiAssetSimulationPaths(
        gross_returns=np.array(assets[0], copy=True),
        inflation_factors=np.array(inflation, copy=True),
        asset_gross_returns=np.array(assets, copy=True),
        source_name="custom",
        source_sha256=first.source_sha256,
        source_metadata=(("version", "1"),),
    )
    assert verified.source_sha256 == first.source_sha256
    with pytest.raises(
        ValueError,
        match="source_sha256 does not match",
    ):
        MultiAssetSimulationPaths(
            gross_returns=np.array(assets[0], copy=True),
            inflation_factors=np.array(inflation, copy=True),
            asset_gross_returns=np.array(assets, copy=True),
            source_name="custom",
            source_sha256="0" * 64,
            source_metadata=(("version", "1"),),
        )


def test_custom_paths_detach_sliced_mutable_bases_before_hashing() -> None:
    asset_base = np.ones((4, 2, 6), dtype=np.float32)
    asset_base[0, 0, ::2] = (1.1, 1.2, 1.3)
    asset_view = asset_base[:, :, ::2]
    gross_view = asset_base[0, :, ::2]
    inflation_base = np.asarray(
        [1.0, 99.0, 1.02, 99.0, 1.04, 99.0],
        dtype=np.float32,
    )
    inflation_view = inflation_base[::2]
    paths = MultiAssetSimulationPaths(
        gross_returns=gross_view,
        inflation_factors=inflation_view,
        asset_gross_returns=asset_view,
        source_name="sliced-mutable-fixture",
        source_metadata=(("layout", "strided"),),
    )
    stored_assets = np.array(paths.asset_gross_returns, copy=True)
    stored_gross = np.array(paths.gross_returns, copy=True)
    stored_inflation = np.array(paths.inflation_factors, copy=True)
    stored_sha256 = paths.source_sha256
    stored_manifest = multi_asset_path_manifest(paths)

    assert paths.asset_gross_returns.dtype == np.float64
    assert paths.asset_gross_returns.flags.c_contiguous is True
    assert paths.asset_gross_returns.flags.owndata is True
    assert paths.gross_returns.flags.owndata is True
    assert paths.inflation_factors.flags.owndata is True

    asset_base[:] = 7
    inflation_base[:] = 8

    np.testing.assert_array_equal(
        paths.asset_gross_returns,
        stored_assets,
    )
    np.testing.assert_array_equal(paths.gross_returns, stored_gross)
    np.testing.assert_array_equal(
        paths.inflation_factors,
        stored_inflation,
    )
    assert paths.source_sha256 == stored_sha256
    assert multi_asset_path_manifest(paths) == stored_manifest


def test_multi_asset_history_bootstraps_aligned_assets_end_to_end(
    tmp_path,
) -> None:
    history_path = tmp_path / "multi-asset.csv"
    history_path.write_text(
        "year,nominal_return,inflation_rate,us_equity_return,"
        "international_equity_return,bonds_return,cash_return\n"
        "2020,0.10,0.02,0.11,0.12,0.01,0.001\n"
        "2021,-0.10,0.03,-0.21,-0.22,0.02,0.002\n"
        "2022,0.05,0.04,0.31,0.32,0.03,0.003\n",
        encoding="utf-8",
    )
    history = load_historical_series(history_path)
    scenario = _scenario(
        return_model="historical_bootstrap",
        historical_block_size=1,
    )

    paths = prepare_simulation_paths(
        scenario,
        historical_returns=history.nominal_returns,
        historical_inflation=history.inflation_rates,
        historical_asset_returns=history.asset_returns,
    )
    try:
        assert paths.asset_gross_returns is not None
        assert paths.generator == (
            "vectorized_multi_asset_stationary_bootstrap_v1"
        )
        us_to_row = {1.11: (1.12, 1.01, 1.001), 0.79: (0.78, 1.02, 1.002), 1.31: (1.32, 1.03, 1.003)}
        for year in range(paths.spec.years):
            for trial in range(paths.spec.trials):
                us = round(
                    float(paths.asset_gross_returns[0, year, trial]),
                    3,
                )
                expected = us_to_row[us]
                np.testing.assert_allclose(
                    paths.asset_gross_returns[1:, year, trial],
                    expected,
                )
        result = simulate(
            scenario,
            100_000,
            historical_series=history,
            prepared_paths=paths,
        )
        assert result.reproducibility["historical"][
            "multi_asset_returns"
        ] is True
    finally:
        paths.close()


def test_named_stress_catalog_selector_is_deterministic_and_manifested() -> None:
    scenario = _scenario()
    definition = resolve_named_stress(NamedStressName.EQUITY_CRASH)

    first = prepare_simulation_paths(
        scenario,
        named_stress=NamedStressName.EQUITY_CRASH,
    )
    second = prepare_simulation_paths(
        scenario,
        named_stress=NamedStressName.EQUITY_CRASH,
    )
    try:
        assert first.asset_gross_returns is not None
        assert second.asset_gross_returns is not None
        np.testing.assert_array_equal(
            first.asset_gross_returns,
            second.asset_gross_returns,
        )
        expected_first = (
            np.asarray(definition.annual_returns[0].as_tuple()) + 1
        )
        np.testing.assert_allclose(
            first.asset_gross_returns[:, 0, 0],
            expected_first,
        )
        result = simulate(scenario, 100_000, prepared_paths=first)
        assert result.assumptions["named_stress"] == "equity_crash"
        assert result.reproducibility["multi_asset_paths"] == {
            "schema_version": 1,
            "source_name": "named_stress_catalog_v1:equity_crash",
            "source_sha256": definition.content_sha256,
            "named_stress": "equity_crash",
        }
    finally:
        first.close()
        second.close()


def test_multi_asset_growth_preserves_progressive_tax_audit_and_buckets() -> None:
    scenario = _scenario(
        current_age=60,
        retirement_age=60,
        end_age=62,
        tax_buckets=[
            {
                "tax_treatment": "tax_deferred",
                "starting_balance": 100_000,
            }
        ],
        tax_assumptions={
            "ordinary_income_tax_rate": 0,
            "long_term_capital_gains_tax_rate": 0,
            "tax_model": "progressive_us_indiana",
            "progressive": {
                "filing_status": "single",
                "simulation_start_year": 2026,
                "taxpayer_birth_year": 1966,
            },
            "apply_required_minimum_distributions": False,
            "withdrawal_order": ["tax_deferred"],
            "retirement_surplus_destination": "tax_deferred",
        },
    )

    result = simulate(scenario, 100_000)

    assert result.assumptions["tax_model"] == "progressive_us_indiana_v1"
    assert result.lifetime_tax_real is not None
    assert result.lifetime_tax_real["p50"] > 0
    assert len(result.annual_tax_audit) == 2
    assert result.annual_tax_audit[0]["tax_year"] == 2026
    assert len(result.annual_allocation_real) == 2
    assert result.reproducibility["portfolio_allocation"] is not None


def test_max_account_and_batch_memory_estimate_covers_transient_tensors() -> None:
    plan_values = _plan()
    plan_values["accounts"] = [
        {
            "account_id": f"account-{index}",
            "portfolio_weight": 1 / MAX_ALLOCATION_ACCOUNTS,
            "target": {
                "us_equity": 0.6,
                "international_equity": 0.2,
                "bonds": 0.15,
                "cash": 0.05,
            },
        }
        for index in range(MAX_ALLOCATION_ACCOUNTS)
    ]
    plan = PortfolioAllocationPlan.model_validate(plan_values)
    trials = 100_000
    batch = 10_000

    estimate = estimate_allocation_state_bytes(
        plan,
        trials,
        batch_size=batch,
    )
    components = MAX_ALLOCATION_ACCOUNTS * ASSET_CLASS_COUNT
    observer_peak_arrays = max(
        ASSET_CLASS_COUNT + 1,
        MAX_ALLOCATION_ACCOUNTS + 1,
    )
    known_live_arrays = (
        trials
        * (
            components
            + MAX_ALLOCATION_ACCOUNTS
            + 3
            + observer_peak_arrays
        )
        * 8
        + batch * components * 4 * 8
        + batch * MAX_ALLOCATION_ACCOUNTS * 5 * 8
        + batch * (ASSET_CLASS_COUNT * 4 + 12) * 8
    )

    assert estimate == known_live_arrays


def test_allocation_memory_estimate_covers_max_account_observer_peak() -> None:
    plan_values = _plan()
    plan_values["accounts"] = [
        {
            "account_id": f"account-{index}",
            "portfolio_weight": 1 / MAX_ALLOCATION_ACCOUNTS,
            "target": {
                "us_equity": 0.6,
                "international_equity": 0.2,
                "bonds": 0.15,
                "cash": 0.05,
            },
        }
        for index in range(MAX_ALLOCATION_ACCOUNTS)
    ]
    plan = PortfolioAllocationPlan.model_validate(plan_values)
    trials = 100_000
    account_count = len(plan.accounts)
    components = account_count * ASSET_CLASS_COUNT
    retained_state_and_audit = (
        trials * (components + account_count + 3) * 8
    )
    observer_peak = (
        trials
        * max(ASSET_CLASS_COUNT + 1, account_count + 1)
        * 8
    )

    estimate = estimate_allocation_state_bytes(
        plan,
        trials,
        batch_size=1,
    )

    assert estimate >= retained_state_and_audit + observer_peak


def test_asset_location_uses_live_capacity_and_conserves_zero_accounts() -> None:
    scenario = _linked_two_account_scenario(
        preferred_equity_treatment="taxable"
    )
    plan = scenario.portfolio_allocation
    assert plan is not None
    state = PortfolioAllocationState(
        plan,
        trials=2,
        current_age=scenario.current_age,
        global_annual_fee_rate=0,
        account_tax_treatments={
            "brokerage": "taxable",
            "ira": "tax_deferred",
        },
        asset_location_preferences=(
            ("us_equity", ("taxable",)),
        ),
    )
    state.align_account_balances(
        slice(0, 2),
        {
            "brokerage": np.asarray([0.0, 25_000.0]),
            "ira": np.asarray([100_000.0, 75_000.0]),
        },
        age=60,
    )

    state.apply_year(
        slice(0, 2),
        asset_gross_returns=np.ones((ASSET_CLASS_COUNT, 2)),
        opening_portfolio_nominal=np.full(2, 100_000.0),
        inflation_factor=1,
        age=60,
        year_index=0,
    )

    account_weights = state.account_weights()
    asset_weights = state.asset_weights()
    assert account_weights["brokerage"] == pytest.approx([0.0, 0.25])
    assert account_weights["ira"] == pytest.approx([1.0, 0.75])
    assert asset_weights["us_equity"] == pytest.approx([0.5, 0.5])
    assert asset_weights["cash"] == pytest.approx([0.5, 0.5])
    assert np.sum(state.weights, axis=(0, 1)) == pytest.approx([1.0, 1.0])


def _linked_two_account_scenario(
    *,
    preferred_equity_treatment: str,
) -> WealthScenario:
    return WealthScenario.model_validate(
        {
            "name": f"equity in {preferred_equity_treatment}",
            "current_age": 60,
            "retirement_age": 60,
            "end_age": 61,
            "accounts": [
                {"id": "brokerage", "role": "taxable"},
                {"id": "ira", "role": "retirement"},
            ],
            "starting_portfolio": 100_000,
            "annual_spending": 1,
            "inflation_rate": 0,
            "return_mean": 0,
            "return_volatility": 0,
            "annual_fee_rate": 0,
            "trials": 100,
            "portfolio_allocation": {
                "market": _market(),
                "accounts": [
                    {
                        "account_id": account_id,
                        "portfolio_weight": 0.5,
                        "target": {
                            "us_equity": 0.5,
                            "international_equity": 0,
                            "bonds": 0,
                            "cash": 0.5,
                        },
                    }
                    for account_id in ("brokerage", "ira")
                ],
                "rebalancing": {"frequency_years": 1},
            },
            "tax_buckets": [
                {
                    "tax_treatment": "taxable",
                    "account_id": "brokerage",
                    "starting_balance": 50_000,
                    "taxable_basis": 50_000,
                },
                {
                    "tax_treatment": "tax_deferred",
                    "account_id": "ira",
                    "starting_balance": 50_000,
                },
            ],
            "tax_assumptions": {
                "ordinary_income_tax_rate": 0,
                "long_term_capital_gains_tax_rate": 0,
                "taxable_account_annual_tax_drag_rate": 0.10,
                "apply_required_minimum_distributions": False,
                "withdrawal_order": ["taxable", "tax_deferred"],
                "retirement_surplus_destination": "taxable",
                "strategy": {
                    "withdrawal_policy": "ordered",
                    "asset_location_preferences": [
                        {
                            "asset_class": "us_equity",
                            "preferred_tax_treatments": [
                                preferred_equity_treatment
                            ],
                        }
                    ],
                },
            },
        }
    )


def test_account_returns_and_asset_location_change_tax_drag_predictably() -> None:
    equity_taxable = _linked_two_account_scenario(
        preferred_equity_treatment="taxable"
    )
    equity_deferred = _linked_two_account_scenario(
        preferred_equity_treatment="tax_deferred"
    )
    asset_returns = np.ones((4, 1, 100), dtype=float)
    asset_returns[0] = 2
    paths = MultiAssetSimulationPaths(
        gross_returns=asset_returns[0],
        inflation_factors=np.ones(2),
        asset_gross_returns=asset_returns,
        source_name="account-location-probe",
    )

    taxable_result = simulate(
        equity_taxable,
        100_000,
        prepared_paths=paths,
    )
    deferred_result = simulate(
        equity_deferred,
        100_000,
        prepared_paths=paths,
    )

    taxable_returns = taxable_result.annual_allocation_real[0][
        "account_gross_returns"
    ]
    deferred_returns = deferred_result.annual_allocation_real[0][
        "account_gross_returns"
    ]
    assert taxable_returns["brokerage"]["p50"] == 2
    assert taxable_returns["ira"]["p50"] == 1
    assert deferred_returns["brokerage"]["p50"] == 1
    assert deferred_returns["ira"]["p50"] == 2
    assert taxable_result.lifetime_tax_real["p50"] == pytest.approx(10_000)
    assert deferred_result.lifetime_tax_real["p50"] == pytest.approx(5_000)
    assert (
        deferred_result.ending_balance_real["p50"]
        - taxable_result.ending_balance_real["p50"]
    ) == pytest.approx(5_000)
    assert taxable_result.reproducibility["tax_policy"][
        "strategy"
    ]["asset_location_execution"] == (
        "capacity_constrained_preference_priority_v1"
    )


def test_duplicate_owner_treatment_accounts_have_distinct_return_keys() -> None:
    values = _linked_two_account_scenario(
        preferred_equity_treatment="taxable"
    ).model_dump(mode="json")
    values["name"] = "duplicate taxable accounts"
    values["accounts"] = [
        {"id": "first", "role": "taxable"},
        {"id": "second", "role": "taxable"},
    ]
    values["portfolio_allocation"]["accounts"] = [
        {
            "account_id": account_id,
            "portfolio_weight": 0.5,
            "target": {
                "us_equity": 1 if account_id == "first" else 0,
                "international_equity": 0,
                "bonds": 0,
                "cash": 0 if account_id == "first" else 1,
            },
        }
        for account_id in ("first", "second")
    ]
    values["tax_buckets"] = [
        {
            "tax_treatment": "taxable",
            "account_id": account_id,
            "starting_balance": 50_000,
            "taxable_basis": 50_000,
        }
        for account_id in ("first", "second")
    ]
    values["tax_assumptions"]["withdrawal_order"] = ["taxable"]
    values["tax_assumptions"]["strategy"] = None
    scenario = WealthScenario.model_validate(values)
    asset_returns = np.ones((4, 1, 100), dtype=float)
    asset_returns[0] = 2
    paths = MultiAssetSimulationPaths(
        gross_returns=asset_returns[0],
        inflation_factors=np.ones(2),
        asset_gross_returns=asset_returns,
        source_name="duplicate-key-probe",
    )

    result = simulate(scenario, 100_000, prepared_paths=paths)

    owner_flows = result.annual_tax_audit[0]["owner_flows"]
    assert {row["account_id"] for row in owner_flows} == {
        "first",
        "second",
    }
    returns = result.annual_allocation_real[0][
        "account_gross_returns"
    ]
    assert returns["first"]["p50"] == 2
    assert returns["second"]["p50"] == 1


def test_account_specific_growth_changes_current_year_strategy_projection() -> None:
    def scenario(preferred_treatment: str) -> WealthScenario:
        values = _linked_two_account_scenario(
            preferred_equity_treatment=preferred_treatment
        ).model_dump(mode="json")
        values["name"] = f"conversion with equity in {preferred_treatment}"
        values["accounts"].append({"id": "roth", "role": "retirement"})
        values["portfolio_allocation"]["accounts"] = [
            {
                "account_id": account_id,
                "portfolio_weight": weight,
                "target": {
                    "us_equity": 0.5,
                    "international_equity": 0,
                    "bonds": 0,
                    "cash": 0.5,
                },
            }
            for account_id, weight in (
                ("brokerage", 0.5),
                ("ira", 0.25),
                ("roth", 0.25),
            )
        ]
        values["tax_buckets"] = [
            {
                "tax_treatment": "taxable",
                "account_id": "brokerage",
                "starting_balance": 50_000,
                "taxable_basis": 50_000,
            },
            {
                "tax_treatment": "tax_deferred",
                "account_id": "ira",
                "starting_balance": 25_000,
            },
            {
                "tax_treatment": "roth",
                "account_id": "roth",
                "starting_balance": 25_000,
            },
        ]
        assumptions = values["tax_assumptions"]
        assumptions.update(
            {
                "tax_model": "progressive_us_indiana",
                "progressive": {
                    "filing_status": "single",
                    "simulation_start_year": 2026,
                    "taxpayer_birth_year": 1966,
                },
                "withdrawal_order": [
                    "taxable",
                    "tax_deferred",
                    "roth",
                ],
            }
        )
        assumptions["strategy"]["roth_conversion"] = {
            "start_age": 60,
            "end_age": 60,
            "target_federal_ordinary_bracket_rate": 0.35,
            "max_annual_conversion_real": 100_000,
        }
        return WealthScenario.model_validate(values)

    asset_returns = np.ones((4, 1, 100), dtype=float)
    asset_returns[0] = 2
    paths = MultiAssetSimulationPaths(
        gross_returns=asset_returns[0],
        inflation_factors=np.ones(2),
        asset_gross_returns=asset_returns,
        source_name="strategy-account-growth-probe",
    )

    taxable_result = simulate(
        scenario("taxable"),
        100_000,
        prepared_paths=paths,
    )
    deferred_result = simulate(
        scenario("tax_deferred"),
        100_000,
        prepared_paths=paths,
    )

    assert taxable_result.annual_tax_strategy_actions[0][
        "roth_conversion_nominal"
    ]["p50"] == pytest.approx(25_000, abs=0.01)
    assert deferred_result.annual_tax_strategy_actions[0][
        "roth_conversion_nominal"
    ]["p50"] == pytest.approx(50_000, abs=0.01)


def test_allocation_requires_exact_liquid_account_coverage() -> None:
    values = _linked_two_account_scenario(
        preferred_equity_treatment="taxable"
    ).model_dump(mode="json")
    values["portfolio_allocation"]["accounts"] = [
        values["portfolio_allocation"]["accounts"][0]
    ]
    values["portfolio_allocation"]["accounts"][0][
        "portfolio_weight"
    ] = 1

    with pytest.raises(ValidationError, match="exactly cover"):
        WealthScenario.model_validate(values)
