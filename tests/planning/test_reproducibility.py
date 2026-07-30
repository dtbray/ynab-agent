from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from ynab_agent.planning.historical import (
    HistoricalGapPolicy,
    HistoricalOrderPolicy,
    load_historical_series,
)
from ynab_agent.planning.models import ValuationProvenance, WealthScenario
from ynab_agent.planning.paths import RunPolicy
from ynab_agent.planning.simulation import (
    REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION,
    SIMULATION_ENGINE_VERSION,
    SIMULATION_RESULT_SCHEMA_VERSION,
    canonical_scenario_sha256,
    simulate,
)


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "reproducible plan",
        "current_age": 40,
        "retirement_age": 43,
        "end_age": 47,
        "starting_portfolio": 100_000,
        "annual_contribution": 10_000,
        "annual_spending": 30_000,
        "inflation_rate": 0.02,
        "return_mean": 0.05,
        "return_volatility": 0.10,
        "trials": 100,
        "seed": 42,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def _write_history(path: Path, observations: str) -> None:
    path.write_text(
        "year,nominal_return,inflation_rate\n" + observations,
        encoding="utf-8",
    )


def test_strict_historical_order_rejects_reordered_rows_deterministically(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "reordered.csv"
    _write_history(
        history_path,
        "2023,0.08,0.03\n2022,-0.04,0.05\n",
    )

    with pytest.raises(ValueError) as error:
        load_historical_series(
            history_path,
            order_policy=HistoricalOrderPolicy.REQUIRE_ASCENDING,
        )

    assert str(error.value) == (
        "historical years must be in strictly ascending order under order policy "
        "'require_ascending'; found 2023 followed by 2022 on line 3"
    )


def test_historical_duplicate_and_invalid_rows_fail_deterministically(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate.csv"
    _write_history(
        duplicate_path,
        "2022,0.08,0.03\n2022,-0.04,0.05\n",
    )
    with pytest.raises(ValueError) as duplicate_error:
        load_historical_series(duplicate_path)
    assert str(duplicate_error.value) == "duplicate historical year: 2022"

    invalid_path = tmp_path / "invalid.csv"
    _write_history(invalid_path, "2022,nan,0.03\n")
    with pytest.raises(ValueError) as invalid_error:
        load_historical_series(invalid_path)
    assert str(invalid_error.value) == "nominal_return must be finite on line 2"


def test_historical_gap_and_pairing_policies_are_explicit(tmp_path: Path) -> None:
    gapped_path = tmp_path / "gapped.csv"
    _write_history(
        gapped_path,
        "2020,0.08,0.03\n2022,-0.04,0.05\n",
    )
    with pytest.raises(
        ValueError,
        match="historical years must be contiguous under gap policy 'reject'",
    ):
        load_historical_series(gapped_path)

    series = load_historical_series(
        gapped_path,
        gap_policy=HistoricalGapPolicy.ALLOW,
    )
    assert series.years == (2020, 2022)
    assert series.gap_policy is HistoricalGapPolicy.ALLOW

    unpaired_path = tmp_path / "unpaired.csv"
    _write_history(
        unpaired_path,
        "2020,0.08,0.03\n2021,-0.04,\n",
    )
    with pytest.raises(ValueError) as pairing_error:
        load_historical_series(unpaired_path)
    assert str(pairing_error.value) == (
        "inflation_rate must be present for every historical observation; missing value on line 3"
    )


def test_history_can_be_shorter_than_the_simulation_horizon(tmp_path: Path) -> None:
    history_path = tmp_path / "short.csv"
    _write_history(
        history_path,
        "2021,0.10,0.02\n2022,-0.05,0.04\n",
    )
    series = load_historical_series(
        history_path,
        order_policy=HistoricalOrderPolicy.REQUIRE_ASCENDING,
    )
    scenario = _scenario(
        return_model="historical_bootstrap",
        historical_block_size=1,
    )

    result = simulate(
        scenario,
        100_000,
        historical_series=series,
    )

    historical = result.reproducibility["historical"]
    assert isinstance(historical, dict)
    assert historical["observation_count"] == 2
    assert historical["date_span"] == {
        "first_year": 2021,
        "last_year": 2022,
    }
    assert len(result.annual_balance_real) == 8


def test_result_has_a_versioned_serializable_reproducibility_contract() -> None:
    scenario = _scenario()
    provenance = ValuationProvenance(
        source="persisted_account_valuations",
        as_of=date(2026, 7, 28),
        account_ids=("retirement", "brokerage"),
        source_sha256="abc123",
    )
    policy = RunPolicy(
        maximum_working_bytes=32 * 1024 * 1024,
        in_memory_path_bytes=16 * 1024 * 1024,
        maximum_temporary_bytes=64 * 1024 * 1024,
        batch_size=17,
    )

    result = simulate(
        scenario,
        100_000,
        run_policy=policy,
        valuation_provenance=provenance,
    )
    payload = result.as_dict()
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    manifest = payload["reproducibility"]

    assert serialized
    assert isinstance(manifest, dict)
    assert manifest["schema_version"] == REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION
    assert manifest["result_schema_version"] == SIMULATION_RESULT_SCHEMA_VERSION
    assert manifest["engine_version"] == SIMULATION_ENGINE_VERSION
    assert manifest["scenario"] == {
        "canonicalization": "json_sort_keys_v1",
        "sha256": canonical_scenario_sha256(scenario),
    }
    assert manifest["random"] == {"seed": 42, "bit_generator": "PCG64"}
    assert manifest["valuation"] == {
        "source": "persisted_account_valuations",
        "as_of": "2026-07-28",
        "account_ids": ["retirement", "brokerage"],
        "source_sha256": "abc123",
    }
    assert manifest["run_policy"] == {
        "maximum_working_bytes": 32 * 1024 * 1024,
        "in_memory_path_bytes": 16 * 1024 * 1024,
        "maximum_temporary_bytes": 64 * 1024 * 1024,
        "process_maximum_working_bytes": 1024 * 1024 * 1024,
        "configured_batch_size": 17,
        "batch_size": 17,
        "storage": "memory",
        "evaluation_state_bytes": 5_600,
        "total_estimated_working_bytes": 15_008,
        "resource_estimate": {
            "retained_path_bytes": 5_664,
            "peak_working_bytes": 9_408,
            "memmap_peak_working_bytes": 3_744,
            "temporary_path_bytes": 5_664,
        },
    }
    assert manifest["historical"] is None
    assert manifest["bootstrap"] is None
    assert result.assumptions["scenario_sha256"] == canonical_scenario_sha256(scenario)
    assert result.engine["schema_version"] == 2
    outcome_semantics = manifest["outcome_semantics"]
    assert isinstance(outcome_semantics, dict)
    assert outcome_semantics["schema_version"] == 2
    assert "pre-outcome-semantics depletion definition" in str(outcome_semantics["success_rate"])
    assert {
        "scenario",
        "starting_portfolio",
        "trials",
        "seed",
        "success_rate",
        "success_rate_ci_95",
        "depleted_trials",
        "median_depletion_age",
        "retirement_balance_real",
        "ending_balance_real",
        "annual_balance_real",
        "assumptions",
        "engine",
    } < payload.keys()
