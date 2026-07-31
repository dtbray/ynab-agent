"""Replayable calibration simulation and deterministic model attribution."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ynab_agent.planning.historical import HistoricalSeries
from ynab_agent.planning.models import WealthScenario
from ynab_agent.planning.simulation import SimulationResult, simulate
from ynab_agent.services.calibration_attribution import (
    AttributionMethod,
    AttributionMethodManifest,
    MAX_ATTRIBUTION_FACTORS,
    ModelAttributionContribution,
    OutcomeMetricValues,
    build_attribution_report,
    metric_delta,
)
from ynab_agent.services.calibration_models import (
    CalibrationSnapshot,
    DriftEvent,
    DriftKind,
    canonical_content_sha256,
)

MAX_CALIBRATION_ATTRIBUTION_FACTORS = MAX_ATTRIBUTION_FACTORS
MAX_CALIBRATION_COMPUTE_UNITS = 200_000_000


class CalibrationResourceLimitError(ValueError):
    """Snapshot is valid but too expensive for unattended execution."""


def run_calibration_snapshot(payload_json: str) -> str:
    """Execute one immutable snapshot in a spawn-safe worker process."""
    snapshot = CalibrationSnapshot.model_validate_json(payload_json)
    baseline_scenario = snapshot.manifest.resolved_scenario.to_wealth_scenario()
    factor_groups = group_calibration_attribution_events(
        snapshot.manifest.drift_events
    )
    enforce_calibration_execution_resources(
        baseline_scenario,
        len(factor_groups),
    )
    historical_series = (
        snapshot.manifest.historical_dataset.to_series()
        if snapshot.manifest.historical_dataset is not None
        else None
    )
    baseline = _simulate_metrics(baseline_scenario, historical_series=historical_series)
    events = snapshot.manifest.drift_events
    factor_scenarios = tuple(
        _apply_drifts(baseline_scenario, factor_events)
        for _, factor_events in factor_groups
    )
    current_scenario = baseline_scenario
    for event in events:
        current_scenario = _apply_drift(current_scenario, event)
    current = _simulate_metrics(current_scenario, historical_series=historical_series)
    contributions = tuple(
        ModelAttributionContribution(
            factor_id=f"drift-group:{kind}",
            label=f"{kind.replace('_', ' ')} calibration context",
            drift_event_sha256=tuple(event.content_sha256 for event in factor_events),
            marginal_delta=metric_delta(
                baseline,
                _simulate_metrics(
                    factor_scenario,
                    historical_series=historical_series,
                ),
            ),
            evaluation_count=1,
            confidence=min(
                (event.confidence for event in factor_events),
                key=lambda row: row.score,
            ),
        )
        for (kind, factor_events), factor_scenario in zip(
            factor_groups,
            factor_scenarios,
            strict=True,
        )
    )
    snapshot_hash = canonical_content_sha256(snapshot.manifest)
    baseline_hash = canonical_content_sha256(baseline_scenario)
    current_hash = canonical_content_sha256(current_scenario)
    method = AttributionMethodManifest(
        method=AttributionMethod.ONE_FACTOR_COUNTERFACTUAL,
        factor_ids=tuple(row.factor_id for row in contributions),
        calibration_snapshot_manifest_sha256=snapshot_hash,
        drift_event_sha256=tuple(event.content_sha256 for event in events),
        baseline_revision_manifest_sha256=baseline_hash,
        current_revision_manifest_sha256=current_hash,
        comparison_manifest_sha256=canonical_content_sha256(
            {
                "baseline": baseline_hash,
                "current": current_hash,
                "factors": tuple(row.factor_id for row in contributions),
            }
        ),
        counterfactual_definition=(
            "Each marginal is the seeded model result after applying one bounded "
            "kind-group of copied drift events to the immutable baseline; "
            "observational-only context is grouped with a zero direct model effect. "
            "The interaction residual reconciles the group sum to the combined result."
        ),
    )
    attribution = build_attribution_report(
        baseline=baseline,
        current=current,
        contributions=contributions,
        method=method,
        calibration_snapshot_manifest=snapshot.manifest,
    )
    result = {
        "schema_version": 1,
        "snapshot_id": snapshot.id,
        "snapshot_manifest_sha256": snapshot.manifest_sha256,
        "baseline": baseline.model_dump(mode="json"),
        "current": current.model_dump(mode="json"),
        "calibrated_scenario": current_scenario.model_dump(mode="json"),
        "calibrated_scenario_sha256": current_hash,
        "attribution": attribution.model_dump(mode="json"),
    }
    return json.dumps(
        result,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def enforce_calibration_execution_resources(
    scenario: WealthScenario,
    factor_count: int,
) -> None:
    if factor_count > MAX_CALIBRATION_ATTRIBUTION_FACTORS:
        raise CalibrationResourceLimitError(
            "calibration drift factors exceed the unattended attribution limit"
        )
    years = scenario.end_age - scenario.current_age
    evaluations = factor_count + 2
    compute_units = scenario.trials * years * evaluations
    if compute_units > MAX_CALIBRATION_COMPUTE_UNITS:
        raise CalibrationResourceLimitError(
            "calibration attribution exceeds the unattended compute limit"
        )


def _simulate_metrics(
    scenario: WealthScenario,
    *,
    historical_series: HistoricalSeries | None,
) -> OutcomeMetricValues:
    if scenario.starting_portfolio is None:
        raise ValueError("calibration snapshot requires resolved starting_portfolio")
    result = simulate(
        scenario,
        scenario.starting_portfolio,
        historical_series=historical_series,
    )
    return _outcome_metrics(result)


def group_calibration_attribution_events(
    events: tuple[DriftEvent, ...],
) -> tuple[tuple[str, tuple[DriftEvent, ...]], ...]:
    grouped: dict[str, list[DriftEvent]] = {}
    driving = {
        DriftKind.SPENDING,
        DriftKind.CONTRIBUTION,
        DriftKind.BALANCE,
        DriftKind.ALLOCATION,
    }
    for event in events:
        key = (
            event.kind.value
            if event.kind in driving
            and event.material
            and not (
                event.kind is DriftKind.CONTRIBUTION
                and event.confidence.score < 1
            )
            else "observational"
        )
        grouped.setdefault(key, []).append(event)
    return tuple(
        (key, tuple(grouped[key]))
        for key in (
            DriftKind.SPENDING.value,
            DriftKind.CONTRIBUTION.value,
            DriftKind.BALANCE.value,
            DriftKind.ALLOCATION.value,
            "observational",
        )
        if key in grouped
    )


def _apply_drifts(
    scenario: WealthScenario,
    events: tuple[DriftEvent, ...],
) -> WealthScenario:
    current = scenario
    for event in events:
        current = _apply_drift(current, event)
    return current


def _outcome_metrics(result: SimulationResult) -> OutcomeMetricValues:
    estate = (
        result.after_tax_estate_value_real or result.estate_value_real or result.ending_balance_real
    )
    return OutcomeMetricValues(
        success_probability=result.success_rate,
        cumulative_shortfall_real_p50=result.cumulative_shortfall_real["p50"],
        lifetime_tax_real_p50=(
            result.lifetime_tax_real["p50"] if result.lifetime_tax_real is not None else None
        ),
        after_tax_estate_value_real_p50=estate["p50"],
    )


def _apply_drift(
    scenario: WealthScenario,
    event: DriftEvent,
) -> WealthScenario:
    if not event.material:
        return scenario
    data = scenario.model_dump(mode="python")
    observed = event.observed_value
    if event.kind is DriftKind.SPENDING and isinstance(observed, int | float):
        data["annual_spending"] = observed
    elif (
        event.kind is DriftKind.CONTRIBUTION
        and event.confidence.score >= 1
        and isinstance(observed, int | float)
    ):
        data["annual_contribution"] = max(0, observed)
    elif event.kind is DriftKind.BALANCE and isinstance(observed, int | float):
        if event.subject_id == "portfolio":
            data["starting_portfolio"] = max(0, observed)
            _apply_portfolio_weights(data, event.details.get("account_balances"))
        else:
            _apply_account_balance(data, event.subject_id, observed)
    elif event.kind is DriftKind.ALLOCATION:
        _apply_account_allocation(data, event.subject_id, observed)
    # Debt-payoff and staleness events intentionally explain a zero direct
    # planning-input marginal until the scenario grows a matching typed field.
    return WealthScenario.model_validate(data)


def _apply_portfolio_weights(
    scenario: dict[str, object],
    raw_balances: object,
) -> None:
    if not isinstance(raw_balances, Mapping):
        return
    balances = {
        str(account_id): max(0, float(balance))
        for account_id, balance in raw_balances.items()
        if not isinstance(balance, bool) and isinstance(balance, int | float)
    }
    total = sum(balances.values())
    allocation = scenario.get("portfolio_allocation")
    if total <= 0 or not isinstance(allocation, dict):
        return
    accounts = allocation.get("accounts")
    if not isinstance(accounts, list | tuple):
        return
    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_id = str(account.get("account_id"))
        if account_id in balances:
            account["portfolio_weight"] = balances[account_id] / total


def _apply_account_balance(
    scenario: dict[str, object],
    account_id: str,
    observed: int | float,
) -> None:
    tax_buckets = scenario.get("tax_buckets")
    if not isinstance(tax_buckets, list | tuple):
        return
    changed = False
    for bucket in tax_buckets:
        if isinstance(bucket, dict) and bucket.get("account_id") == account_id:
            bucket["starting_balance"] = max(0, observed)
            changed = True
            break
    if not changed:
        return
    balances = {
        str(bucket["account_id"]): float(bucket["starting_balance"])
        for bucket in tax_buckets
        if isinstance(bucket, dict)
        and bucket.get("account_id") is not None
        and isinstance(bucket.get("starting_balance"), int | float)
    }
    total = sum(balances.values())
    scenario["starting_portfolio"] = total
    allocation = scenario.get("portfolio_allocation")
    if not isinstance(allocation, dict) or total <= 0:
        return
    accounts = allocation.get("accounts")
    if not isinstance(accounts, list | tuple):
        return
    for account in accounts:
        if isinstance(account, dict):
            selected_id = str(account.get("account_id"))
            if selected_id in balances:
                account["portfolio_weight"] = balances[selected_id] / total


def _apply_account_allocation(
    scenario: dict[str, object],
    account_id: str,
    observed: object,
) -> None:
    if not isinstance(observed, Mapping):
        return
    allocation = scenario.get("portfolio_allocation")
    if not isinstance(allocation, dict):
        return
    accounts = allocation.get("accounts")
    if not isinstance(accounts, list | tuple):
        return
    for account in accounts:
        if isinstance(account, dict) and account.get("account_id") == account_id:
            account["target"] = dict(observed)
            return
