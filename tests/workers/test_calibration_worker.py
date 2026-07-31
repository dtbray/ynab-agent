from __future__ import annotations

import pytest

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.calibration_execution import (
    _apply_drift,
    CalibrationResourceLimitError,
    MAX_CALIBRATION_ATTRIBUTION_FACTORS,
    enforce_calibration_execution_resources,
    group_calibration_attribution_events,
)
from ynab_agent.services.calibration_drift import (
    ScalarDriftInput,
    detect_balance_drift,
)
from ynab_agent.services.calibration_models import (
    ConfidenceAssessment,
    MaterialityThreshold,
    ObservationUnit,
)


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "calibration resource fixture",
        "current_age": 30,
        "retirement_age": 65,
        "end_age": 95,
        "starting_portfolio": 250_000,
        "annual_spending": 48_000,
        "trials": 100,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


def test_calibration_execution_admits_bounded_work_and_rejects_factor_fanout() -> None:
    enforce_calibration_execution_resources(_scenario(), factor_count=5)

    with pytest.raises(
        CalibrationResourceLimitError,
        match="drift factors",
    ):
        enforce_calibration_execution_resources(
            _scenario(),
            factor_count=MAX_CALIBRATION_ATTRIBUTION_FACTORS + 1,
        )


def test_calibration_execution_rejects_compute_before_allocating_paths() -> None:
    with pytest.raises(
        CalibrationResourceLimitError,
        match="compute limit",
    ):
        enforce_calibration_execution_resources(
            _scenario(trials=100_000, current_age=18, end_age=130),
            factor_count=MAX_CALIBRATION_ATTRIBUTION_FACTORS,
        )


def test_many_account_events_are_grouped_before_resource_admission() -> None:
    events = tuple(
        detect_balance_drift(
            ScalarDriftInput(
                subject_id=f"account-{index}",
                baseline_value=100,
                observed_value=101 + index,
                unit=ObservationUnit.DOLLARS,
                threshold=MaterialityThreshold(absolute=1),
                confidence=ConfidenceAssessment(score=1, basis="test evidence"),
                observation_ids=(f"baseline-{index}", f"current-{index}"),
            )
        )
        for index in range(32)
    )

    groups = group_calibration_attribution_events(events)

    assert len(groups) == 1
    assert groups[0][0] == "balance"
    assert groups[0][1] == events
    enforce_calibration_execution_resources(
        _scenario(),
        factor_count=len(groups),
    )


def test_immaterial_event_is_evidence_only_and_does_not_churn_scenario() -> None:
    event = detect_balance_drift(
        ScalarDriftInput(
            subject_id="portfolio",
            baseline_value=250_000,
            observed_value=250_001,
            unit=ObservationUnit.DOLLARS,
            threshold=MaterialityThreshold(absolute=2_500, relative=0.05),
            confidence=ConfidenceAssessment(score=1, basis="test evidence"),
            observation_ids=("baseline", "current"),
        )
    )
    scenario = _scenario()

    assert event.material is False
    assert group_calibration_attribution_events((event,))[0][0] == "observational"
    assert _apply_drift(scenario, event) is scenario
