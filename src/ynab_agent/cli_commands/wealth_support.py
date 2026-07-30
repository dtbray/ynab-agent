"""Shared parsing helpers for wealth-planning CLI adapters."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import typer

from ynab_agent.planning.historical import (
    HistoricalOrderPolicy,
    HistoricalSeries,
    load_historical_series,
)
from ynab_agent.planning.models import ReturnModel, WealthScenario


def load_scenario(scenario_path: Path) -> WealthScenario:
    """Load one private scenario file with a CLI-specific error."""
    try:
        return WealthScenario.model_validate_json(
            scenario_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            f"invalid scenario file: {exc}",
            param_hint="--scenario",
        ) from exc


def load_history(
    scenario: WealthScenario,
    returns_path: Path | None,
) -> HistoricalSeries | None:
    """Load required historical observations without leaking file paths."""
    if (
        scenario.return_model is ReturnModel.HISTORICAL_BOOTSTRAP
        and returns_path is None
    ):
        raise typer.BadParameter(
            "--returns is required for a historical_bootstrap scenario",
            param_hint="--returns",
        )
    if returns_path is None:
        return None
    try:
        return load_historical_series(
            returns_path,
            order_policy=HistoricalOrderPolicy.REQUIRE_ASCENDING,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--returns") from exc
