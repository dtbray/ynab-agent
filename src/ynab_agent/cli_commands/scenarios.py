"""Typer adapter for immutable wealth scenario revisions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from ynab_agent.cli_commands.wealth_support import load_history, load_scenario
from ynab_agent.cli_support import console, format_dollars, print_json, render_rows
from ynab_agent.config import settings
from ynab_agent.planning.stress import NamedStressName
from ynab_agent.runtime import open_scenario_comparison_service
from ynab_agent.services.scenario_comparison import (
    ScenarioComparisonError,
    historical_snapshot_from_series,
)


scenario_app = typer.Typer(
    help="Save and compare immutable wealth scenario revisions",
)


def _optional_dollars(value: float | None) -> str:
    return format_dollars(value) if value is not None else ""


@scenario_app.command("save")
def save_scenario_revision(
    scenario_path: Annotated[
        Path,
        typer.Option(
            "--scenario",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to a private wealth scenario JSON file",
        ),
    ],
    returns_path: Annotated[
        Path | None,
        typer.Option(
            "--returns",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Historical returns CSV required by historical_bootstrap scenarios",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the immutable revision as JSON"),
    ] = False,
) -> None:
    """Resolve live account values and append an immutable revision."""
    scenario = load_scenario(scenario_path)
    history = load_history(scenario, returns_path)
    historical_dataset = (
        historical_snapshot_from_series(
            history,
            dataset_id=f"local-{history.observations_sha256[:16]}",
        )
        if history is not None
        else None
    )

    async def run() -> None:
        try:
            async with open_scenario_comparison_service(settings) as service:
                revision = await service.save_revision(
                    scenario,
                    historical_dataset=historical_dataset,
                )
        except (RuntimeError, ValueError, ScenarioComparisonError) as exc:
            raise typer.BadParameter(
                str(exc),
                param_hint="--scenario",
            ) from exc
        if json_output:
            print_json(revision.model_dump(mode="json"))
            return
        render_rows(
            [
                {
                    "id": revision.id,
                    "scenario": revision.scenario_name,
                    "revision": revision.revision,
                    "manifest": revision.manifest_sha256[:12],
                    "created": revision.created_at.isoformat(),
                }
            ],
            [
                ("id", "Revision ID"),
                ("scenario", "Scenario"),
                ("revision", "Revision"),
                ("manifest", "Manifest"),
                ("created", "Created"),
            ],
        )

    asyncio.run(run())


@scenario_app.command("compare")
def compare_scenario_revisions(
    baseline_revision_id: Annotated[
        str,
        typer.Option(
            "--baseline",
            help="Saved revision UUID used as the comparison baseline",
        ),
    ],
    alternative_revision_ids: Annotated[
        list[str],
        typer.Option(
            "--alternative",
            help="Alternative revision UUID; repeat for multiple alternatives",
        ),
    ],
    named_stress: Annotated[
        NamedStressName | None,
        typer.Option(
            "--named-stress",
            help="Apply one registered deterministic market stress",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the stable comparison model as JSON"),
    ] = False,
) -> None:
    """Compare saved revisions against identical economic return paths."""

    async def run() -> None:
        try:
            async with open_scenario_comparison_service(settings) as service:
                comparison = await service.compare(
                    baseline_revision_id=baseline_revision_id,
                    alternative_revision_ids=alternative_revision_ids,
                    named_stress=named_stress,
                )
        except (RuntimeError, ValueError, ScenarioComparisonError) as exc:
            raise typer.BadParameter(
                str(exc),
                param_hint="--alternative",
            ) from exc
        if json_output:
            print_json(comparison.model_dump(mode="json"))
            return
        render_rows(
            [
                {
                    "scenario": alternative.scenario_name,
                    "revision": alternative.revision,
                    "success_delta": (
                        f"{alternative.delta.success_probability:+.1%}"
                    ),
                    "spending_delta": format_dollars(
                        alternative.delta.requested_annual_spending_real
                    ),
                    "tax_delta": (
                        format_dollars(alternative.delta.lifetime_tax_real_p50)
                        if alternative.delta.lifetime_tax_real_p50 is not None
                        else ""
                    ),
                    "estate_p50_delta": format_dollars(
                        alternative.delta.estate_value_real_p50
                    ),
                    "estate_p10_delta": format_dollars(
                        alternative.delta.estate_value_real_p10
                    ),
                    "dominated": (
                        ", ".join(alternative.dominated_by_revision_ids)
                    ),
                }
                for alternative in comparison.alternatives
            ],
            [
                ("scenario", "Alternative"),
                ("revision", "Revision"),
                ("success_delta", "Success Δ"),
                ("spending_delta", "Requested Annual Δ"),
                ("tax_delta", "Tax Δ"),
                ("estate_p50_delta", "Estate P50 Δ"),
                ("estate_p10_delta", "Estate P10 Δ"),
                ("dominated", "Dominated By"),
            ],
        )
        render_rows(
            [
                {
                    "scenario": alternative.scenario_name,
                    "funded_delta": _optional_dollars(
                        alternative.delta.funded_spending_real_p50
                    ),
                    "shortfall_delta": _optional_dollars(
                        alternative.delta.cumulative_shortfall_real_p50
                    ),
                    "guardrail_delta": _optional_dollars(
                        alternative.delta.guardrail_reduction_real_p50
                    ),
                    "after_tax_estate_p50_delta": _optional_dollars(
                        alternative.delta.after_tax_estate_value_real_p50
                    ),
                    "after_tax_estate_p10_delta": _optional_dollars(
                        alternative.delta.after_tax_estate_value_real_p10
                    ),
                }
                for alternative in comparison.alternatives
            ],
            [
                ("scenario", "Alternative"),
                ("funded_delta", "Funded P50 Δ"),
                ("shortfall_delta", "Shortfall P50 Δ"),
                ("guardrail_delta", "Guardrail Cuts P50 Δ"),
                ("after_tax_estate_p50_delta", "After-Tax Estate P50 Δ"),
                ("after_tax_estate_p10_delta", "After-Tax Estate P10 Δ"),
            ],
        )
        render_rows(
            [
                {
                    "scenario": alternative.scenario_name,
                    "irmaa_delta": _optional_dollars(
                        alternative.delta.lifetime_irmaa_surcharge_real_p50
                    ),
                    "irmaa_exposure_delta": (
                        f"{alternative.delta.irmaa_exposure_probability:+.1%}"
                        if alternative.delta.irmaa_exposure_probability is not None
                        else ""
                    ),
                }
                for alternative in comparison.alternatives
            ],
            [
                ("scenario", "Alternative"),
                ("irmaa_delta", "IRMAA P50 Δ"),
                ("irmaa_exposure_delta", "IRMAA Exposure Δ"),
            ],
        )
        console.print(
            f"Comparison {comparison.id}; manifest "
            f"{comparison.manifest_sha256[:12]}"
        )

    asyncio.run(run())
