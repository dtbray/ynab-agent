"""Typer validation commands for multi-asset portfolio allocations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import ValidationError
import typer

from ynab_agent.cli_support import print_json, render_rows
from ynab_agent.planning.allocation import (
    PortfolioAllocationPlan,
    validate_portfolio_allocation,
)
from ynab_agent.planning.stress import named_stress_catalog


allocation_app = typer.Typer(
    help="Validate multi-asset market and account allocation assumptions",
)


@allocation_app.command("stresses")
def list_stresses(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete catalog as JSON"),
    ] = False,
) -> None:
    """List bounded named stresses accepted by simulation surfaces."""
    catalog = named_stress_catalog()
    payload = [
        definition.model_dump(mode="json")
        for definition in catalog
    ]
    if json_output:
        print_json(payload)
        return
    render_rows(
        [
            {
                "name": definition.name.value,
                "title": definition.title,
                "years": len(definition.annual_returns),
                "sha256": definition.content_sha256,
            }
            for definition in catalog
        ],
        [
            ("name", "Selector"),
            ("title", "Stress"),
            ("years", "Years"),
            ("sha256", "Definition SHA-256"),
        ],
    )


@allocation_app.command("validate")
def validate_allocation(
    plan_path: Annotated[
        Path,
        typer.Option(
            "--plan",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Portfolio allocation plan JSON",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the canonical JSON representation"),
    ] = False,
) -> None:
    """Validate covariance, account weights, fees, and glide paths."""
    try:
        plan = PortfolioAllocationPlan.model_validate_json(
            plan_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            f"invalid allocation plan: {exc}",
            param_hint="--plan",
        ) from exc
    validation = validate_portfolio_allocation(plan)
    payload = validation.model_dump(mode="json")
    if json_output:
        print_json(payload)
        return
    render_rows(
        [
            {
                "valid": validation.valid,
                "accounts": len(plan.accounts),
                "assets": 4,
                "rebalancing_years": (
                    plan.rebalancing.frequency_years or "disabled"
                ),
                "strategy": validation.manifest.strategy_identity,
                "sha256": validation.manifest.plan_sha256,
            }
        ],
        [
            ("valid", "Valid"),
            ("accounts", "Accounts"),
            ("assets", "Assets"),
            ("rebalancing_years", "Rebalance Every"),
            ("strategy", "Strategy"),
            ("sha256", "Plan SHA-256"),
        ],
    )
