"""Housing decision projection commands."""

from pathlib import Path
from typing import Annotated, cast
from collections.abc import Mapping, Sequence

import typer

from ynab_agent.cli_commands.wealth_support import load_scenario
from ynab_agent.cli_support import print_json, render_rows
from ynab_agent.planning.housing import project_housing_plan


housing_app = typer.Typer(help="Validate and audit illiquid housing decisions")


@housing_app.command("project")
def project(
    scenario_path: Annotated[
        Path,
        typer.Option("--scenario", exists=True, dir_okay=False, readable=True),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Project the deterministic housing ledger embedded in a scenario."""
    projection = project_housing_plan(load_scenario(scenario_path))
    if json_output:
        print_json(projection)
        return
    render_rows(
        cast(Sequence[Mapping[str, object]], projection["annual_housing"]),
        [
            ("age", "Age"),
            ("action", "Action"),
            ("ending_home_value", "Home"),
            ("ending_home_equity", "Equity"),
            ("portfolio_spending", "Portfolio cost"),
            ("liquid_deposit", "Liquid deposit"),
        ],
    )
