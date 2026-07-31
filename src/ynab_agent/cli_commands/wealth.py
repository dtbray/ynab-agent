"""Wealth-planning Typer commands."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import ValidationError
import typer

from ynab_agent.cli_commands.scenarios import scenario_app
from ynab_agent.cli_commands.wealth_support import load_history, load_scenario
from ynab_agent.cli_support import (
    console,
    format_dollars,
    format_milliunits,
    print_json,
    render_rows,
)
from ynab_agent.config import settings
from ynab_agent.planning.models import TaxTreatment, WealthScenario
from ynab_agent.planning.progressive_tax import (
    TaxCalculationInput,
    calculate_annual_tax,
)
from ynab_agent.planning.reporting import write_simulation_html
from ynab_agent.planning.simulation import SimulationResult, simulate
from ynab_agent.planning.social_security_optimizer import optimize_social_security
from ynab_agent.planning.solver import SolveVariable, solve_scenario
from ynab_agent.planning.stress import NamedStressName
from ynab_agent.runtime import open_wealth_service
from ynab_agent.services.wealth import ResolvedStartingPortfolio


wealth_app = typer.Typer(
    help="Build retirement scenarios from selected YNAB balances",
)
wealth_app.add_typer(scenario_app, name="scenarios")


def _load_tax_request(request_path: Path) -> TaxCalculationInput:
    try:
        return TaxCalculationInput.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            f"invalid tax request file: {exc}",
            param_hint="--input",
        ) from exc


def _action_p50(action: dict[str, object], field: str) -> float:
    value = action.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"tax-strategy action is missing {field}")
    percentile = value.get("p50")
    if not isinstance(percentile, (float, int)):
        raise ValueError(f"tax-strategy action has invalid {field}")
    return float(percentile)


def _withdrawal_p50(
    action: dict[str, object],
    treatment: TaxTreatment,
) -> float:
    withdrawals = action.get("withdrawals_nominal")
    if not isinstance(withdrawals, dict):
        raise ValueError("tax-strategy action is missing withdrawals_nominal")
    value = withdrawals.get(treatment.value)
    if not isinstance(value, dict):
        return 0.0
    percentile = value.get("p50")
    if not isinstance(percentile, (float, int)):
        raise ValueError(f"tax-strategy action has invalid {treatment.value} withdrawal")
    return float(percentile)


async def _starting_portfolio(
    scenario: WealthScenario,
) -> ResolvedStartingPortfolio:
    async with open_wealth_service(settings) as service:
        return await service.resolve_starting_portfolio_with_provenance(scenario)


def _render_goal_outcomes(result: SimulationResult) -> None:
    render_rows(
        [
            {
                "goal": goal.name,
                "kind": goal.kind.value,
                "target_real": format_dollars(goal.target_real),
                "attainment": (
                    f"{goal.attainment_probability:.1%}"
                    if goal.attainment_probability is not None
                    else "not evaluated"
                ),
                "basis": goal.evaluation_basis,
            }
            for goal in result.goal_outcomes
        ],
        [
            ("goal", "Goal"),
            ("kind", "Kind"),
            ("target_real", "Target (real)"),
            ("attainment", "Attainment"),
            ("basis", "Evaluation Basis"),
        ],
    )


@wealth_app.command("accounts")
def accounts(
    tracking_only: Annotated[
        bool,
        typer.Option(
            "--tracking-only/--all",
            help="Show only off-budget tracking accounts",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """List active accounts and balance freshness for scenario classification."""

    async def run() -> None:
        async with open_wealth_service(settings) as service:
            rows = await service.list_account_freshness(
                tracking_only=tracking_only
            )

        today = date.today()
        display_rows: list[dict[str, object]] = []
        for row in rows:
            latest_date = (
                date.fromisoformat(row.latest_transaction[:10])
                if row.latest_transaction
                else None
            )
            display_rows.append(
                {
                    "account_id": row.account.id,
                    "name": row.account.name,
                    "type": row.account.type,
                    "balance": format_milliunits(row.account.balance_milliunits),
                    "latest_transaction": latest_date.isoformat() if latest_date else "",
                    "stale_days": (today - latest_date).days if latest_date else "",
                    "role": "unassigned",
                }
            )

        if json_output:
            print_json(display_rows)
            return
        render_rows(
            display_rows,
            [
                ("account_id", "Account ID"),
                ("name", "Account"),
                ("type", "Type"),
                ("balance", "Balance"),
                ("latest_transaction", "Latest Transaction"),
                ("stale_days", "Stale Days"),
                ("role", "Scenario Role"),
            ],
        )

    asyncio.run(run())


@wealth_app.command("tax")
def tax_command(
    request_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to an annual tax calculation JSON file",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete auditable calculation"),
    ] = False,
) -> None:
    """Calculate versioned federal and Indiana income tax for one year."""
    result = calculate_annual_tax(_load_tax_request(request_path))
    if json_output:
        print_json(result.model_dump(mode="json"))
        return
    render_rows(
        [
            {
                "tax_year": result.tax_year,
                "federal": format_dollars(result.federal_income_tax),
                "indiana": format_dollars(result.indiana_income_tax),
                "total": format_dollars(result.total_income_tax),
                "effective_rate": f"{result.effective_income_tax_rate:.2%}",
                "marginal_ordinary": (
                    f"{result.marginal_ordinary_income_tax_rate:.2%}"
                ),
                "marginal_ltcg": (
                    f"{result.marginal_long_term_capital_gains_tax_rate:.2%}"
                ),
                "policy": result.policy_manifest.policy_id,
            }
        ],
        [
            ("tax_year", "Tax Year"),
            ("federal", "Federal"),
            ("indiana", "Indiana"),
            ("total", "Total"),
            ("effective_rate", "Effective Rate"),
            ("marginal_ordinary", "Marginal Ordinary"),
            ("marginal_ltcg", "Marginal LTCG"),
            ("policy", "Policy"),
        ],
    )


@wealth_app.command("simulate")
def simulate_command(
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
            help="CSV with year, nominal_return, and optional inflation_rate columns",
        ),
    ] = None,
    named_stress: Annotated[
        NamedStressName | None,
        typer.Option(
            "--stress",
            help="Run one bounded named multi-asset stress instead of random paths",
        ),
    ] = None,
    html_path: Annotated[
        Path | None,
        typer.Option(
            "--html",
            dir_okay=False,
            help="Write a self-contained interactive fan chart",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a summary"),
    ] = False,
) -> None:
    """Run a seeded parametric or historical retirement simulation."""
    scenario = load_scenario(scenario_path)
    history = load_history(scenario, returns_path, named_stress)

    async def run() -> None:
        try:
            starting_portfolio = await _starting_portfolio(scenario)
            result = simulate(
                scenario,
                starting_portfolio.value,
                historical_series=history,
                valuation_provenance=starting_portfolio.provenance,
                named_stress=named_stress,
            )
            written_html = (
                write_simulation_html(result, html_path)
                if html_path is not None
                else None
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--scenario") from exc

        if json_output:
            payload = result.as_dict()
            if written_html is not None:
                payload["html_path"] = str(written_html)
            print_json(payload)
            return

        render_rows(
            [
                {
                    "scenario": result.scenario,
                    "starting_portfolio": format_dollars(result.starting_portfolio),
                    "success_rate": f"{result.success_rate:.1%}",
                    "success_ci_95": (
                        f"{result.success_rate_ci_95['low']:.1%}–"
                        f"{result.success_rate_ci_95['high']:.1%}"
                    ),
                    "retirement_p50_real": format_dollars(
                        result.retirement_balance_real["p50"]
                    ),
                    "ending_p10_real": format_dollars(
                        result.ending_balance_real["p10"]
                    ),
                    "ending_p50_real": format_dollars(
                        result.ending_balance_real["p50"]
                    ),
                    "ending_p90_real": format_dollars(
                        result.ending_balance_real["p90"]
                    ),
                    "funded_spending_p50": (
                        f"{result.funded_spending_ratio['p50']:.1%}"
                    ),
                    "shortfall_p50_real": format_dollars(
                        result.cumulative_shortfall_real["p50"]
                    ),
                    "failure_years_p50": (
                        f"{result.failure_duration_years['p50']:.1f}"
                        if result.failure_duration_years is not None
                        else ""
                    ),
                    "recovery_probability": (
                        f"{result.recovery_probability:.1%}"
                        if result.recovery_probability is not None
                        else ""
                    ),
                    "median_depletion_age": (
                        format_dollars(result.median_depletion_age)
                        if result.median_depletion_age is not None
                        else ""
                    ),
                }
            ],
            [
                ("scenario", "Scenario"),
                ("starting_portfolio", "Starting Portfolio"),
                ("success_rate", "Success"),
                ("success_ci_95", "95% CI"),
                ("retirement_p50_real", "Retirement P50 (real)"),
                ("ending_p10_real", "Ending P10 (real)"),
                ("ending_p50_real", "Ending P50 (real)"),
                ("ending_p90_real", "Ending P90 (real)"),
                ("funded_spending_p50", "Funded Spending P50"),
                ("shortfall_p50_real", "Cumulative Shortfall P50 (real)"),
                ("failure_years_p50", "Failure Duration P50 (years)"),
                ("recovery_probability", "Recovery After Failure"),
                ("median_depletion_age", "Median Depletion Age"),
            ],
        )
        _render_goal_outcomes(result)
        if written_html is not None:
            console.print(f"Interactive report: {written_html}")

    asyncio.run(run())


@wealth_app.command("tax-strategy")
def tax_strategy_command(
    scenario_path: Annotated[
        Path,
        typer.Option(
            "--scenario",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Private scenario containing versioned tax strategy assumptions",
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
        typer.Option("--json", help="Emit the complete replayable simulation result"),
    ] = False,
) -> None:
    """Evaluate and print a deterministic annual tax-strategy schedule."""
    scenario = load_scenario(scenario_path)
    if scenario.tax_assumptions is None or scenario.tax_assumptions.strategy is None:
        raise typer.BadParameter(
            "scenario must configure tax_assumptions.strategy",
            param_hint="--scenario",
        )
    history = load_history(scenario, returns_path)

    async def run() -> None:
        try:
            starting_portfolio = await _starting_portfolio(scenario)
            result = simulate(
                scenario,
                starting_portfolio.value,
                historical_series=history,
                valuation_provenance=starting_portfolio.provenance,
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--scenario") from exc
        if json_output:
            print_json(result.as_dict())
            return
        render_rows(
            [
                {
                    "year": action["tax_year"],
                    "age": action["age"],
                    "conversion": format_dollars(
                        _action_p50(action, "roth_conversion_nominal")
                    ),
                    "gain_harvest": format_dollars(
                        _action_p50(
                            action,
                            "harvested_long_term_capital_gains_nominal",
                        )
                    ),
                }
                for action in result.annual_tax_strategy_actions
            ],
            [
                ("year", "Tax Year"),
                ("age", "Age"),
                ("conversion", "Roth Conversion P50"),
                ("gain_harvest", "Gain Harvest P50"),
            ],
        )
        render_rows(
            [
                {
                    "year": action["tax_year"],
                    "age": action["age"],
                    "tax_treatment": treatment.value,
                    "withdrawal": format_dollars(
                        _withdrawal_p50(action, treatment)
                    ),
                }
                for action in result.annual_tax_strategy_actions
                for treatment in TaxTreatment
            ],
            [
                ("year", "Tax Year"),
                ("age", "Age"),
                ("tax_treatment", "Tax Treatment"),
                ("withdrawal", "Withdrawal P50"),
            ],
        )
        render_rows(
            [
                {
                    "lifetime_tax": (
                        format_dollars(result.lifetime_tax_real["p50"])
                        if result.lifetime_tax_real is not None
                        else ""
                    ),
                    "irmaa": (
                        format_dollars(result.lifetime_irmaa_surcharge_real["p50"])
                        if result.lifetime_irmaa_surcharge_real is not None
                        else ""
                    ),
                    "irmaa_exposure": (
                        f"{result.irmaa_exposure_probability:.1%}"
                        if result.irmaa_exposure_probability is not None
                        else ""
                    ),
                    "funded_spending": f"{result.funded_spending_ratio['p50']:.1%}",
                    "shortfall": format_dollars(
                        result.cumulative_shortfall_real["p50"]
                    ),
                    "after_tax_estate": (
                        format_dollars(result.after_tax_ending_balance_real["p50"])
                        if result.after_tax_ending_balance_real is not None
                        else ""
                    ),
                }
            ],
            [
                ("lifetime_tax", "Lifetime Tax P50"),
                ("irmaa", "IRMAA P50"),
                ("irmaa_exposure", "IRMAA Exposure"),
                ("funded_spending", "Funded Spending P50"),
                ("shortfall", "Shortfall P50"),
                ("after_tax_estate", "After-Tax Estate P50"),
            ],
        )

    asyncio.run(run())


@wealth_app.command("optimize-social-security")
def optimize_social_security_command(
    scenario_path: Annotated[
        Path,
        typer.Option(
            "--scenario",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to a private household wealth scenario JSON file",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete ranked comparison"),
    ] = False,
) -> None:
    """Compare ages 62–70 on household portfolio outcomes."""
    scenario = load_scenario(scenario_path)

    async def run() -> None:
        try:
            starting_portfolio = await _starting_portfolio(scenario)
            result = optimize_social_security(
                scenario,
                starting_portfolio.value,
                valuation_provenance=starting_portfolio.provenance,
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--scenario") from exc
        if json_output:
            print_json(result.model_dump(mode="json"))
            return
        render_rows(
            [
                {
                    "person": person_id,
                    "claim_age": claim_age,
                    "success": f"{result.recommended.success_rate:.1%}",
                    "funded_spending_p50": (
                        f"{result.recommended.funded_spending_ratio_p50:.1%}"
                    ),
                    "ending_portfolio_p50": format_dollars(
                        result.recommended.tax_adjusted_ending_portfolio_p50
                    ),
                }
                for person_id, claim_age in (
                    result.recommended.claim_age_by_person.items()
                )
            ],
            [
                ("person", "Person"),
                ("claim_age", "Claim Age"),
                ("success", "Success"),
                ("funded_spending_p50", "Funded Spending p50"),
                ("ending_portfolio_p50", "After-tax Ending p50"),
            ],
        )

    asyncio.run(run())


@wealth_app.command("solve")
def solve_command(
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
    variable: Annotated[
        SolveVariable,
        typer.Option("--for", help="Scenario value to solve for"),
    ],
    lower: Annotated[float, typer.Option("--lower", help="Lower search bound")],
    upper: Annotated[float, typer.Option("--upper", help="Upper search bound")],
    target_success_rate: Annotated[
        float,
        typer.Option(
            "--target-success",
            min=0.001,
            max=0.999,
            help="Required probability of funding every planned withdrawal",
        ),
    ] = 0.9,
    resolution: Annotated[
        float,
        typer.Option(
            "--resolution",
            min=0.01,
            help="Dollar increment for money-variable searches",
        ),
    ] = 100.0,
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
    named_stress: Annotated[
        NamedStressName | None,
        typer.Option(
            "--stress",
            help="Solve against one bounded named multi-asset stress",
        ),
    ] = None,
    html_path: Annotated[
        Path | None,
        typer.Option(
            "--html",
            dir_okay=False,
            help="Write the solved simulation as an interactive fan chart",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Solve a spending, contribution, portfolio, or retirement-age threshold."""
    scenario = load_scenario(scenario_path)
    history = load_history(scenario, returns_path, named_stress)

    async def run() -> None:
        try:
            starting_portfolio = await _starting_portfolio(scenario)
            result = solve_scenario(
                scenario,
                starting_portfolio.value,
                variable=variable,
                lower=lower,
                upper=upper,
                target_success_rate=target_success_rate,
                resolution=resolution,
                historical_series=history,
                valuation_provenance=starting_portfolio.provenance,
                named_stress=named_stress,
            )
            written_html = (
                write_simulation_html(result.simulation, html_path)
                if html_path is not None
                else None
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--for") from exc

        if json_output:
            payload = result.as_dict()
            if written_html is not None:
                payload["html_path"] = str(written_html)
            print_json(payload)
            return

        render_rows(
            [
                {
                    "variable": result.variable,
                    "value": (
                        str(int(result.value))
                        if variable is SolveVariable.RETIREMENT_AGE
                        else format_dollars(result.value)
                    ),
                    "target": f"{result.target_success_rate:.1%}",
                    "achieved": f"{result.achieved_success_rate:.1%}",
                    "success_ci_95": (
                        f"{result.simulation.success_rate_ci_95['low']:.1%}–"
                        f"{result.simulation.success_rate_ci_95['high']:.1%}"
                    ),
                    "ending_p50_real": format_dollars(
                        result.simulation.ending_balance_real["p50"]
                    ),
                    "funded_spending_p50": (
                        f"{result.simulation.funded_spending_ratio['p50']:.1%}"
                    ),
                    "shortfall_p50_real": format_dollars(
                        result.simulation.cumulative_shortfall_real["p50"]
                    ),
                }
            ],
            [
                ("variable", "Solved Variable"),
                ("value", "Value"),
                ("target", "Target Success"),
                ("achieved", "Achieved"),
                ("success_ci_95", "95% CI"),
                ("ending_p50_real", "Ending P50 (real)"),
                ("funded_spending_p50", "Funded Spending P50"),
                ("shortfall_p50_real", "Cumulative Shortfall P50 (real)"),
            ],
        )
        _render_goal_outcomes(result.simulation)
        if written_html is not None:
            console.print(f"Interactive report: {written_html}")

    asyncio.run(run())


@wealth_app.command("performance")
def performance(
    account_ids: Annotated[
        list[str],
        typer.Option(
            "--account-id",
            help="Investment tracking account ID; repeat for a combined return",
        ),
    ],
    valuation_date: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            help="Date assigned to the cached ending balance",
        ),
    ] = None,
    ending_balance: Annotated[
        float | None,
        typer.Option(
            "--ending-balance",
            min=0,
            help="Reviewed combined terminal value; required for a historical as-of date",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Calculate a money-weighted return from irregular tracking-account flows."""
    try:
        as_of_date = (
            date.fromisoformat(valuation_date)
            if valuation_date is not None
            else date.today()
        )
    except ValueError as exc:
        raise typer.BadParameter(
            "as-of must use YYYY-MM-DD",
            param_hint="--as-of",
        ) from exc

    async def run() -> None:
        try:
            async with open_wealth_service(settings) as service:
                result = await service.calculate_money_weighted_performance(
                    account_ids=account_ids,
                    valuation_date=as_of_date,
                    current_date=date.today(),
                    ending_balance=ending_balance,
                )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--account-id") from exc

        if json_output:
            print_json(result.as_dict())
            return
        render_rows(
            [
                {
                    "accounts": len(result.account_ids),
                    "period": (
                        f"{result.first_cash_flow_date} → {result.valuation_date}"
                    ),
                    "contributions": format_dollars(result.contributions),
                    "withdrawals": format_dollars(result.withdrawals),
                    "ending_balance": format_dollars(result.ending_balance),
                    "excluded": result.excluded_transaction_count,
                    "xirr": f"{result.xirr:.2%}",
                }
            ],
            [
                ("accounts", "Accounts"),
                ("period", "Period"),
                ("contributions", "Contributions"),
                ("withdrawals", "Withdrawals"),
                ("ending_balance", "Ending Balance"),
                ("excluded", "Excluded Activity"),
                ("xirr", "Money-weighted Return"),
            ],
        )
        console.print(f"Valuation: {result.valuation_note}")
        reason_counts = Counter(
            item.reason.value for item in result.cash_flow_classifications
        )
        if reason_counts:
            render_rows(
                [
                    {"reason": reason, "transactions": count}
                    for reason, count in sorted(reason_counts.items())
                ],
                [("reason", "Cash-flow treatment"), ("transactions", "Transactions")],
            )

    asyncio.run(run())
