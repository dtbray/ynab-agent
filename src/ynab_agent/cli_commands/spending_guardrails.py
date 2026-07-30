"""Typer commands for YNAB-backed retirement spending guardrails."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Annotated

from pydantic import TypeAdapter, ValidationError
import typer

from ynab_agent.cli_support import (
    format_dollars,
    print_json,
    render_rows,
)
from ynab_agent.config import settings
from ynab_agent.planning.spending_guardrails import (
    AnnualSpendingContext,
    RetirementSpendingPlan,
    SpendingPolicy,
    SpendingTier,
    evaluate_guardrails,
)
from ynab_agent.runtime import open_spending_guardrail_service
from ynab_agent.services.spending_guardrails import (
    SpendingTierAssignment,
    UnknownSpendingCategoryError,
)


spending_guardrails_app = typer.Typer(
    help="Map YNAB categories and preview retirement spending policies",
)


def _dollars_to_milliunits(value: str) -> int:
    try:
        dollars = Decimal(value)
    except InvalidOperation as exc:
        raise typer.BadParameter("must be a dollar amount") from exc
    if not dollars.is_finite() or dollars < 0:
        raise typer.BadParameter("must be a non-negative finite dollar amount")
    return int(
        (dollars * Decimal("1000")).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _load_plan(path: Path) -> RetirementSpendingPlan:
    try:
        return RetirementSpendingPlan.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            f"invalid spending plan: {exc}",
            param_hint="--plan",
        ) from exc


def _load_policy(path: Path | None) -> SpendingPolicy | None:
    if path is None:
        return None
    try:
        return TypeAdapter(SpendingPolicy).validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise typer.BadParameter(
            f"invalid spending policy: {exc}",
            param_hint="--policy",
        ) from exc


def _parse_month(value: str) -> date:
    try:
        return date.fromisoformat(
            f"{value}-01" if len(value) == 7 else value
        )
    except ValueError as exc:
        raise typer.BadParameter(
            "must use YYYY-MM or YYYY-MM-DD",
            param_hint="--through-month",
        ) from exc


@spending_guardrails_app.command("map")
def map_category(
    budget_id: Annotated[
        str,
        typer.Option("--budget-id", help="YNAB budget identity"),
    ],
    category_id: Annotated[
        str,
        typer.Option("--category-id", help="Stable YNAB category identity"),
    ],
    tier: Annotated[
        SpendingTier,
        typer.Option("--tier", help="Retirement spending priority"),
    ],
    essential_floor: Annotated[
        str | None,
        typer.Option(
            "--essential-floor",
            help="Monthly real-dollar floor; required for essential categories",
        ),
    ] = None,
    note: Annotated[
        str | None,
        typer.Option("--note", help="Optional mapping rationale"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Create or replace a category-to-tier mapping."""
    floor_milliunits = (
        _dollars_to_milliunits(essential_floor)
        if essential_floor is not None
        else None
    )
    try:
        assignment = SpendingTierAssignment(
            budget_id=budget_id,
            category_id=category_id,
            tier=tier,
            essential_floor_milliunits=floor_milliunits,
            note=note,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc), param_hint="--tier") from exc

    async def run() -> None:
        try:
            async with open_spending_guardrail_service(settings) as service:
                mapping = await service.assign_tier(assignment)
        except UnknownSpendingCategoryError as exc:
            raise typer.BadParameter(
                str(exc),
                param_hint="--category-id",
            ) from exc
        payload = asdict(mapping)
        if json_output:
            print_json(payload)
            return
        render_rows(
            [payload],
            [
                ("category_id", "Category ID"),
                ("category_name", "Category"),
                ("tier", "Tier"),
                ("essential_floor_milliunits", "Monthly Floor (milliunits)"),
                ("updated_at", "Updated"),
            ],
        )

    asyncio.run(run())


@spending_guardrails_app.command("list")
def list_mappings(
    budget_id: Annotated[
        str,
        typer.Option("--budget-id", help="YNAB budget identity"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """List durable mappings with their latest cached labels."""

    async def run() -> None:
        async with open_spending_guardrail_service(settings) as service:
            mappings = await service.list_mappings(budget_id=budget_id)
        rows = [asdict(mapping) for mapping in mappings]
        if json_output:
            print_json({"mappings": rows})
            return
        render_rows(
            rows,
            [
                ("category_id", "Category ID"),
                ("category_group_name", "Group"),
                ("category_name", "Category"),
                ("tier", "Tier"),
                ("essential_floor_milliunits", "Monthly Floor"),
                ("category_deleted", "Deleted"),
            ],
        )

    asyncio.run(run())


@spending_guardrails_app.command("baseline")
def derive_baseline(
    budget_id: Annotated[
        str,
        typer.Option("--budget-id", help="YNAB budget identity"),
    ],
    through_month: Annotated[
        str,
        typer.Option(
            "--through-month",
            help="Exclusive month boundary, such as 2026-08",
        ),
    ],
    lookback_months: Annotated[
        int,
        typer.Option("--months", min=1, max=120),
    ] = 12,
    policy_path: Annotated[
        Path | None,
        typer.Option(
            "--policy",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional policy JSON; defaults to fixed-real",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a table"),
    ] = False,
) -> None:
    """Annualize mapped spending without coupling it to a policy."""
    policy = _load_policy(policy_path)
    parsed_month = _parse_month(through_month)

    async def run() -> None:
        async with open_spending_guardrail_service(settings) as service:
            plan = await service.derive_plan(
                budget_id=budget_id,
                through_month=parsed_month,
                lookback_months=lookback_months,
                policy=policy,
            )
        if json_output:
            print_json({"plan": plan.model_dump(mode="json")})
            return
        render_rows(
            [
                {
                    "policy": plan.policy.kind,
                    "essential": format_dollars(plan.baseline.essential),
                    "lifestyle": format_dollars(plan.baseline.lifestyle),
                    "discretionary": format_dollars(
                        plan.baseline.discretionary
                    ),
                    "one_time": format_dollars(plan.baseline.one_time),
                    "essential_floor": format_dollars(
                        plan.essential_floor
                    ),
                    "total": format_dollars(plan.baseline.total),
                }
            ],
            [
                ("policy", "Policy"),
                ("essential", "Essential"),
                ("lifestyle", "Lifestyle"),
                ("discretionary", "Discretionary"),
                ("one_time", "One-time"),
                ("essential_floor", "Essential Floor"),
                ("total", "Total"),
            ],
        )

    asyncio.run(run())


@spending_guardrails_app.command("preview")
def preview_guardrails(
    plan_path: Annotated[
        Path,
        typer.Option(
            "--plan",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Retirement spending plan JSON",
        ),
    ],
    opening_portfolio: Annotated[
        list[float],
        typer.Option(
            "--opening-portfolio",
            min=0,
            help="One real portfolio value per retirement year",
        ),
    ],
    start_age: Annotated[
        int,
        typer.Option("--start-age", min=0, max=130),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of an audit table"),
    ] = False,
) -> None:
    """Preview deterministic annual reductions and restorations."""
    if start_age + len(opening_portfolio) > 131:
        raise typer.BadParameter(
            "preview extends beyond age 130",
            param_hint="--opening-portfolio",
        )
    summary = evaluate_guardrails(
        _load_plan(plan_path),
        (
            AnnualSpendingContext(
                age=start_age + offset,
                opening_portfolio_real=value,
            )
            for offset, value in enumerate(opening_portfolio)
        ),
    )
    if json_output:
        print_json({"summary": summary.model_dump(mode="json")})
        return
    render_rows(
        [
            {
                "age": row.age,
                "action": row.action,
                "reason": row.reason,
                "spending": format_dollars(row.applied_spending.total),
                "delta": format_dollars(row.spending_delta.total),
                "essential": format_dollars(row.applied_spending.essential),
            }
            for row in summary.annual_path
        ],
        [
            ("age", "Age"),
            ("action", "Action"),
            ("reason", "Reason"),
            ("spending", "Real Spending"),
            ("delta", "Change"),
            ("essential", "Essential"),
        ],
    )
