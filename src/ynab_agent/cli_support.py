"""Rendering helpers owned exclusively by the command-line adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
import json
import logging

from rich.console import Console
from rich.table import Table

from ynab_agent.config import Settings


console = Console()


def configure_logging(app_settings: Settings) -> None:
    """Configure command logging from settings resolved at invocation time."""
    logging.basicConfig(
        level=getattr(logging, app_settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def print_json(data: object) -> None:
    """Render JSON consistently with the legacy CLI."""
    console.print_json(json.dumps(data, default=str))


def format_dollars(amount: int | float | Decimal | None) -> str:
    """Format a dollar value for a human-readable table."""
    if amount is None:
        return ""
    return f"{float(amount):,.2f}"


def format_milliunits(amount: int | None) -> str:
    """Format YNAB milliunits as dollars."""
    if amount is None:
        return ""
    return f"{amount / 1000:,.2f}"


def render_rows(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
) -> None:
    """Render mappings as a Rich table."""
    table = Table(show_header=True, header_style="bold magenta")
    for _, title in columns:
        table.add_column(title)
    for row in rows:
        table.add_row(*[str(row.get(key, "")) for key, _ in columns])
    console.print(table)
