"""Natural-language budget query Typer command."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.table import Table

from ynab_agent.cli_support import configure_logging, console
from ynab_agent.config import settings
from ynab_agent.queries.natural_language import NaturalLanguageQuerier
from ynab_agent.runtime import open_database


natural_language_app = typer.Typer()


@natural_language_app.command("query")
def query(
    question: Annotated[
        str,
        typer.Argument(
            help="Natural language question about your budget",
        ),
    ],
) -> None:
    """Ask a natural language question about your budget."""
    configure_logging(settings)

    async def run() -> None:
        async with open_database(settings, initialize=True) as database:
            result = await NaturalLanguageQuerier(database).query(question)

        console.print(f"\n[bold cyan]{result.summary}[/bold cyan]\n")
        if not result.data:
            return

        table = Table(show_header=True, header_style="bold magenta")
        for key in result.data[0]:
            table.add_column(key.replace("_", " ").title())
        for row in result.data[:20]:
            table.add_row(*[str(value) for value in row.values()])
        console.print(table)
        if len(result.data) > 20:
            console.print(f"\n... and {len(result.data) - 20} more rows")

    asyncio.run(run())
