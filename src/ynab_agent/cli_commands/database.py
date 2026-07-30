"""Database lifecycle Typer commands."""

from __future__ import annotations

import asyncio

import typer

from ynab_agent.cli_support import configure_logging, console
from ynab_agent.config import settings
from ynab_agent.runtime import open_database


database_app = typer.Typer()


@database_app.command("init")
def initialize_database() -> None:
    """Initialize database schema."""
    configure_logging(settings)

    async def run() -> None:
        async with open_database(settings, initialize=True):
            console.print(
                f"✅ Database initialized at [bold]{settings.effective_database_url}[/bold]"
            )

    asyncio.run(run())
