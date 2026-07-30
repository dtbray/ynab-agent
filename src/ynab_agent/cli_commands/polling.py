"""Transaction polling Typer command."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated, cast

import typer

from ynab_agent.cli_support import configure_logging, console
from ynab_agent.config import settings
from ynab_agent.poller import TransactionPoller
from ynab_agent.runtime import open_database


polling_app = typer.Typer()


@polling_app.command("poll")
def poll(
    once: Annotated[
        bool,
        typer.Option("--once", "-o", help="Run once and exit"),
    ] = False,
    interval: Annotated[
        int | None,
        typer.Option(
            "--interval",
            "-i",
            help="Polling interval in minutes",
        ),
    ] = None,
) -> None:
    """Poll YNAB for unapproved transactions."""
    configure_logging(settings)
    resolved_interval = settings.poll_interval_minutes if interval is None else interval

    async def run() -> None:
        async with open_database(settings, initialize=True) as database:
            poller = TransactionPoller(database)

            if once:
                console.print("🔍 Running single poll...")
                count = await poller.run_once()
                console.print(f"✅ Sent {count} reminders")
                return

            console.print(f"🔄 Starting continuous polling (interval: {resolved_interval} min)...")
            console.print("Press Ctrl+C to stop")
            try:
                await poller.run_continuous(resolved_interval)
            except KeyboardInterrupt:
                cast(Callable[[], None], poller.stop)()
                console.print("\n👋 Stopped")

    asyncio.run(run())
