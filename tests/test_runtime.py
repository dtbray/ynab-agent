import asyncio
from collections.abc import Callable
from typing import cast

import pytest

from ynab_agent.config import Settings
from ynab_agent.runtime import DatabaseFactory, open_database, open_wealth_service
from ynab_agent.services.wealth import WealthService


class FakeDatabase:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.initialize_calls = 0
        self.close_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


def fake_factory(
    created: list[FakeDatabase],
) -> Callable[[str], FakeDatabase]:
    def create(database_url: str) -> FakeDatabase:
        database = FakeDatabase(database_url)
        created.append(database)
        return database

    return create


def as_database_factory(
    factory: Callable[[str], FakeDatabase],
) -> DatabaseFactory:
    return cast(DatabaseFactory, cast(object, factory))


@pytest.mark.asyncio
async def test_open_database_initializes_when_requested_and_closes() -> None:
    created: list[FakeDatabase] = []
    settings = Settings(database_url="sqlite+aiosqlite:///runtime.db")

    async with open_database(
        settings,
        initialize=True,
        factory=as_database_factory(fake_factory(created)),
    ) as database:
        assert database.database_url == "sqlite+aiosqlite:///runtime.db"
        assert created[0].initialize_calls == 1
        assert created[0].close_calls == 0

    assert created[0].close_calls == 1


@pytest.mark.asyncio
async def test_open_database_closes_after_body_failure() -> None:
    created: list[FakeDatabase] = []
    settings = Settings(database_url="sqlite+aiosqlite:///runtime.db")

    with pytest.raises(RuntimeError, match="operation failed"):
        async with open_database(
            settings,
            factory=as_database_factory(fake_factory(created)),
        ):
            raise RuntimeError("operation failed")

    assert created[0].close_calls == 1


@pytest.mark.asyncio
async def test_open_database_closes_after_cancellation() -> None:
    created: list[FakeDatabase] = []
    settings = Settings(database_url="sqlite+aiosqlite:///runtime.db")
    entered = asyncio.Event()

    async def operation() -> None:
        async with open_database(
            settings,
            factory=as_database_factory(fake_factory(created)),
        ):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(operation())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert created[0].close_calls == 1


@pytest.mark.asyncio
async def test_open_wealth_service_uses_scoped_database() -> None:
    created: list[FakeDatabase] = []
    settings = Settings(database_url="sqlite+aiosqlite:///runtime.db")

    async with open_wealth_service(
        settings,
        database_factory=as_database_factory(fake_factory(created)),
    ) as service:
        assert isinstance(service, WealthService)
        assert created[0].close_calls == 0

    assert created[0].close_calls == 1
