from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from datetime import date

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.wealth import (
    AccountFreshness,
    CashFlowCandidate,
    WealthAccount,
    WealthService,
)


class _Repository:
    def __init__(self, accounts: Sequence[WealthAccount] = ()) -> None:
        self.accounts = tuple(accounts)

    async def get_accounts(
        self,
        account_ids: Collection[str],
    ) -> Sequence[WealthAccount]:
        selected = set(account_ids)
        return tuple(account for account in self.accounts if account.id in selected)

    async def list_account_freshness(
        self,
        *,
        tracking_only: bool,
    ) -> Sequence[AccountFreshness]:
        return ()

    async def get_cash_flow_candidates(
        self,
        account_ids: Collection[str],
        *,
        through: date,
    ) -> Sequence[CashFlowCandidate]:
        return ()


def _scenario(**overrides: object) -> WealthScenario:
    values: dict[str, object] = {
        "name": "portfolio provenance",
        "current_age": 40,
        "retirement_age": 60,
        "end_age": 95,
        "starting_portfolio": 125_000,
        "annual_spending": 40_000,
    }
    values.update(overrides)
    return WealthScenario.model_validate(values)


async def _resolve_explicit() -> None:
    service = WealthService(_Repository())

    resolved = await service.resolve_starting_portfolio_with_provenance(_scenario())

    assert resolved.value == 125_000
    assert resolved.provenance.source == "explicit_scenario_input"
    assert resolved.provenance.as_of is None
    assert resolved.provenance.account_ids == ()
    assert resolved.provenance.source_sha256 is not None
    assert len(resolved.provenance.source_sha256) == 64
    assert await service.resolve_starting_portfolio(_scenario()) == 125_000


def test_explicit_starting_portfolio_has_typed_provenance() -> None:
    asyncio.run(_resolve_explicit())


async def _resolve_cached() -> None:
    accounts = (
        WealthAccount(
            id="taxable",
            name="Taxable",
            type="otherAsset",
            on_budget=False,
            balance_milliunits=25_000_000,
            closed=False,
            deleted=False,
            last_reconciled_at="2025-12-01T00:00:00+00:00",
        ),
        WealthAccount(
            id="retirement",
            name="Retirement",
            type="otherAsset",
            on_budget=False,
            balance_milliunits=100_000_000,
            closed=False,
            deleted=False,
            last_reconciled_at=None,
        ),
    )
    service = WealthService(_Repository(accounts))
    first_scenario = _scenario(
        starting_portfolio=None,
        accounts=[
            {"id": "taxable", "role": "taxable"},
            {"id": "retirement", "role": "retirement"},
        ],
    )
    reordered_scenario = _scenario(
        starting_portfolio=None,
        accounts=[
            {"id": "retirement", "role": "retirement"},
            {"id": "taxable", "role": "taxable"},
        ],
    )

    first = await service.resolve_starting_portfolio_with_provenance(first_scenario)
    reordered = await service.resolve_starting_portfolio_with_provenance(reordered_scenario)

    assert first.value == 125_000
    assert first.provenance.source == "cached_liquid_accounts"
    assert first.provenance.as_of is None
    assert first.provenance.account_ids == ("retirement", "taxable")
    assert first.provenance.source_sha256 == reordered.provenance.source_sha256
    assert first.provenance.source_sha256 is not None
    assert len(first.provenance.source_sha256) == 64


def test_cached_liquid_accounts_have_stable_input_provenance() -> None:
    asyncio.run(_resolve_cached())


async def _project_cached_mortgage() -> None:
    mortgage = WealthAccount(
        id="mortgage",
        name="Home",
        type="mortgage",
        on_budget=False,
        balance_milliunits=-184_229_470,
        closed=False,
        deleted=False,
        last_reconciled_at="2026-07-28T00:00:00+00:00",
        debt_interest_rates='{"2021-10-01": 2750}',
        debt_minimum_payments='{"2025-07-01": 1157200}',
        debt_escrow_amounts='{"2025-07-01": 292340}',
    )

    projection = await WealthService(
        _Repository((mortgage,))
    ).project_mortgage(
        mortgage.id,
        as_of=date(2026, 7, 29),
        current_age=30,
    )

    assert projection.estimated_payoff_age == 55
    assert round(projection.annual_principal_and_interest, 2) == 10_378.32


def test_cached_mortgage_terms_produce_a_payoff_projection() -> None:
    asyncio.run(_project_cached_mortgage())
