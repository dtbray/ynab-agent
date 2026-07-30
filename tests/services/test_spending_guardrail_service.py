from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from ynab_agent.planning.spending_guardrails import (
    SpendingTier,
    WithdrawalRateGuardrailPolicy,
)
from ynab_agent.services.spending_guardrails import (
    CategorySpendingAggregate,
    SpendingGuardrailService,
    SpendingTierAssignment,
    SpendingTierMapping,
)


class FakeSpendingTierRepository:
    def __init__(
        self,
        *,
        mappings: Sequence[SpendingTierMapping],
        spending: Sequence[CategorySpendingAggregate],
    ) -> None:
        self.mappings = list(mappings)
        self.spending = tuple(spending)
        self.window: tuple[date, date] | None = None

    async def upsert_mapping(
        self,
        assignment: SpendingTierAssignment,
        *,
        updated_at: str,
    ) -> SpendingTierMapping | None:
        return None

    async def list_mappings(
        self,
        *,
        budget_id: str,
    ) -> Sequence[SpendingTierMapping]:
        return tuple(
            row for row in self.mappings if row.budget_id == budget_id
        )

    async def list_spending(
        self,
        *,
        budget_id: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[CategorySpendingAggregate]:
        self.window = (start_date, end_date)
        return self.spending


def _mapping(
    category_id: str,
    tier: SpendingTier,
    *,
    floor: int | None = None,
    name: str = "Original name",
) -> SpendingTierMapping:
    return SpendingTierMapping(
        budget_id="budget-1",
        category_id=category_id,
        tier=tier,
        essential_floor_milliunits=floor,
        note=None,
        category_name=name,
        category_group_name="Group",
        category_hidden=False,
        category_deleted=False,
        updated_at="2026-07-30T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_derived_baseline_annualizes_observed_spending_and_explicit_floor() -> None:
    repository = FakeSpendingTierRepository(
        mappings=[
            _mapping(
                "food",
                SpendingTier.ESSENTIAL,
                floor=80_000,
            ),
            _mapping("fun", SpendingTier.LIFESTYLE),
        ],
        spending=[
            CategorySpendingAggregate(
                category_id="food",
                tier=SpendingTier.ESSENTIAL,
                spending_milliunits=100_000,
            ),
            CategorySpendingAggregate(
                category_id="fun",
                tier=SpendingTier.LIFESTYLE,
                spending_milliunits=50_000,
            ),
        ],
    )
    policy = WithdrawalRateGuardrailPolicy(
        lower_withdrawal_rate=0.03,
        upper_withdrawal_rate=0.05,
    )

    plan = await SpendingGuardrailService(repository).derive_plan(
        budget_id="budget-1",
        through_month=date(2026, 8, 15),
        lookback_months=2,
        policy=policy,
    )

    assert repository.window == (
        date(2026, 6, 1),
        date(2026, 8, 1),
    )
    assert plan.baseline.essential == 960
    assert plan.baseline.lifestyle == 300
    assert plan.essential_floor == 960
    assert plan.policy == policy
    assert plan.source is not None
    assert plan.source.through_month == "2026-08"
    assert plan.source.mapped_category_count == 2


@pytest.mark.asyncio
async def test_mapping_fingerprint_survives_category_rename() -> None:
    repository = FakeSpendingTierRepository(
        mappings=[
            _mapping(
                "food-id",
                SpendingTier.ESSENTIAL,
                floor=50_000,
                name="Groceries",
            )
        ],
        spending=[],
    )
    service = SpendingGuardrailService(repository)

    before = await service.derive_plan(
        budget_id="budget-1",
        through_month=date(2026, 8, 1),
    )
    repository.mappings[0] = _mapping(
        "food-id",
        SpendingTier.ESSENTIAL,
        floor=50_000,
        name="Food at home",
    )
    after = await service.derive_plan(
        budget_id="budget-1",
        through_month=date(2026, 8, 1),
    )

    assert before.source is not None
    assert after.source is not None
    assert before.source.mapping_sha256 == after.source.mapping_sha256


def test_essential_mapping_requires_an_explicit_floor() -> None:
    with pytest.raises(ValueError, match="explicit monthly floor"):
        SpendingTierAssignment(
            budget_id="budget-1",
            category_id="food",
            tier=SpendingTier.ESSENTIAL,
        )
