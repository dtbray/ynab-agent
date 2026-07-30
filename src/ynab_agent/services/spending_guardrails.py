"""Application service for durable YNAB spending-tier mappings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ynab_agent.planning.spending_guardrails import (
    FixedRealSpendingPolicy,
    RetirementSpendingPlan,
    SpendingPlanSource,
    SpendingPolicy,
    SpendingTier,
    SpendingTierAmounts,
)


MAX_MAPPING_NOTE_LENGTH = 500


class SpendingTierAssignment(BaseModel):
    """Validated user choice attached to a stable category ID."""

    model_config = ConfigDict(frozen=True)

    budget_id: str = Field(min_length=1, max_length=64)
    category_id: str = Field(min_length=1, max_length=64)
    tier: SpendingTier
    essential_floor_milliunits: int | None = Field(default=None, ge=0)
    note: str | None = Field(
        default=None,
        max_length=MAX_MAPPING_NOTE_LENGTH,
    )

    @model_validator(mode="after")
    def validate_floor(self) -> SpendingTierAssignment:
        if (
            self.tier is SpendingTier.ESSENTIAL
            and self.essential_floor_milliunits is None
        ):
            raise ValueError(
                "essential categories require an explicit monthly floor"
            )
        if (
            self.tier is not SpendingTier.ESSENTIAL
            and self.essential_floor_milliunits is not None
        ):
            raise ValueError(
                "essential floors are only valid for essential categories"
            )
        return self


@dataclass(frozen=True)
class SpendingTierMapping:
    """Persisted mapping resolved against the latest cached category name."""

    budget_id: str
    category_id: str
    tier: SpendingTier
    essential_floor_milliunits: int | None
    note: str | None
    category_name: str | None
    category_group_name: str | None
    category_hidden: bool
    category_deleted: bool
    updated_at: str


@dataclass(frozen=True)
class CategorySpendingAggregate:
    """Observed net outflow for one mapped category and bounded window."""

    category_id: str
    tier: SpendingTier
    spending_milliunits: int


class SpendingTierRepository(Protocol):
    """Persistence and read-model seam for YNAB-derived spending."""

    async def upsert_mapping(
        self,
        assignment: SpendingTierAssignment,
        *,
        updated_at: str,
    ) -> SpendingTierMapping | None: ...

    async def list_mappings(
        self,
        *,
        budget_id: str,
    ) -> Sequence[SpendingTierMapping]: ...

    async def list_spending(
        self,
        *,
        budget_id: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[CategorySpendingAggregate]: ...


class UnknownSpendingCategoryError(ValueError):
    """Raised when a stable category identity is absent from the plan cache."""


class SpendingGuardrailService:
    """Manage tier identities and derive a policy-independent baseline."""

    def __init__(self, repository: SpendingTierRepository) -> None:
        self.repository = repository

    async def assign_tier(
        self,
        assignment: SpendingTierAssignment,
    ) -> SpendingTierMapping:
        row = await self.repository.upsert_mapping(
            assignment,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        if row is None:
            raise UnknownSpendingCategoryError(
                f"category {assignment.category_id!r} is not cached "
                f"for plan {assignment.budget_id!r}"
            )
        return row

    async def list_mappings(
        self,
        *,
        budget_id: str,
    ) -> tuple[SpendingTierMapping, ...]:
        return tuple(
            await self.repository.list_mappings(budget_id=budget_id)
        )

    async def derive_plan(
        self,
        *,
        budget_id: str,
        through_month: date,
        lookback_months: int = 12,
        policy: SpendingPolicy | None = None,
    ) -> RetirementSpendingPlan:
        if not 1 <= lookback_months <= 120:
            raise ValueError("lookback months must be between 1 and 120")
        end_month = _first_of_month(through_month)
        start_month = _add_months(end_month, -lookback_months)
        mappings = tuple(
            await self.repository.list_mappings(budget_id=budget_id)
        )
        spending = tuple(
            await self.repository.list_spending(
                budget_id=budget_id,
                start_date=start_month,
                end_date=end_month,
            )
        )
        totals = {tier: 0 for tier in SpendingTier}
        for row in spending:
            totals[row.tier] += max(0, row.spending_milliunits)
        annualized = {
            tier: round(total / lookback_months * 12) / 1000
            for tier, total in totals.items()
        }
        essential_floor = (
            sum(
                mapping.essential_floor_milliunits or 0
                for mapping in mappings
                if mapping.tier is SpendingTier.ESSENTIAL
            )
            * 12
            / 1000
        )
        mapping_sha = _mapping_sha256(mappings)
        return RetirementSpendingPlan(
            baseline=SpendingTierAmounts(
                essential=max(
                    annualized[SpendingTier.ESSENTIAL],
                    essential_floor,
                ),
                lifestyle=annualized[SpendingTier.LIFESTYLE],
                discretionary=annualized[SpendingTier.DISCRETIONARY],
                one_time=annualized[SpendingTier.ONE_TIME],
            ),
            essential_floor=essential_floor,
            policy=policy or FixedRealSpendingPolicy(),
            source=SpendingPlanSource(
                budget_id=budget_id,
                through_month=f"{end_month:%Y-%m}",
                lookback_months=lookback_months,
                mapping_sha256=mapping_sha,
                mapped_category_count=len(mappings),
            ),
        )


def _mapping_sha256(
    mappings: Sequence[SpendingTierMapping],
) -> str:
    """Fingerprint identities and assumptions, excluding renameable labels."""
    material = [
        {
            "budget_id": row.budget_id,
            "category_id": row.category_id,
            "tier": row.tier.value,
            "essential_floor_milliunits": (
                row.essential_floor_milliunits
            ),
        }
        for row in sorted(
            mappings,
            key=lambda row: (row.budget_id, row.category_id),
        )
    ]
    canonical = json.dumps(
        material,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(canonical).hexdigest()


def _first_of_month(value: date) -> date:
    return value.replace(day=1)


def _add_months(month: date, delta: int) -> date:
    month_index = month.year * 12 + month.month - 1 + delta
    year, zero_based_month = divmod(month_index, 12)
    return date(year, zero_based_month + 1, 1)
