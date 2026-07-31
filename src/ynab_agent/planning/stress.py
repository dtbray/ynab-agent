"""Bounded, versioned named market stresses for scenario comparisons."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_STRESS_YEARS = 10


class NamedStressName(StrEnum):
    """Stable public selectors for the built-in stress catalog."""

    EQUITY_CRASH = "equity_crash"
    STAGFLATION = "stagflation"
    LOST_DECADE = "lost_decade"


class StressReturnVector(BaseModel):
    """One deterministic year's simple returns in public asset order."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    us_equity: float = Field(gt=-1, le=5)
    international_equity: float = Field(gt=-1, le=5)
    bonds: float = Field(gt=-1, le=5)
    cash: float = Field(gt=-1, le=5)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.us_equity,
            self.international_equity,
            self.bonds,
            self.cash,
        )


class NamedStressDefinition(BaseModel):
    """Auditable deterministic return and inflation sequence."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    schema_version: int = 1
    name: NamedStressName
    title: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    annual_returns: tuple[StressReturnVector, ...] = Field(
        min_length=1,
        max_length=MAX_STRESS_YEARS,
    )
    annual_inflation: tuple[float, ...] = Field(
        min_length=1,
        max_length=MAX_STRESS_YEARS,
    )

    @model_validator(mode="after")
    def validate_years(self) -> NamedStressDefinition:
        if len(self.annual_returns) != len(self.annual_inflation):
            raise ValueError(
                "named stress must have one inflation rate per return year"
            )
        if any(value <= -1 or value > 1 for value in self.annual_inflation):
            raise ValueError(
                "named stress inflation rates must be greater than -1 and at most 1"
            )
        return self

    @property
    def content_sha256(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return sha256(canonical).hexdigest()


_STRESS_CATALOG = {
    definition.name: definition
    for definition in (
        NamedStressDefinition(
            name=NamedStressName.EQUITY_CRASH,
            title="Equity crash and partial rebound",
            description=(
                "A sharp global equity drawdown followed by a two-year "
                "partial recovery."
            ),
            annual_returns=(
                StressReturnVector(
                    us_equity=-0.37,
                    international_equity=-0.43,
                    bonds=0.05,
                    cash=0.02,
                ),
                StressReturnVector(
                    us_equity=0.18,
                    international_equity=0.16,
                    bonds=0.03,
                    cash=0.02,
                ),
                StressReturnVector(
                    us_equity=0.10,
                    international_equity=0.09,
                    bonds=0.03,
                    cash=0.02,
                ),
            ),
            annual_inflation=(0.01, 0.02, 0.02),
        ),
        NamedStressDefinition(
            name=NamedStressName.STAGFLATION,
            title="Three-year stagflation",
            description=(
                "Persistent inflation with weak equities and an initial "
                "bond drawdown."
            ),
            annual_returns=(
                StressReturnVector(
                    us_equity=-0.12,
                    international_equity=-0.15,
                    bonds=-0.10,
                    cash=0.03,
                ),
                StressReturnVector(
                    us_equity=-0.04,
                    international_equity=-0.06,
                    bonds=-0.02,
                    cash=0.05,
                ),
                StressReturnVector(
                    us_equity=0.03,
                    international_equity=0.02,
                    bonds=0.01,
                    cash=0.05,
                ),
            ),
            annual_inflation=(0.08, 0.07, 0.05),
        ),
        NamedStressDefinition(
            name=NamedStressName.LOST_DECADE,
            title="Low-return lost decade",
            description=(
                "A five-year sequence of alternating losses and weak "
                "recoveries that repeats for longer horizons."
            ),
            annual_returns=(
                StressReturnVector(
                    us_equity=-0.18,
                    international_equity=-0.21,
                    bonds=0.04,
                    cash=0.02,
                ),
                StressReturnVector(
                    us_equity=0.08,
                    international_equity=0.06,
                    bonds=0.03,
                    cash=0.02,
                ),
                StressReturnVector(
                    us_equity=-0.10,
                    international_equity=-0.13,
                    bonds=0.02,
                    cash=0.02,
                ),
                StressReturnVector(
                    us_equity=0.06,
                    international_equity=0.05,
                    bonds=0.01,
                    cash=0.02,
                ),
                StressReturnVector(
                    us_equity=0.02,
                    international_equity=0.01,
                    bonds=0.02,
                    cash=0.02,
                ),
            ),
            annual_inflation=(0.03, 0.03, 0.04, 0.03, 0.03),
        ),
    )
}


def named_stress_catalog() -> tuple[NamedStressDefinition, ...]:
    """Return the complete bounded catalog in stable selector order."""
    return tuple(_STRESS_CATALOG[name] for name in NamedStressName)


def resolve_named_stress(
    name: NamedStressName,
) -> NamedStressDefinition:
    """Resolve a validated public selector."""
    return _STRESS_CATALOG[name]
