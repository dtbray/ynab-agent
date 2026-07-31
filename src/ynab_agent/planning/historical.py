"""Load paired annual market-return and inflation observations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import io
import json
import math
from pathlib import Path


ASSET_RETURN_COLUMNS = (
    "us_equity_return",
    "international_equity_return",
    "bonds_return",
    "cash_return",
)
AssetHistoricalReturns = tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]


class HistoricalOrderPolicy(StrEnum):
    """How a loader handles observations that are not in ascending year order."""

    NORMALIZE = "normalize"
    REQUIRE_ASCENDING = "require_ascending"


class HistoricalGapPolicy(StrEnum):
    """How a loader handles missing years between observations."""

    REJECT = "reject"
    ALLOW = "allow"


@dataclass(frozen=True)
class HistoricalSeries:
    """Annual observations used for paired block-bootstrap sampling."""

    years: tuple[int, ...]
    nominal_returns: tuple[float, ...]
    inflation_rates: tuple[float, ...] | None
    source: str
    sha256: str
    asset_returns: AssetHistoricalReturns | None = None
    observations_sha256: str = ""
    order_policy: HistoricalOrderPolicy = HistoricalOrderPolicy.NORMALIZE
    gap_policy: HistoricalGapPolicy = HistoricalGapPolicy.REJECT

    def __post_init__(self) -> None:
        if not self.years:
            raise ValueError("historical series must contain at least one observation")
        if len(self.nominal_returns) != len(self.years):
            raise ValueError("historical series must have one return per year")
        if self.inflation_rates is not None and len(self.inflation_rates) != len(self.years):
            raise ValueError("historical series must have one inflation rate per year")
        if self.asset_returns is not None:
            if len(self.asset_returns) != len(ASSET_RETURN_COLUMNS):
                raise ValueError(
                    "historical series must contain exactly four asset return series"
                )
            if any(
                len(values) != len(self.years)
                for values in self.asset_returns
            ):
                raise ValueError(
                    "historical series must have one return per asset and year"
                )
            if any(
                not math.isfinite(value) or value <= -1
                for values in self.asset_returns
                for value in values
            ):
                raise ValueError(
                    "historical asset returns must be finite and greater than -1"
                )
        for previous, current in zip(self.years, self.years[1:]):
            if current <= previous:
                raise ValueError("historical series years must be unique and ascending")
            if self.gap_policy is HistoricalGapPolicy.REJECT and current != previous + 1:
                raise ValueError(
                    "historical years must be contiguous under gap policy "
                    f"'{self.gap_policy.value}'; found {previous} followed by {current}"
                )
        if not self.observations_sha256:
            object.__setattr__(
                self,
                "observations_sha256",
                _observation_fingerprint(
                    self.years,
                    self.nominal_returns,
                    self.inflation_rates,
                    self.asset_returns,
                ),
            )

    @property
    def first_year(self) -> int:
        return self.years[0]

    @property
    def last_year(self) -> int:
        return self.years[-1]


def _observation_fingerprint(
    years: tuple[int, ...],
    nominal_returns: tuple[float, ...],
    inflation_rates: tuple[float, ...] | None,
    asset_returns: AssetHistoricalReturns | None = None,
) -> str:
    payload: dict[str, object] = {
        "inflation_rates": inflation_rates,
        "nominal_returns": nominal_returns,
        "years": years,
    }
    # Preserve legacy one-asset fingerprints exactly.
    if asset_returns is not None:
        payload["asset_returns"] = asset_returns
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_historical_series(
    path: Path,
    *,
    order_policy: HistoricalOrderPolicy = HistoricalOrderPolicy.NORMALIZE,
    gap_policy: HistoricalGapPolicy = HistoricalGapPolicy.REJECT,
) -> HistoricalSeries:
    """Load year, nominal_return, and optional inflation_rate columns from CSV."""
    try:
        content = path.read_bytes()
        reader = csv.DictReader(
            io.StringIO(content.decode("utf-8-sig"), newline=""),
        )
        rows = list(reader)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read historical returns: {exc}") from exc

    fieldnames = set(reader.fieldnames or ())
    required = {"year", "nominal_return"}
    missing = sorted(required - fieldnames)
    if missing:
        raise ValueError(f"historical returns CSV is missing columns: {', '.join(missing)}")
    if not rows:
        raise ValueError("historical returns CSV has no observations")
    present_asset_columns = fieldnames.intersection(ASSET_RETURN_COLUMNS)
    if present_asset_columns and present_asset_columns != set(
        ASSET_RETURN_COLUMNS
    ):
        missing_assets = sorted(
            set(ASSET_RETURN_COLUMNS) - present_asset_columns
        )
        raise ValueError(
            "multi-asset historical returns require all asset columns; "
            f"missing: {', '.join(missing_assets)}"
        )

    years: list[int] = []
    returns: list[float] = []
    inflation: list[float] = []
    asset_rows: list[tuple[float, float, float, float]] = []
    has_inflation = "inflation_rate" in fieldnames
    has_asset_returns = bool(present_asset_columns)
    for line_number, row in enumerate(rows, start=2):
        if has_inflation and row.get("inflation_rate") in (None, ""):
            raise ValueError(
                "inflation_rate must be present for every historical observation; "
                f"missing value on line {line_number}"
            )
        try:
            year = int(row["year"])
            nominal_return = float(row["nominal_return"])
            inflation_rate = float(row["inflation_rate"]) if has_inflation else None
            asset_return = (
                (
                    float(row[ASSET_RETURN_COLUMNS[0]]),
                    float(row[ASSET_RETURN_COLUMNS[1]]),
                    float(row[ASSET_RETURN_COLUMNS[2]]),
                    float(row[ASSET_RETURN_COLUMNS[3]]),
                )
                if has_asset_returns
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid historical value on CSV line {line_number}") from exc
        if year in years:
            raise ValueError(f"duplicate historical year: {year}")
        if order_policy is HistoricalOrderPolicy.REQUIRE_ASCENDING and years and year < years[-1]:
            raise ValueError(
                "historical years must be in strictly ascending order under "
                f"order policy '{order_policy.value}'; found {years[-1]} "
                f"followed by {year} on line {line_number}"
            )
        if not math.isfinite(nominal_return):
            raise ValueError(f"nominal_return must be finite on line {line_number}")
        if nominal_return <= -1:
            raise ValueError(f"nominal_return must be greater than -1 on line {line_number}")
        if inflation_rate is not None:
            if not math.isfinite(inflation_rate):
                raise ValueError(f"inflation_rate must be finite on line {line_number}")
            if inflation_rate <= -1:
                raise ValueError(f"inflation_rate must be greater than -1 on line {line_number}")
        if asset_return is not None and any(
            not math.isfinite(value) or value <= -1
            for value in asset_return
        ):
            raise ValueError(
                "asset returns must be finite and greater than -1 "
                f"on line {line_number}"
            )
        years.append(year)
        returns.append(nominal_return)
        if inflation_rate is not None:
            inflation.append(inflation_rate)
        if asset_return is not None:
            asset_rows.append(asset_return)

    observations = sorted(
        zip(
            years,
            returns,
            inflation if has_inflation else [None] * len(years),
            asset_rows if has_asset_returns else [None] * len(years),
        ),
        key=lambda observation: observation[0],
    )
    sorted_years = tuple(observation[0] for observation in observations)
    for previous, current in zip(sorted_years, sorted_years[1:]):
        if gap_policy is HistoricalGapPolicy.REJECT and current != previous + 1:
            raise ValueError(
                "historical years must be contiguous under gap policy "
                f"'{gap_policy.value}'; found {previous} followed by {current}"
            )

    sorted_inflation: tuple[float, ...] | None = None
    if has_inflation:
        sorted_inflation = tuple(
            observation[2] for observation in observations if observation[2] is not None
        )
        if len(sorted_inflation) != len(observations):
            raise ValueError("inflation_rate must be present for every historical observation")

    sorted_returns = tuple(observation[1] for observation in observations)
    sorted_asset_returns: AssetHistoricalReturns | None = None
    if has_asset_returns:
        sorted_asset_returns = (
            tuple(observation[3][0] for observation in observations if observation[3] is not None),
            tuple(observation[3][1] for observation in observations if observation[3] is not None),
            tuple(observation[3][2] for observation in observations if observation[3] is not None),
            tuple(observation[3][3] for observation in observations if observation[3] is not None),
        )
        if any(
            len(values) != len(observations)
            for values in sorted_asset_returns
        ):
            raise ValueError(
                "asset returns must be present for every historical observation"
            )

    return HistoricalSeries(
        years=sorted_years,
        nominal_returns=sorted_returns,
        inflation_rates=sorted_inflation,
        source=str(path.expanduser().resolve()),
        sha256=hashlib.sha256(content).hexdigest(),
        asset_returns=sorted_asset_returns,
        observations_sha256=_observation_fingerprint(
            sorted_years,
            sorted_returns,
            sorted_inflation,
            sorted_asset_returns,
        ),
        order_policy=order_policy,
        gap_policy=gap_policy,
    )
