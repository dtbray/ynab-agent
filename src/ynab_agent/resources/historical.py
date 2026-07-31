"""Eager registry for server-owned historical planner datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ynab_agent.planning.historical import (
    HistoricalOrderPolicy,
    load_historical_series,
)
from ynab_agent.services.planner_jobs import HistoricalDatasetSnapshot


class RegisteredHistoricalDatasets:
    """Resolve path-free snapshots by configured public identifier."""

    def __init__(
        self,
        datasets: Mapping[str, HistoricalDatasetSnapshot] | None = None,
    ) -> None:
        self._datasets = dict(datasets or {})

    @classmethod
    def from_paths(
        cls,
        paths: Mapping[str, Path],
    ) -> RegisteredHistoricalDatasets:
        datasets: dict[str, HistoricalDatasetSnapshot] = {}
        for dataset_id, path in paths.items():
            if not dataset_id or len(dataset_id) > 64:
                raise ValueError(
                    "historical dataset IDs must contain between 1 and 64 characters"
                )
            series = load_historical_series(
                path,
                order_policy=HistoricalOrderPolicy.REQUIRE_ASCENDING,
            )
            datasets[dataset_id] = HistoricalDatasetSnapshot(
                dataset_id=dataset_id,
                years=series.years,
                nominal_returns=series.nominal_returns,
                inflation_rates=series.inflation_rates,
                asset_returns=series.asset_returns,
                content_sha256=series.sha256,
                observations_sha256=series.observations_sha256,
                order_policy=series.order_policy,
                gap_policy=series.gap_policy,
            )
        return cls(datasets)

    def resolve(
        self,
        dataset_id: str,
    ) -> HistoricalDatasetSnapshot | None:
        return self._datasets.get(dataset_id)
