"""Configuration owned by the inbound HTTP adapter."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HttpApiSettings(BaseSettings):
    """Security settings for the HTTP interface."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="YNAB_AGENT_HTTP_",
        extra="ignore",
    )

    api_key: SecretStr | None = None
    allow_unauthenticated_loopback: bool = True
    planner_max_workers: int = Field(default=1, ge=1, le=8)
    planner_max_pending_jobs: int = Field(default=4, ge=0, le=100)
    planner_maximum_total_working_bytes: int = Field(
        default=768 * 1024 * 1024,
        gt=0,
    )
    planner_maximum_working_bytes: int = Field(
        default=512 * 1024 * 1024,
        gt=0,
    )
    planner_in_memory_path_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=0,
    )
    planner_maximum_temporary_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,
        gt=0,
    )
    planner_batch_size: int = Field(default=10_000, gt=0)
    planner_maximum_compute_units: int = Field(
        default=200_000_000,
        gt=0,
    )
    planner_max_request_body_bytes: int = Field(
        default=64 * 1024,
        ge=1024,
        le=1024 * 1024,
    )
    planner_historical_datasets: dict[str, Path] = Field(default_factory=dict)
    calibration_poll_interval_seconds: float = Field(
        default=30,
        gt=0,
        le=3_600,
    )
