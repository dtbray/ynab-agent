"""Frozen, replay-safe contracts for continuous plan calibration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
import hashlib
import json
import math
import unicodedata
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema
from pydantic.json_schema import JsonSchemaValue

from ynab_agent.planning.models import WealthScenario
from ynab_agent.services.planner_jobs import HistoricalDatasetSnapshot


CALIBRATION_SNAPSHOT_SCHEMA_VERSION = 1
MAX_OBSERVATIONS_PER_SNAPSHOT = 10_000
MAX_SOURCE_LINKS_PER_OBSERVATION = 100
MAX_SOURCE_WATERMARKS = 32
MAX_DRIFT_EVENTS_PER_SNAPSHOT = 2_000
MAX_OBSERVATION_VALUE_FIELDS = 64
MAX_CONFIDENCE_NOTES = 8
MAX_IDENTIFIER_LENGTH = 200
MAX_SOURCE_URI_LENGTH = 1_024
MAX_JSON_DEPTH = 16
MAX_JSON_CONTAINER_ITEMS = 1_024
MAX_JSON_NODES = 4_096
MAX_JSON_STRING_LENGTH = 4_096
MAX_OBSERVATION_JSON_BYTES = 65_536
MAX_CANONICAL_JSON_DEPTH = 64
MAX_CANONICAL_JSON_CONTAINER_ITEMS = 10_000
MAX_CANONICAL_JSON_NODES = 20_000_000
MAX_CANONICAL_JSON_BYTES = 67_108_864
MAX_EVIDENCE_TEXT_LENGTH = 500
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FrozenJsonObject(Mapping[str, object]):
    """Static, pickle-safe JSON object backed only by immutable tuples."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object] | None = None) -> None:
        normalized: dict[str, object] = {}
        for raw_key, item in (value or {}).items():
            key = _normalize_text(raw_key)
            if key in normalized:
                raise ValueError("JSON keys collide after NFC normalization")
            normalized[key] = _freeze_json(item)
        self._items = tuple(sorted(normalized.items()))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __reduce__(
        self,
    ) -> tuple[type[FrozenJsonObject], tuple[dict[str, object]]]:
        return type(self), (self.to_dict(),)

    def to_dict(self) -> dict[str, object]:
        return {key: _thaw_json(item) for key, item in self._items}

    @classmethod
    def _validate(cls, value: object) -> FrozenJsonObject:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("immutable JSON objects require a mapping")
        return cls(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            json_schema_input_schema=core_schema.dict_schema(
                keys_schema=core_schema.str_schema(),
                values_schema=core_schema.any_schema(),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value.to_dict(),
                when_used="always",
            ),
        )


type ImmutableJson = None | bool | int | float | str | tuple[ImmutableJson, ...] | FrozenJsonObject


class FrozenFloatMap(Mapping[str, float]):
    """Static normalized mapping for bounded allocation weights."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
        normalized: dict[str, float] = {}
        for raw_key, raw_value in value.items():
            key = _normalize_text(raw_key)
            if key in normalized:
                raise ValueError("allocation classes collide after NFC normalization")
            if isinstance(raw_value, bool) or not isinstance(
                raw_value,
                int | float,
            ):
                raise TypeError("allocation weights must be numeric")
            weight = _lossless_float(raw_value, label="allocation weight")
            if not math.isfinite(weight):
                raise ValueError("allocation weights must be finite")
            normalized[key] = weight
        self._items = tuple(sorted(normalized.items()))

    def __getitem__(self, key: str) -> float:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(
        self,
    ) -> tuple[type[FrozenFloatMap], tuple[dict[str, float]]]:
        return type(self), (dict(self._items),)

    @classmethod
    def _validate(cls, value: object) -> FrozenFloatMap:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("allocation weights require a mapping")
        return cls(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            json_schema_input_schema=core_schema.dict_schema(
                keys_schema=core_schema.str_schema(),
                values_schema=core_schema.float_schema(),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: dict(value.items()),
                when_used="always",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {
            "type": "object",
            "additionalProperties": {"type": "number"},
        }


class FrozenWealthScenario(FrozenJsonObject):
    """Validated immutable scenario snapshot with explicit thawing."""

    __slots__ = ()

    @classmethod
    def _validate(cls, value: object) -> FrozenWealthScenario:
        if isinstance(value, cls):
            return value
        if isinstance(value, WealthScenario):
            scenario = value
        elif isinstance(value, Mapping):
            scenario = WealthScenario.model_validate(_thaw_json(value))
        else:
            raise TypeError("resolved_scenario requires a WealthScenario")
        return cls(scenario.model_dump(mode="python"))

    def to_wealth_scenario(self) -> WealthScenario:
        """Return a detached mutable scenario for simulation code."""
        return WealthScenario.model_validate(self.to_dict())


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("JSON object keys must be strings")
    return unicodedata.normalize("NFC", value)


def _exact_decimal(value: int | float, *, label: str) -> Decimal:
    """Preserve a JSON number exactly before any bounded float computation."""
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        decimal = Decimal(value) if isinstance(value, int) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not decimal.is_finite():
        raise ValueError(f"{label} must be finite")
    return decimal


def _lossless_float(value: int | float, *, label: str) -> float:
    """Convert only values whose exact decimal identity survives binary float."""
    decimal = _exact_decimal(value, label=label)
    try:
        converted = float(decimal)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} is outside the detector numeric range") from error
    if not math.isfinite(converted) or Decimal(str(converted)) != decimal:
        raise ValueError(f"{label} cannot be represented exactly by the detector")
    return converted


def _freeze_json(value: object) -> ImmutableJson:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Enum):
        return _freeze_json(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("JSON datetimes must include a timezone")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Mapping):
        return FrozenJsonObject(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"unsupported immutable JSON value: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, FrozenJsonObject):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _normalize_input(value: object) -> object:
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, item in value.items():
            key = _normalize_text(raw_key)
            if key in normalized:
                raise ValueError("input keys collide after NFC normalization")
            normalized[key] = _normalize_input(item)
        return normalized
    if isinstance(value, list | tuple):
        return tuple(_normalize_input(item) for item in value)
    if isinstance(value, str):
        return _normalize_text(value)
    return value


def _assert_static_storage(value: object) -> None:
    if isinstance(value, dict | list):
        raise TypeError("calibration models cannot retain mutable containers")
    if isinstance(value, FrozenJsonObject | FrozenFloatMap):
        return
    if isinstance(value, tuple | frozenset):
        for item in value:
            _assert_static_storage(item)
        return
    if isinstance(value, FrozenCalibrationModel):
        for item in value.__dict__.values():
            _assert_static_storage(item)


class FrozenCalibrationModel(BaseModel):
    """Base configuration shared by public calibration contracts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        return _normalize_input(value)

    @model_validator(mode="after")
    def require_static_storage(self) -> FrozenCalibrationModel:
        for field_value in self.__dict__.values():
            _assert_static_storage(field_value)
        return self


class _CanonicalBudget:
    __slots__ = ("bytes", "max_bytes", "max_container_items", "max_depth", "max_nodes", "nodes")

    def __init__(
        self,
        *,
        max_depth: int,
        max_container_items: int,
        max_nodes: int,
        max_bytes: int,
    ) -> None:
        self.max_depth = max_depth
        self.max_container_items = max_container_items
        self.max_nodes = max_nodes
        self.max_bytes = max_bytes
        self.nodes = 0
        self.bytes = 0

    def add_node(self, *, depth: int) -> None:
        if depth > self.max_depth:
            raise ValueError("canonical JSON exceeds the depth limit")
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ValueError("canonical JSON exceeds the node limit")

    def add_bytes(self, value: bytes) -> bytes:
        self.bytes += len(value)
        if self.bytes > self.max_bytes:
            raise ValueError("canonical JSON exceeds the byte limit")
        return value


def canonical_content_sha256(
    value: object,
    *,
    exclude: frozenset[str] = frozenset(),
) -> str:
    """Stream one normalized JSON representation into a content digest."""
    budget = _CanonicalBudget(
        max_depth=MAX_CANONICAL_JSON_DEPTH,
        max_container_items=MAX_CANONICAL_JSON_CONTAINER_ITEMS,
        max_nodes=MAX_CANONICAL_JSON_NODES,
        max_bytes=MAX_CANONICAL_JSON_BYTES,
    )
    digest = hashlib.sha256()
    for chunk in _canonical_chunks(
        value,
        budget=budget,
        depth=0,
        root_exclude=exclude,
    ):
        digest.update(chunk)
    return digest.hexdigest()


def _canonical_chunks(
    value: object,
    *,
    budget: _CanonicalBudget,
    depth: int,
    root_exclude: frozenset[str] = frozenset(),
) -> Iterator[bytes]:
    if isinstance(value, BaseModel):
        value = {
            name: getattr(value, name)
            for name in type(value).model_fields
            if name not in root_exclude
        }
        root_exclude = frozenset()
    budget.add_node(depth=depth)
    if value is None or isinstance(value, bool):
        yield budget.add_bytes(json.dumps(value, separators=(",", ":")).encode("utf-8"))
        return
    if isinstance(value, Enum):
        yield from _canonical_chunks(
            value.value,
            budget=budget,
            depth=depth,
        )
        return
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical datetimes must include a timezone")
        value = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    elif isinstance(value, date):
        value = value.isoformat()
    if isinstance(value, str):
        normalized = _normalize_text(value)
        if len(normalized) > MAX_JSON_STRING_LENGTH:
            raise ValueError("canonical JSON string exceeds the length limit")
        yield budget.add_bytes(
            json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return
    if isinstance(value, int):
        yield budget.add_bytes(str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        if value == 0:
            value = 0
        elif value.is_integer():
            value = int(value)
        yield budget.add_bytes(str(value).encode("ascii"))
        return
    if isinstance(value, Mapping):
        if len(value) > budget.max_container_items:
            raise ValueError("canonical JSON object exceeds the item limit")
        normalized_items: dict[str, object] = {}
        for raw_key, item in value.items():
            key = _normalize_text(raw_key)
            if len(key) > MAX_JSON_STRING_LENGTH:
                raise ValueError("canonical JSON key exceeds the length limit")
            if key in normalized_items:
                raise ValueError("canonical JSON keys collide after normalization")
            normalized_items[key] = item
        yield budget.add_bytes(b"{")
        for index, key in enumerate(sorted(normalized_items)):
            if index:
                yield budget.add_bytes(b",")
            yield budget.add_bytes(
                json.dumps(
                    key,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            yield budget.add_bytes(b":")
            yield from _canonical_chunks(
                normalized_items[key],
                budget=budget,
                depth=depth + 1,
            )
        yield budget.add_bytes(b"}")
        return
    if isinstance(value, list | tuple):
        if len(value) > budget.max_container_items:
            raise ValueError("canonical JSON array exceeds the item limit")
        yield budget.add_bytes(b"[")
        for index, item in enumerate(value):
            if index:
                yield budget.add_bytes(b",")
            yield from _canonical_chunks(
                item,
                budget=budget,
                depth=depth + 1,
            )
        yield budget.add_bytes(b"]")
        return
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _validate_bounded_json(value: object, *, label: str) -> None:
    budget = _CanonicalBudget(
        max_depth=MAX_JSON_DEPTH,
        max_container_items=MAX_JSON_CONTAINER_ITEMS,
        max_nodes=MAX_JSON_NODES,
        max_bytes=MAX_OBSERVATION_JSON_BYTES,
    )
    try:
        for _ in _canonical_chunks(value, budget=budget, depth=0):
            pass
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: {error}") from error


class ObservationKind(StrEnum):
    """Financial fact represented by one immutable observation."""

    SPENDING = "spending"
    CONTRIBUTION = "contribution"
    DEBT_BALANCE = "debt_balance"
    ACCOUNT_BALANCE = "account_balance"
    ASSET_ALLOCATION = "asset_allocation"
    ACCOUNT_FRESHNESS = "account_freshness"


class ObservationSourceKind(StrEnum):
    """Origin of an observation copied into a calibration snapshot."""

    YNAB_SYNC = "ynab_sync"
    USER_REVIEWED = "user_reviewed"
    IMPORTED_REVIEWED = "imported_reviewed"
    DERIVED_CACHE_QUERY = "derived_cache_query"


class FreshnessState(StrEnum):
    """Evidence-based freshness state without implying valuation accuracy."""

    FRESH = "fresh"
    STALE = "stale"
    FROZEN = "frozen"
    UNKNOWN = "unknown"


def classify_freshness(
    *,
    evaluated_at: datetime,
    latest_activity_at: datetime | None,
    last_reconciled_at: datetime | None,
    latest_reviewed_at: datetime | None,
    unchanged_since: datetime | None,
    stale_after_days: int,
    frozen_after_days: int,
) -> tuple[
    FreshnessState,
    datetime | None,
    int | None,
    int | None,
    tuple[str, ...],
]:
    """Replay a freshness state from every decisive source timestamp."""
    evidence_by_name = {
        "latest_activity": latest_activity_at,
        "last_reconciled": last_reconciled_at,
        "reviewed_valuation": latest_reviewed_at,
        "unchanged_balance_window": unchanged_since,
    }
    if any(
        timestamp is not None and timestamp > evaluated_at
        for timestamp in evidence_by_name.values()
    ):
        raise ValueError("freshness evidence cannot postdate evaluated_at")
    latest_evidence = max(
        (
            timestamp
            for timestamp in (
                latest_activity_at,
                last_reconciled_at,
                latest_reviewed_at,
            )
            if timestamp is not None
        ),
        default=None,
    )
    age_days = (
        int((evaluated_at - latest_evidence).total_seconds() // 86_400)
        if latest_evidence is not None
        else None
    )
    unchanged_days = (
        int((evaluated_at - unchanged_since).total_seconds() // 86_400)
        if unchanged_since is not None
        else None
    )
    valuation_review = max(
        (
            timestamp
            for timestamp in (last_reconciled_at, latest_reviewed_at)
            if timestamp is not None
        ),
        default=None,
    )
    frozen = (
        unchanged_since is not None
        and unchanged_days is not None
        and unchanged_days >= frozen_after_days
        and (valuation_review is None or valuation_review <= unchanged_since)
    )
    if frozen:
        state = FreshnessState.FROZEN
    elif latest_evidence is None:
        state = FreshnessState.UNKNOWN
    elif age_days is not None and age_days >= stale_after_days:
        state = FreshnessState.STALE
    else:
        state = FreshnessState.FRESH
    evidence = tuple(
        name for name, timestamp in sorted(evidence_by_name.items()) if timestamp is not None
    )
    return state, latest_evidence, age_days, unchanged_days, evidence


class DriftKind(StrEnum):
    """Supported plan-versus-observation drift domains."""

    SPENDING = "spending"
    CONTRIBUTION = "contribution"
    DEBT_PAYOFF = "debt_payoff"
    BALANCE = "balance"
    ALLOCATION = "allocation"
    STALENESS = "staleness"


class ObservationUnit(StrEnum):
    """Unit carried by scalar and structured observations."""

    DOLLARS = "dollars"
    DOLLARS_PER_YEAR = "dollars_per_year"
    FRACTION = "fraction"
    DAYS = "days"
    STRUCTURED = "structured"


_OBSERVATION_UNIT_BY_KIND = {
    ObservationKind.SPENDING: ObservationUnit.DOLLARS_PER_YEAR,
    ObservationKind.CONTRIBUTION: ObservationUnit.DOLLARS_PER_YEAR,
    ObservationKind.DEBT_BALANCE: ObservationUnit.DOLLARS,
    ObservationKind.ACCOUNT_BALANCE: ObservationUnit.DOLLARS,
    ObservationKind.ASSET_ALLOCATION: ObservationUnit.FRACTION,
    ObservationKind.ACCOUNT_FRESHNESS: ObservationUnit.DAYS,
}
_FRESHNESS_REQUIRED_KINDS = {
    ObservationKind.DEBT_BALANCE,
    ObservationKind.ACCOUNT_BALANCE,
    ObservationKind.ASSET_ALLOCATION,
    ObservationKind.ACCOUNT_FRESHNESS,
}


class ThresholdMode(StrEnum):
    """How configured absolute and relative materiality tests combine."""

    ANY = "any"
    ALL = "all"


class SourceWatermark(FrozenCalibrationModel):
    """YNAB delta checkpoint copied at snapshot time."""

    budget_id: str = Field(min_length=1, max_length=64)
    resource: str = Field(min_length=1, max_length=64)
    server_knowledge: int = Field(ge=0)
    synced_at: datetime
    change_batch_id: str = Field(min_length=1, max_length=64)

    @field_validator("synced_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source watermark synced_at must include a timezone")
        return value


def _watermark_identity(
    watermark: SourceWatermark,
) -> tuple[str, str, int, datetime, str]:
    return (
        watermark.budget_id,
        watermark.resource,
        watermark.server_knowledge,
        watermark.synced_at,
        watermark.change_batch_id,
    )


class SourceObservationLink(FrozenCalibrationModel):
    """Stable pointer plus copied digest for a mutable upstream fact."""

    source_kind: ObservationSourceKind
    source_id: str = Field(min_length=1, max_length=256)
    source_uri: str = Field(min_length=1, max_length=MAX_SOURCE_URI_LENGTH)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    observed_at: datetime
    change_batch_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source observation observed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_sync_batch(self) -> SourceObservationLink:
        if self.source_kind is ObservationSourceKind.YNAB_SYNC and self.change_batch_id is None:
            raise ValueError("YNAB source links require a change_batch_id")
        return self


class FreshnessAssessment(FrozenCalibrationModel):
    """Auditable stale/frozen classification for one tracked subject."""

    state: FreshnessState
    evaluated_at: datetime
    latest_activity_at: datetime | None = None
    last_reconciled_at: datetime | None = None
    latest_reviewed_at: datetime | None = None
    latest_evidence_at: datetime | None = None
    unchanged_since: datetime | None = None
    age_days: int | None = Field(default=None, ge=0)
    unchanged_days: int | None = Field(default=None, ge=0)
    stale_after_days: int = Field(gt=0, le=3_650)
    frozen_after_days: int = Field(gt=0, le=3_650)
    evidence: tuple[str, ...] = Field(default=(), max_length=MAX_CONFIDENCE_NOTES)

    @field_validator(
        "evaluated_at",
        "latest_activity_at",
        "last_reconciled_at",
        "latest_reviewed_at",
        "latest_evidence_at",
        "unchanged_since",
    )
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("freshness timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_state_evidence(self) -> FreshnessAssessment:
        if self.frozen_after_days < self.stale_after_days:
            raise ValueError("frozen_after_days must be at least stale_after_days")
        (
            expected_state,
            expected_latest,
            expected_age,
            expected_unchanged,
            expected_evidence,
        ) = classify_freshness(
            evaluated_at=self.evaluated_at,
            latest_activity_at=self.latest_activity_at,
            last_reconciled_at=self.last_reconciled_at,
            latest_reviewed_at=self.latest_reviewed_at,
            unchanged_since=self.unchanged_since,
            stale_after_days=self.stale_after_days,
            frozen_after_days=self.frozen_after_days,
        )
        if self.state is not expected_state:
            raise ValueError("freshness state does not match source evidence")
        if self.latest_evidence_at != expected_latest:
            raise ValueError("latest_evidence_at must match decisive source timestamps")
        if self.age_days != expected_age:
            raise ValueError("freshness age_days must match latest_evidence_at")
        if self.unchanged_days != expected_unchanged:
            raise ValueError("freshness unchanged_days must match unchanged_since")
        if self.evidence != expected_evidence:
            raise ValueError("freshness evidence labels must match timestamps")
        return self


class ObservationPayload(FrozenCalibrationModel):
    """Bounded copied value used to reproduce later calibration decisions."""

    unit: ObservationUnit
    value: ImmutableJson
    fields: FrozenJsonObject = Field(default_factory=FrozenJsonObject)

    @model_validator(mode="before")
    @classmethod
    def reject_oversized_input(cls, value: object) -> object:
        if isinstance(value, Mapping):
            _validate_bounded_json(
                {
                    "value": value.get("value"),
                    "fields": value.get("fields", {}),
                },
                label="observation payload",
            )
        return value

    @model_validator(mode="after")
    def validate_payload_bounds(self) -> ObservationPayload:
        if len(self.fields) > MAX_OBSERVATION_VALUE_FIELDS:
            raise ValueError("observation fields exceed the item limit")
        _validate_bounded_json(
            {"value": self.value, "fields": self.fields},
            label="observation payload",
        )
        return self


class CalibrationObservation(FrozenCalibrationModel):
    """One content-addressed financial observation and its source evidence."""

    id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    kind: ObservationKind
    subject_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    effective_date: date
    observed_at: datetime
    payload: ObservationPayload
    freshness: FreshnessAssessment | None = None
    source_links: tuple[SourceObservationLink, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_LINKS_PER_OBSERVATION,
    )
    source_watermarks: tuple[SourceWatermark, ...] = Field(
        default=(),
        max_length=MAX_SOURCE_WATERMARKS,
    )
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibration observation observed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def verify_content_hash(self) -> CalibrationObservation:
        if self.payload.unit is not _OBSERVATION_UNIT_BY_KIND[self.kind]:
            raise ValueError("observation unit does not match observation kind")
        if self.kind in _FRESHNESS_REQUIRED_KINDS and self.freshness is None:
            raise ValueError("valuation and allocation observations require freshness")
        if self.effective_date > self.observed_at.astimezone(timezone.utc).date():
            raise ValueError("observation effective_date cannot postdate observed_at")
        if any(link.observed_at > self.observed_at for link in self.source_links):
            raise ValueError("source links cannot postdate observation observed_at")
        if any(watermark.synced_at > self.observed_at for watermark in self.source_watermarks):
            raise ValueError("observation watermarks cannot postdate observation observed_at")
        link_keys = tuple((link.source_kind, link.source_id) for link in self.source_links)
        if len(set(link_keys)) != len(link_keys):
            raise ValueError("observation source links must be unique")
        watermark_keys = tuple(
            (watermark.budget_id, watermark.resource) for watermark in self.source_watermarks
        )
        if len(set(watermark_keys)) != len(watermark_keys):
            raise ValueError("observation source watermarks must be unique")
        if self.freshness is not None and self.freshness.evaluated_at != self.observed_at:
            raise ValueError("observation freshness must be evaluated at observation observed_at")
        if self.kind is ObservationKind.ACCOUNT_FRESHNESS and self.freshness is not None:
            expected_age = (
                self.freshness.unchanged_days
                if self.freshness.state is FreshnessState.FROZEN
                else self.freshness.age_days
            )
            if self.payload.value != expected_age:
                raise ValueError("account freshness observation value must match its decisive age")
        object.__setattr__(
            self,
            "source_links",
            tuple(
                sorted(
                    self.source_links,
                    key=lambda link: (
                        link.source_kind.value,
                        link.source_id,
                        link.source_uri,
                        link.observed_at,
                        link.content_sha256,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "source_watermarks",
            tuple(
                sorted(
                    self.source_watermarks,
                    key=_watermark_identity,
                )
            ),
        )
        expected = canonical_content_sha256(
            self,
            exclude=frozenset({"content_sha256"}),
        )
        if self.content_sha256 != expected:
            raise ValueError("calibration observation content hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        kind: ObservationKind,
        subject_id: str,
        effective_date: date,
        observed_at: datetime,
        payload: ObservationPayload,
        source_links: tuple[SourceObservationLink, ...],
        freshness: FreshnessAssessment | None = None,
        source_watermarks: tuple[SourceWatermark, ...] = (),
    ) -> CalibrationObservation:
        canonical_links = tuple(
            sorted(
                source_links,
                key=lambda link: (
                    link.source_kind.value,
                    link.source_id,
                    link.source_uri,
                    link.observed_at,
                    link.content_sha256,
                ),
            )
        )
        canonical_watermarks = tuple(sorted(source_watermarks, key=_watermark_identity))
        draft = cls.model_construct(
            id=id,
            kind=kind,
            subject_id=subject_id,
            effective_date=effective_date,
            observed_at=observed_at,
            payload=payload,
            freshness=freshness,
            source_links=canonical_links,
            source_watermarks=canonical_watermarks,
            content_sha256="0" * 64,
        )
        material = draft.model_dump(
            mode="python",
            exclude={"content_sha256"},
        )
        return cls.model_validate(
            {
                **material,
                "content_sha256": canonical_content_sha256(material),
            }
        )


class ConfidenceAssessment(FrozenCalibrationModel):
    """Bounded evidence quality attached to drift and attribution."""

    score: float = Field(ge=0, le=1)
    basis: str = Field(min_length=1, max_length=500)
    limitations: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_CONFIDENCE_NOTES,
    )

    @model_validator(mode="after")
    def validate_limitations(self) -> ConfidenceAssessment:
        if len(set(self.limitations)) != len(self.limitations):
            raise ValueError("confidence limitations must be unique")
        if any(not item or len(item) > MAX_EVIDENCE_TEXT_LENGTH for item in self.limitations):
            raise ValueError("confidence limitations are invalid")
        object.__setattr__(
            self,
            "limitations",
            tuple(sorted(self.limitations)),
        )
        return self


class MaterialityThreshold(FrozenCalibrationModel):
    """Configurable threshold applied consistently across drift detectors."""

    absolute: float | None = Field(default=None, gt=0)
    relative: float | None = Field(default=None, gt=0, le=10)
    relative_floor: float = Field(default=1, gt=0)
    mode: ThresholdMode = ThresholdMode.ANY
    minimum_confidence: float = Field(default=0, ge=0, le=1)

    @field_validator(
        "absolute",
        "relative",
        "relative_floor",
        "minimum_confidence",
        mode="before",
    )
    @classmethod
    def reject_lossy_numeric_coercion(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("materiality threshold values must be JSON numbers")
        return _lossless_float(value, label="materiality threshold")

    @model_validator(mode="after")
    def require_one_threshold(self) -> MaterialityThreshold:
        if self.absolute is None and self.relative is None:
            raise ValueError("materiality requires an absolute or relative threshold")
        return self


class CalibrationPolicy(FrozenCalibrationModel):
    """Versioned drift and freshness policy copied into every snapshot."""

    spending: MaterialityThreshold
    contribution: MaterialityThreshold
    debt_payoff: MaterialityThreshold
    balance: MaterialityThreshold
    allocation: MaterialityThreshold
    staleness: MaterialityThreshold
    stale_after_days: int = Field(default=45, gt=0, le=3_650)
    frozen_after_days: int = Field(default=90, gt=0, le=3_650)
    lookback_months: int = Field(default=12, ge=1, le=120)

    @model_validator(mode="after")
    def validate_freshness_windows(self) -> CalibrationPolicy:
        if self.frozen_after_days < self.stale_after_days:
            raise ValueError("frozen_after_days must be at least stale_after_days")
        return self


_DRIFT_OBSERVATION_KIND = {
    DriftKind.SPENDING: ObservationKind.SPENDING,
    DriftKind.CONTRIBUTION: ObservationKind.CONTRIBUTION,
    DriftKind.DEBT_PAYOFF: ObservationKind.DEBT_BALANCE,
    DriftKind.BALANCE: ObservationKind.ACCOUNT_BALANCE,
    DriftKind.ALLOCATION: ObservationKind.ASSET_ALLOCATION,
    DriftKind.STALENESS: ObservationKind.ACCOUNT_FRESHNESS,
}
_DRIFT_POLICY_FIELD = {
    DriftKind.SPENDING: "spending",
    DriftKind.CONTRIBUTION: "contribution",
    DriftKind.DEBT_PAYOFF: "debt_payoff",
    DriftKind.BALANCE: "balance",
    DriftKind.ALLOCATION: "allocation",
    DriftKind.STALENESS: "staleness",
}


class DriftEvent(FrozenCalibrationModel):
    """One observed association between a plan input and copied source data."""

    kind: DriftKind
    subject_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    baseline_value: ImmutableJson
    observed_value: ImmutableJson
    unit: ObservationUnit
    absolute_delta: float = Field(ge=0)
    signed_delta: float
    relative_delta: float | None = None
    material: bool
    threshold: MaterialityThreshold
    confidence: ConfidenceAssessment
    observation_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_LINKS_PER_OBSERVATION,
    )
    freshness: FreshnessAssessment | None = None
    interpretation: Literal["observed_association_not_causal"] = "observed_association_not_causal"
    details: FrozenJsonObject = Field(default_factory=FrozenJsonObject)
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def reject_oversized_input(cls, value: object) -> object:
        if isinstance(value, Mapping):
            _validate_bounded_json(
                {
                    "baseline_value": value.get("baseline_value"),
                    "observed_value": value.get("observed_value"),
                    "details": value.get("details", {}),
                },
                label="drift event values",
            )
        return value

    @field_validator(
        "absolute_delta",
        "signed_delta",
        "relative_delta",
        mode="before",
    )
    @classmethod
    def reject_lossy_delta_coercion(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("drift deltas must be JSON numbers")
        return _lossless_float(value, label="drift delta")

    @model_validator(mode="after")
    def verify_content_hash(self) -> DriftEvent:
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("drift observation IDs must be unique")
        if any(not item or len(item) > MAX_IDENTIFIER_LENGTH for item in self.observation_ids):
            raise ValueError("drift observation IDs are invalid")
        object.__setattr__(
            self,
            "observation_ids",
            tuple(sorted(self.observation_ids)),
        )
        if len(self.details) > MAX_OBSERVATION_VALUE_FIELDS:
            raise ValueError("drift event details exceed the item limit")
        _validate_bounded_json(
            {
                "baseline_value": self.baseline_value,
                "observed_value": self.observed_value,
                "details": self.details,
            },
            label="drift event values",
        )
        expected = canonical_content_sha256(
            self,
            exclude=frozenset({"content_sha256"}),
        )
        if self.content_sha256 != expected:
            raise ValueError("drift event content hash mismatch")
        return self


class CalibrationSnapshotManifest(FrozenCalibrationModel):
    """Complete immutable source and scenario state preceding an automated run."""

    schema_version: int = Field(
        default=CALIBRATION_SNAPSHOT_SCHEMA_VERSION,
        ge=1,
    )
    profile_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    scenario_revision_id: str = Field(min_length=36, max_length=36)
    scenario_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    previous_snapshot_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
    )
    source_sync_batch_id: str = Field(min_length=1, max_length=64)
    as_of: datetime
    resolved_scenario: FrozenWealthScenario
    resolved_scenario_sha256: str = Field(pattern=SHA256_PATTERN)
    historical_dataset: HistoricalDatasetSnapshot | None = None
    policy: CalibrationPolicy
    observations: tuple[CalibrationObservation, ...] = Field(
        min_length=1,
        max_length=MAX_OBSERVATIONS_PER_SNAPSHOT,
    )
    drift_events: tuple[DriftEvent, ...] = Field(
        default=(),
        max_length=MAX_DRIFT_EVENTS_PER_SNAPSHOT,
    )
    source_watermarks: tuple[SourceWatermark, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_WATERMARKS,
    )

    @field_validator("as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibration snapshot as_of must include a timezone")
        return value

    @model_validator(mode="after")
    def verify_resolved_scenario_hash(self) -> CalibrationSnapshotManifest:
        expected = canonical_content_sha256(self.resolved_scenario)
        if self.resolved_scenario_sha256 != expected:
            raise ValueError("resolved scenario content hash mismatch")
        observation_ids = tuple(observation.id for observation in self.observations)
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("snapshot observation IDs must be unique")
        upstream_links: set[tuple[object, ...]] = set()
        evidence_rows: set[tuple[object, ...]] = set()
        for observation in self.observations:
            if (
                observation.observed_at > self.as_of
                or observation.effective_date > self.as_of.astimezone(timezone.utc).date()
            ):
                raise ValueError("snapshot observations cannot postdate snapshot as_of")
            if any(link.observed_at > self.as_of for link in observation.source_links):
                raise ValueError("snapshot source links cannot postdate snapshot as_of")
            if any(watermark.synced_at > self.as_of for watermark in observation.source_watermarks):
                raise ValueError("snapshot observation watermarks cannot postdate snapshot as_of")
            if (
                observation.freshness is not None
                and observation.freshness.evaluated_at != self.as_of
            ):
                raise ValueError(
                    "snapshot observation freshness must be evaluated at snapshot as_of"
                )
            if observation.freshness is not None and (
                observation.freshness.stale_after_days != self.policy.stale_after_days
                or observation.freshness.frozen_after_days != self.policy.frozen_after_days
            ):
                raise ValueError("observation freshness windows must match snapshot policy")
            for link in observation.source_links:
                link_identity = (
                    link.source_kind,
                    link.source_id,
                    link.content_sha256,
                    link.observed_at,
                    link.change_batch_id,
                )
                if link_identity in upstream_links:
                    raise ValueError(
                        "distinct observation IDs cannot reuse an upstream source link"
                    )
                upstream_links.add(link_identity)
            evidence_identity = (
                observation.kind,
                observation.subject_id,
                observation.effective_date,
                canonical_content_sha256(observation.payload),
                tuple(sorted(link.content_sha256 for link in observation.source_links)),
                observation.payload.fields.get("drift_group_id"),
            )
            if evidence_identity in evidence_rows:
                raise ValueError("distinct observation IDs cannot duplicate payload/group evidence")
            evidence_rows.add(evidence_identity)
        if any(watermark.synced_at > self.as_of for watermark in self.source_watermarks):
            raise ValueError("snapshot watermarks cannot postdate snapshot as_of")
        if any(
            event.freshness is not None and event.freshness.evaluated_at != self.as_of
            for event in self.drift_events
        ):
            raise ValueError("drift freshness assessments must be evaluated at snapshot as_of")
        watermark_keys = tuple(
            (watermark.budget_id, watermark.resource) for watermark in self.source_watermarks
        )
        if len(set(watermark_keys)) != len(watermark_keys):
            raise ValueError("snapshot source watermarks must be unique")
        if any(
            watermark.change_batch_id != self.source_sync_batch_id
            for watermark in self.source_watermarks
        ):
            raise ValueError("snapshot watermark batches must match source_sync_batch_id")
        top_level_watermarks = {
            _watermark_identity(watermark) for watermark in self.source_watermarks
        }
        for observation in self.observations:
            if any(
                link.change_batch_id is not None
                and link.change_batch_id != self.source_sync_batch_id
                for link in observation.source_links
            ):
                raise ValueError("source link batches must match source_sync_batch_id")
            if any(
                _watermark_identity(watermark) not in top_level_watermarks
                for watermark in observation.source_watermarks
            ):
                raise ValueError("observation watermarks must match snapshot watermarks")
        observation_id_set = set(observation_ids)
        if any(
            observation_id not in observation_id_set
            for event in self.drift_events
            for observation_id in event.observation_ids
        ):
            raise ValueError("drift events must reference snapshot observations")
        observation_by_id = {observation.id: observation for observation in self.observations}
        for event in self.drift_events:
            expected_threshold = getattr(
                self.policy,
                _DRIFT_POLICY_FIELD[event.kind],
            )
            if event.threshold != expected_threshold:
                raise ValueError("drift event threshold must match snapshot policy")
            expected_kind = _DRIFT_OBSERVATION_KIND[event.kind]
            expected_unit = _OBSERVATION_UNIT_BY_KIND[expected_kind]
            if event.unit is not expected_unit:
                raise ValueError("drift event unit does not match drift kind")
            references = tuple(
                observation_by_id[observation_id] for observation_id in event.observation_ids
            )
            if any(
                observation.kind is not expected_kind
                or observation.subject_id != event.subject_id
                or observation.payload.unit is not event.unit
                for observation in references
            ):
                raise ValueError("drift event context does not match referenced observations")
            _verify_drift_evidence(
                event,
                references,
                resolved_scenario=self.resolved_scenario,
            )
            if event.freshness is not None and (
                event.freshness.stale_after_days != self.policy.stale_after_days
                or event.freshness.frozen_after_days != self.policy.frozen_after_days
            ):
                raise ValueError("drift freshness windows must match snapshot policy")
            if event.kind is DriftKind.STALENESS:
                if event.freshness is None:
                    raise ValueError("staleness drift requires a freshness assessment")
                if any(observation.freshness != event.freshness for observation in references):
                    raise ValueError("staleness drift freshness must match referenced observations")
                expected_baseline = FrozenJsonObject(
                    {
                        "stale_after_days": event.freshness.stale_after_days,
                        "frozen_after_days": event.freshness.frozen_after_days,
                    }
                )
                expected_observed = FrozenJsonObject(
                    {
                        "state": event.freshness.state.value,
                        "age_days": event.freshness.age_days,
                        "unchanged_days": event.freshness.unchanged_days,
                    }
                )
                if (
                    event.baseline_value != expected_baseline
                    or event.observed_value != expected_observed
                ):
                    raise ValueError("staleness drift values must match freshness evidence")
                observed_days = (
                    event.freshness.unchanged_days or 0
                    if event.freshness.state is FreshnessState.FROZEN
                    else (
                        event.freshness.age_days or 0
                        if event.freshness.state is FreshnessState.STALE
                        else 0
                    )
                )
                expected_relative = observed_days / event.threshold.relative_floor
                if (
                    event.absolute_delta != float(observed_days)
                    or event.signed_delta != float(observed_days)
                    or event.relative_delta != expected_relative
                ):
                    raise ValueError("staleness drift deltas must match freshness evidence age")
                threshold_tests = tuple(
                    test
                    for test in (
                        (
                            event.absolute_delta >= event.threshold.absolute
                            if event.threshold.absolute is not None
                            else None
                        ),
                        (
                            abs(expected_relative) >= event.threshold.relative
                            if event.threshold.relative is not None
                            else None
                        ),
                    )
                    if test is not None
                )
                passes_threshold = (
                    all(threshold_tests)
                    if event.threshold.mode is ThresholdMode.ALL
                    else any(threshold_tests)
                )
                expected_material = (
                    event.freshness.state in {FreshnessState.STALE, FreshnessState.FROZEN}
                    and event.confidence.score >= event.threshold.minimum_confidence
                    and passes_threshold
                )
                if event.material is not expected_material:
                    raise ValueError("staleness drift materiality must match freshness evidence")
            elif event.freshness is not None:
                raise ValueError("only staleness drift may include freshness")
        drift_hashes = tuple(event.content_sha256 for event in self.drift_events)
        if len(set(drift_hashes)) != len(drift_hashes):
            raise ValueError("snapshot drift events must be unique")
        object.__setattr__(
            self,
            "observations",
            tuple(
                sorted(
                    self.observations,
                    key=lambda observation: (
                        observation.id,
                        observation.content_sha256,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "drift_events",
            tuple(
                sorted(
                    self.drift_events,
                    key=lambda event: event.content_sha256,
                )
            ),
        )
        object.__setattr__(
            self,
            "source_watermarks",
            tuple(
                sorted(
                    self.source_watermarks,
                    key=_watermark_identity,
                )
            ),
        )
        canonical_content_sha256(self)
        return self


def _verify_drift_evidence(
    event: DriftEvent,
    references: tuple[CalibrationObservation, ...],
    *,
    resolved_scenario: FrozenWealthScenario,
) -> None:
    """Replay an event from its copied observations before it can drive automation."""
    if event.kind in {DriftKind.SPENDING, DriftKind.CONTRIBUTION}:
        by_role = _references_by_role(
            event,
            references,
            roles={"baseline", "observed"},
        )
        baseline_rows = by_role["baseline"]
        observed_rows = by_role["observed"]
        _require_shared_source_chronology(
            event,
            earlier=baseline_rows,
            later=observed_rows,
            earlier_role="baseline",
            later_role="observed",
        )
        if max(row.effective_date for row in baseline_rows) >= min(
            row.effective_date for row in observed_rows
        ):
            raise ValueError(
                f"{event.kind.value} drift baseline evidence must predate observed evidence"
            )
        baseline_value = _aggregate_scalar_evidence(
            event.kind,
            baseline_rows,
            role="baseline",
        )
        observed_value = _aggregate_scalar_evidence(
            event.kind,
            observed_rows,
            role="observed",
        )
        _verify_scalar_plan_baseline(
            event,
            baseline_rows,
            baseline_value=baseline_value,
            resolved_scenario=resolved_scenario,
        )
        _verify_replayed_event(
            event,
            baseline_value=baseline_value,
            observed_value=observed_value,
        )
        return
    if event.kind is DriftKind.BALANCE:
        by_role = _references_by_role(
            event,
            references,
            roles={"baseline", "current"},
            one_per_role=True,
        )
        baseline = by_role["baseline"][0]
        current = by_role["current"][0]
        _require_shared_source_chronology(
            event,
            earlier=(baseline,),
            later=(current,),
            earlier_role="baseline",
            later_role="current",
        )
        if baseline.effective_date >= current.effective_date:
            raise ValueError("balance baseline evidence must predate current evidence")
        baseline_value = _numeric_observation_value(baseline)
        _verify_balance_plan_baseline(
            event,
            baseline,
            baseline_value=baseline_value,
            resolved_scenario=resolved_scenario,
        )
        _verify_replayed_event(
            event,
            baseline_value=baseline_value,
            observed_value=_numeric_observation_value(current),
            replay_details=(
                {"account_balances": current.payload.fields["account_balances"]}
                if event.subject_id == "portfolio"
                and "account_balances" in current.payload.fields
                else None
            ),
        )
        return
    if event.kind is DriftKind.ALLOCATION:
        by_role = _references_by_role(
            event,
            references,
            roles={"target", "current"},
            one_per_role=True,
        )
        target = by_role["target"][0]
        current = by_role["current"][0]
        _require_shared_source_chronology(
            event,
            earlier=(target,),
            later=(current,),
            earlier_role="target",
            later_role="current",
        )
        if target.effective_date >= current.effective_date:
            raise ValueError("allocation target evidence must predate current evidence")
        if not isinstance(target.payload.value, Mapping) or not isinstance(
            current.payload.value,
            Mapping,
        ):
            raise ValueError("allocation drift evidence must contain an allocation mapping")
        try:
            target_weights = FrozenFloatMap(target.payload.value)
            current_weights = FrozenFloatMap(current.payload.value)
        except (TypeError, ValueError) as error:
            raise ValueError("allocation drift evidence is invalid") from error
        _verify_allocation_plan_target(
            event,
            target,
            resolved_scenario=resolved_scenario,
        )
        # Imported lazily because the pure detector depends on these evidence
        # contracts while snapshot verification must still replay its result.
        from ynab_agent.services.calibration_drift import (  # noqa: PLC0415
            AllocationDriftInput,
            detect_allocation_drift,
        )

        replayed = detect_allocation_drift(
            AllocationDriftInput(
                subject_id=event.subject_id,
                target_weights=target_weights,
                observed_weights=current_weights,
                threshold=event.threshold,
                confidence=event.confidence,
                observation_ids=event.observation_ids,
            )
        )
        if replayed != event:
            raise ValueError("allocation drift does not replay from copied evidence")
        return
    if event.kind is DriftKind.DEBT_PAYOFF:
        if len(references) != 2:
            raise ValueError(
                "debt payoff drift requires exactly one previous and one current observation"
            )
        debt_by_role: dict[str, CalibrationObservation] = {}
        for observation in references:
            role = observation.payload.fields.get("drift_role")
            if role not in {"previous", "current"} or role in debt_by_role:
                raise ValueError(
                    "debt payoff drift requires unambiguous previous/current evidence roles"
                )
            debt_by_role[role] = observation
        if set(debt_by_role) != {"previous", "current"}:
            raise ValueError(
                "debt payoff drift requires unambiguous previous/current evidence roles"
            )
        debt_previous = debt_by_role["previous"]
        debt_current = debt_by_role["current"]
        _require_shared_source_chronology(
            event,
            earlier=(debt_previous,),
            later=(debt_current,),
            earlier_role="previous",
            later_role="current",
        )
        if debt_previous.effective_date >= debt_current.effective_date:
            raise ValueError("debt payoff previous evidence must predate current evidence")
        _verify_replayed_event(
            event,
            baseline_value=_numeric_observation_value(debt_previous),
            observed_value=_numeric_observation_value(debt_current),
        )
        return
    if event.kind is DriftKind.STALENESS:
        _require_single_reference(event, references)
        return
    raise ValueError("unsupported drift kind cannot be bound to evidence")


def _references_by_role(
    event: DriftEvent,
    references: tuple[CalibrationObservation, ...],
    *,
    roles: set[str],
    one_per_role: bool = False,
) -> dict[str, tuple[CalibrationObservation, ...]]:
    grouped: dict[str, list[CalibrationObservation]] = {role: [] for role in roles}
    for observation in references:
        role = observation.payload.fields.get("drift_role")
        if not isinstance(role, str) or role not in roles:
            raise ValueError(
                f"{event.kind.value} drift requires unambiguous "
                f"{'/'.join(sorted(roles))} evidence roles"
            )
        grouped[role].append(observation)
    if any(not rows for rows in grouped.values()) or (
        one_per_role and any(len(rows) != 1 for rows in grouped.values())
    ):
        raise ValueError(
            f"{event.kind.value} drift requires unambiguous "
            f"{'/'.join(sorted(roles))} evidence roles"
        )
    return {role: tuple(rows) for role, rows in grouped.items()}


def _require_shared_source_chronology(
    event: DriftEvent,
    *,
    earlier: tuple[CalibrationObservation, ...],
    later: tuple[CalibrationObservation, ...],
    earlier_role: str,
    later_role: str,
) -> None:
    earlier_versions = _source_version_times(earlier)
    later_versions = _source_version_times(later)
    shared_sources = earlier_versions.keys() & later_versions.keys()
    if any(
        max(earlier_versions[source]) >= min(later_versions[source]) for source in shared_sources
    ):
        raise ValueError(
            f"{event.kind.value} drift shared source versions require "
            f"{earlier_role} observed_at before {later_role} observed_at"
        )


def _source_version_times(
    references: tuple[CalibrationObservation, ...],
) -> dict[tuple[ObservationSourceKind, str], tuple[datetime, ...]]:
    collected: dict[tuple[ObservationSourceKind, str], list[datetime]] = {}
    for observation in references:
        for link in observation.source_links:
            collected.setdefault(
                (link.source_kind, link.source_id),
                [],
            ).append(link.observed_at)
    return {source: tuple(timestamps) for source, timestamps in collected.items()}


def _verify_scalar_plan_baseline(
    event: DriftEvent,
    references: tuple[CalibrationObservation, ...],
    *,
    baseline_value: float,
    resolved_scenario: FrozenWealthScenario,
) -> None:
    expected_selector = {
        DriftKind.SPENDING: "annual_spending",
        DriftKind.CONTRIBUTION: "annual_contribution",
    }[event.kind]
    if len(references) != 1 or references[0].payload.fields.get("drift_aggregation") != "exact":
        raise ValueError(f"{event.kind.value} plan baseline requires one exact typed plan field")
    selector = references[0].payload.fields.get("plan_field")
    if selector != expected_selector:
        raise ValueError(
            f"{event.kind.value} plan baseline must select resolved_scenario.{expected_selector}"
        )
    expected_value = resolved_scenario.get(expected_selector)
    if isinstance(expected_value, bool) or not isinstance(expected_value, int | float):
        raise ValueError(f"resolved_scenario.{expected_selector} must be a numeric plan field")
    if _lossless_float(expected_value, label=f"resolved_scenario.{expected_selector}") != (
        baseline_value
    ):
        raise ValueError(
            f"{event.kind.value} plan baseline must equal resolved_scenario.{expected_selector}"
        )


def _verify_balance_plan_baseline(
    event: DriftEvent,
    reference: CalibrationObservation,
    *,
    baseline_value: float,
    resolved_scenario: FrozenWealthScenario,
) -> None:
    selector = reference.payload.fields.get("plan_field")
    if event.subject_id == "portfolio" and selector == "starting_portfolio":
        expected_value = resolved_scenario.get("starting_portfolio")
        field_label = "resolved_scenario.starting_portfolio"
    elif (
        isinstance(selector, str)
        and selector == f"tax_buckets[account_id={event.subject_id}].starting_balance"
    ):
        tax_buckets = resolved_scenario.get("tax_buckets")
        matching = (
            tuple(
                bucket
                for bucket in tax_buckets
                if isinstance(bucket, Mapping) and bucket.get("account_id") == event.subject_id
            )
            if isinstance(tax_buckets, tuple)
            else ()
        )
        if len(matching) != 1:
            raise ValueError("account balance plan baseline requires one linked tax bucket")
        expected_value = matching[0].get("starting_balance")
        field_label = (
            f"resolved_scenario.tax_buckets[account_id={event.subject_id}].starting_balance"
        )
    else:
        raise ValueError(
            "balance plan baseline requires a typed portfolio or linked-account "
            "starting-balance selector"
        )
    if isinstance(expected_value, bool) or not isinstance(expected_value, int | float):
        raise ValueError(f"balance plan baseline requires numeric {field_label}")
    if (
        _lossless_float(
            expected_value,
            label=field_label,
        )
        != baseline_value
    ):
        raise ValueError(f"balance plan baseline must equal {field_label}")


def _verify_allocation_plan_target(
    event: DriftEvent,
    reference: CalibrationObservation,
    *,
    resolved_scenario: FrozenWealthScenario,
) -> None:
    selector = reference.payload.fields.get("plan_field")
    expected_selector = f"portfolio_allocation.accounts[account_id={event.subject_id}].target"
    if selector != expected_selector:
        raise ValueError(
            "allocation drift requires a typed target-allocation field using "
            "the per-account allocation selector"
        )
    allocation = resolved_scenario.get("portfolio_allocation")
    accounts = allocation.get("accounts") if isinstance(allocation, Mapping) else None
    matching = (
        tuple(
            account
            for account in accounts
            if isinstance(account, Mapping) and account.get("account_id") == event.subject_id
        )
        if isinstance(accounts, tuple)
        else ()
    )
    if len(matching) != 1:
        raise ValueError("allocation plan target requires one matching allocation account")
    expected = matching[0].get("target")
    if not isinstance(expected, Mapping):
        raise ValueError("allocation plan target must be a typed weight mapping")
    reference_value = reference.payload.value
    if not isinstance(reference_value, Mapping):
        raise ValueError("allocation evidence target must be a weight mapping")
    if FrozenFloatMap(expected) != FrozenFloatMap(reference_value):
        raise ValueError("allocation plan target must equal the resolved per-account target")


def _aggregate_scalar_evidence(
    kind: DriftKind,
    references: tuple[CalibrationObservation, ...],
    *,
    role: str,
) -> float:
    modes = tuple(observation.payload.fields.get("drift_aggregation") for observation in references)
    if modes == ("exact",):
        return _numeric_observation_value(references[0])
    if not modes or set(modes) != {"sum"}:
        raise ValueError(f"{kind.value} drift {role} evidence requires an exact or sum aggregation")
    group_ids = tuple(
        observation.payload.fields.get("drift_group_id") for observation in references
    )
    if (
        any(not isinstance(group_id, str) or not group_id for group_id in group_ids)
        or len(set(group_ids)) != 1
    ):
        raise ValueError(f"{kind.value} drift {role} sum evidence requires one explicit group")
    stable_sources = tuple(
        (link.source_kind, link.source_id)
        for observation in references
        for link in observation.source_links
    )
    if len(set(stable_sources)) != len(stable_sources):
        raise ValueError(f"{kind.value} drift {role} sum evidence cannot reuse a stable source")
    evidence_keys = tuple(
        (
            canonical_content_sha256(observation.payload),
            tuple(sorted(link.content_sha256 for link in observation.source_links)),
        )
        for observation in references
    )
    if len(set(evidence_keys)) != len(evidence_keys):
        raise ValueError(f"{kind.value} drift sum evidence cannot be double counted")
    total = sum(
        (_numeric_observation_decimal(observation) for observation in references),
        start=Decimal(0),
    )
    return _lossless_float(total_as_number(total), label=f"{kind.value} drift sum")


def _require_single_reference(
    event: DriftEvent,
    references: tuple[CalibrationObservation, ...],
) -> CalibrationObservation:
    if len(references) != 1:
        raise ValueError(f"{event.kind.value} drift requires exactly one observation")
    return references[0]


def _numeric_observation_value(observation: CalibrationObservation) -> float:
    value = observation.payload.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{observation.kind.value} drift evidence must contain a numeric value")
    return _lossless_float(value, label=f"{observation.kind.value} drift evidence")


def _numeric_observation_decimal(observation: CalibrationObservation) -> Decimal:
    value = observation.payload.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{observation.kind.value} drift evidence must contain a numeric value")
    return _exact_decimal(value, label=f"{observation.kind.value} drift evidence")


def total_as_number(value: Decimal) -> int | float:
    """Retain exact integral sums, otherwise use canonical decimal text."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _verify_replayed_event(
    event: DriftEvent,
    *,
    observed_value: float | FrozenFloatMap,
    baseline_value: float | FrozenFloatMap | None = None,
    replay_details: Mapping[str, object] | None = None,
) -> None:
    # The local import keeps the detector layer dependent on contracts, not vice versa
    # at module import time, while ensuring one implementation owns all drift math.
    from ynab_agent.services.calibration_drift import (
        AllocationDriftInput,
        DebtPayoffInput,
        ScalarDriftInput,
        detect_allocation_drift,
        detect_balance_drift,
        detect_contribution_drift,
        detect_debt_payoff,
        detect_spending_drift,
    )

    if event.kind is DriftKind.DEBT_PAYOFF:
        payoff_tolerance = event.details.get("payoff_tolerance")
        if (
            baseline_value is None
            or isinstance(baseline_value, FrozenFloatMap)
            or isinstance(payoff_tolerance, bool)
            or not isinstance(payoff_tolerance, int | float)
            or isinstance(observed_value, FrozenFloatMap)
        ):
            raise ValueError("debt payoff drift derivation metadata is incomplete")
        replayed = detect_debt_payoff(
            DebtPayoffInput(
                subject_id=event.subject_id,
                previous_liability=baseline_value,
                current_liability=observed_value,
                payoff_tolerance=payoff_tolerance,
                threshold=event.threshold,
                confidence=event.confidence,
                observation_ids=event.observation_ids,
            )
        )
    elif event.kind is DriftKind.ALLOCATION:
        if not isinstance(baseline_value, FrozenFloatMap) or not isinstance(
            observed_value,
            FrozenFloatMap,
        ):
            raise ValueError("allocation drift derivation values are invalid")
        replayed = detect_allocation_drift(
            AllocationDriftInput(
                subject_id=event.subject_id,
                target_weights=baseline_value,
                observed_weights=observed_value,
                threshold=event.threshold,
                confidence=event.confidence,
                observation_ids=event.observation_ids,
            )
        )
    else:
        if (
            baseline_value is None
            or isinstance(baseline_value, FrozenFloatMap)
            or isinstance(observed_value, FrozenFloatMap)
        ):
            raise ValueError("scalar drift derivation values are invalid")
        candidate = ScalarDriftInput(
            subject_id=event.subject_id,
            baseline_value=baseline_value,
            observed_value=observed_value,
            unit=event.unit,
            threshold=event.threshold,
            confidence=event.confidence,
            observation_ids=event.observation_ids,
        )
        detector = {
            DriftKind.SPENDING: detect_spending_drift,
            DriftKind.CONTRIBUTION: detect_contribution_drift,
            DriftKind.BALANCE: detect_balance_drift,
        }[event.kind]
        replayed = detector(candidate)
    if replay_details is None and replayed != event:
        raise ValueError(
            "drift event values and derivation must match referenced observation evidence"
        )
    if replay_details is not None:
        expected_details = FrozenJsonObject(
            {
                **replayed.details.to_dict(),
                **replay_details,
            }
        )
        if (
            event.details != expected_details
            or event.model_dump(
                mode="python",
                exclude={"details", "content_sha256"},
            )
            != replayed.model_dump(
                mode="python",
                exclude={"details", "content_sha256"},
            )
        ):
            raise ValueError(
                "drift event values and derivation must match referenced observation evidence"
            )


class CalibrationSnapshot(FrozenCalibrationModel):
    """Content-addressed wrapper for an immutable calibration manifest."""

    id: str = Field(min_length=36, max_length=36)
    manifest: CalibrationSnapshotManifest
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calibration snapshot created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def verify_manifest_hash(self) -> CalibrationSnapshot:
        if self.manifest_sha256 != canonical_content_sha256(self.manifest):
            raise ValueError("calibration snapshot manifest hash mismatch")
        if self.created_at < self.manifest.as_of:
            raise ValueError("calibration snapshot cannot be created before as_of")
        return self
