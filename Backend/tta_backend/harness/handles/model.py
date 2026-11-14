"""Typed handle records: what a handle is, where it came from and where its data is."""
from __future__ import annotations

import datetime
import math
import re
import secrets
from typing import Annotated, Any, Literal, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, ValidationInfo, field_validator, model_validator

from tta_backend.harness.release import HARNESS_RELEASE
from tta_backend.harness.results import RUN_ID_PATTERN

HandleKind = Literal["grid", "series", "points", "polygons", "table", "scalar"]
KINDS: tuple[HandleKind, ...] = get_args(HandleKind)

Origin = Literal["derived", "retrieval"]
PayloadFormat = Literal["zarr", "parquet", "geojson"]

# How each file-backed kind is stored; scalars live inline in the record.
PAYLOAD_FORMAT: dict[HandleKind, PayloadFormat] = {
    "grid": "zarr",
    "series": "parquet",
    "points": "parquet",
    "table": "parquet",
    "polygons": "geojson",
}

# The earthdata service's handle prefixes. Derived ids start with a kind,
# which is none of these, so the two can never collide.
RETRIEVAL_PREFIXES = ("dataset", "aoi", "obs", "cube", "job", "query", "preview")

# Anchored, and only characters that cannot form a path, so an id is never
# usable for traversal once it passes.
_DERIVED_ID = re.compile(r"(?P<kind>grid|series|points|polygons|table|scalar)_[0-9a-f]{32}")
_RETRIEVAL_ID = re.compile(r"(?:" + "|".join(RETRIEVAL_PREFIXES) + r")_[A-Za-z0-9_-]{1,128}")

# Validation context for a record read back from a registry. It was checked
# when written, so only its shape is parsed: a check tightened since then
# cannot make a stored record unreadable.
STORED: dict[str, Any] = {"stored": True}


def _stored(info: ValidationInfo) -> bool:
    return bool(info.context and info.context.get("stored"))


def new_handle_id(kind: HandleKind) -> str:
    return f"{kind}_{secrets.token_hex(16)}"


def is_derived_id(handle_id: str) -> bool:
    return _DERIVED_ID.fullmatch(handle_id) is not None


def is_retrieval_id(handle_id: str) -> bool:
    return _RETRIEVAL_ID.fullmatch(handle_id) is not None


def is_handle_id(handle_id: str) -> bool:
    return is_derived_id(handle_id) or is_retrieval_id(handle_id)


def _sanitize_non_finite(value: Any) -> Any:
    """Non-finite floats become None: Postgres JSONB rejects them, and both registries must agree."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_finite(v) for v in value]
    return value


def _has_nul(value: Any) -> bool:
    """Whether any string, key or value, holds NUL, which Postgres text and JSONB cannot store."""
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(key) or _has_nul(inner) for key, inner in value.items())
    if isinstance(value, list | tuple):
        return any(_has_nul(inner) for inner in value)
    return False


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Period(_Frozen):
    """ISO 8601 dates or datetimes, as given."""

    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _iso(cls, value: str, info: ValidationInfo) -> str:
        if not _stored(info):
            datetime.datetime.fromisoformat(value)
        return value


class Descriptor(_Frozen):
    """What the model and the provenance pane are told about a handle."""

    summary: str
    quantity: str | None = None
    unit: str | None = None
    product: str | None = None
    region: JsonValue = None
    period: Period | None = None
    masking: JsonValue = None

    @field_validator("summary")
    @classmethod
    def _one_line(cls, value: str, info: ValidationInfo) -> str:
        if _stored(info):
            return value
        if not 1 <= len(value) <= 500:
            raise ValueError("summary is 1 to 500 characters")
        if "\n" in value or "\r" in value:
            raise ValueError("summary must be one line")
        return value

    @field_validator("region", "masking")
    @classmethod
    def _finite(cls, value: JsonValue) -> JsonValue:
        result: JsonValue = _sanitize_non_finite(value)
        return result


class StoredPayload(_Frozen):
    state: Literal["stored"] = "stored"
    format: PayloadFormat
    # Relative to the payload store's root; the store re-derives and checks it.
    path: str


class InlineScalar(_Frozen):
    state: Literal["inline"] = "inline"
    value: int | float | None
    # Any JSON the operation reports beside the value, such as a standard
    # deviation or an interval; nothing here reads inside it.
    uncertainty: JsonValue = None

    @field_validator("value", "uncertainty")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        return _sanitize_non_finite(value)


class EvictedPayload(_Frozen):
    state: Literal["evicted"] = "evicted"
    format: PayloadFormat
    evicted_at: AwareDatetime


class ServicePayload(_Frozen):
    """Held by the earthdata service, which opens it."""

    state: Literal["service"] = "service"


Payload = Annotated[StoredPayload | InlineScalar | EvictedPayload | ServicePayload, Field(discriminator="state")]


class HandleRecord(_Frozen):
    """A handle and how it was made. Identity is (user_id, id)."""

    id: str
    kind: HandleKind
    origin: Origin
    user_id: str
    conversation_id: str
    descriptor: Descriptor
    parents: tuple[str, ...] = ()
    operation: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    release: str = HARNESS_RELEASE
    # The run that minted a derived handle; None for retrieval records and for
    # handles minted outside a run.
    run_id: str | None = None
    # Passed in from the run's clock so benchmark runs repeat.
    created_at: AwareDatetime
    payload: Payload

    @field_validator("parameters")
    @classmethod
    def _finite(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        # The JsonValue type is what keeps anything but plain data, credential
        # objects included, out of parameters.
        result: dict[str, JsonValue] = _sanitize_non_finite(value)
        return result

    @model_validator(mode="after")
    def _consistent(self, info: ValidationInfo) -> HandleRecord:
        if _stored(info):
            return self
        if not self.user_id:
            raise ValueError("a handle has an owner")
        if self.run_id is not None and RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("a run id is 'run_<32 lowercase hex>'")
        if _has_nul(self.model_dump()):
            raise ValueError("no string in a handle record may contain NUL")
        if self.origin == "derived":
            match = _DERIVED_ID.fullmatch(self.id)
            if match is None or match["kind"] != self.kind:
                raise ValueError("a derived id is '<kind>_<32 hex>' and its prefix is its kind")
            if not self.operation:
                raise ValueError("a derived handle names the operation that made it")
            if self.kind == "scalar":
                if not isinstance(self.payload, InlineScalar):
                    raise ValueError("a scalar's value is stored inline")
            elif not isinstance(self.payload, StoredPayload | EvictedPayload) or self.payload.format != PAYLOAD_FORMAT[self.kind]:
                raise ValueError(f"a {self.kind} payload is stored as {PAYLOAD_FORMAT[self.kind]}")
        else:
            if not is_retrieval_id(self.id):
                raise ValueError("a retrieval handle keeps the earthdata service's id")
            if self.parents or self.operation is not None:
                raise ValueError("a retrieval handle's lineage belongs to the earthdata service")
            if not isinstance(self.payload, ServicePayload):
                raise ValueError("a retrieval handle's data is held by the earthdata service")
        if len(set(self.parents)) != len(self.parents) or self.id in self.parents:
            raise ValueError("parents are distinct and never the handle itself")
        if not all(is_handle_id(parent) for parent in self.parents):
            raise ValueError("every parent is a handle id")
        return self
