"""Payload bytes for derived handles, in files under a root the host chooses."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from pydantic import JsonValue

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import (
    PAYLOAD_FORMAT,
    EvictedPayload,
    HandleKind,
    HandleRecord,
    InlineScalar,
    PayloadFormat,
    StoredPayload,
    is_derived_id,
)

_SUFFIX: dict[PayloadFormat, str] = {"zarr": ".zarr", "parquet": ".parquet", "geojson": ".geojson"}
_STAGING = ".staging-"


@dataclass(frozen=True)
class ScalarInput:
    """What a caller hands in to mint a scalar; the unit lives on the descriptor."""

    value: int | float | None
    uncertainty: JsonValue = None


@dataclass(frozen=True)
class Scalar:
    value: int | float | None
    unit: str | None
    uncertainty: JsonValue


def _user_dir(user_id: str) -> str:
    # Hashed so no user id, whatever its characters, can name a path or reach
    # another user's directory.
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]


def relative_path(user_id: str, handle_id: str, fmt: PayloadFormat) -> str:
    if not is_derived_id(handle_id):
        raise ValueError("payloads are stored only for derived handle ids")
    return f"{_user_dir(user_id)}/{handle_id}{_SUFFIX[fmt]}"


class PayloadStore:
    """Grids as Zarr, series/points/tables as Parquet, polygons as GeoJSON.

    Every write lands in a staging name and is renamed into place, so a reader
    sees either the whole payload or none of it.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).resolve()

    def _full(self, relative: str) -> Path:
        full = (self._root / relative).resolve()
        if not full.is_relative_to(self._root) or full == self._root:
            raise HandleError("not_found")
        return full

    async def write(self, user_id: str, handle_id: str, kind: HandleKind, data: Any) -> StoredPayload | InlineScalar:
        if kind == "scalar":
            if not isinstance(data, ScalarInput):
                raise TypeError("a scalar is minted from a ScalarInput")
            return InlineScalar(value=data.value, uncertainty=data.uncertainty)
        fmt = PAYLOAD_FORMAT[kind]
        relative = relative_path(user_id, handle_id, fmt)
        await asyncio.to_thread(_write, self._full(relative), fmt, data)
        return StoredPayload(format=fmt, path=relative)

    async def open(self, record: HandleRecord) -> Any:
        """The payload's data. A grid is lazily loaded: the caller loads what it
        needs before the handle can be evicted, or later reads fail in Zarr."""
        payload = record.payload
        if isinstance(payload, InlineScalar):
            return Scalar(value=payload.value, unit=record.descriptor.unit, uncertainty=payload.uncertainty)
        if isinstance(payload, EvictedPayload):
            raise HandleError("evicted", record.id)
        if not isinstance(payload, StoredPayload):
            raise TypeError("only derived payloads are opened from the store")
        # Re-derived rather than trusted, so a record can only ever point at
        # its own owner's file for its own id.
        if payload.path != relative_path(record.user_id, record.id, payload.format):
            raise HandleError("not_found", record.id)
        full = self._full(payload.path)
        try:
            return await asyncio.to_thread(_read, full, payload.format)
        except FileNotFoundError:
            # No bytes behind a stored record: they were deleted by an eviction.
            raise HandleError("evicted", record.id) from None

    async def delete(self, user_id: str, handle_id: str, fmt: PayloadFormat) -> None:
        await asyncio.to_thread(_delete, self._full(relative_path(user_id, handle_id, fmt)))


def _staging(final: Path) -> Path:
    return final.with_name(f"{_STAGING}{secrets.token_hex(8)}-{final.name}")


def _write(final: Path, fmt: PayloadFormat, data: Any) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(final)
    staging = _staging(final)
    try:
        if fmt == "zarr":
            if isinstance(data, xr.DataArray):
                data = data.to_dataset(name=data.name if data.name is not None else "value")
            if not isinstance(data, xr.Dataset):
                raise TypeError("a grid is an xarray Dataset or DataArray")
            data.to_zarr(staging, mode="w-", consolidated=False)
        elif fmt == "parquet":
            if not isinstance(data, pd.DataFrame):
                raise TypeError("series, points and tables are pandas DataFrames")
            data.to_parquet(staging)
            _fsync(staging)
        else:
            if not isinstance(data, dict) or not isinstance(data.get("type"), str):
                raise TypeError("polygons are a GeoJSON object")
            staging.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
            _fsync(staging)
        # Ids are random, so the final name is not expected to exist; the check
        # above refuses one that does. POSIX rename would replace it, Windows
        # would fail. The rename makes the whole payload appear at once.
        os.rename(staging, final)
    except BaseException:
        _remove(staging)
        raise


def _fsync(path: Path) -> None:
    # On disk before the rename, so a record committed after it never points
    # at a file a crash left truncated.
    with open(path, "r+b") as file:
        os.fsync(file.fileno())


def _read(full: Path, fmt: PayloadFormat) -> Any:
    if fmt == "zarr":
        if not full.is_dir():
            raise FileNotFoundError(full)
        return xr.open_zarr(full, consolidated=False)
    if fmt == "parquet":
        return pd.read_parquet(full)
    return json.loads(full.read_text(encoding="utf-8"))


def _delete(full: Path) -> None:
    if not full.exists():
        return
    # Renamed away first so a concurrent open finds nothing rather than a
    # half-deleted store.
    doomed = _staging(full)
    try:
        os.rename(full, doomed)
    except FileNotFoundError:
        return
    _remove(doomed)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)
