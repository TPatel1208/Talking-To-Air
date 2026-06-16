"""Normalization helpers for append-safe Zarr writes."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import xarray as xr


class ZarrNormalizationError(ValueError):
    """Raised when a dataset window cannot be made append-compatible."""


def normalize_for_zarr_append(
    datasets: list[xr.Dataset],
    *,
    append_dim: str = "time",
    template: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Combine one granule window and normalize it for append-safe Zarr writes.

    The first normalized window can be used as the template for later windows.
    Subsequent windows are cast and encoded to match that template so coordinate
    dtypes, variable dtypes, dimensions, and chunk encodings do not drift.
    """
    if not datasets:
        raise ZarrNormalizationError("Cannot normalize an empty granule window")

    valid = [ds for ds in datasets if append_dim in ds.dims or append_dim in ds.coords]
    if not valid:
        raise ZarrNormalizationError(
            f"No parsed granules contain append dimension {append_dim!r}"
        )

    try:
        combined = valid[0] if len(valid) == 1 else xr.concat(valid, dim=append_dim)
    except Exception as exc:
        raise ZarrNormalizationError(
            f"Could not concatenate {len(valid)} granule(s) on {append_dim!r}: {exc}"
        ) from exc

    combined = _strip_zarr_unsafe_coord_attrs(combined)
    if template is not None:
        combined = _match_template(combined, template, append_dim)
    combined = _normalize_chunk_encoding(combined, append_dim, template)
    return combined


def _strip_zarr_unsafe_coord_attrs(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for coord in ds.coords:
        ds[coord].attrs.pop("units", None)
        ds[coord].attrs.pop("calendar", None)
    return ds


def _match_template(
    ds: xr.Dataset,
    template: xr.Dataset,
    append_dim: str,
) -> xr.Dataset:
    missing = set(template.data_vars) - set(ds.data_vars)
    extra = set(ds.data_vars) - set(template.data_vars)
    if missing or extra:
        raise ZarrNormalizationError(
            "Data variables differ from first Zarr window: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    template_dims = {name: size for name, size in template.sizes.items() if name != append_dim}
    dims = {name: size for name, size in ds.sizes.items() if name != append_dim}
    if dims != template_dims:
        raise ZarrNormalizationError(
            "Non-append dimensions differ from first Zarr window: "
            f"expected={template_dims} actual={dims}"
        )

    for coord in template.coords:
        if coord not in ds.coords:
            raise ZarrNormalizationError(f"Missing coordinate {coord!r} in Zarr append window")
        expected = template[coord].dtype
        actual = ds[coord].dtype
        if actual != expected:
            try:
                ds = ds.assign_coords({coord: ds[coord].astype(expected)})
            except Exception as exc:
                raise ZarrNormalizationError(
                    f"Coordinate {coord!r} dtype {actual} cannot be cast to {expected}"
                ) from exc

    for name in template.data_vars:
        expected = template[name].dtype
        actual = ds[name].dtype
        if actual != expected:
            try:
                ds[name] = ds[name].astype(expected)
            except Exception as exc:
                raise ZarrNormalizationError(
                    f"Variable {name!r} dtype {actual} cannot be cast to {expected}"
                ) from exc
    return ds


def _normalize_chunk_encoding(
    ds: xr.Dataset,
    append_dim: str,
    template: xr.Dataset | None,
) -> xr.Dataset:
    ds = ds.copy()
    template_chunks = _template_chunks(template) if template is not None else {}

    for name, array in ds.variables.items():
        if not array.dims:
            continue
        chunks = template_chunks.get(name)
        if chunks is None:
            chunks = tuple(_default_chunk_size(dim, size, append_dim) for dim, size in array.sizes.items())
        ds[name].encoding["chunks"] = chunks
    return ds


def _template_chunks(template: xr.Dataset | None) -> Mapping[str, tuple[int, ...]]:
    if template is None:
        return {}
    chunks: dict[str, tuple[int, ...]] = {}
    for name, array in template.variables.items():
        encoded = array.encoding.get("chunks")
        if encoded:
            chunks[name] = tuple(int(value) for value in encoded)
    return chunks


def _default_chunk_size(dim: str, size: int, append_dim: str) -> int:
    if size < 1:
        return 1
    if dim == append_dim:
        return 1
    if np.issubdtype(type(size), np.integer):
        return int(size)
    return size
