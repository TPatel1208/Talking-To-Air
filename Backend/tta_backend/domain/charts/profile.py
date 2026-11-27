"""The vertical profile: one regional mean per layer, against pressure or altitude.

``reduce_profile`` reduces an already masked layered field; ``profile_payload``
turns the result into the ``profile`` chart payload. ``profile_table`` and
``profile_from_table`` carry a profile as a table: ``layer``, ``value`` and
``valid_fraction``, then for each physical axis ``{kind}``, ``{kind}_spread``
and ``{kind}_unit``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.domain import xarray_ops
from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.aggregation.regional_reduction import reduce_keeping_axes
from tta_backend.domain.charts.provenance import crop_band_to_region
from tta_backend.domain.geo import find_lat_coord, find_lon_coord, vertical_axis_kind

# Ceiling on the cells a profile reduces over, after narrowing to the region.
MAX_PROFILE_CELLS = 40_000_000

AXIS_KINDS = ("pressure", "altitude")

_aggregation_service = AggregationService()


def vertical_dim(da: xr.DataArray, time_dim: str | None) -> tuple[str | None, list[str]]:
    """The one dimension left after latitude, longitude and time, and every
    candidate, so an ambiguous field can be refused by naming them."""
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    spatial = {name for name in (lat_coord, lon_coord, time_dim) if name}
    candidates = [str(d) for d in da.dims if d not in spatial]
    return (candidates[0] if len(candidates) == 1 else None), candidates


def vertical_axis_candidates(
    narrowed: xr.DataArray, ds: xr.Dataset | None, vertical_dim: str, region: dict[str, Any] | None,
) -> dict[str, xr.DataArray]:
    """The physical vertical axes for ``vertical_dim``, keyed ``pressure`` or ``altitude``.

    The narrowed array's own coordinates are searched first. A product that
    publishes an axis as a data variable is found in the opened Dataset and
    narrowed to ``region`` the same way companion bands are. Coordinates ride
    the crop but not the geometry mask, so callers restrict them to the cells
    the field kept (``profile_axis_block``'s ``region_mask``).
    """
    found: dict[str, xr.DataArray] = {}
    for var in narrowed.coords.values():
        if vertical_dim not in getattr(var, "dims", ()):
            continue
        kind = vertical_axis_kind(var)
        if kind and kind not in found:
            found[kind] = var
    if ds is None or getattr(ds, "data_vars", None) is None or region is None:
        return found
    for name, var in ds.data_vars.items():
        if vertical_dim not in getattr(var, "dims", ()) or name == narrowed.name:
            continue
        kind = vertical_axis_kind(var)
        if not kind or kind in found:
            continue
        cropped, in_region = crop_band_to_region(var, region)
        if cropped is not None and in_region:
            found[kind] = cropped
    return found


def layer_order(axis_values: list[Any], kind: str) -> str:
    """Whether layer 0 is the top of the atmosphere or the bottom, measured off the axis."""
    finite = [v for v in axis_values if v is not None]
    if len(finite) < 2 or finite[0] == finite[-1]:
        return "unknown"
    # Pressure falls with height; altitude rises.
    rising = finite[-1] > finite[0]
    if kind == "pressure":
        return "top_down" if rising else "bottom_up"
    return "bottom_up" if rising else "top_down"


def rounded(values: Any) -> list[Any]:
    """Six significant digits, non-finite as None."""
    return [None if not np.isfinite(v) else float(f"{float(v):.6g}") for v in np.asarray(values).ravel()]


def profile_axis_block(axis_da: xr.DataArray, vertical_dim: str, kind: str, region_mask: xr.DataArray | None = None) -> dict[str, Any]:
    """One vertical axis reduced to the region, with each layer's spread
    (max minus min across the cells the field kept)."""
    if region_mask is not None:
        axis_da = axis_da.where(region_mask.broadcast_like(axis_da))

    keep = (vertical_dim,)
    mean = reduce_keeping_axes(axis_da, keep=keep, stat="mean")
    highest = reduce_keeping_axes(axis_da, keep=keep, stat="max")
    lowest = reduce_keeping_axes(axis_da, keep=keep, stat="min")
    spread = np.asarray(highest.values, dtype="float64") - np.asarray(lowest.values, dtype="float64")
    return {
        "kind": kind,
        "units": axis_da.attrs.get("units", "") or "",
        "values": rounded(mean.values),
        "spread": rounded(spread),
        "layer_order": layer_order(rounded(mean.values), kind),
    }


def per_layer_valid_fraction(masked: xr.DataArray, vertical_dim: str, region_cells: int | None = None) -> list[Any]:
    """The fraction of the region's cells, times the timesteps reduced, that
    held a value at each layer. ``region_cells`` is the region's footprint on
    one slice; without it the array's own size is the denominator. Clamped to 1."""
    finite = xarray_ops.isfinite(masked)
    collapsed = [d for d in masked.dims if d != vertical_dim]
    counts = finite.sum(collapsed).values
    spatial_cells = region_cells if region_cells else None
    if spatial_cells is None:
        spatial_cells = 1
        for dim in collapsed:
            spatial_cells *= int(masked.sizes[dim])
        total = spatial_cells
    else:
        lat_coord, lon_coord = find_lat_coord(masked), find_lon_coord(masked)
        total = spatial_cells
        for dim in collapsed:
            if dim not in (lat_coord, lon_coord):
                total *= int(masked.sizes[dim])
    if total <= 0:
        return [0.0] * int(masked.sizes[vertical_dim])
    return [min(1.0, round(float(c) / total, 6)) for c in np.atleast_1d(counts)]


def profile_scale_guard(narrowed: xr.DataArray) -> str | None:
    """Why a profile is too large to reduce, or None."""
    cells = int(getattr(narrowed, "size", 0))
    if cells <= MAX_PROFILE_CELLS:
        return None
    return (
        f"This profile would reduce over ~{cells:,} cells, above the "
        f"{MAX_PROFILE_CELLS:,}-cell limit for a single request."
    )


@dataclass(frozen=True)
class Profile:
    """A reduced profile: per-layer values, the physical axes and coverage.

    ``valid_indices`` are the timesteps that held data, for the aggregation meta.
    """

    vertical_dim: str
    layers: list[int]
    values: list[Any]
    vertical: dict[str, dict[str, Any]]
    default_axis: str | None
    layer_order: str
    valid_fraction: list[Any]
    valid_indices: list[int]


def reduce_profile(
    masked: xr.DataArray,
    *,
    vertical_dim: str,
    time_dim: str | None,
    cadence: str,
    ds: xr.Dataset | None = None,
    region: dict[str, Any] | None = None,
    region_cells: int | None = None,
) -> Profile | None:
    """The profile of a masked layered field, or None when no layer holds data.

    Space then time: one area-weighted mean per (timestep, layer), then the
    cadence-weighted mean over the timesteps that held data. Pressure is the
    default axis when there is one.
    """
    keep = tuple(d for d in (time_dim, vertical_dim) if d and d in masked.dims)
    per_slice = reduce_keeping_axes(masked, keep=keep, stat="mean")

    if time_dim and time_dim in per_slice.dims:
        spatial_of_matrix = [d for d in per_slice.dims if d != time_dim]
        valid_indices = [
            i for i, ok in enumerate(
                np.atleast_1d(xarray_ops.isfinite(per_slice).any(spatial_of_matrix).values)
            ) if bool(ok)
        ]
        if not valid_indices:
            return None
        per_slice = per_slice.isel({time_dim: valid_indices})
        profile = _aggregation_service.temporal_mean(per_slice, time_dim, cadence)
    else:
        valid_indices = [0]
        profile = per_slice

    values = rounded(profile.values)
    if not any(v is not None for v in values):
        return None

    axes = vertical_axis_candidates(masked, ds, vertical_dim, region)
    vertical = {
        kind: profile_axis_block(axis_da, vertical_dim, kind, region_mask=xarray_ops.isfinite(masked))
        for kind, axis_da in axes.items()
    }
    default_axis = "pressure" if "pressure" in vertical else next(iter(vertical), None)
    return Profile(
        vertical_dim=vertical_dim,
        layers=list(range(int(masked.sizes[vertical_dim]))),
        values=values,
        vertical=vertical,
        default_axis=default_axis,
        layer_order=(vertical.get(default_axis or "") or {}).get("layer_order", "unknown"),
        valid_fraction=per_layer_valid_fraction(masked, vertical_dim, region_cells),
        valid_indices=valid_indices,
    )


def profile_payload(
    profile: Profile,
    *,
    title: str,
    variable: str,
    units: str,
    masking: Any,
    agg_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``profile`` chart payload, before its reproducibility blocks."""
    payload: dict[str, Any] = {
        "type": "profile",
        "title": title,
        "variable": variable,
        "units": units,
        "stat": "mean",
        "vertical_dim": profile.vertical_dim,
        "layers": profile.layers,
        "values": profile.values,
        "vertical": profile.vertical,
        "default_axis": profile.default_axis,
        "layer_order": profile.layer_order,
        "valid_fraction": profile.valid_fraction,
        "masking": masking,
    }
    if agg_meta is not None:
        payload["aggregation_meta"] = agg_meta
    return payload


def profile_export(payload: dict[str, Any]) -> dict[str, Any]:
    """The arrays a profile's export carries itself: they are its full resolution."""
    return {
        "layers": payload["layers"],
        "values": payload["values"],
        "valid_fraction": payload["valid_fraction"],
        "vertical": payload["vertical"],
        "default_axis": payload["default_axis"],
        "layer_order": payload["layer_order"],
    }


PROFILE_COLUMNS = ("layer", "value", "valid_fraction")


def profile_table(profile: Profile) -> pd.DataFrame:
    """A profile as one row per layer."""
    columns: dict[str, list[Any]] = {
        "layer": list(profile.layers),
        "value": [np.nan if v is None else v for v in profile.values],
        "valid_fraction": list(profile.valid_fraction),
    }
    for kind in AXIS_KINDS:
        block = profile.vertical.get(kind)
        if block is None:
            continue
        columns[kind] = [np.nan if v is None else v for v in block["values"]]
        columns[f"{kind}_spread"] = [np.nan if v is None else v for v in block["spread"]]
        columns[f"{kind}_unit"] = [block["units"]] * len(profile.layers)
    frame = pd.DataFrame(columns)
    frame["layer"] = frame["layer"].astype("int64")
    for name in frame.columns:
        if name != "layer" and not name.endswith("_unit"):
            frame[name] = frame[name].astype("float64")
    return frame


def _column(frame: pd.DataFrame, name: str) -> list[Any]:
    return rounded(frame[name].to_numpy(dtype="float64"))


def profile_from_table(frame: pd.DataFrame) -> Profile:
    """The profile a table holds. Raises ValueError when it lacks PROFILE_COLUMNS."""
    missing = [c for c in PROFILE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"a profile table needs the columns {', '.join(missing)}")
    frame = frame.sort_values("layer", kind="stable")
    vertical: dict[str, dict[str, Any]] = {}
    for kind in AXIS_KINDS:
        if kind not in frame.columns:
            continue
        values = _column(frame, kind)
        units = frame[f"{kind}_unit"].dropna() if f"{kind}_unit" in frame.columns else pd.Series(dtype="string")
        vertical[kind] = {
            "kind": kind,
            "units": str(units.iloc[0]) if len(units) else "",
            "values": values,
            "spread": _column(frame, f"{kind}_spread") if f"{kind}_spread" in frame.columns else [],
            "layer_order": layer_order(values, kind),
        }
    default_axis = "pressure" if "pressure" in vertical else next(iter(vertical), None)
    return Profile(
        vertical_dim="layer",
        layers=[int(v) for v in frame["layer"]],
        values=_column(frame, "value"),
        vertical=vertical,
        default_axis=default_axis,
        layer_order=(vertical.get(default_axis or "") or {}).get("layer_order", "unknown"),
        valid_fraction=_column(frame, "valid_fraction"),
        valid_indices=[],
    )
