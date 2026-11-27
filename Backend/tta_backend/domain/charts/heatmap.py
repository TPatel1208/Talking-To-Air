"""The ``heatmap`` payload: a 2-D (lat x lon) field, its colour scale, statistics and overlay bounds.

The overlay PNG is drawn by the caller: pass ``overlay_renderer`` to add its
fields to ``payload["overlay"]``. Without one the overlay holds only its
bounds, and the frontend draws the map from ``values``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import xarray as xr

from tta_backend.domain import xarray_ops
from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.charts.colormaps import resolve as resolve_colormap
from tta_backend.domain.earthdata.opening import normalize_longitudes
from tta_backend.domain.geo import find_lat_coord, find_lon_coord
from tta_backend.domain.masking.geometry import half_cell

# Ceiling on the cells serialized into a payload's ``values`` grid. It bounds
# the payload's size only: the statistics and vmin/vmax are computed on the
# full field before thinning, and the overlay PNG is drawn at full resolution.
MAX_GRID_CELLS = 8_000

# (lats, lons, field, lut, vmin, vmax) -> the overlay fields to record.
OverlayRenderer = Callable[[np.ndarray, np.ndarray, np.ndarray, list[Any], float, float], dict[str, Any]]


class Colormap(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def lut(self) -> list[list[int]]: ...


def percentile_bounds(arr: np.ndarray) -> tuple[float, float]:
    """The 2nd and 98th percentiles of the finite cells; (0, 1) when there are none."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmin == vmax:
        delta = abs(vmin) * 0.01 or 1.0
        return vmin - delta, vmax + delta
    return vmin, vmax


# How a diverging scale is set, for the legend's saturation warning.
DIVERGING_SCALE_DISCLOSURE: dict[str, Any] = {"method": "percentile_magnitude", "p": 98}


def diverging_bounds(arr: np.ndarray) -> tuple[float, float]:
    """A scale centred on zero, sized to the 98th-percentile magnitude of the finite cells."""
    vals = np.asarray(arr, dtype=float)
    valid = vals[np.isfinite(vals)]
    if valid.size == 0:
        return -1.0, 1.0
    max_abs = float(np.percentile(np.abs(valid), 98))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    return -max_abs, max_abs


def downsample_grid(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin a 2-D (lat x lon) grid to at most MAX_GRID_CELLS cells by keeping
    every N-th row and column, the same step on both axes."""
    n_rows, n_cols = arr.shape
    total = n_rows * n_cols
    if total <= MAX_GRID_CELLS:
        return lats, lons, arr
    scale = (total / MAX_GRID_CELLS) ** 0.5
    row_step = max(1, int(np.ceil(scale)))
    col_step = max(1, int(np.ceil(scale)))
    return lats[::row_step], lons[::col_step], arr[::row_step, ::col_step]


def area_weighted_mean(arr: np.ndarray, lats: np.ndarray, finite: np.ndarray) -> float:
    """cos(latitude)-weighted mean over the finite cells of a (lat, lon) grid,
    the same regional mean the statistics tools report.

    Reduced row by row against the 1-D weights, accumulating in float64.
    """
    lat_vals = np.asarray(lats, dtype=float)
    if lat_vals.ndim != 1 or lat_vals.size != arr.shape[0]:
        return float(arr[finite].mean(dtype=np.float64))
    weights = np.clip(np.cos(np.deg2rad(lat_vals)), 0.0, None)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    contribution = np.where(finite, arr, 0.0).sum(axis=1, dtype=np.float64)
    counted = finite.sum(axis=1, dtype=np.float64)
    total = float((counted * weights).sum())
    if total <= 0.0:
        return float(arr[finite].mean(dtype=np.float64))
    return float((contribution * weights).sum() / total)


def field_statistics(arr: np.ndarray, lats: np.ndarray) -> dict[str, Any]:
    """Count, valid fraction, area-weighted mean, min and max of the full-resolution field."""
    finite = np.isfinite(arr)
    count = int(finite.sum())
    if count == 0:
        return {"count": 0, "valid_fraction": 0.0}
    values = arr[finite]
    return {
        "count": count,
        "valid_fraction": round(float(count) / float(arr.size), 6),
        "mean": float(f"{area_weighted_mean(arr, lats, finite):.6e}"),
        "min": float(f"{values.min():.6e}"),
        "max": float(f"{values.max():.6e}"),
    }


def time_mean_field(da: xr.DataArray, time_dim: str, cadence: str) -> tuple[xr.DataArray, list[int]]:
    """The field a map shows for data with a time axis: the cadence-weighted
    mean over the timesteps that hold a finite value, and those timesteps.
    With none, the field is all NaN."""
    spatial = [d for d in da.dims if d != time_dim]
    found = np.atleast_1d(np.asarray(xarray_ops.isfinite(da).any(spatial).values))
    valid = [i for i, ok in enumerate(found) if bool(ok)]
    if not valid:
        return da.isel({time_dim: slice(0, 0)}).mean(dim=time_dim, skipna=True), []
    return AggregationService().temporal_mean(da.isel({time_dim: valid}), time_dim, cadence), valid


def heatmap_payload(
    da: xr.DataArray,
    title: str,
    variable: str,
    units: str,
    *,
    diverging: bool = False,
    value_range: tuple[float, float] | None = None,
    scale_disclosure: dict[str, Any] | None = None,
    colormap: Colormap | None = None,
    overlay_renderer: OverlayRenderer | None = None,
) -> dict[str, Any]:
    """The heatmap payload for a 2-D field.

    The colour scale is the 2-98 percentile range unless ``value_range`` is
    given; ``scale_disclosure`` says how an imposed range was set. The field
    is carried at float32; the mean accumulates in float64.
    """
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    if lat_coord is None or lon_coord is None:
        raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")

    da = normalize_longitudes(da, lon_coord)
    if da.dims.index(lat_coord) != 0:
        da = da.transpose(lat_coord, lon_coord)

    arr = da.values.astype(np.float32, copy=False)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    vmin, vmax = value_range if value_range is not None else percentile_bounds(arr)
    if value_range is not None:
        scale = scale_disclosure if scale_disclosure is not None else {"method": "explicit"}
    else:
        scale = {"method": "percentile", "p": [2, 98]}

    lats_out = da[lat_coord].values
    lons_out = da[lon_coord].values

    # Pixel-edge bounds of the full-resolution grid, which is how the overlay
    # PNG is rasterized.
    lon_half = half_cell(lons_out)
    lat_half = half_cell(lats_out)
    overlay_bounds = [
        float(np.nanmin(lons_out)) - lon_half, float(np.nanmin(lats_out)) - lat_half,
        float(np.nanmax(lons_out)) + lon_half, float(np.nanmax(lats_out)) + lat_half,
    ]

    resolved = colormap if colormap is not None else resolve_colormap(variable, diverging=diverging)

    overlay: dict[str, Any] = {"bounds": overlay_bounds}
    if overlay_renderer is not None:
        overlay.update(overlay_renderer(lats_out, lons_out, arr, resolved.lut, vmin, vmax))

    statistics = field_statistics(arr, lats_out)

    lats_out, lons_out, arr = downsample_grid(lats_out, lons_out, arr)

    values_json = [
        [None if not np.isfinite(v) else float(f"{v:.6e}") for v in row]
        for row in arr
    ]

    return {
        "type":     "heatmap",
        "title":    title,
        "variable": variable,
        "units":    units,
        "lats":     [round(float(v), 6) for v in lats_out],
        "lons":     [round(float(v), 6) for v in lons_out],
        "values":   values_json,
        "statistics": statistics,
        "vmin": float(f"{vmin:.6e}"),
        "vmax": float(f"{vmax:.6e}"),
        "scale": scale,
        "colormap": {"name": resolved.name, "lut": resolved.lut},
        "overlay": overlay,
    }
