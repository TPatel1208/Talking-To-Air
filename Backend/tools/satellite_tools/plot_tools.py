"""
plot_tools.py
-------------
Satellite plotting tools.

Returns chart payloads (JSON) instead of PNG files so the frontend can
render interactive Plotly charts. The payload schema is:

Spatial heatmap
---------------
{
  "type":     "heatmap",
  "title":    str,
  "variable": str,
  "units":    str,
  "lats":     [float, ...],        # 1-D latitude axis
  "lons":     [float, ...],        # 1-D longitude axis
  "values":   [[float, ...], ...], # 2-D row-major grid (lat × lon), NaN → null
  "vmin":     float,
  "vmax":     float,
}

Multi-panel comparison (list of heatmaps)
------------------------------------------
{
  "type":   "heatmap_multi",
  "panels": [ <heatmap payload>, ... ]
}

Time-series
-----------
{
  "type":      "timeseries",
  "title":     str,
  "variable":  str,
  "units":     str,
  "stat":      str,
  "times":     [ISO str, ...],
  "values":    [float, ...],
}
"""
import json
import os
import sys
import numpy as np
from langchain.tools import tool
from typing import Annotated,  List, Optional
from pydantic import Field
from tools.satellite_tools.models import DataDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_utils import _load_data
from utils.plotting import _normalize_to_2d, mask_data_by_geometry, RegionResolver
from tools.satellite_tools.harmony_api import COLLECTIONS

_resolver = RegionResolver()


def _sel_bounds(da, lat_coord, lon_coord, bounds):
    """
    Crop a DataArray to (minx, miny, maxx, maxy) bounds in a coordinate-order-
    safe way.  xarray slice() requires start <= stop when coords are increasing
    and start >= stop when decreasing.  We detect the direction and swap if needed
    so the crop never silently returns an empty array.
    """
    lat_vals = da[lat_coord].values
    lon_vals = da[lon_coord].values

    lat_min, lat_max = bounds[1], bounds[3]   # miny, maxy
    lon_min, lon_max = bounds[0], bounds[2]   # minx, maxx

    # If latitude is stored N→S (decreasing), slice must be (max, min)
    if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # Longitude is almost always W→E (increasing), but handle both
    if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    return da.sel({lat_coord: lat_slice, lon_coord: lon_slice})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _percentile_bounds(arr: np.ndarray):
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, 2)), float(np.percentile(valid, 98))


_MAX_GRID_CELLS = 8_000   # match the frontend MAX_POINTS constant


def _downsample_grid(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray):
    """
    Uniformly thin a 2-D (lat × lon) grid so it contains at most _MAX_GRID_CELLS
    non-null cells.  Returns (lats_ds, lons_ds, arr_ds).

    Strategy: keep every N-th row and every M-th column where N and M are chosen
    so that rows*cols ≈ _MAX_GRID_CELLS.  This is done *before* JSON serialisation
    so the payload written to disk is already small rather than forcing the browser
    to parse a multi-MB string and then discard most of it in flattenGrid.
    """
    n_rows, n_cols = arr.shape
    total = n_rows * n_cols
    if total <= _MAX_GRID_CELLS:
        return lats, lons, arr

    # Scale both axes by the same factor to preserve aspect ratio
    scale = (total / _MAX_GRID_CELLS) ** 0.5
    row_step = max(1, int(np.ceil(scale)))
    col_step = max(1, int(np.ceil(scale)))

    return lats[::row_step], lons[::col_step], arr[::row_step, ::col_step]


def _da_to_heatmap_payload(da, title: str, variable: str, units: str) -> dict:
    lat_coord = next((c for c in ["lat", "latitude", "Latitude"] if c in da.coords), None)
    lon_coord = next((c for c in ["lon", "longitude", "Longitude"] if c in da.coords), None)
    if lat_coord is None or lon_coord is None:
        raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")

    if da.dims.index(lat_coord) != 0:
        da = da.transpose(lat_coord, lon_coord)

    arr = da.values.astype(float)
    vmin, vmax = _percentile_bounds(arr)

    # ── Fix #1: downsample BEFORE serialising so the JSON payload is small ──
    lats_out = da[lat_coord].values
    lons_out = da[lon_coord].values
    lats_out, lons_out, arr = _downsample_grid(lats_out, lons_out, arr)

    values_json = [
        [None if np.isnan(v) else float(f"{v:.6e}") for v in row]
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
        "vmin": float(f"{vmin:.6e}"),
        "vmax": float(f"{vmax:.6e}"),
    }

def _save_chart(payload: dict, name: str) -> str:
    """Write chart payload to /outputs/<name>.chart.json and return the path."""
    import re
    safe = re.sub(r'[^\w\-]', '_', name)[:80]
    path = os.path.join(OUTPUT_DIR, f"{safe}.chart.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path

# ── Tools ─────────────────────────────────────────────────────────────────────


def _get(data_dict, key, default=None):
    """Access a field from a DataDict object or plain dict interchangeably."""
    if isinstance(data_dict, dict):
        return data_dict.get(key, default)
    return getattr(data_dict, key, default)

@tool
def plot_singular(data_dict: Annotated[dict, Field(description="The complete JSON object returned by fetch_environmental_data. Pass the entire object — do not extract fields or convert to a string.")], variable: str, location: str,
                  title: str = "", cmap: Optional[str] = "Spectral_r") -> str:
    """
    Plot a spatial heatmap of a variable over a single location at one point in time.
    Use when the user asks for a "map", "plot", or "show" for a single snapshot.

    Do NOT use this for time series, trends, or requests involving change over time —
    use conduct_temporal_statistic instead.

    Args:
        data_dict : dict from fetch_environmental_data.
        variable  : Variable name e.g. 'NO2', 'CO', 'CO2', etc.
        location  : Place name e.g. 'New York City', 'California'.
        title     : Plot title. Auto-generated from variable + location if omitted.
        cmap      : Colormap hint for the frontend (default 'Spectral_r').

    Returns:
        JSON string — chart payload for the frontend to render interactively.
    """
    try:
        da = _load_data(data_dict)
    except Exception as e:
        return json.dumps({"error": f"Failed to load data: {e}"})

    da = _normalize_to_2d(da)

    region = _resolver.resolve_location(location)
    if region is None:
        return json.dumps({"error": f"Could not geocode location: '{location}'"})

    try:
        da = mask_data_by_geometry(da, region["geometry"])
        bounds = region["bounds"]  # (minx, miny, maxx, maxy)
        lat_coord = next(c for c in ["lat", "latitude", "Latitude"] if c in da.coords)
        lon_coord = next(c for c in ["lon", "longitude", "Longitude"] if c in da.coords)
        da = _sel_bounds(da, lat_coord, lon_coord, bounds

)
    except Exception as e:
        return json.dumps({"error": f"Masking failed: {e}"})

    col    = COLLECTIONS.get(variable.upper(), {})
    units  = _get(data_dict, "units") or col.get("units", "")
    resolved_title = title or f"{variable} over {region['name']}"

    try:
        payload = _da_to_heatmap_payload(da, resolved_title, variable, units)
        payload["cmap"]   = cmap or "Spectral_r"
        payload["bounds"] = list(region["bounds"])  # (minx, miny, maxx, maxy)
    except Exception as e:
        return json.dumps({"error": f"Failed to build chart payload: {e}"})

    return _save_chart(payload, resolved_title)


@tool
def plot_multiple(
    data_dicts: Annotated[List[dict], Field(description="List of complete JSON objects, each returned by a separate fetch_environmental_data call.")],
    variable: str,
    locations: List[str],
    title: str = "",
    cmap: Optional[str] = "Spectral_r",
) -> str:
    """
    Plot the same environmental variable across multiple locations side by side.

    IMPORTANT — call fetch_environmental_data separately for each location first,
    collecting each result into a list. If a fetch fails for one dataset, try a
    fallback dataset before adding to the list. Only call this tool once you have
    a successful data_dict for every location.

    Args:
        data_dicts : list of dicts from fetch_environmental_data, one per location.
        variable   : Variable name e.g. 'NO2'.
        locations  : List of place names matching data_dicts order.
        title      : Overall title (optional).
        cmap       : Colormap hint for the frontend (default 'Spectral_r').

    Returns:
        JSON string — multi-panel chart payload for the frontend to render.
    """
    if len(data_dicts) != len(locations):
        return json.dumps({"error": f"len(data_dicts)={len(data_dicts)} != len(locations)={len(locations)}"})

    panels = []
    for data_dict, location in zip(data_dicts, locations):
        try:
            da = _load_data(data_dict)
        except Exception as e:
            return json.dumps({"error": f"Failed to load data for '{location}': {e}"})

        da = _normalize_to_2d(da)

        region = _resolver.resolve_location(location)
        if region is None:
            return json.dumps({"error": f"Could not geocode location: '{location}'"})

        try:
            da = mask_data_by_geometry(da, region["geometry"])
        except Exception as e:
            return json.dumps({"error": f"Masking failed for '{location}': {e}"})

        lat_coord = next(c for c in ["lat", "latitude", "Latitude"] if c in da.coords)
        lon_coord = next(c for c in ["lon", "longitude", "Longitude"] if c in da.coords)
        bounds = region["bounds"]
        da = _sel_bounds(da, lat_coord, lon_coord, bounds
        )

        col   = COLLECTIONS.get(variable.upper(), {})
        units = _get(data_dict, "units") or col.get("units", "")

        try:
            panel = _da_to_heatmap_payload(da, region["name"], variable, units)
            panel["cmap"]   = cmap or "Spectral_r"
            panel["bounds"] = list(region["bounds"])
        except Exception as e:
            return json.dumps({"error": f"Failed to build panel for '{location}': {e}"})

        panels.append(panel)

    multi_payload = {"type": "heatmap_multi", "title": title or f"{variable} Comparison", "panels": panels}
    return _save_chart(multi_payload, title or f"{variable}_comparison")


@tool
def conduct_temporal_statistic(
    data_dict: Annotated[dict, Field(description="The complete JSON object returned by fetch_environmental_data. Pass the entire object — do not extract fields or convert to a string.")],
    location: str,
    stat: str = "mean",
) -> str:
    """
    Produce a time-series line chart showing how a variable changes over time.

    Use this tool when the user asks for a "time series", "trend", "how X changed over time",
    "monthly values", or anything involving change across multiple time steps.
    Do NOT use plot_singular for these requests — plot_singular only shows a single snapshot.

    Args:
        data_dict: dict directly from fetch_environmental_data
                   (must cover a multi-day or multi-month range with multiple granules)
        location:  place name to spatially mask before computing e.g. 'New Jersey'
        stat:      statistic to compute at each time step.
                   One of: 'mean', 'median', 'max', 'min', 'std'  (default: 'mean')

    Returns:
        JSON string — time-series chart payload for the frontend to render interactively.
    """
    import pandas as pd

    try:
        da = _load_data(data_dict)
    except Exception as e:
        return json.dumps({"error": f"Failed to load data: {e}"})

    if "time" not in da.dims:
        return json.dumps({"error": f"No time dimension found. dims={list(da.dims)}"})

    region = _resolver.resolve_location(location)
    if region is None:
        return json.dumps({"error": f"Could not resolve location: '{location}'"})

    da = mask_data_by_geometry(da, region["geometry"])

    lat_coord = next(c for c in ["lat", "latitude", "Latitude"] if c in da.coords)
    lon_coord = next(c for c in ["lon", "longitude", "Longitude"] if c in da.coords)
    bounds = region["bounds"]
    da = _sel_bounds(da, lat_coord, lon_coord, bounds
    )

    var        = _get(data_dict, "variable", "")
    col_info   = COLLECTIONS.get(var, {})
    fill_value = col_info.get("fill_value", -1.267651e+30)
    max_valid  = col_info.get("valid_max", 1e18)
    min_valid  = col_info.get("valid_min", -1e15)

    stat_fn = {
        "mean":   np.nanmean,
        "median": np.nanmedian,
        "max":    np.nanmax,
        "min":    np.nanmin,
        "std":    np.nanstd,
    }.get(stat)

    if stat_fn is None:
        return json.dumps({"error": f"Unknown stat '{stat}'. Use: mean, median, max, min, std"})

    times, values = [], []
    for i in range(da.sizes["time"]):
        slice_2d = da.isel(time=i).values
        valid = slice_2d[
            np.isfinite(slice_2d) &
            (slice_2d != fill_value) &
            (slice_2d > min_valid) &
            (slice_2d < max_valid)
        ]
        if len(valid) == 0:
            continue
        raw_time = da["time"].values[i]
        timestamp = pd.Timestamp(raw_time).isoformat()
        times.append(timestamp)
        values.append(round(float(stat_fn(valid)), 6))

    if not times:
        return json.dumps({"error": f"No valid data found for '{location}' across any time step."})

    # Sort by time
    paired = sorted(zip(times, values))
    times, values = zip(*paired)

    ts_payload = {
        "type":     "timeseries",
        "title":    f"{var} {stat} over {location}",
        "variable": var,
        "units":    _get(data_dict, "units", ""),
        "stat":     stat,
        "times":    list(times),
        "values":   list(values),
    }
    return _save_chart(ts_payload, f"{var}_{stat}_{location}")