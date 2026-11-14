"""The layout every capability writes and reads for each handle kind.

grid      an xarray Dataset with one data variable, ``lat``/``lon`` coordinates,
          an optional ``time`` coordinate, and ``units`` in the variable's attrs.
series    a DataFrame of ``time`` (UTC), ``value`` and ``series`` (which line a
          row belongs to: a monitor id, a region name, a product).
points    a DataFrame of ``id``, ``name``, ``lat``, ``lon``; ``value`` when the
          points carry a measurement.
polygons  a GeoJSON FeatureCollection.
table     a DataFrame with any columns.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import xarray as xr

SERIES_COLUMNS = ("time", "value", "series")
POINTS_COLUMNS = ("id", "name", "lat", "lon")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], kind: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"a {kind} needs the columns {', '.join(missing)}")
    return frame


def series_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A series in the shared layout, sorted by line and time."""
    frame = _require_columns(frame, SERIES_COLUMNS, "series").copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["value"] = frame["value"].astype("float64")
    frame["series"] = frame["series"].astype("string")
    return frame.sort_values(["series", "time"], kind="stable").reset_index(drop=True)


def points_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Points in the shared layout; latitude and longitude are checked."""
    frame = _require_columns(frame, POINTS_COLUMNS, "points").copy()
    frame["lat"] = frame["lat"].astype("float64")
    frame["lon"] = frame["lon"].astype("float64")
    if not frame["lat"].between(-90, 90).all() or not frame["lon"].between(-180, 180).all():
        raise ValueError("points need latitude in -90..90 and longitude in -180..180")
    frame["id"] = frame["id"].astype("string")
    return frame.reset_index(drop=True)


def grid_variable(grid: xr.Dataset) -> xr.DataArray:
    """The one data variable of a grid."""
    names = list(grid.data_vars)
    if len(names) != 1:
        raise ValueError("a grid holds exactly one data variable")
    return grid[names[0]]


def polygons_collection(value: Any) -> dict[str, Any]:
    """Polygons as a GeoJSON FeatureCollection."""
    if not isinstance(value, dict) or value.get("type") != "FeatureCollection" or not isinstance(value.get("features"), list):
        raise ValueError("polygons are a GeoJSON FeatureCollection")
    return value
