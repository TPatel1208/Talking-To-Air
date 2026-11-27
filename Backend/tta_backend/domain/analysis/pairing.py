"""Pairing a gridded series with point readings: nearest cell, daily pairs, correlation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.geo import find_lat_coord, find_lon_coord

_aggregation_service = AggregationService()


def nearest_cell_series(da: xr.DataArray, lat: float, lon: float) -> xr.DataArray:
    """The nearest grid cell to (lat, lon) of a lat/lon-indexed DataArray."""
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    return da.sel({lat_coord: lat, lon_coord: lon}, method="nearest")


def nearest_cell_dataset(ds: xr.Dataset | None, da: xr.DataArray, lat: float, lon: float) -> xr.Dataset | None:
    """``ds`` narrowed to the cell ``nearest_cell_series`` takes from ``da``.

    The sibling quality flag must be narrowed with the science variable: a
    full-grid flag condition would broadcast the single cell back up to the
    whole grid at ``.where()``. ``ds`` is returned untouched when it lacks the
    science array's horizontal coordinates.
    """
    if ds is None:
        return None
    selection = {
        name: value
        for name, value in ((find_lat_coord(da), lat), (find_lon_coord(da), lon))
        if name and name in ds.coords
    }
    return ds.sel(selection, method="nearest") if selection else ds


def finite_series(series: xr.DataArray) -> tuple[list[str], list[float]]:
    """ISO times and values of a masked single-cell series; untimed and non-finite values are left out.

    An untimed reading cannot be paired with a ground day.
    """
    raw_values = np.atleast_1d(series.values)
    raw_times = np.atleast_1d(series["time"].values) if "time" in series.coords else [None] * raw_values.size
    times, values = [], []
    for t, v in zip(raw_times, raw_values, strict=True):
        if not np.isfinite(v) or t is None:
            continue
        times.append(pd.Timestamp(t).isoformat())
        values.append(float(v))
    return times, values


def within_grid(da: xr.DataArray, lat: float, lon: float) -> bool:
    """Whether (lat, lon) lies inside the grid's footprint, half a cell past its outermost cell centres.

    A point outside would otherwise take the nearest edge cell as its own.
    """
    for name, value in ((find_lat_coord(da), lat), (find_lon_coord(da), lon)):
        centres = np.sort(np.asarray(da[name].values, dtype=float))
        half = float(np.median(np.diff(centres))) / 2 if centres.size > 1 else 0.0
        if not centres[0] - half <= value <= centres[-1] + half:
            return False
    return True


def extract_monitor_series(
    da: xr.DataArray, lat: float, lon: float, col_info: dict[str, Any] | None = None,
    source_ds: xr.Dataset | None = None,
) -> tuple[list[str], list[float], dict[str, Any], dict[str, Any]]:
    """The satellite series at the nearest cell to (lat, lon), masked.

    Narrowed to the cell before masking, so the quality pass rate describes
    that one cell. Fill, out-of-range, flagged and untimed values are left out
    and counted. Returns (ISO times, values, coverage, masking provenance),
    coverage being {n_total, n_valid, n_excluded, coverage_fraction}.
    """
    series, masking = _aggregation_service.resolve_and_mask(
        nearest_cell_series(da, lat, lon),
        col_info=col_info,
        source_ds=nearest_cell_dataset(source_ds, da, lat, lon),
    )

    n_total = series.sizes.get("time", 1)
    times, values = finite_series(series)

    n_valid = len(values)
    coverage = {
        "n_total": int(n_total),
        "n_valid": n_valid,
        "n_excluded": int(n_total) - n_valid,
        "coverage_fraction": (n_valid / n_total) if n_total else 0.0,
    }
    return times, values, coverage, masking


def pair_daily(times: list[str], values: list[float], ground_daily: dict[str, float]) -> list[dict[str, Any]]:
    """Daily means of a (possibly sub-daily) series paired with ground values by date.

    ``ground_daily`` maps "YYYY-MM-DD" to a value. A date on only one side is
    dropped. Records {date, satellite, ground}, sorted by date.
    """
    daily_sat: dict[str, list[float]] = {}
    for t, v in zip(times, values, strict=True):
        date = t[:10]
        daily_sat.setdefault(date, []).append(v)

    paired = []
    for date in sorted(daily_sat):
        if date not in ground_daily:
            continue
        paired.append({
            "date": date,
            "satellite": float(np.mean(daily_sat[date])),
            "ground": ground_daily[date],
        })
    return paired


def correlation_stats(paired: list[dict[str, Any]], total_ground_days: int | None = None) -> dict[str, Any]:
    """Pearson r, N and coverage fraction of paired {satellite, ground} records.

    ``coverage_fraction`` is the paired count over ``total_ground_days``
    (the paired count itself when not given). r is None for fewer than two
    pairs or a constant side.
    """
    n = len(paired)
    r = None
    if n >= 2:
        sat = np.array([p["satellite"] for p in paired])
        grd = np.array([p["ground"] for p in paired])
        if np.std(sat) > 0 and np.std(grd) > 0:
            r = float(np.corrcoef(sat, grd)[0, 1])

    total = total_ground_days if total_ground_days is not None else n
    return {
        "r": r,
        "n": n,
        "coverage_fraction": (n / total) if total else 0.0,
    }
