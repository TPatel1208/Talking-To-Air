"""Each day's peak of a field or a series: its value, when and where."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

GRID_PEAK_COLUMNS = ["date", "time", "value", "lat", "lon"]
SERIES_PEAK_COLUMNS = ["date", "time", "value", "series"]


def _days(times: np.ndarray) -> list[str]:
    return [pd.Timestamp(t).date().isoformat() for t in times]


def daily_peaks_grid(da: xr.DataArray) -> pd.DataFrame:
    """One row per day with a valid value: the highest cell over that day's time
    steps, the step it was seen at, and the cell's latitude and longitude.

    ``da`` is a masked (time, lat, lon) field. A day without a valid value has no row.
    """
    field = da.transpose("time", "lat", "lon")
    values = np.asarray(field.values, dtype=float)
    times = np.atleast_1d(np.asarray(field["time"].values))
    lats = np.asarray(field["lat"].values, dtype=float)
    lons = np.asarray(field["lon"].values, dtype=float)
    days = _days(times)
    rows = []
    for day in sorted(set(days)):
        steps = [i for i, d in enumerate(days) if d == day]
        block = values[steps]
        finite = np.isfinite(block)
        if not finite.any():
            continue
        flat = int(np.nanargmax(np.where(finite, block, np.nan)))
        step, lat, lon = np.unravel_index(flat, block.shape)
        rows.append({
            "date": day,
            "time": pd.Timestamp(times[steps[step]]).isoformat(),
            "value": float(block[step, lat, lon]),
            "lat": float(lats[lat]),
            "lon": float(lons[lon]),
        })
    return pd.DataFrame(rows, columns=GRID_PEAK_COLUMNS)


def daily_peaks_series(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per day with a valid value: the highest reading across every line,
    its time, and the line it belongs to. Days are UTC dates."""
    valid = frame[np.isfinite(frame["value"].astype(float))]
    rows = []
    if not valid.empty:
        times = pd.to_datetime(valid["time"], utc=True)
        for day, group in valid.groupby(times.dt.date.astype(str), sort=True):
            best = group.loc[group["value"].astype(float).idxmax()]
            rows.append({
                "date": str(day),
                "time": pd.Timestamp(best["time"]).isoformat(),
                "value": float(best["value"]),
                "series": str(best["series"]),
            })
    return pd.DataFrame(rows, columns=SERIES_PEAK_COLUMNS)
