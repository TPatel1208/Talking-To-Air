"""Values above a threshold: in a field, a series, or AQS daily records.

A value exceeds a level when it is above it. Against a pollutant's
standard, values are first truncated to its reporting precision, as AQI
classification truncates them (``truncate_for_standard``); against a level
given as a number they are compared raw. Nothing here computes a standard's
averaging period or form.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.datasets.standards import NaaqsStandard, load_standards, truncate_to_reporting_precision
from tta_backend.domain.aggregation.aggregation_service import cos_lat_weights

GRID_EXCEEDANCE_COLUMNS = ["date", "valid_cells", "cells_above", "area_fraction_above", "max", "max_lat", "max_lon"]
SERIES_EXCEEDANCE_COLUMNS = ["date", "time", "series", "value"]


class UnknownStandard(LookupError):
    """A pollutant or averaging period the standards file does not hold."""


def naaqs_for(pollutant: str, averaging_period: str | None) -> NaaqsStandard | list[str]:
    """The pollutant's NAAQS for ``averaging_period``; without one, its only
    standard, or the averaging periods to choose from when it has several.

    Raises UnknownStandard for an unknown pollutant or averaging period.
    """
    standards = load_standards()
    if pollutant not in standards:
        raise UnknownStandard(pollutant)
    naaqs = standards[pollutant].naaqs
    if averaging_period is None:
        return naaqs[0] if len(naaqs) == 1 else [n.averaging_period for n in naaqs]
    for standard in naaqs:
        if standard.averaging_period == averaging_period:
            return standard
    raise UnknownStandard(averaging_period)


def reporting_decimals(pollutant: str) -> int:
    """How many decimals the pollutant's concentrations are truncated to."""
    return load_standards()[pollutant].decimals


def truncate_for_standard(pollutant: str, values: Any) -> np.ndarray:
    """``values`` truncated to the pollutant's reporting precision
    (``truncate_to_reporting_precision``), in its breakpoint unit; non-finite
    values are kept as they are."""
    arr = np.asarray(values, dtype=float)
    truncate = np.frompyfunc(lambda v: truncate_to_reporting_precision(pollutant, v), 1, 1)
    return np.asarray(truncate(arr), dtype=float)


def _compared(values: np.ndarray, pollutant: str | None) -> np.ndarray:
    return truncate_for_standard(pollutant, values) if pollutant is not None else values


def _area_weights(field: xr.DataArray) -> np.ndarray:
    weights = cos_lat_weights(field)
    if weights is None:
        return np.ones(field.shape[-2:], dtype=float)
    lat_weights = np.asarray(weights.values, dtype=float)
    return np.broadcast_to(lat_weights[:, None], field.shape[-2:])


def grid_exceedances(
    da: xr.DataArray, level: float, *, truncate_for: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per day, the valid cells, the cells above ``level`` at any time step that
    day, their cos(latitude) area fraction and the day's highest cell; and over
    the whole field, the days and the area above it at least once.

    ``da`` is a masked field with ``lat``, ``lon`` and optionally ``time``; a
    field without time is one row with no date. Days without a valid cell have no row.
    With ``truncate_for`` (a pollutant key), values are truncated to its
    reporting precision before comparing; the reported maxima are not truncated.
    """
    field = da if "time" in da.dims else da.expand_dims(time=[np.datetime64("NaT", "ns")])
    field = field.transpose("time", "lat", "lon")
    values = np.asarray(field.values, dtype=float)
    compared = _compared(values, truncate_for)
    times = np.atleast_1d(np.asarray(field["time"].values))
    lats = np.asarray(field["lat"].values, dtype=float)
    lons = np.asarray(field["lon"].values, dtype=float)
    weights = _area_weights(field)
    days = [None if pd.isna(t) else pd.Timestamp(t).date().isoformat() for t in times]

    rows = []
    ever_valid = np.zeros(values.shape[1:], dtype=bool)
    ever_above = np.zeros(values.shape[1:], dtype=bool)
    for day in dict.fromkeys(sorted(days, key=lambda d: d or "")):
        steps = [i for i, d in enumerate(days) if d == day]
        block = values[steps]
        finite = np.isfinite(block)
        valid = finite.any(axis=0)
        if not valid.any():
            continue
        above = (np.where(finite, compared[steps], -np.inf) > level).any(axis=0)
        ever_valid |= valid
        ever_above |= above
        day_max = np.where(finite, block, -np.inf).max(axis=0)
        lat, lon = np.unravel_index(int(np.argmax(np.where(valid, day_max, -np.inf))), valid.shape)
        rows.append({
            "date": day,
            "valid_cells": int(valid.sum()),
            "cells_above": int(above.sum()),
            "area_fraction_above": float(weights[above].sum() / weights[valid].sum()),
            "max": float(day_max[lat, lon]),
            "max_lat": float(lats[lat]),
            "max_lon": float(lons[lon]),
        })
    table = pd.DataFrame(rows, columns=GRID_EXCEEDANCE_COLUMNS)
    overall = {
        "days_with_data": len(rows),
        "days_above": int((table["cells_above"] > 0).sum()) if rows else 0,
        "cells_with_data": int(ever_valid.sum()),
        "cells_ever_above": int(ever_above.sum()),
        "area_fraction_ever_above": float(weights[ever_above].sum() / weights[ever_valid].sum()) if ever_valid.any() else 0.0,
    }
    return table, overall


def series_exceedances(
    frame: pd.DataFrame, level: float, *, truncate_for: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The readings above ``level`` (date, time, series line, value), and how many
    readings were checked, how many and on how many days were above, and on how many lines.

    With ``truncate_for`` (a pollutant key), values are truncated to its
    reporting precision before comparing; the listed values are not truncated.
    """
    values = frame["value"].astype(float)
    valid = frame[np.isfinite(values)]
    compared = _compared(valid["value"].to_numpy(dtype=float), truncate_for)
    above = valid[compared > level]
    times = pd.to_datetime(above["time"], utc=True)
    table = pd.DataFrame({
        "date": times.dt.date.astype(str).to_numpy(),
        "time": [t.isoformat() for t in times],
        "series": above["series"].astype(str).to_numpy(),
        "value": above["value"].astype(float).to_numpy(),
    }, columns=SERIES_EXCEEDANCE_COLUMNS)
    all_days = pd.to_datetime(valid["time"], utc=True).dt.date
    overall = {
        "values_checked": len(valid),
        "values_above": len(table),
        "days_with_data": int(all_days.nunique()),
        "days_above": int(table["date"].nunique()),
        "lines_above": int(table["series"].nunique()),
    }
    return table, overall


def exceedance_days(
    records: list[dict[str, Any]],
    measurement_field: str,
    hard_threshold: float | None,
    percentile_threshold: float | None,
) -> set[str]:
    """The ``date_local`` values of AQS records whose ``measurement_field`` is
    above ``hard_threshold`` or at or above the ``percentile_threshold`` cutoff
    (90.0 is the top 10%). Records without a value or a date are not flagged.
    """
    valid = [
        float(r[measurement_field])
        for r in records
        if r.get(measurement_field) is not None
    ]
    percentile_cutoff = None
    if percentile_threshold is not None and valid:
        sorted_vals = sorted(valid)
        idx = min(int(len(sorted_vals) * percentile_threshold / 100), len(sorted_vals) - 1)
        percentile_cutoff = sorted_vals[idx]

    exceeded: set[str] = set()
    for r in records:
        raw = r.get(measurement_field)
        date = r.get("date_local")
        # An undated record cannot name an exceedance day.
        if raw is None or date is None:
            continue
        v = float(raw)
        if hard_threshold is not None and v > hard_threshold:
            exceeded.add(date)
        elif percentile_cutoff is not None and v >= percentile_cutoff:
            exceeded.add(date)
    return exceeded
