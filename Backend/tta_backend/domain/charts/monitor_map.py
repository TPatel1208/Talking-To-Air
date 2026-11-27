"""Monitor points coloured by AQI category, the rows of the ground-monitor map.

A point's colour is its pollutant's AQI category from datasets/standards.yaml,
never its position between the lowest and highest value shown. A value the
breakpoint table does not cover is labelled as such, not given the nearest
category. Values are in the pollutant's breakpoint unit.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tta_backend.datasets.standards import AqiBreakpoint, NaaqsStandard, OutOfTable, aqi_category, load_standards
from tta_backend.domain.analysis.thresholds import truncate_for_standard

# The EPA AQI category colours.
AQI_COLORS: dict[str, str] = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
}
BEYOND_LABEL = "Beyond the AQI table"
BEYOND_COLOR = "#3F0011"
BELOW_LABEL = "Below this period's AQI table"
BELOW_COLOR = "#9CA3AF"
NO_VALUE_LABEL = "No value"
NO_VALUE_COLOR = "#CBD5E1"


def pollutant_for(quantity: str | None) -> str | None:
    """The standards key whose display name starts ``quantity``, as the monitor sources name it."""
    if not quantity:
        return None
    text = quantity.strip().lower()
    matches = [key for key, info in load_standards().items() if text.startswith(info.name.lower())]
    return matches[0] if len(matches) == 1 else None


def averaging_period(pollutant: str, requested: str | None = None) -> str:
    """``requested`` when the pollutant has breakpoints for it, else its default.
    Raises ValueError for a period it has no breakpoints for."""
    info = load_standards()[pollutant]
    period = requested or info.default_averaging
    if not info.breakpoints_for(period):
        raise ValueError(f"{pollutant} has no {period!r} breakpoints")
    return period


def standard_for(pollutant: str, period: str) -> NaaqsStandard | None:
    """The NAAQS with the same averaging period, when there is one."""
    for entry in load_standards()[pollutant].naaqs:
        if entry.averaging_period == period:
            return entry
    return None


def classify(pollutant: str, value: float | None, period: str) -> dict[str, Any]:
    """A value's AQI category, its AQI range and its colour."""
    if value is None or not math.isfinite(value):
        return {"aqi_category": NO_VALUE_LABEL, "aqi_low": None, "aqi_high": None, "color": NO_VALUE_COLOR}
    found = aqi_category(pollutant, value, period)
    if isinstance(found, AqiBreakpoint):
        return {
            "aqi_category": found.category, "aqi_low": found.aqi_low, "aqi_high": found.aqi_high,
            "color": AQI_COLORS.get(found.category, BEYOND_COLOR),
        }
    if found is OutOfTable.BEYOND:
        return {"aqi_category": BEYOND_LABEL, "aqi_low": None, "aqi_high": None, "color": BEYOND_COLOR}
    return {"aqi_category": BELOW_LABEL, "aqi_low": None, "aqi_high": None, "color": BELOW_COLOR}


def legend(pollutant: str, period: str) -> list[dict[str, Any]]:
    """Every category the map can show, in table order, with its concentration and AQI range."""
    info = load_standards()[pollutant]
    rows = info.breakpoints_for(period)
    entries: list[dict[str, Any]] = []
    if rows and rows[0].concentration_low > 0:
        entries.append({"category": BELOW_LABEL, "color": BELOW_COLOR, "concentration_high": rows[0].concentration_low})
    for row in rows:
        entries.append({
            "category": row.category, "color": AQI_COLORS.get(row.category, BEYOND_COLOR),
            "concentration_low": row.concentration_low, "concentration_high": row.concentration_high,
            "aqi_low": row.aqi_low, "aqi_high": row.aqi_high,
        })
    entries.append({"category": BEYOND_LABEL, "color": BEYOND_COLOR, "concentration_low": rows[-1].concentration_high})
    return entries


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def monitor_rows(points: pd.DataFrame, *, pollutant: str, period: str, unit: str) -> list[dict[str, Any]]:
    """One row per point in the ground-monitor map's layout (``site_id``,
    ``local_site_name``, ``latitude``, ``longitude``), with its ``value``,
    AQI category, colour and whether it is above the NAAQS level of the same
    period, compared after truncating to the reporting precision as the AQI
    category is."""
    standard = standard_for(pollutant, period)
    rows: list[dict[str, Any]] = []
    for record in points.to_dict(orient="records"):
        value = _finite_or_none(record.get("value"))
        row: dict[str, Any] = {
            "site_id": str(record["id"]),
            "local_site_name": None if pd.isna(record.get("name")) else str(record.get("name")),
            "latitude": float(record["lat"]),
            "longitude": float(record["lon"]),
            "value": value,
            "units": unit,
            "averaging_period": period,
            **classify(pollutant, value, period),
        }
        if standard is not None and standard.unit == unit:
            row["standard_level"] = standard.level
            row["above_standard_level"] = (
                None if value is None else float(truncate_for_standard(pollutant, value)) > standard.level
            )
        rows.append(row)
    return rows


SITE_TABLE_COLUMNS = ("site_id", "name", "lat", "lon", "value", "unit")
SITE_STATS = ("max", "mean")


def missing_site_columns(frame: pd.DataFrame) -> list[str]:
    """The ``SITE_TABLE_COLUMNS`` a table lacks."""
    return [column for column in SITE_TABLE_COLUMNS if column not in frame.columns]


def site_units(frame: pd.DataFrame) -> list[str]:
    """The distinct units named in a site table's ``unit`` column, in order seen."""
    return list(dict.fromkeys(str(u) for u in frame["unit"].dropna() if str(u).strip()))


def one_point_per_site(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    """A site table reduced to one point per ``site_id``, in the points layout
    (``id``, ``name``, ``lat``, ``lon``, ``value``) with ``rows``, how many
    rows the site had.

    The value is the ``stat`` (max or mean) of the site's finite values, or
    missing when it has none; name, latitude and longitude are its first row's.
    """
    if stat not in SITE_STATS:
        raise ValueError(f"stat is one of {SITE_STATS}")
    values = pd.to_numeric(frame["value"], errors="coerce")
    finite = frame.assign(value=values.where(np.isfinite(values)))
    grouped = finite.groupby(frame["site_id"].astype(str), sort=False)
    first = grouped[["name", "lat", "lon"]].first()
    reduced = grouped["value"].max() if stat == "max" else grouped["value"].mean()
    return pd.DataFrame({
        "id": first.index.astype(str),
        "name": first["name"].to_numpy(),
        "lat": first["lat"].astype(float).to_numpy(),
        "lon": first["lon"].astype(float).to_numpy(),
        "value": reduced.reindex(first.index).to_numpy(dtype=float),
        "rows": grouped.size().reindex(first.index).to_numpy(dtype=int),
    })


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["aqi_category"]] = counts.get(row["aqi_category"], 0) + 1
    return counts
