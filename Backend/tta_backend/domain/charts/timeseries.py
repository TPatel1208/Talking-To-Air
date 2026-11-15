"""The ``timeseries`` payload: one line, or several on shared or paired axes.

A one-line payload is ``times``/``values``. A payload of several lines keeps
the first line in ``times``/``values`` and lists every line in ``series``,
each naming the y axis it is drawn against (``y`` or ``y2``) from ``axes``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Lines with different units are drawn against at most this many y axes.
MAX_AXES = 2
AXIS_IDS = ("y", "y2")


def timeseries_points(reduced: Any, time_values: Any) -> tuple[list[str], list[float], list[int]]:
    """The finite values of a per-timestep reduction with their ISO times
    and timestep indices, sorted by time. A NaN timestep is left off the line."""
    times: list[str] = []
    values: list[float] = []
    indices: list[int] = []
    for i, value in enumerate(np.atleast_1d(np.asarray(reduced))):
        if not np.isfinite(value):
            continue
        times.append(pd.Timestamp(time_values[i]).isoformat())
        values.append(round(float(value), 6))
        indices.append(i)
    if not times:
        return [], [], []
    paired = sorted(zip(times, values, indices, strict=True))
    sorted_times, sorted_values, sorted_indices = zip(*paired, strict=True)
    return list(sorted_times), list(sorted_values), list(sorted_indices)


def timeseries_payload(title: str, variable: str, units: str, stat: str, times: list[str], values: list[float]) -> dict[str, Any]:
    """A one-line ``timeseries`` payload, before its reproducibility blocks."""
    return {
        "type":     "timeseries",
        "title":    title,
        "variable": variable,
        "units":    units,
        "stat":     stat,
        "times":    list(times),
        "values":   list(values),
    }


@dataclass(frozen=True)
class Line:
    """One line of a multi-line chart; ``units`` None means the unit is not known."""

    label: str
    units: str | None
    times: list[str]
    values: list[float]
    source: dict[str, Any]


def axis_units(lines: Sequence[Line]) -> list[str | None]:
    """The distinct units of ``lines``, in first-seen order: one y axis each."""
    return list(dict.fromkeys(line.units for line in lines))


def line_points(frame: pd.DataFrame) -> tuple[list[str], list[float]]:
    """A line's finite values and their ISO times, sorted by time, from ``time`` and ``value`` columns."""
    kept = frame[np.isfinite(frame["value"].to_numpy(dtype="float64"))].sort_values("time", kind="stable")
    times = [pd.Timestamp(t).isoformat() for t in kept["time"]]
    values = [round(float(v), 6) for v in kept["value"]]
    return times, values


def multi_series_payload(lines: Sequence[Line], *, title: str, variable: str, stat: str) -> dict[str, Any]:
    """A ``timeseries`` payload of several lines. Each distinct unit gets its
    own y axis, up to MAX_AXES; raises ValueError beyond that or with no lines."""
    if not lines:
        raise ValueError("a time series needs at least one line")
    units = axis_units(lines)
    if len(units) > MAX_AXES:
        raise ValueError(f"lines in {len(units)} units cannot share {MAX_AXES} axes")
    axis_of = {unit: AXIS_IDS[i] for i, unit in enumerate(units)}
    first = lines[0]
    return {
        "type": "timeseries",
        "title": title,
        "variable": variable,
        "units": first.units or "",
        "stat": stat,
        "times": list(first.times),
        "values": list(first.values),
        "axes": [{"id": axis_of[unit], "units": unit or ""} for unit in units],
        "series": [
            {
                "label": line.label,
                "units": line.units or "",
                "axis": axis_of[line.units],
                "times": list(line.times),
                "values": list(line.values),
                **line.source,
            }
            for line in lines
        ],
    }
