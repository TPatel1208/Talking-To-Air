"""Figures and data files drawn from a stored chart payload alone.

A chart a harness run drew may render a handle the backend derived, which
the data service cannot open, so these read nothing but the payload: a PNG of
a heatmap from the grid the payload carries, of a time series from its lines
and of a profile from its layers; CSV rows of a time series's lines and of a
profile's layers, which the payload holds in full. Any other chart type has
neither, and the functions say so by returning None.
"""
from __future__ import annotations

import asyncio
import csv
import io
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from tta_backend.services.export_service import ExportService

# Chart types each function can draw or tabulate from the payload.
FIGURE_TYPES = frozenset({"heatmap", "timeseries", "profile"})
ROW_TYPES = frozenset({"timeseries", "profile"})

_PROFILES = ExportService()


def _figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height))


def _png(fig: Figure) -> bytes:
    fig.tight_layout()
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=160, bbox_inches="tight")
    return output.getvalue()


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else float("nan")


def _colormap(payload: dict[str, Any]) -> Any:
    lut = (payload.get("colormap") or {}).get("lut")
    if isinstance(lut, list) and lut and all(isinstance(c, list) and len(c) == 4 for c in lut):
        return ListedColormap(np.asarray(lut, dtype=float) / 255.0)
    return "viridis"


def _heatmap(payload: dict[str, Any]) -> bytes | None:
    lats, lons, values = payload.get("lats"), payload.get("lons"), payload.get("values")
    if not (isinstance(lats, list) and isinstance(lons, list) and isinstance(values, list) and lats and lons):
        return None
    grid = np.array([[_number(v) for v in row] for row in values], dtype=float)
    if grid.shape != (len(lats), len(lons)):
        return None
    fig = _figure(8, 6)
    ax = fig.subplots()
    vmin, vmax = payload.get("vmin"), payload.get("vmax")
    mesh = ax.pcolormesh(
        [_number(v) for v in lons], [_number(v) for v in lats], grid, shading="auto", cmap=_colormap(payload),
        vmin=vmin if isinstance(vmin, int | float) else None, vmax=vmax if isinstance(vmax, int | float) else None,
    )
    points = ((payload.get("overlays") or {}).get("points") or {}).get("points") or []
    if points:
        ax.scatter([_number(p.get("lon")) for p in points], [_number(p.get("lat")) for p in points],
                   s=12, c="black", marker="o", linewidths=0)
    fig.colorbar(mesh, ax=ax, label=str(payload.get("units") or ""))
    ax.set_title(str(payload.get("title") or payload.get("variable") or "Map"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    return _png(fig)


def _lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    series = payload.get("series")
    if isinstance(series, list) and series and all(isinstance(line, dict) for line in series):
        return series
    return [{
        "label": payload.get("variable") or "value", "units": payload.get("units") or "",
        "times": payload.get("times") or [], "values": payload.get("values") or [],
    }]


def _timeseries(payload: dict[str, Any]) -> bytes | None:
    lines = [line for line in _lines(payload) if line.get("times")]
    if not lines:
        return None
    fig = _figure(10, 5)
    ax = fig.subplots()
    axes: dict[str, Any] = {}
    for line in lines:
        units = str(line.get("units") or "")
        if units not in axes:
            axes[units] = ax if not axes else ax.twinx()
            axes[units].set_ylabel(units)
        times = pd.to_datetime(line.get("times"), utc=True, errors="coerce")
        axes[units].plot(times, [_number(v) for v in line.get("values") or []], linewidth=1.3, label=str(line.get("label") or ""))
    ax.set_title(str(payload.get("title") or "Time series"))
    ax.set_xlabel("Time (UTC)")
    ax.tick_params(axis="x", rotation=30)
    handles = [h for target in axes.values() for h in target.get_legend_handles_labels()[0]]
    if len(handles) > 1:
        ax.legend(handles=handles, loc="best", fontsize="small")
    return _png(fig)


async def figure_png(payload: dict[str, Any]) -> bytes | None:
    """The chart as a PNG drawn from its payload, or None for a type it cannot draw."""
    chart_type = payload.get("type")
    if chart_type == "heatmap":
        return await asyncio.to_thread(_heatmap, payload)
    if chart_type == "timeseries":
        return await asyncio.to_thread(_timeseries, payload)
    if chart_type == "profile" and payload.get("export"):
        return await _PROFILES.build_chart_png(payload, {})
    return None


async def chart_rows(payload: dict[str, Any]) -> list[list[Any]] | None:
    """The chart's data as CSV rows, header first, or None for a type whose payload does not hold its data in full."""
    chart_type = payload.get("type")
    if chart_type == "timeseries":
        rows: list[list[Any]] = [["series", "time", "value", "units", "handle"]]
        for line in _lines(payload):
            for time, value in zip(line.get("times") or [], line.get("values") or [], strict=False):
                rows.append([line.get("label") or "", time, value, line.get("units") or "", line.get("handle") or ""])
        return rows
    if chart_type == "profile" and payload.get("export"):
        return [row async for row in _PROFILES.iter_chart_csv_rows(payload, {})]
    return None


def csv_bytes(rows: Iterable[Iterable[Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


__all__ = ["FIGURE_TYPES", "ROW_TYPES", "chart_rows", "csv_bytes", "figure_png"]
