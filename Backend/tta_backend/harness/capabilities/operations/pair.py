"""Pairing a gridded satellite series with ground monitors' daily readings.

For each monitor with readings, the grid cell nearest it supplies the satellite
values (already masked, so a cell the quality rules or the area's outline
masked has none). Satellite values are averaged to a day and paired with the
monitor's reading for the same day; a day on one side only is left out. Only
the quantity and pollutant pairs in ``ACCEPTED_PAIRS`` are paired, because
the others have not been checked to measure comparable things.

A day is the monitor's local standard-time day, which is what AQS calls
date_local. AQS gives no UTC offset on a monitor or a daily row, so the
offset is ``round(longitude / 15)`` hours; a satellite time is shifted by it
before it is bucketed into a day. A series stamped only at midnight UTC
holds one value per date, not instants, so its dates are used as they are.

A correlation needs ``MIN_PAIRS_FOR_R`` paired days; below that r is not given
(two pairs always correlate perfectly), though the counts still are.

The result is a series handle of the satellite's daily value at each monitor's
cell on its paired days, in the grid's unit, one line per monitor. The ground
values stay in the readings handle: a column and a surface reading differ in
unit, so there is no bias and the lines are drawn on separate axes.
"""
from __future__ import annotations

import math
from typing import Annotated, Any

import pandas as pd
import xarray as xr
from pydantic import Field, JsonValue

from tta_backend.domain.analysis.pairing import (
    correlation_stats,
    finite_series,
    nearest_cell_series,
    pair_daily,
    within_grid,
)
from tta_backend.domain.charts.monitor_map import pollutant_for
from tta_backend.domain.geo import find_lat_coord, find_lon_coord
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import number, plain, refusal
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.operations.common import (
    GridInput,
    derived,
    open_grid_input,
    refuse,
)
from tta_backend.harness.capabilities.operations.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import points_frame, series_frame
from tta_backend.harness.trace import FailureCategory, StepFailure

__all__ = ["ACCEPTED_PAIRS", "CAPABILITIES", "GROUP", "MIN_PAIRS_FOR_R", "pair_with_monitors"]

# Satellite quantity id to the pollutant key of the monitors it may be paired with.
ACCEPTED_PAIRS: dict[str, str] = {"no2_tropospheric_column": "no2"}

UNSUPPORTED_PAIR = (
    "{name}: {quantity} cannot be paired with {pollutant} readings; the accepted pairs are {pairs} "
    "(satellite quantity with monitor pollutant)."
)
NO_PAIRS = (
    "No day had both a valid satellite value at a monitor's nearest cell and a ground reading, "
    "across {monitors} monitors."
)
COLUMN_VS_SURFACE = (
    "The satellite values are a tropospheric column ({satellite_unit}) and the monitors report surface "
    "concentrations ({ground_unit}). The two are in different units, so the correlation shows agreement in "
    "pattern over time, not in level, and no bias is given."
)
R_NEEDS_PAIRS = (
    "A correlation r needs at least {minimum} paired days, so none is given for {where}; the counts are still reported."
)
SHARED_CELL = (
    "Monitors {ids} share one nearest satellite cell, so that cell's values count once for each of them in the "
    "pooled correlation."
)
PER_MONITOR = "Per monitor, correlation r over its paired days: {lines}."
WITHOUT_PAIRS = "{count} monitor(s) had no paired day and are left out of the statistics: {ids}."
MAX_LISTED = 10
# The fewest paired days a correlation is reported for.
MIN_PAIRS_FOR_R = 5
_EARTH_RADIUS_KM = 6371.0


class PairParams(Params):
    grid: Annotated[str, handle_param("grid"), Field(description="A satellite grid handle with a time axis, covering the monitors.")]
    readings: Annotated[str, handle_param("series"), Field(description="Daily ground readings, one series line per monitor.")]
    monitors: Annotated[str, handle_param("points"), Field(description="The monitors' locations (id, lat, lon), as sources.monitors.find returns them.")]


def _unsupported(call: Call[Any], quantity: str | None, pollutant: str | None) -> StepFailure:
    return refusal(
        call, UNSUPPORTED_PAIR, "unsupported_pair", category=FailureCategory.BAD_INPUT,
        quantity=quantity or "a quantity not recorded", pollutant=pollutant or "an unrecorded",
        pairs=", ".join(f"{q} with {p}" for q, p in ACCEPTED_PAIRS.items()),
    )


def _km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Great-circle distance in kilometres."""
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    half = (
        math.sin((phi_b - phi_a) / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(math.radians(lon_b - lon_a) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(half))


def _ground_by_day(rows: pd.DataFrame) -> dict[str, float]:
    """A monitor's readings as {YYYY-MM-DD: value}, the mean where a day has several."""
    days = rows["time"].dt.strftime("%Y-%m-%d")
    return {str(day): float(value) for day, value in rows.groupby(days, sort=True)["value"].mean().items()}


def _local_offset_hours(lon: float) -> int:
    """Hours from UTC to a longitude's standard time."""
    return round(lon / 15)


def _local_days(times: list[str], lon: float) -> list[str]:
    """ISO times (UTC) moved to the local standard time of ``lon``, so their date is the local day.

    Times that are all midnight are date labels and are left as they are.
    """
    if all(pd.Timestamp(t) == pd.Timestamp(t).normalize() for t in times):
        return times
    shift = pd.Timedelta(hours=_local_offset_hours(lon))
    return [(pd.Timestamp(t) + shift).isoformat() for t in times]


def _r_of(stats: dict[str, Any]) -> float | None:
    return stats["r"] if stats["n"] >= MIN_PAIRS_FOR_R else None


def _pair_all(data: xr.DataArray, readings: pd.DataFrame, monitors: pd.DataFrame) -> dict[str, Any]:
    """Each monitor's paired days, its statistics and the satellite lines to store."""
    where = {str(row.id): row for row in monitors.itertuples()}
    lat_name, lon_name = find_lat_coord(data), find_lon_coord(data)
    by_id = {str(m): rows for m, rows in readings.groupby(readings["series"].astype(str))}
    paired_by_monitor: dict[str, list[dict[str, Any]]] = {}
    per_monitor: list[dict[str, Any]] = []
    for monitor_id in dict.fromkeys([*where, *by_id]):
        ground = _ground_by_day(by_id[monitor_id]) if monitor_id in by_id else {}
        entry: dict[str, Any] = {"id": monitor_id, "ground_days": len(ground), "n": 0, "r": None}
        site = where.get(monitor_id)
        if site is None:
            entry["reason"] = "no_location"
        elif not ground:
            entry["reason"] = "no_readings"
        elif not within_grid(data, float(site.lat), float(site.lon)):
            entry["reason"] = "outside_grid"
        else:
            cell = nearest_cell_series(data, float(site.lat), float(site.lon))
            times, values = finite_series(cell)
            paired = pair_daily(_local_days(times, float(site.lon)), values, ground)
            cell_lat, cell_lon = float(cell[lat_name]), float(cell[lon_name])
            entry.update(
                name=str(site.name), cell=(cell_lat, cell_lon),
                cell_km=_km(float(site.lat), float(site.lon), cell_lat, cell_lon),
            )
            if paired:
                paired_by_monitor[monitor_id] = paired
                stats = correlation_stats(paired, total_ground_days=len(ground))
                entry.update(n=stats["n"], r=_r_of(stats), coverage_fraction=stats["coverage_fraction"])
            else:
                entry["reason"] = "no_paired_day"
        per_monitor.append(entry)
    pooled_pairs = [pair for pairs in paired_by_monitor.values() for pair in pairs]
    pooled = correlation_stats(pooled_pairs)
    pooled["r"] = _r_of(pooled)
    pooled["days"] = len({pair["date"] for pair in pooled_pairs})
    by_cell: dict[tuple[float, float], list[str]] = {}
    for entry in per_monitor:
        if entry["n"]:
            by_cell.setdefault(entry["cell"], []).append(entry["id"])
    lines = [
        {"time": pd.Timestamp(pair["date"], tz="UTC"), "value": pair["satellite"], "series": monitor_id}
        for monitor_id, pairs in paired_by_monitor.items() for pair in pairs
    ]
    return {
        "per_monitor": per_monitor, "pooled": pooled, "lines": lines,
        "shared_cells": [ids for ids in by_cell.values() if len(ids) > 1],
    }


def _r_text(r: float | None) -> str:
    return "not defined" if r is None else f"{r:.2f}"


def _per_monitor_line(entries: list[dict[str, Any]]) -> str:
    shown = entries[:MAX_LISTED]
    text = "; ".join(f"{e['id']} r {_r_text(e['r'])} over {e['n']} days" for e in shown)
    return text + (f"; and {len(entries) - len(shown)} more" if len(entries) > len(shown) else "")


async def _pair(call: Call[PairParams]) -> CapabilityResult:
    grid: GridInput = await open_grid_input(call, "grid")
    if not grid.has_time:
        raise refuse(call, "no_time_axis", param="grid")
    readings_record = call.handle("readings")
    quantity = grid.record.descriptor.quantity
    pollutant = pollutant_for(readings_record.descriptor.quantity)
    if quantity not in ACCEPTED_PAIRS or ACCEPTED_PAIRS[quantity] != pollutant:
        raise _unsupported(call, quantity, pollutant)

    readings = series_frame(await call.open("readings"))
    monitors = points_frame(await call.open("monitors"))
    out = await call.compute(_pair_all, grid.data, readings, monitors)
    per_monitor: list[dict[str, Any]] = out["per_monitor"]
    pooled = out["pooled"]
    if not out["lines"]:
        return CapabilityResult(
            status="insufficient_evidence", reason=NO_PAIRS.format(monitors=len(per_monitor)),
            values={"situation": "no_pairs"},
        )

    paired = [e for e in per_monitor if e["n"]]
    unpaired = [e for e in per_monitor if not e["n"]]
    frame = series_frame(pd.DataFrame(out["lines"]))
    unit = grid.unit
    record = await call.mint(
        "series", frame,
        derived(
            grid.record,
            f"{grid.record.descriptor.quantity} at the cell nearest each of {len(paired)} monitors on days with a ground reading",
            unit=unit, masking=grid.masking,
        ),
    )
    ground_unit = readings_record.descriptor.unit or "an unrecorded unit"
    disclosures = [COLUMN_VS_SURFACE.format(satellite_unit=unit or "an unrecorded unit", ground_unit=ground_unit)]
    disclosures.append(PER_MONITOR.format(lines=_per_monitor_line(paired)))
    short_of_r = [e["id"] for e in paired if e["n"] < MIN_PAIRS_FOR_R]
    short_where = [f"monitor {i}" for i in short_of_r[:MAX_LISTED]]
    if pooled["n"] < MIN_PAIRS_FOR_R:
        short_where.append("the pooled result")
    if short_where:
        disclosures.append(R_NEEDS_PAIRS.format(minimum=MIN_PAIRS_FOR_R, where=", ".join(short_where)))
    for ids in out["shared_cells"]:
        disclosures.append(SHARED_CELL.format(ids=", ".join(ids[:MAX_LISTED])))
    if unpaired:
        disclosures.append(WITHOUT_PAIRS.format(count=len(unpaired), ids=", ".join(e["id"] for e in unpaired[:MAX_LISTED])))

    summary: dict[str, JsonValue] = {
        "monitors": len(per_monitor),
        "monitors_paired": len(paired),
        "monitors_without_pairs": len(unpaired),
        "paired_days": pooled["days"],
        "pairs": pooled["n"],
        "pooled_r": number(pooled["r"]),
        "per_monitor": [
            {
                "id": e["id"], "r": number(e["r"]), "n": e["n"], "ground_days": e["ground_days"],
                "cell_km": None if "cell_km" not in e else round(e["cell_km"], 3),
                "reason": e.get("reason"),
            }
            for e in per_monitor
        ],
    }
    values: dict[str, JsonValue] = {"pairing": "daily", "ground_unit": ground_unit, "pollutant": plain(pollutant)}
    return call.done(record, summary=summary, values=values, disclosures=tuple(disclosures))


pair_with_monitors = Capability(
    name="operations.pair_with_monitors",
    description=(
        "Pair a satellite grid with ground monitors: at each monitor's nearest cell, daily satellite values matched "
        "to its daily readings, with correlation r and paired-day counts per monitor and pooled."
    ),
    params=PairParams,
    output_kind="series",
    label="Pairing satellite data with ground monitors",
    method=(
        "Each monitor's nearest grid cell was paired with its daily readings by local day: {summary.pairs} monitor-days "
        "over {summary.paired_days} days, at {summary.monitors_paired} of {summary.monitors} monitors."
    ),
    retrieves=False,
    retry_safe=True,
    run=_pair,
)

CAPABILITIES: tuple[Capability[Any], ...] = (pair_with_monitors,)
