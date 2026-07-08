"""
validation_tools.py
--------------------
Satellite<->ground validation tools (PRD T07).

Co-locates a retrieved satellite cube with EPA AQS ground monitors: one cube
retrieval over the AOI, satellite values extracted at every monitor location
by nearest-cell selection (TTA-side, never per-monitor AppEEARS jobs), paired
with the AQS daily series in time, and summarized with correlation and
coverage statistics. Renders as overlay `timeseries` artifacts, one per
monitor, per T06's TimeseriesArtifactMetadata.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_core.tools import BaseTool

from config.workflow_stages import STAGE_RENDER
from datasets.mask_info import override_for
from earthdata_mcp.results import MCPToolError
from preprocessing.aggregation_service import AggregationService
from services.artifact_registry import build_artifact_reference
from services.open_handle import OpenHandleError, open_handle
from tools.ground_sensor_tools import epa_aqs_tools as aqs
from utils.geo_utils import find_lat_coord, find_lon_coord
from utils.plotting import RegionResolver
from utils.streaming import emit_status

_aggregation_service = AggregationService()
_resolver = RegionResolver()


def _nearest_cell_series(da, lat: float, lon: float):
    """Select the nearest grid cell to (lat, lon) from a lat/lon-indexed DataArray."""
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    return da.sel({lat_coord: lat, lon_coord: lon}, method="nearest")


def _extract_monitor_series(da, lat: float, lon: float, col_info: dict | None = None):
    """Extract the satellite time series at the nearest cell to (lat, lon).

    Fill values and out-of-range cells (per T03 masking, AggregationService.
    apply_quality_mask) are excluded from the returned series and counted
    toward coverage stats — never silently included as valid readings.

    Returns (times: list[ISO str], values: list[float], coverage: dict) where
    coverage = {n_total, n_valid, n_excluded, coverage_fraction}.
    """
    masked = _aggregation_service.apply_quality_mask(da, col_info=col_info or {})
    series = _nearest_cell_series(masked, lat, lon)

    n_total = series.sizes.get("time", 1)
    raw_values = np.atleast_1d(series.values)
    raw_times = np.atleast_1d(series["time"].values) if "time" in series.coords else [None] * n_total

    times, values = [], []
    for t, v in zip(raw_times, raw_values):
        if not np.isfinite(v):
            continue
        times.append(pd.Timestamp(t).isoformat() if t is not None else None)
        values.append(float(v))

    n_valid = len(values)
    coverage = {
        "n_total": int(n_total),
        "n_valid": n_valid,
        "n_excluded": int(n_total) - n_valid,
        "coverage_fraction": (n_valid / n_total) if n_total else 0.0,
    }
    return times, values, coverage


def _pair_daily(times: list[str], values: list[float], ground_daily: dict[str, float]) -> list[dict]:
    """Aggregate a (possibly sub-daily) satellite series to daily means and pair
    with ground daily values by date.

    ``ground_daily`` maps an ISO date string ("YYYY-MM-DD") to a ground value.
    Dates present on only one side are dropped — pairing is honest about
    temporal mismatch rather than inventing a value for a missing side.
    Returns records sorted by date: {date, satellite, ground}.
    """
    daily_sat: dict[str, list[float]] = {}
    for t, v in zip(times, values):
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


def _correlation_stats(paired: list[dict], total_ground_days: int | None = None) -> dict:
    """Compute Pearson r, N, and coverage fraction for a list of paired
    {satellite, ground} records (as produced by ``_pair_daily``, one
    monitor's worth or a pooled concatenation across monitors).

    ``coverage_fraction`` is the paired-day count over ``total_ground_days``
    (defaults to the paired count itself, i.e. full coverage, when the
    caller doesn't know the ground record's total day count).
    r is None when fewer than 2 points or either side is constant (undefined).
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


def _mask_col_info(da) -> dict:
    short_name = da.attrs.get("short_name") or da.name or ""
    return override_for(str(short_name).upper())


def _time_range(da) -> tuple[str, str]:
    if "time" not in da.coords:
        return "", ""
    times = sorted(str(t) for t in np.atleast_1d(da["time"].values))
    if not times:
        return "", ""
    return times[0], times[-1]


def _station_id(monitor: dict) -> str:
    return "-".join(str(monitor.get(k, "??")) for k in ("state_code", "county_code", "site_number"))


def _exceedance_days(
    records: list[dict],
    measurement_field: str,
    hard_threshold: float | None,
    percentile_threshold: float | None,
) -> set[str]:
    """Return the set of ``date_local`` values whose ``measurement_field``
    exceeds ``hard_threshold`` and/or falls at/above the ``percentile_threshold``
    cutoff (top N%, e.g. 90.0 = top 10%). Mirrors find_exceedance_days'
    flagging logic (epa_aqs_tools.py) so both tools agree on what "exceeded"
    means; records with no value for the field are ignored, not flagged.
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

    exceeded = set()
    for r in records:
        raw = r.get(measurement_field)
        if raw is None:
            continue
        v = float(raw)
        if hard_threshold is not None and v > hard_threshold:
            exceeded.add(r.get("date_local"))
        elif percentile_cutoff is not None and v >= percentile_cutoff:
            exceeded.add(r.get("date_local"))
    return exceeded


def make_validate_against_ground(mcp_tools: dict[str, BaseTool]):
    @tool
    async def validate_against_ground(
        handle: str,
        location: str,
        param_code: str = "42602",
        pollutant_standard: Optional[str] = None,
        k: int = 10,
    ) -> str:
        """
        Compare a retrieved satellite cube against EPA AQS ground monitors over
        the same area — the validation question at the heart of TEMPO-era
        research. Co-locates monitors in the AOI, extracts satellite values at
        each monitor by nearest-cell selection (one retrieval, not one job per
        monitor), pairs with the AQS daily series, and reports per-monitor and
        pooled correlation/coverage statistics.

        Satellite column density and ground surface concentration are
        different physical quantities — never imply they measure the same
        thing; this tool always reports both units explicitly.

        Args:
            handle: cube_/obs_ handle from a retrieval tool, covering the AOI
                and time range to validate.
            location: place name defining the AOI, e.g. 'New Jersey'.
            param_code: AQS parameter code (default '42602' = NO2).
            pollutant_standard: AQS pollutant_standard to filter ground
                records to (recommended — see ground prompt table).
            k: max number of monitors to co-locate (default 10).

        Returns:
            JSON string: per-monitor paired series + stats, pooled stats, and
            timeseries artifact refs (one per monitor, overlaying the
            satellite and ground series).
        """
        try:
            ds = await open_handle(handle, mcp_tools)
            da = _aggregation_service.to_dataarray(ds)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        region = await _resolver.aresolve_location(location)
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        minx, miny, maxx, maxy = region["bounds"]
        bbox = [miny, maxy, minx, maxx]  # [south, north, west, east]

        start_time, end_time = _time_range(da)
        if not start_time:
            return json.dumps({
                "error": f"Handle '{handle}' has no time dimension to validate against ground data."
            })
        bdate_obj = datetime.date.fromisoformat(start_time[:10])
        edate_obj = datetime.date.fromisoformat(end_time[:10])
        bdate_str = bdate_obj.strftime("%Y%m%d")
        edate_str = edate_obj.strftime("%Y%m%d")

        monitors = await aqs._fetch_active_monitors(bbox, param_code, bdate_str, edate_str, k)
        if not monitors:
            return json.dumps({"error": f"No active {param_code} monitors found over '{location}'."})

        try:
            records, _, _ = await aqs._fetch_summary(
                "dailyData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
                None, None, None, None, miny, maxy, minx, maxx, None, None, pollutant_standard,
            )
        except RuntimeError as e:
            return json.dumps({"error": f"No ground data found over '{location}': {e}"})

        daily_rows, _, _ = aqs._aggregate_summary_records(records, "daily")
        ground_by_site: dict[str, dict[str, float]] = {}
        ground_units_by_site: dict[str, str] = {}
        for row in daily_rows:
            ground_by_site.setdefault(row["site_id"], {})[row["period"]] = row["mean"]
            ground_units_by_site.setdefault(row["site_id"], row.get("units") or "")

        col_info = _mask_col_info(da)
        satellite_units = da.attrs.get("units", "")
        variable_name = da.name or ""

        def _extract_and_pair_monitors():
            # CPU-bound per-monitor mask/extraction/pairing loop (T16), run
            # off the event loop via asyncio.to_thread below.
            monitor_results = []
            artifact_refs = []
            pooled_paired = []
            monitor_ids = []

            for monitor in monitors:
                station_id = _station_id(monitor)
                ground_daily = ground_by_site.get(station_id)
                if not ground_daily:
                    continue

                lat, lon = float(monitor["latitude"]), float(monitor["longitude"])
                times, values, coverage = _extract_monitor_series(da, lat, lon, col_info)
                paired = _pair_daily(times, values, ground_daily)
                if not paired:
                    continue

                stats = _correlation_stats(paired, total_ground_days=len(ground_daily))
                pooled_paired.extend(paired)
                monitor_ids.append(station_id)

                station_name = monitor.get("local_site_name") or monitor.get("address") or station_id
                ts_payload = {
                    "type": "timeseries",
                    "title": f"{variable_name} vs {station_name} ({station_id})",
                    "times": [p["date"] for p in paired],
                    "satellite_values": [p["satellite"] for p in paired],
                    "ground_values": [p["ground"] for p in paired],
                    "satellite_units": satellite_units,
                    "ground_units": ground_units_by_site.get(station_id, ""),
                    "stats": stats,
                    "coverage": coverage,
                    "chart_id": f"ts_{uuid.uuid4().hex[:12]}",
                    "metadata": {
                        "source_handles": [handle],
                        "series": [
                            {"label": f"{variable_name} (satellite)", "source_kind": "satellite"},
                            {
                                "label": f"EPA {station_id} ({station_name})",
                                "source_kind": "ground",
                                "station_id": station_id,
                            },
                        ],
                    },
                }
                ref = build_artifact_reference(ts_payload)
                if ref is not None:
                    artifact_refs.append(ref.model_dump(exclude_none=True))

                monitor_results.append({
                    "station_id": station_id,
                    "station_name": station_name,
                    "latitude": lat,
                    "longitude": lon,
                    "ground_units": ground_units_by_site.get(station_id, ""),
                    "stats": stats,
                    "coverage": coverage,
                    "source_handles": [handle],
                    "chart_id": ts_payload["chart_id"],
                })

            return monitor_results, artifact_refs, pooled_paired, monitor_ids

        emit_status("Validating against ground monitors...", stage=STAGE_RENDER)
        monitor_results, artifact_refs, pooled_paired, monitor_ids = await asyncio.to_thread(
            _extract_and_pair_monitors
        )

        if not monitor_results:
            return json.dumps({
                "error": (
                    f"No monitors in '{location}' had both satellite and ground data "
                    "over the requested period."
                )
            })

        pooled_stats = _correlation_stats(pooled_paired)

        return json.dumps({
            "location": region.get("name", location),
            "variable": variable_name,
            "satellite_units": satellite_units,
            "param_code": param_code,
            "pollutant_standard": pollutant_standard,
            "pairing": "daily",
            "monitors_matched": len(monitor_results),
            "monitor_ids": monitor_ids,
            "monitors": monitor_results,
            "pooled_stats": pooled_stats,
            "_artifact_refs": artifact_refs,
        })

    return validate_against_ground


def make_exceedance_overlay(mcp_tools: dict[str, BaseTool]):
    @tool
    async def exceedance_overlay(
        handle: str,
        location: str,
        param_code: str = "42602",
        hard_threshold: Optional[float] = None,
        percentile_threshold: Optional[float] = None,
        k: int = 10,
    ) -> str:
        """
        Mark days a ground monitor exceeded a pollutant standard on the
        satellite series for the same AOI — regulatory-relevant events
        anchoring a satellite<->ground comparison.

        hard_threshold: fixed value (defaults to the regulatory limit for
        known param_codes, same table as find_exceedance_days).
        percentile_threshold: top N% of the period, e.g. 90.0 = top 10%.
        Both can combine; at least one must be resolvable.

        Args:
            handle: cube_/obs_ handle from a retrieval tool, covering the AOI
                and time range to check.
            location: place name defining the AOI, e.g. 'New Jersey'.
            param_code: AQS parameter code (default '42602' = NO2).
            hard_threshold: fixed exceedance value, overrides the regulatory default.
            percentile_threshold: top N% cutoff, 0-100.
            k: max number of monitors to check (default 10).

        Returns:
            JSON string: per-monitor exceedance dates + satellite series, and
            timeseries artifact refs (one per monitor with an exceedance day).
        """
        try:
            ds = await open_handle(handle, mcp_tools)
            da = _aggregation_service.to_dataarray(ds)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        region = await _resolver.aresolve_location(location)
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        minx, miny, maxx, maxy = region["bounds"]
        bbox = [miny, maxy, minx, maxx]  # [south, north, west, east]

        start_time, end_time = _time_range(da)
        if not start_time:
            return json.dumps({
                "error": f"Handle '{handle}' has no time dimension to check for exceedance days."
            })
        bdate_obj = datetime.date.fromisoformat(start_time[:10])
        edate_obj = datetime.date.fromisoformat(end_time[:10])
        bdate_str = bdate_obj.strftime("%Y%m%d")
        edate_str = edate_obj.strftime("%Y%m%d")

        monitors = await aqs._fetch_active_monitors(bbox, param_code, bdate_str, edate_str, k)
        if not monitors:
            return json.dumps({"error": f"No active {param_code} monitors found over '{location}'."})

        reg = aqs._REGULATORY_THRESHOLDS.get(param_code)
        if reg is None and hard_threshold is None and percentile_threshold is None:
            return json.dumps({
                "error": (
                    f"No regulatory threshold known for param_code '{param_code}'. "
                    "Provide hard_threshold or percentile_threshold explicitly."
                )
            })
        pollutant_standard = reg[0] if reg else None
        measurement_field = reg[1] if reg else "first_max_value"
        regulatory_limit = reg[2] if reg else None
        effective_hard = hard_threshold if hard_threshold is not None else regulatory_limit

        try:
            records, _, _ = await aqs._fetch_summary(
                "dailyData", param_code, bdate_obj, edate_obj, bdate_str, edate_str,
                None, None, None, None, miny, maxy, minx, maxx, None, None, pollutant_standard,
            )
        except RuntimeError as e:
            return json.dumps({"error": f"No ground data found over '{location}': {e}"})

        col_info = _mask_col_info(da)
        satellite_units = da.attrs.get("units", "")
        variable_name = da.name or ""

        def _extract_exceedance_monitors():
            # CPU-bound per-monitor mask/extraction loop (T16), run off the
            # event loop via asyncio.to_thread below.
            monitor_results = []
            artifact_refs = []

            for monitor in monitors:
                station_id = _station_id(monitor)
                site_records = [r for r in records if aqs._site_id(r) == station_id]
                if not site_records:
                    continue

                exceeded_dates = _exceedance_days(
                    site_records, measurement_field, effective_hard, percentile_threshold
                )
                if not exceeded_dates:
                    continue

                lat, lon = float(monitor["latitude"]), float(monitor["longitude"])
                times, values, coverage = _extract_monitor_series(da, lat, lon, col_info)

                station_name = monitor.get("local_site_name") or monitor.get("address") or station_id
                ts_payload = {
                    "type": "timeseries",
                    "title": f"{variable_name} with {station_id} exceedance days",
                    "times": times,
                    "values": values,
                    "exceedance_dates": sorted(exceeded_dates),
                    "satellite_units": satellite_units,
                    "chart_id": f"ts_{uuid.uuid4().hex[:12]}",
                    "metadata": {
                        "source_handles": [handle],
                        "series": [{"label": f"{variable_name} (satellite)", "source_kind": "satellite"}],
                    },
                }
                ref = build_artifact_reference(ts_payload)
                if ref is not None:
                    artifact_refs.append(ref.model_dump(exclude_none=True))

                monitor_results.append({
                    "station_id": station_id,
                    "station_name": station_name,
                    "exceedance_dates": sorted(exceeded_dates),
                    "coverage": coverage,
                    "source_handles": [handle],
                    "chart_id": ts_payload["chart_id"],
                })

            return monitor_results, artifact_refs

        emit_status("Checking exceedance days against the satellite series...", stage=STAGE_RENDER)
        monitor_results, artifact_refs = await asyncio.to_thread(_extract_exceedance_monitors)

        if not monitor_results:
            return json.dumps({"error": f"No exceedance days found for monitors in '{location}'."})

        return json.dumps({
            "location": region.get("name", location),
            "variable": variable_name,
            "satellite_units": satellite_units,
            "param_code": param_code,
            "pollutant_standard": pollutant_standard,
            "measurement_field": measurement_field,
            "hard_threshold": effective_hard,
            "percentile_threshold": percentile_threshold,
            "monitors_matched": len(monitor_results),
            "monitors": monitor_results,
            "_artifact_refs": artifact_refs,
        })

    return exceedance_overlay
