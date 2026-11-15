"""
validation_tools.py
--------------------
Satellite<->ground validation tools.

Co-locates a retrieved satellite cube with EPA AQS ground monitors: one cube
retrieval over the AOI, satellite values extracted at every monitor location
by nearest-cell selection (TTA-side, never per-monitor AppEEARS jobs), paired
with the AQS daily series in time, and summarized with correlation and
coverage statistics. Renders as overlay `timeseries` artifacts, one per
monitor, per TimeseriesArtifactMetadata.
"""
from __future__ import annotations

import datetime
import json
import uuid
from typing import TYPE_CHECKING, Any, Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    import xarray as xr

from tta_backend.services import admission
from tta_backend.config.workflow_stages import STAGE_RENDER
from tta_backend.datasets.mask_info import col_info_for_variable
from tta_backend.earthdata_mcp.results import MCPToolError
from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.analysis.checks import time_range
from tta_backend.domain.analysis.pairing import correlation_stats, extract_monitor_series, pair_daily
from tta_backend.domain.analysis.thresholds import exceedance_days
from tta_backend.services.artifact_registry import build_artifact_reference
from tta_backend.services import variable_choice_registry
from tta_backend.services.open_handle import OpenHandleError, open_handle
from tta_backend.tools.ground_sensor_tools import epa_aqs_tools as aqs
from tta_backend.utils.plotting import RegionResolver
from tta_backend.utils.streaming import emit_status, ground_validation_allowed

_aggregation_service = AggregationService()
_resolver = RegionResolver()


def _out_of_scope_error() -> str | None:
    """The refusal these satellite<->ground tools return when the task in
    progress is not a comparison. Under the supervisor's dispatch,
    ground data is the supervisor's to request as a ground request, where
    the dispatch log records it; checked before any data is opened."""
    if ground_validation_allowed():
        return None
    return json.dumps({"error": (
        "Ground validation is not part of this task: only a satellite request with "
        "intent compare may query ground monitors. Ground readings are the "
        "supervisor's to request; answer from the satellite data."
    )})


def _time_range(da: xr.DataArray) -> tuple[str, str]:
    start, end = time_range(da)
    return start or "", end or ""


def _station_id(monitor: dict[str, Any]) -> str:
    return "-".join(str(monitor.get(k, "??")) for k in ("state_code", "county_code", "site_number"))


def make_validate_against_ground(mcp_tools: dict[str, BaseTool]) -> BaseTool:
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
        out_of_scope = _out_of_scope_error()
        if out_of_scope is not None:
            return out_of_scope
        try:
            ds = await open_handle(handle, mcp_tools)
            da = _aggregation_service.to_dataarray(ds, recorded_variable=variable_choice_registry.get(handle))
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        # A composite region that can't be built raises an error naming the
        # bad token (a None return couldn't say which one).
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
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

        col_info = col_info_for_variable(da, ds)
        satellite_units = da.attrs.get("units", "")
        variable_name = da.name or ""

        def _extract_and_pair_monitors() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
            # CPU-bound per-monitor mask/extraction/pairing loop, run
            # off the event loop via admission.run_heavy below, which also bounds
                # how many such reductions may hold memory at once.
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
                times, values, coverage, masking = extract_monitor_series(
                    da, lat, lon, col_info, source_ds=ds,
                )
                paired = pair_daily(times, values, ground_daily)
                if not paired:
                    continue

                stats = correlation_stats(paired, total_ground_days=len(ground_daily))
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
                    "masking": masking,
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
        monitor_results, artifact_refs, pooled_paired, monitor_ids = await admission.run_heavy(
            _extract_and_pair_monitors
        )

        if not monitor_results:
            return json.dumps({
                "error": (
                    f"No monitors in '{location}' had both satellite and ground data "
                    "over the requested period."
                )
            })

        pooled_stats = correlation_stats(pooled_paired)

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


def make_exceedance_overlay(mcp_tools: dict[str, BaseTool]) -> BaseTool:
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
        out_of_scope = _out_of_scope_error()
        if out_of_scope is not None:
            return out_of_scope
        try:
            ds = await open_handle(handle, mcp_tools)
            da = _aggregation_service.to_dataarray(ds, recorded_variable=variable_choice_registry.get(handle))
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        # A composite region that can't be built raises an error naming the
        # bad token (a None return couldn't say which one).
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
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

        col_info = col_info_for_variable(da, ds)
        satellite_units = da.attrs.get("units", "")
        variable_name = da.name or ""

        def _extract_exceedance_monitors() -> tuple[list[Any], list[Any]]:
            # CPU-bound per-monitor mask/extraction loop, run off the
            # event loop via admission.run_heavy below, which also bounds how
            # many such reductions may hold memory at once.
            monitor_results = []
            artifact_refs = []

            for monitor in monitors:
                station_id = _station_id(monitor)
                site_records = [r for r in records if aqs._site_id(r) == station_id]
                if not site_records:
                    continue

                exceeded_dates = exceedance_days(
                    site_records, measurement_field, effective_hard, percentile_threshold
                )
                if not exceeded_dates:
                    continue

                lat, lon = float(monitor["latitude"]), float(monitor["longitude"])
                times, values, coverage, masking = extract_monitor_series(
                    da, lat, lon, col_info, source_ds=ds,
                )

                station_name = monitor.get("local_site_name") or monitor.get("address") or station_id
                ts_payload = {
                    "type": "timeseries",
                    "title": f"{variable_name} with {station_id} exceedance days",
                    "times": times,
                    "values": values,
                    "exceedance_dates": sorted(exceeded_dates),
                    "coverage": coverage,
                    "masking": masking,
                    "satellite_units": satellite_units,
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
                    "exceedance_dates": sorted(exceeded_dates),
                    "coverage": coverage,
                    "source_handles": [handle],
                    "chart_id": ts_payload["chart_id"],
                })

            return monitor_results, artifact_refs

        emit_status("Checking exceedance days against the satellite series...", stage=STAGE_RENDER)
        monitor_results, artifact_refs = await admission.run_heavy(_extract_exceedance_monitors)

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
