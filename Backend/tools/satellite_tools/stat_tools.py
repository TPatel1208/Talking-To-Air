import asyncio
import json
import os
import numpy as np
from langchain.tools import tool
from langchain_core.tools import BaseTool
from typing import Annotated
from pydantic import Field

from config.workflow_stages import STAGE_RENDER
from datasets.mask_info import override_for
from earthdata_mcp.results import MCPToolError
from services.open_handle import OpenHandleError, open_handle
from utils.plotting import _normalize_to_2d, mask_data_by_geometry, RegionResolver
from utils.streaming import emit_status
from preprocessing.aggregation_service import AggregationService

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
_resolver = RegionResolver()
_aggregation_service = AggregationService()

VALID_STATS = {"mean", "median", "max", "min", "std"}


def _mask_col_info(da) -> dict:
    short_name = da.attrs.get("short_name") or da.name or ""
    return override_for(str(short_name).upper())


def make_compute_statistic_tool(mcp_tools: dict[str, BaseTool]):
    @tool
    async def compute_statistic_tool(
        handle: Annotated[str, Field(description="An obs_/cube_ handle from a retrieval or transform tool.")],
        location: str,
        stats: list[str] = ["mean", "median", "max", "min"],
    ) -> str:
        """
        Compute basic statistics (mean, median, max, min, std) over a region
        for a single retrieved dataset.

        Use when the user asks questions like:
          - 'What is the average NO2 in Texas?'
          - 'What was the max pollution in California on April 8?'
          - 'Give me summary statistics for NO2 over New York'

        Args:
            handle:   obs_/cube_ handle from a retrieval or transform tool
            location: place name to spatially mask before computing e.g. 'Texas'
            stats:    list of statistics to compute.
                      Any of: 'mean', 'median', 'max', 'min', 'std'

        Returns:
            JSON string with each requested statistic and its value.
        """
        try:
            ds = await open_handle(handle, mcp_tools)
            da = _aggregation_service.to_dataarray(ds)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        region = _resolver.resolve_location(location)
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        emit_status("Computing statistics...", stage=STAGE_RENDER)

        def _mask_aggregate_stats():
            # CPU-bound mask -> aggregate -> stats chain (T16), run off the
            # event loop via asyncio.to_thread below.
            masked = mask_data_by_geometry(da, region['geometry'])

            col_info = _mask_col_info(masked)
            try:
                aggregation = _aggregation_service.aggregate(
                    masked,
                    variable=masked.name,
                    stat="mean",
                    col_info=col_info,
                )
            except ValueError as e:
                return "error", str(e)
            reduced = next(iter(aggregation.ds.data_vars.values()))
            reduced = _normalize_to_2d(reduced)

            values = reduced.values
            valid = values[np.isfinite(values)]
            if len(valid) == 0:
                return "error", f"No valid data found for '{location}'. The region may be outside the data bbox."

            invalid_stats = [s for s in stats if s not in VALID_STATS]
            if invalid_stats:
                return "error", f"Unknown stats: {invalid_stats}. Valid: {sorted(VALID_STATS)}"

            result = {
                "location": location,
                "variable": reduced.name or "",
                "units":    reduced.attrs.get("units", ""),
                "n_pixels": int(len(valid)),
                "aggregation_meta": aggregation.meta,
                "source_handles": [handle],
            }
            for s in stats:
                result[s] = _aggregation_service.compute_values_stat(valid, s)

            return None, result

        status, result = await asyncio.to_thread(_mask_aggregate_stats)
        if status == "error":
            return json.dumps({"error": result})
        return json.dumps(result)

    return compute_statistic_tool


def make_find_daily_peak(mcp_tools: dict[str, BaseTool]):
    @tool
    async def find_daily_peak(
        handle: Annotated[str, Field(description="An obs_/cube_ handle from a retrieval or transform tool.")],
        location: str,
    ) -> str:
        """
        Find the peak (maximum) value and its lat/lon location within a region.

        Use when the user asks questions like:
          - 'Where was NO2 highest in Texas on April 8?'
          - 'What was the worst pollution point in California?'
          - 'Find the peak NO2 location in New York'

        Args:
            handle:   obs_/cube_ handle from a retrieval or transform tool
            location: place name to spatially mask before searching e.g. 'Texas'

        Returns:
            JSON string with peak value, lat, lon, and metadata.
        """
        try:
            ds = await open_handle(handle, mcp_tools)
            da = _aggregation_service.to_dataarray(ds)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        region = _resolver.resolve_location(location)
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        emit_status("Finding peak value...", stage=STAGE_RENDER)

        def _mask_aggregate_peak():
            # CPU-bound mask -> aggregate -> peak search chain (T16), run
            # off the event loop via asyncio.to_thread below.
            masked = mask_data_by_geometry(da, region['geometry'])

            col_info = _mask_col_info(masked)
            try:
                aggregation = _aggregation_service.aggregate(
                    masked,
                    variable=masked.name,
                    stat="mean",
                    col_info=col_info,
                )
            except ValueError as e:
                return "error", str(e)
            reduced = next(iter(aggregation.ds.data_vars.values()))
            reduced = _normalize_to_2d(reduced)

            # Resolve dim names and positions early
            lat_dim = next((d for d in reduced.dims if d.lower() in ['lat', 'latitude']), None)
            lon_dim = next((d for d in reduced.dims if d.lower() in ['lon', 'longitude']), None)

            if lat_dim is None or lon_dim is None:
                return "error", f"Could not find lat/lon dimensions. Available dims: {list(reduced.dims)}"

            lat_array = reduced[lat_dim].values
            lon_array = reduced[lon_dim].values

            # Filter
            values     = reduced.values
            valid_mask = np.isfinite(values)

            if not np.any(valid_mask):
                return "error", f"No valid data found for '{location}'. The region may be outside the data bbox."

            # Find peak
            masked_values = np.where(valid_mask, values, np.nan)
            flat_idx      = np.nanargmax(masked_values)
            dim0_idx, dim1_idx = np.unravel_index(flat_idx, masked_values.shape)

            # Determine which axis corresponds to lat and lon
            dims    = list(reduced.dims)
            lat_pos = dims.index(lat_dim)
            lon_pos = dims.index(lon_dim)
            indices = [dim0_idx, dim1_idx]
            lat_idx = indices[lat_pos]
            lon_idx = indices[lon_pos]

            try:
                peak_lat = float(lat_array[lat_idx] if lat_array.ndim == 1 else lat_array[lat_idx, lon_idx])
                peak_lon = float(lon_array[lon_idx] if lon_array.ndim == 1 else lon_array[lat_idx, lon_idx])
            except (IndexError, TypeError) as e:
                return "error", f"Failed to extract peak coordinates: {e}"

            peak_val = float(masked_values[dim0_idx, dim1_idx])

            return None, {
                "location":   location,
                "variable":   reduced.name or "",
                "units":      reduced.attrs.get("units", ""),
                "peak_value": peak_val,
                "peak_lat":   peak_lat,
                "peak_lon":   peak_lon,
                "aggregation_meta": aggregation.meta,
                "source_handles": [handle],
            }

        status, result = await asyncio.to_thread(_mask_aggregate_peak)
        if status == "error":
            return json.dumps({"error": result})
        return json.dumps(result)

    return find_daily_peak
