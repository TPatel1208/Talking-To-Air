import asyncio
import json
import numpy as np
from langchain.tools import tool
from langchain_core.tools import BaseTool
from typing import Annotated, Optional
from pydantic import Field

from tta_backend.services import admission
from tta_backend.config.workflow_stages import STAGE_RENDER
from tta_backend.datasets.mask_info import col_info_for_variable
from tta_backend.earthdata_mcp.results import MCPToolError
from tta_backend.services.open_handle import OpenHandleError, open_handle
from tta_backend.tools.satellite_tools.plot_tools import _normalize_longitudes
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord
from tta_backend.utils.plotting import _normalize_to_2d, apply_mask_region_type, mask_data_by_geometry, RegionResolver
from tta_backend.utils.streaming import emit_status
from tta_backend.preprocessing.aggregation_service import (
    VARIABLE_RESOLUTION_ATTR,
    AggregationService,
    VariableChoiceRequired,
    area_weighted_mean,
)
from tta_backend.preprocessing.variable_choice_builder import emit_variable_choice_payload

_resolver = RegionResolver()
_aggregation_service = AggregationService()

VALID_STATS = {"mean", "median", "max", "min", "std"}


def _build_dim_selector(dimension: str | None, dimension_value: float | None) -> dict | None:
    if dimension is None or dimension_value is None:
        return None
    return {dimension: dimension_value}


def make_compute_statistic_tool(mcp_tools: dict[str, BaseTool]):
    @tool
    async def compute_statistic_tool(
        handle: Annotated[str, Field(description="An obs_/cube_ handle from a retrieval or transform tool.")],
        location: str,
        stats: list[str] = ["mean", "median", "max", "min"],
        variable: Optional[str] = None,
        dimension: Optional[str] = None,
        dimension_value: Optional[float] = None,
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
            variable  : Science variable to use, for a multi-variable file with no
                        variable chosen at retrieval time.
            dimension       : Name of an extra non-spatial, non-time dimension to
                               select a single value from (e.g. a vertical level).
            dimension_value : Coordinate value to select from ``dimension`` (nearest match).

        Returns:
            JSON string with each requested statistic and its value.
        """
        try:
            ds = await open_handle(handle, mcp_tools)
            # Normalize longitude on the whole opened Dataset, before
            # extracting the science DataArray -- the plot_singular
            # convention. A 0..360 grid otherwise rasterizes a western-
            # hemisphere region entirely outside the data ("No valid data
            # found" for data that's fully present), and normalizing only
            # the extracted array would leave ds's sibling QA-flag variable
            # on 0..360, so QA alignment would hit an empty intersection.
            ds_lon_coord = find_lon_coord(ds)
            if ds_lon_coord:
                ds = _normalize_longitudes(ds, ds_lon_coord)
            da = _aggregation_service.to_dataarray(ds, handle=handle, variable=variable)
        except VariableChoiceRequired as e:
            emit_variable_choice_payload(e.resolution, ds)
            emit_status("Waiting for a variable choice.", stage=STAGE_RENDER)
            return json.dumps({"error": e.mcp_error.to_dict()})
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        # aresolve_location, never the sync resolve_location: the sync path
        # does a blocking HTTP geocode (plus a rate-limit sleep) directly on
        # the event loop, freezing every concurrent stream for its duration.
        # T60 D14: a composite that cannot be built raises the taxonomy's
        # error naming the offending token -- a ``None`` return could never
        # carry which token failed. Same shape as the open_handle catch.
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        emit_status("Computing statistics...", stage=STAGE_RENDER)

        def _mask_aggregate_stats():
            # CPU-bound mask -> aggregate -> stats chain (T16), run off the
            # event loop via admission.run_heavy below, which also bounds how
            # many such reductions may hold memory at once.
            masked = mask_data_by_geometry(da, region['geometry'])
            # T42: an empty mask that self-healed to boundary cells downgrades
            # the disclosed region_type -- read it off the masked array so the
            # result names what was actually computed.
            apply_mask_region_type(masked, region)
            col_info = col_info_for_variable(masked, ds)
            dim_selector = _build_dim_selector(dimension, dimension_value)

            def _reduced_field(stat):
                """Mask, reduce over time with ``stat`` per cell, and squeeze
                to a 2-D (lat, lon) field."""
                aggregation = _aggregation_service.aggregate(
                    masked,
                    variable=masked.name,
                    stat=stat,
                    col_info=col_info,
                    source_ds=ds,
                )
                reduced = next(iter(aggregation.ds.data_vars.values()))
                return aggregation, _normalize_to_2d(reduced, dim_selector=dim_selector)

            invalid_stats = [s for s in stats if s not in VALID_STATS]
            if invalid_stats:
                return "error", f"Unknown stats: {invalid_stats}. Valid: {sorted(VALID_STATS)}"

            try:
                aggregation, mean_field = _reduced_field("mean")
                # T48: masking stripped the resolver's stash off ``masked``
                # before aggregate saw it -- carry it from the pre-mask ``da``
                # into meta so the chosen-variable disclosure isn't lost on the
                # stat path (one resolver, disclosed identically everywhere).
                if da.attrs.get(VARIABLE_RESOLUTION_ATTR):
                    aggregation.meta["variable_resolution"] = da.attrs[VARIABLE_RESOLUTION_ATTR]

                values = mean_field.values
                valid = values[np.isfinite(values)]
                if len(valid) == 0:
                    return "error", f"No valid data found for '{location}'. The region may be outside the data bbox."

                result = {
                    "location": location,
                    "variable": mean_field.name or "",
                    "units":    mean_field.attrs.get("units", ""),
                    "n_pixels": int(len(valid)),
                    "aggregation_meta": aggregation.meta,
                    "source_handles": [handle],
                    # T42 region fidelity: the region we actually masked.
                    "region_name": region.get("display_name") or region.get("name"),
                    "display_name": region.get("display_name") or region.get("name"),
                    "region_type": region.get("region_type"),
                    "region_origin": region.get("region_origin"),  # T60 D10a
                }
                # Each statistic is computed on the basis that makes it true
                # to its name, and that basis is disclosed:
                #   - max/min compose exactly across time (the max of per-cell
                #     maxima IS the max over every observation), so they reduce
                #     time with the same stat — the old max-of-the-time-MEAN-
                #     field systematically understated multi-day extremes.
                #   - mean is the cos(latitude) area-weighted mean of the
                #     temporal-mean field (area_weighted_mean).
                #   - median/std don't compose across time; they are computed
                #     over the temporal-mean field and say so.
                stat_basis: dict[str, str] = {}
                for s in stats:
                    if s in ("max", "min"):
                        _, extremum_field = _reduced_field(s)
                        field_values = extremum_field.values
                        result[s] = _aggregation_service.compute_values_stat(
                            field_values[np.isfinite(field_values)], s,
                        )
                        stat_basis[s] = f"{s} over every valid observation (all timesteps)"
                    elif s == "mean":
                        result[s] = area_weighted_mean(mean_field)
                        stat_basis[s] = "cos(latitude) area-weighted spatial mean of the temporal-mean field"
                    else:
                        result[s] = _aggregation_service.compute_values_stat(valid, s)
                        stat_basis[s] = f"{s} of the temporal-mean field (per-cell mean over time first)"
                result["stat_basis"] = stat_basis
            except ValueError as e:
                return "error", str(e)
            except MCPToolError as e:
                return "error", e.to_dict()

            return None, result

        # The geometry mask itself can refuse (unsupported/unidentifiable
        # grid, T24/T44) — that classified answer must reach the agent as a
        # structured error, never escape the tool off-taxonomy (QA
        # 2026-07-17: GPM stats surfaced as a generic internal error).
        try:
            status, result = await admission.run_heavy(_mask_aggregate_stats)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if status == "error":
            return json.dumps({"error": result})
        return json.dumps(result)

    return compute_statistic_tool


def make_find_daily_peak(mcp_tools: dict[str, BaseTool]):
    @tool
    async def find_daily_peak(
        handle: Annotated[str, Field(description="An obs_/cube_ handle from a retrieval or transform tool.")],
        location: str,
        variable: Optional[str] = None,
        dimension: Optional[str] = None,
        dimension_value: Optional[float] = None,
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
            variable  : Science variable to use, for a multi-variable file with no
                        variable chosen at retrieval time.
            dimension       : Name of an extra non-spatial, non-time dimension to
                               select a single value from (e.g. a vertical level).
            dimension_value : Coordinate value to select from ``dimension`` (nearest match).

        Returns:
            JSON string with peak value, lat, lon, and metadata.
        """
        try:
            ds = await open_handle(handle, mcp_tools)
            # See compute_statistic_tool: normalize the whole Dataset's
            # longitude to -180..180 before extraction, so geometry masking
            # and QA-flag alignment both see one coordinate convention.
            ds_lon_coord = find_lon_coord(ds)
            if ds_lon_coord:
                ds = _normalize_longitudes(ds, ds_lon_coord)
            da = _aggregation_service.to_dataarray(ds, handle=handle, variable=variable)
        except VariableChoiceRequired as e:
            emit_variable_choice_payload(e.resolution, ds)
            emit_status("Waiting for a variable choice.", stage=STAGE_RENDER)
            return json.dumps({"error": e.mcp_error.to_dict()})
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        # aresolve_location, never the sync resolve_location — see
        # compute_statistic_tool: the sync path blocks the event loop on a
        # geocoding HTTP call plus a rate-limit sleep.
        # T60 D14: a composite that cannot be built raises the taxonomy's
        # error naming the offending token -- a ``None`` return could never
        # carry which token failed. Same shape as the open_handle catch.
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        emit_status("Finding peak value...", stage=STAGE_RENDER)

        def _mask_aggregate_peak():
            # CPU-bound mask -> aggregate -> peak search chain (T16), run
            # off the event loop via admission.run_heavy below, which also bounds
                # how many such reductions may hold memory at once.
            masked = mask_data_by_geometry(da, region['geometry'])
            apply_mask_region_type(masked, region)  # T42: disclose boundary_cells self-heal

            col_info = col_info_for_variable(masked, ds)
            try:
                # stat="max": the peak of the per-cell MAX field is the true
                # peak over every observation in the period (max composes
                # exactly across time). The old stat="mean" reduced time
                # first, so "worst pollution point" was the peak of the
                # period-AVERAGE map — systematically understating multi-day
                # extremes and possibly misplacing them.
                aggregation = _aggregation_service.aggregate(
                    masked,
                    variable=masked.name,
                    stat="max",
                    col_info=col_info,
                    source_ds=ds,
                )
                reduced = next(iter(aggregation.ds.data_vars.values()))
                reduced = _normalize_to_2d(reduced, dim_selector=_build_dim_selector(dimension, dimension_value))
            except ValueError as e:
                return "error", str(e)
            except MCPToolError as e:
                return "error", e.to_dict()

            # T48: carry the resolver's chosen-variable disclosure across the
            # attr-stripping mask (see compute_statistic_tool).
            if da.attrs.get(VARIABLE_RESOLUTION_ATTR):
                aggregation.meta["variable_resolution"] = da.attrs[VARIABLE_RESOLUTION_ATTR]

            # Resolve dim names via the canonical CF-metadata identifier
            # (T24), so an axis named 'row'/'y' is found by its metadata, not
            # a hardcoded name list. On a rectilinear grid the lat/lon coord
            # names are also dimension names.
            lat_name = find_lat_coord(reduced)
            lon_name = find_lon_coord(reduced)
            lat_dim = lat_name if lat_name in reduced.dims else None
            lon_dim = lon_name if lon_name in reduced.dims else None

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
                # T42 region fidelity: the region we actually masked.
                "region_name": region.get("display_name") or region.get("name"),
                "display_name": region.get("display_name") or region.get("name"),
                "region_type": region.get("region_type"),
                "region_origin": region.get("region_origin"),  # T60 D10a
            }

        # See compute_statistic_tool: a mask refusal is a classified answer,
        # not an exception to escape off-taxonomy.
        try:
            status, result = await admission.run_heavy(_mask_aggregate_peak)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if status == "error":
            return json.dumps({"error": result})
        return json.dumps(result)

    return find_daily_peak
