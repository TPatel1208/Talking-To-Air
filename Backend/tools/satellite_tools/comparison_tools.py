"""
comparison_tools.py
--------------------
Region/period comparison tools (PRD T08).

Compares two already-retrieved handles of the same variable — two AOIs over
one period (mode="region") or one AOI over two periods (mode="period") — and
renders the result as a `comparison` artifact (T06). Region mode never
differences (the two domains aren't spatially comparable cell-by-cell): it
renders side-by-side panels on a shared color scale plus per-region stats.
Period mode grid-aligns both retrievals via the MCP's `align` transform,
differences B minus A cell-by-cell (cells missing on either side excluded
from both the map and the stats), and reports a diverging difference map
plus anomaly statistics.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import numpy as np
from langchain.tools import tool
from langchain_core.tools import BaseTool

from config.workflow_stages import STAGE_RENDER
from datasets.mask_info import override_for
from earthdata_mcp.results import MCPToolError, parse_tool_result
from preprocessing.aggregation_service import AggregationService
from services.open_handle import OpenHandleError, open_handle
from tools.satellite_tools.plot_tools import _da_to_heatmap_payload, _percentile_bounds, _save_chart
from utils.geo_utils import find_lat_coord, find_lon_coord
from utils.plotting import _normalize_to_2d
from utils.streaming import emit_status

_aggregation_service = AggregationService()


def _difference(da_a, da_b):
    """period B minus period A, cell-by-cell.

    xarray subtraction already propagates NaN — a cell missing (fill-masked)
    on either side becomes NaN in the difference, i.e. excluded from both
    the rendered map and any stat computed over it, never a fabricated 0.
    """
    return da_b - da_a


def _split_aligned(aligned):
    """Split the MCP `align` transform's output into its two source arrays.

    ``align(source_handles=[a, b])`` grid-aligns >=2 gridded inputs into one
    cube_ handle; TTA's convention for a two-source align (the MCP's own
    layout for the merged cube isn't otherwise specified) is a leading
    ``source`` dimension of length 2, in the same order the handles were
    passed — ``isel(source=0)`` is A, ``isel(source=1)`` is B.
    """
    if "source" not in aligned.dims:
        raise ValueError(f"Aligned result has no 'source' dimension: dims={list(aligned.dims)}")
    n_sources = aligned.sizes["source"]
    if n_sources != 2:
        raise ValueError(f"Expected an aligned result with 2 sources, found {n_sources}.")
    return aligned.isel(source=0), aligned.isel(source=1)


def _anomaly_stats(da_a, da_b, diff, threshold: float | None) -> dict:
    """Mean difference, percent change (relative to period A's mean), and
    optionally the area exceeding a threshold change magnitude.

    All computed only over cells valid on both sides (``diff`` already
    carries NaN wherever either side was missing — see ``_difference``).
    """
    diff_vals = np.asarray(diff.values, dtype=float)
    valid_diff = diff_vals[np.isfinite(diff_vals)]

    a_vals = np.asarray(da_a.values, dtype=float)
    diff_mask = np.isfinite(diff_vals)
    a_paired = a_vals[diff_mask]
    a_paired_valid = a_paired[np.isfinite(a_paired)]

    mean_difference = float(np.mean(valid_diff)) if valid_diff.size else None
    mean_a = float(np.mean(a_paired_valid)) if a_paired_valid.size else None
    percent_change = (
        (mean_difference / mean_a) * 100.0
        if mean_difference is not None and mean_a not in (None, 0.0)
        else None
    )

    stats = {
        "n_cells": int(valid_diff.size),
        "mean_difference": mean_difference,
        "percent_change": percent_change,
    }

    if threshold is not None:
        exceeding = valid_diff[np.abs(valid_diff) >= threshold]
        stats["area_exceeding_threshold"] = {
            "threshold": threshold,
            "n_cells": int(exceeding.size),
            "fraction": (exceeding.size / valid_diff.size) if valid_diff.size else 0.0,
        }

    return stats


def _region_stats(da) -> dict | None:
    """Basic descriptive stats over da's valid cells, or None if none are valid."""
    values = np.asarray(da.values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "max": float(np.max(valid)),
        "min": float(np.min(valid)),
        "n_pixels": int(valid.size),
    }


def _empty_overlap_error(da, label: str) -> str | None:
    """Reject a side with no valid data at all — no overlap with its requested window."""
    values = np.asarray(da.values, dtype=float)
    if not np.isfinite(values).any():
        return f"No valid data found for {label} — its retrieval has no overlap with the requested window."
    return None


def _time_range(da) -> tuple[str, str] | tuple[None, None]:
    if "time" not in da.coords:
        return None, None
    times = sorted(str(t) for t in np.atleast_1d(da["time"].values))
    if not times:
        return None, None
    return times[0], times[-1]


def _disjoint_periods_error(da_a, da_b) -> str | None:
    """Region mode compares two AOIs over what should be the same period —
    reject if their time windows don't even overlap (a plain-language guard,
    not a proxy for period-mode's own aligned differencing)."""
    start_a, end_a = _time_range(da_a)
    start_b, end_b = _time_range(da_b)
    if start_a is None or start_b is None:
        return None
    if end_a < start_b or end_b < start_a:
        return (
            f"The two retrievals cover disjoint time periods ({start_a[:10]}..{end_a[:10]} vs "
            f"{start_b[:10]}..{end_b[:10]}) — region mode compares two areas over the same "
            "period; use mode=\"period\" to compare two periods instead."
        )
    return None


def _variable_mismatch_error(da_a, da_b) -> str | None:
    """Return an error message if da_a/da_b are different variables, else None."""
    name_a = (da_a.name or "").strip()
    name_b = (da_b.name or "").strip()
    if name_a and name_b and name_a != name_b:
        return (
            f"Cannot compare different variables: '{name_a}' (A) vs '{name_b}' (B). "
            "compare only supports comparing the same variable across two regions or periods."
        )
    return None


def _mask_col_info(da) -> dict:
    short_name = da.attrs.get("short_name") or da.name or ""
    return override_for(str(short_name).upper())


def _prepare_2d(da, variable_name: str):
    """Apply quality masking and collapse to a single 2-D (lat, lon) snapshot
    (time-mean, matching plot_singular's default) so every side of a
    comparison renders as one map."""
    col_info = _mask_col_info(da)
    aggregation = _aggregation_service.aggregate(da, variable=variable_name, stat="mean", col_info=col_info)
    reduced = next(iter(aggregation.ds.data_vars.values()))
    return _normalize_to_2d(reduced)


def _bbox_from_da(da) -> list[float]:
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    lats = np.asarray(da[lat_coord].values, dtype=float)
    lons = np.asarray(da[lon_coord].values, dtype=float)
    return [float(np.nanmin(lons)), float(np.nanmin(lats)), float(np.nanmax(lons)), float(np.nanmax(lats))]


def _shared_bounds(da_a, da_b) -> tuple[float, float]:
    combined = np.concatenate([
        np.asarray(da_a.values, dtype=float).ravel(),
        np.asarray(da_b.values, dtype=float).ravel(),
    ])
    return _percentile_bounds(combined)


def _diverging_bounds(diff_da) -> tuple[float, float]:
    """Symmetric, zero-centered scale sized to the diff's 98th-percentile magnitude."""
    vals = np.asarray(diff_da.values, dtype=float)
    valid = vals[np.isfinite(vals)]
    if valid.size == 0:
        return -1.0, 1.0
    max_abs = float(np.percentile(np.abs(valid), 98))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    return -max_abs, max_abs


def _region_panel(da, handle: str, title: str, variable_name: str, units: str, vmin: float, vmax: float) -> dict:
    panel = _da_to_heatmap_payload(
        da, title, variable_name, units, render_overlay=True, value_range=(vmin, vmax),
    )
    panel["bounds"] = _bbox_from_da(da)
    panel["metadata"] = {"source_handles": [handle]}
    return panel


def make_compare(mcp_tools: dict[str, BaseTool]):
    @tool
    async def compare(
        handle_a: str,
        handle_b: str,
        mode: str,
        label_a: str = "A",
        label_b: str = "B",
        threshold: Optional[float] = None,
        variable: Optional[str] = None,
    ) -> str:
        """
        Compare two retrievals of the same variable — either two regions
        over one period (mode="region") or one region across two periods
        (mode="period") — and render the result as a comparison artifact.

        region mode: side-by-side panels on a shared color scale plus
        per-region summary statistics. Never differences — the two domains
        aren't spatially comparable cell-by-cell.
        period mode: grid-aligns both retrievals via the MCP's `align`
        transform, then differences period B minus period A cell-by-cell
        (cells missing on either side excluded from both the map and the
        stats), rendering a diverging, zero-centered difference map plus
        mean difference / percent change / area exceeding a threshold change.

        Both retrievals must be the same variable, and (region mode) must
        cover overlapping time windows — mismatches are rejected with a
        plain explanation rather than silently attempted.

        Args:
            handle_a: obs_/cube_ handle for the first region or period.
            handle_b: obs_/cube_ handle for the second region or period.
            mode: "region" (two AOIs, same period) or "period" (one AOI, two periods).
            label_a: panel/legend label for handle_a, e.g. a region or period name.
            label_b: panel/legend label for handle_b.
            threshold: period mode only — absolute change magnitude defining
                a "significant" change, for the area-exceeding-threshold stat.
            variable: Science variable to use for both handles, for a
                multi-variable file with no variable chosen at retrieval time.

        Returns:
            JSON string — comparison chart payload (frontend-renderable) with
            an embedded artifact ref and summary statistics.
        """
        if mode not in ("region", "period"):
            return json.dumps({"error": f"Unknown mode '{mode}'. Use 'region' or 'period'."})

        try:
            ds_a = await open_handle(handle_a, mcp_tools)
            da_a = _aggregation_service.to_dataarray(ds_a, handle=handle_a, variable=variable)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle_a}' (A): {e}"})
        try:
            ds_b = await open_handle(handle_b, mcp_tools)
            da_b = _aggregation_service.to_dataarray(ds_b, handle=handle_b, variable=variable)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle_b}' (B): {e}"})

        mismatch = _variable_mismatch_error(da_a, da_b)
        if mismatch:
            return json.dumps({"error": mismatch})

        empty_a = _empty_overlap_error(da_a, f"{label_a} ({handle_a})")
        if empty_a:
            return json.dumps({"error": empty_a})
        empty_b = _empty_overlap_error(da_b, f"{label_b} ({handle_b})")
        if empty_b:
            return json.dumps({"error": empty_b})

        variable_name = da_a.name or ""
        units = da_a.attrs.get("units", "")

        emit_status("Building comparison...", stage=STAGE_RENDER)

        if mode == "region":
            disjoint = _disjoint_periods_error(da_a, da_b)
            if disjoint:
                return json.dumps({"error": disjoint})
            # CPU-bound mask -> aggregate -> payload chain (T16), run off
            # the event loop.
            return await asyncio.to_thread(
                _build_region_comparison, handle_a, handle_b, da_a, da_b, label_a, label_b, variable_name, units,
            )

        try:
            align_raw = await mcp_tools["align"].ainvoke({"source_handles": [handle_a, handle_b]})
            align_result = parse_tool_result(align_raw)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        aligned_handle = align_result.get("handle")
        if align_result.get("status") == "error" or not aligned_handle:
            return json.dumps({
                "error": align_result.get("message")
                or "Failed to grid-align the two retrievals onto a common grid."
            })

        try:
            aligned_ds = await open_handle(aligned_handle, mcp_tools)
            # variable_name is already resolved from A/B above -- pass it
            # explicitly so the aligned (multi-source) dataset doesn't hit
            # its own ambiguous-variable error for a choice already made.
            aligned_da = _aggregation_service.to_dataarray(aligned_ds, handle=aligned_handle, variable=variable_name)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open the aligned result '{aligned_handle}': {e}"})

        try:
            aligned_a, aligned_b = _split_aligned(aligned_da)
        except ValueError as e:
            return json.dumps({"error": f"Grid alignment produced an unusable result: {e}"})

        # CPU-bound mask -> aggregate -> payload chain (T16), run off the
        # event loop.
        return await asyncio.to_thread(
            _build_period_comparison, handle_a, handle_b, aligned_handle, aligned_a, aligned_b,
            label_a, label_b, variable_name, units, threshold,
        )

    return compare


def _build_region_comparison(handle_a, handle_b, da_a, da_b, label_a, label_b, variable_name, units) -> str:
    da_a_2d = _prepare_2d(da_a, variable_name)
    da_b_2d = _prepare_2d(da_b, variable_name)
    vmin, vmax = _shared_bounds(da_a_2d, da_b_2d)

    panel_a = _region_panel(da_a_2d, handle_a, label_a, variable_name, units, vmin, vmax)
    panel_b = _region_panel(da_b_2d, handle_b, label_b, variable_name, units, vmin, vmax)
    stats_a = _region_stats(da_a_2d)
    stats_b = _region_stats(da_b_2d)

    payload = {
        "type": "heatmap_multi",
        "mode": "n-panel",
        "title": f"{variable_name}: {label_a} vs {label_b}",
        "variable": variable_name,
        "units": units,
        "panels": [panel_a, panel_b],
        "stats": {label_a: stats_a, label_b: stats_b},
        "metadata": {"source_handles": [handle_a, handle_b]},
    }
    return _save_chart(payload, f"{variable_name}_{label_a}_vs_{label_b}")


def _build_period_comparison(
    handle_a, handle_b, aligned_handle, aligned_a, aligned_b, label_a, label_b, variable_name, units, threshold,
) -> str:
    da_a_2d = _prepare_2d(aligned_a, variable_name)
    da_b_2d = _prepare_2d(aligned_b, variable_name)
    diff = _difference(da_a_2d, da_b_2d)
    stats = _anomaly_stats(da_a_2d, da_b_2d, diff, threshold)

    vmin, vmax = _diverging_bounds(diff)
    diff_payload = _da_to_heatmap_payload(
        diff, f"{variable_name}: {label_b} - {label_a}", variable_name, units, diverging=True,
        render_overlay=True, value_range=(vmin, vmax),
    )
    diff_payload["bounds"] = _bbox_from_da(diff)

    payload = {
        "type": "heatmap_multi",
        "mode": "difference",
        "title": f"{variable_name}: {label_b} vs {label_a} (difference)",
        "variable": variable_name,
        "units": units,
        "panels": [
            {"title": label_a, "metadata": {"source_handles": [handle_a]}},
            {"title": label_b, "metadata": {"source_handles": [handle_b]}},
        ],
        "difference": diff_payload,
        "stats": stats,
        "sign_convention": f"{label_b} minus {label_a}",
        "metadata": {"source_handles": [handle_a, handle_b, aligned_handle]},
    }
    return _save_chart(payload, f"{variable_name}_{label_b}_minus_{label_a}")
