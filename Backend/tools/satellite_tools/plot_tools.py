"""
plot_tools.py
-------------
Satellite plotting tools.

Data access is one seam: every tool takes an ``obs_``/``cube_`` handle and
calls ``open_handle`` (services.open_handle) to get an opened xarray
Dataset — never a parameter dict to re-fetch by. ``build_satellite_tools``
(tools.satellite_tools.factory) binds the MCP tools these need via closure
before registering them with the agent, since the model itself only ever
supplies a handle.

Returns chart payloads (JSON) instead of PNG files so the frontend can
render interactive Plotly charts. The API persists these payloads durably
in PostgreSQL when they are attached to a session. The payload schema is:

Spatial heatmap
---------------
{
  "type":     "heatmap",
  "title":    str,
  "variable": str,
  "units":    str,
  "lats":     [float, ...],        # 1-D latitude axis
  "lons":     [float, ...],        # 1-D longitude axis
  "values":   [[float, ...], ...], # 2-D row-major grid (lat × lon), NaN → null
  "vmin":     float,
  "vmax":     float,
}

Multi-panel comparison (list of heatmaps)
------------------------------------------
{
  "type":   "heatmap_multi",
  "panels": [ <heatmap payload>, ... ]
}

Time-series
-----------
{
  "type":      "timeseries",
  "title":     str,
  "variable":  str,
  "units":     str,
  "stat":      str,
  "times":     [ISO str, ...],
  "values":    [float, ...],
}
"""
import asyncio
import json
import logging
import os
import uuid
import numpy as np
from langchain.tools import tool
from langchain_core.tools import BaseTool
from typing import Annotated, List, Optional
from pydantic import Field

from config.workflow_stages import STAGE_RENDER
from datasets.mask_info import col_info_for_short_name, short_name_from_attrs
from earthdata_mcp.results import MCPToolError
from services.artifact_registry import build_artifact_reference
from services.open_handle import OpenHandleError, open_handle
from utils.geo_utils import find_lat_coord, find_lon_coord
from utils.colormaps import resolve as resolve_colormap
from utils.overlay_render import render_overlay_png
from utils.plotting import _normalize_to_2d, mask_data_by_geometry, RegionResolver
from utils.streaming import emit_chart, emit_status
from preprocessing.aggregation_service import AggregationService

logger = logging.getLogger(__name__)

_RENDER_TYPE_TO_ARTIFACT_PREFIX = {"heatmap": "map", "heatmap_multi": "cmp", "timeseries": "ts"}

_resolver = RegionResolver()
_aggregation_service = AggregationService()


def _sel_bounds(da, lat_coord, lon_coord, bounds):
    """
    Crop a DataArray to (minx, miny, maxx, maxy) bounds in a coordinate-order-
    safe way.  xarray slice() requires start <= stop when coords are increasing
    and start >= stop when decreasing.  We detect the direction and swap if needed
    so the crop never silently returns an empty array.
    """
    lat_vals = da[lat_coord].values
    lon_vals = da[lon_coord].values

    lat_min, lat_max = bounds[1], bounds[3]   # miny, maxy
    lon_min, lon_max = bounds[0], bounds[2]   # minx, maxx

    # If latitude is stored N→S (decreasing), slice must be (max, min)
    if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # Longitude is almost always W→E (increasing), but handle both
    if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    return da.sel({lat_coord: lat_slice, lon_coord: lon_slice})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Overlay PNGs live outside OUTPUT_DIR on purpose: OUTPUT_DIR is mounted
# unauthenticated at /outputs (api.py), and overlays must only be reachable
# through the authenticated /chart/{id}/overlay.png route (T23).
OVERLAY_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "overlay_store", "overlays")
os.makedirs(OVERLAY_STORE_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _percentile_bounds(arr: np.ndarray):
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmin == vmax:
        delta = abs(vmin) * 0.01 or 1.0
        return vmin - delta, vmax + delta
    return vmin, vmax


_MAX_GRID_CELLS = 8_000   # match the frontend MAX_POINTS constant


def _normalize_longitudes(da, lon_coord):
    """Convert 0..360 longitude coordinates to -180..180 and keep them sorted."""
    lon_vals = np.asarray(da[lon_coord].values)
    finite_lons = lon_vals[np.isfinite(lon_vals)]
    if finite_lons.size == 0 or finite_lons.min() < 0 or finite_lons.max() <= 180:
        return da

    normalized = ((lon_vals + 180) % 360) - 180
    return da.assign_coords({lon_coord: normalized}).sortby(lon_coord)


def _downsample_grid(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray):
    """
    Uniformly thin a 2-D (lat × lon) grid so it contains at most _MAX_GRID_CELLS
    non-null cells.  Returns (lats_ds, lons_ds, arr_ds).

    Strategy: keep every N-th row and every M-th column where N and M are chosen
    so that rows*cols ≈ _MAX_GRID_CELLS.  This is done *before* JSON serialisation
    so the payload written to disk is already small rather than forcing the browser
    to parse a multi-MB string and then discard most of it in flattenGrid.
    """
    n_rows, n_cols = arr.shape
    total = n_rows * n_cols
    if total <= _MAX_GRID_CELLS:
        return lats, lons, arr

    # Scale both axes by the same factor to preserve aspect ratio
    scale = (total / _MAX_GRID_CELLS) ** 0.5
    row_step = max(1, int(np.ceil(scale)))
    col_step = max(1, int(np.ceil(scale)))

    return lats[::row_step], lons[::col_step], arr[::row_step, ::col_step]


def _points_from_grid(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray):
    row_idx, col_idx = np.where(np.isfinite(arr))
    count = len(row_idx)
    if count == 0:
        return {"lats": [], "lons": [], "values": []}

    if count > _MAX_GRID_CELLS:
        take = np.linspace(0, count - 1, _MAX_GRID_CELLS, dtype=int)
        row_idx = row_idx[take]
        col_idx = col_idx[take]

    point_values = arr[row_idx, col_idx]
    return {
        "lats": [round(float(lats[i]), 6) for i in row_idx],
        "lons": [round(float(lons[i]), 6) for i in col_idx],
        "values": [float(f"{v:.6e}") for v in point_values],
    }


def _render_and_store_overlay(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray, lut: list, vmin: float, vmax: float) -> str | None:
    """Render the full-native-resolution overlay PNG and persist it to
    OVERLAY_STORE_DIR. Returns the stored path, or None on failure -- a
    failed render must degrade the chart (no overlay.url; the frontend
    falls back to canvas-from-arrays), never fail the whole tool call."""
    try:
        png_bytes = render_overlay_png(lats, lons, arr, lut, vmin, vmax)
        path = os.path.join(OVERLAY_STORE_DIR, f"{uuid.uuid4().hex}.png")
        with open(path, "wb") as f:
            f.write(png_bytes)
        return path
    except Exception:
        logger.warning("overlay_render_failed", exc_info=True)
        return None


def _da_to_heatmap_payload(
    da, title: str, variable: str, units: str, *,
    diverging: bool = False, render_overlay: bool = False, value_range: tuple[float, float] | None = None,
) -> dict:
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    if lat_coord is None or lon_coord is None:
        raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")

    da = _normalize_longitudes(da, lon_coord)

    if da.dims.index(lat_coord) != 0:
        da = da.transpose(lat_coord, lon_coord)

    arr = da.values.astype(float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    # A caller may impose a shared/diverging scale across multiple panels
    # (comparison_tools) -- the overlay below must colorize against that
    # same range, or the rendered map and its legend would disagree about
    # what a color means (T23's anti-drift guarantee).
    vmin, vmax = value_range if value_range is not None else _percentile_bounds(arr)

    lats_out = da[lat_coord].values
    lons_out = da[lon_coord].values
    points = _points_from_grid(lats_out, lons_out, arr)

    # Full-native-resolution extent, captured before downsampling, for the
    # server-rendered overlay PNG (T23) — visual fidelity and interaction
    # resolution are deliberately decoupled.
    overlay_bounds = [
        float(np.nanmin(lons_out)), float(np.nanmin(lats_out)),
        float(np.nanmax(lons_out)), float(np.nanmax(lats_out)),
    ]

    colormap = resolve_colormap(variable, diverging=diverging)

    overlay = {"bounds": overlay_bounds}
    if render_overlay:
        # Must run on the full-native-resolution grid, before _downsample_grid
        # below thins lats_out/lons_out/arr for the JSON payload -- visual
        # fidelity (PNG) is deliberately decoupled from interaction resolution
        # (arrays).
        overlay_path = _render_and_store_overlay(lats_out, lons_out, arr, colormap.lut, vmin, vmax)
        if overlay_path is not None:
            overlay["_path"] = overlay_path

    lats_out, lons_out, arr = _downsample_grid(lats_out, lons_out, arr)

    values_json = [
        [None if not np.isfinite(v) else float(f"{v:.6e}") for v in row]
        for row in arr
    ]

    return {
        "type":     "heatmap",
        "title":    title,
        "variable": variable,
        "units":    units,
        "lats":     [round(float(v), 6) for v in lats_out],
        "lons":     [round(float(v), 6) for v in lons_out],
        "values":   values_json,
        "points":   points,
        "vmin": float(f"{vmin:.6e}"),
        "vmax": float(f"{vmax:.6e}"),
        "colormap": {"name": colormap.name, "lut": colormap.lut},
        "overlay": overlay,
    }

def _heatmap_dims(payload: dict | None) -> list[int] | None:
    if not payload:
        return None
    lats, lons = payload.get("lats"), payload.get("lons")
    if isinstance(lats, list) and isinstance(lons, list):
        return [len(lats), len(lons)]
    return None


def _summary_dims_and_range(payload: dict, render_type: str | None):
    """Grid/point dimensions and value range for the compact model-facing
    summary — enough for the agent to describe the chart (T13 story #4)
    without re-reading the raw grid/points arrays."""
    if render_type == "heatmap":
        return _heatmap_dims(payload), payload.get("vmin"), payload.get("vmax")

    if render_type == "heatmap_multi":
        if payload.get("mode") == "difference" and isinstance(payload.get("difference"), dict):
            diff = payload["difference"]
            return _heatmap_dims(diff), diff.get("vmin"), diff.get("vmax")
        panels = [p for p in (payload.get("panels") or []) if isinstance(p, dict)]
        first = next((p for p in panels if p.get("lats")), None)
        dims = _heatmap_dims(first)
        if dims:
            dims = [len(panels), *dims]
        return dims, (first.get("vmin") if first else None), (first.get("vmax") if first else None)

    if render_type == "timeseries":
        times = payload.get("times") or []
        values = [v for v in (payload.get("values") or []) if isinstance(v, (int, float))]
        dims = [len(times)] if times else None
        vmin = min(values) if values else None
        vmax = max(values) if values else None
        return dims, vmin, vmax

    return None, payload.get("vmin"), payload.get("vmax")


def _chart_model_summary(payload: dict) -> dict:
    """The compact, model-facing view of a chart payload (T13): render type,
    title, variable, units, dimensions, value range, artifact id, and source
    handles — everything the agent needs to describe the chart and cite it,
    never the raw grid/points the frontend renders from ``emit_chart``."""
    render_type = payload.get("type")
    grid_dims, vmin, vmax = _summary_dims_and_range(payload, render_type)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    summary = {
        "render_type": render_type,
        "title": payload.get("title"),
        "variable": payload.get("variable"),
        "units": payload.get("units"),
        "grid_dims": grid_dims,
        "vmin": vmin,
        "vmax": vmax,
        "chart_id": payload.get("chart_id"),
        "source_handles": metadata.get("source_handles"),
    }
    summary = {k: v for k, v in summary.items() if v is not None}
    if payload.get("_artifact_refs"):
        summary["_artifact_refs"] = payload["_artifact_refs"]
    return summary


def _wire_overlay_url(overlay: dict | None, url: str) -> None:
    if isinstance(overlay, dict) and overlay.get("_path"):
        overlay["url"] = url


def _wire_overlay_urls(payload: dict) -> None:
    """Turn each rendered overlay's internal `_path` into a servable `url`,
    now that `_save_chart` has minted `chart_id` -- the render happens
    earlier (asyncio.to_thread, before chart_id exists) with a `_path` that
    only this process can read, never handed to the frontend directly."""
    chart_id = payload.get("chart_id")
    if not chart_id:
        return

    if payload.get("type") == "heatmap_multi":
        for i, panel in enumerate(payload.get("panels") or []):
            if isinstance(panel, dict):
                _wire_overlay_url(panel.get("overlay"), f"/chart/{chart_id}/overlay.png?panel={i}")
        difference = payload.get("difference")
        if isinstance(difference, dict):
            _wire_overlay_url(difference.get("overlay"), f"/chart/{chart_id}/overlay.png")
    else:
        _wire_overlay_url(payload.get("overlay"), f"/chart/{chart_id}/overlay.png")


def _save_chart(payload: dict, name: str) -> str:
    """Emit the full chart payload out-of-band (frontend chart/artifact
    pipeline) and return a compact model-facing summary (T13).

    Mints a stable artifact id for render types the T06 artifact vocabulary
    covers (map/comparison/timeseries) and embeds an `_artifact_refs` entry
    so the id is visible to both the calling LLM (to cite in its envelope,
    see config/earthdata_agent_prompt.py) and the gallery — mirroring the
    `_artifact_refs` convention EPA table tools already use. The full
    payload (grid/points/provenance/query/export) is emitted via
    ``emit_chart`` for the existing chart/artifact pipeline to persist and
    render; the model only ever sees the compact summary.
    """
    payload.setdefault("metadata", {})
    payload["metadata"].setdefault("name", name)

    prefix = _RENDER_TYPE_TO_ARTIFACT_PREFIX.get(payload.get("type"))
    if prefix is not None:
        payload["chart_id"] = f"{prefix}_{uuid.uuid4().hex[:12]}"
        try:
            ref = build_artifact_reference(payload)
        except Exception:
            logger.warning("artifact_reference_build_failed", extra={"_render_type": payload.get("type")})
            ref = None
        if ref is not None:
            payload["_artifact_refs"] = [ref.model_dump(exclude_none=True)]

    _wire_overlay_urls(payload)

    emit_chart(payload)
    return json.dumps(_chart_model_summary(payload))

# ── Handle / masking helpers ───────────────────────────────────────────────────


def _open_dataarray(ds, handle: str | None = None, variable: str | None = None):
    """Pick the science variable off an opened Dataset, unmasked.

    Resolution (T25): explicit ``variable`` -> the choice recorded for
    ``handle`` at retrieval time -> the file's only data variable -> the
    collection's pinned ``primary_var`` (via short_name attr) -> a structured
    candidate-listing error (AggregationService.to_dataarray).
    """
    return _aggregation_service.to_dataarray(ds, handle=handle, variable=variable)


def _build_dim_selector(dimension: str | None, dimension_value: float | None) -> dict | None:
    """A single-entry {dim_name: value} selector from a tool's optional
    ``dimension``/``dimension_value`` params, or None when no dimension was
    named -- the shape utils.plotting._normalize_to_2d's dim_selector expects."""
    if dimension is None or dimension_value is None:
        return None
    return {dimension: dimension_value}


def _mask_col_info(da, ds=None) -> dict:
    """The collections.yaml/registry masking metadata for a variable,
    resolved by the collection's short_name global attribute (T25
    masking-execution fix) -- collection identity is a dataset-level fact,
    so ``ds.attrs`` (the opened Dataset) is checked before the per-variable
    ``da.attrs``. ``short_name_from_attrs`` tolerates both the lowercase
    ``short_name`` and the CF/ACDD ``ShortName`` spelling real granules use.
    Falls back to the variable's own name when neither carries an identity
    marker, so a MASK_OVERRIDES quirk keyed on it still applies.
    """
    short_name = (
        short_name_from_attrs(ds.attrs if ds is not None else None)
        or short_name_from_attrs(da.attrs)
        or da.name
        or ""
    )
    return col_info_for_short_name(str(short_name).upper())


def _time_range(da) -> tuple[str, str]:
    if "time" not in da.coords:
        return "", ""
    times = sorted(str(t) for t in np.atleast_1d(da["time"].values))
    if not times:
        return "", ""
    return times[0], times[-1]


def _query_definition(da, region: dict | None, aggregation: str, chart_parameters: dict | None = None) -> dict:
    start_date, end_date = _time_range(da)
    query = {
        "dataset": da.name or "",
        "start_date": start_date,
        "end_date": end_date,
        "bbox": list(region["bounds"]) if region else None,
        "aggregation": aggregation,
    }
    if chart_parameters:
        query["chart_parameters"] = chart_parameters
    return {k: v for k, v in query.items() if v not in (None, "", [])}


def _provenance(handles: list[str], da, region_name: str, aggregation: str, agg_meta: dict | None = None) -> dict:
    start_date, end_date = _time_range(da)
    provenance = {
        "variable": da.name or "",
        "start_date": start_date,
        "end_date": end_date,
        "region_name": region_name,
        "aggregation": aggregation,
        "units": da.attrs.get("units", ""),
        "source_handles": list(handles),
    }
    if agg_meta:
        provenance["aggregation"] = agg_meta["aggregation_label"]
        provenance["n_granules"] = agg_meta["n_granules"]
        provenance["cadence"] = agg_meta["cadence"]
        provenance["granule_dates"] = agg_meta["granule_dates"]
        if agg_meta.get("masking"):
            # T25 Phase 3: qa_status (verified/cf-deterministic/inferred, not
            # verified/not applied) travels into the answer the agent sees,
            # not just internal meta -- an inferred QA mask must be a
            # disclosed fact, never a silent guess.
            provenance["masking"] = agg_meta["masking"]
    return provenance


def _attach_reproducibility(
    payload: dict,
    handles: list[str],
    da,
    region_name: str,
    aggregation: str,
    chart_parameters: dict | None = None,
    agg_meta: dict | None = None,
    region: dict | None = None,
) -> dict:
    aggregation_label = agg_meta["aggregation_label"] if agg_meta else aggregation
    payload["provenance"] = _provenance(handles, da, region_name, aggregation_label, agg_meta)
    payload["query"] = _query_definition(da, region, aggregation_label, chart_parameters)
    payload["export"] = {
        "type": payload.get("type"),
        "variable": da.name or "",
        "units": da.attrs.get("units", ""),
        "region_name": region_name,
        "aggregation": aggregation_label,
        "aggregation_meta": agg_meta or payload.get("aggregation_meta") or {},
        "chart_parameters": chart_parameters or {},
        "source_handles": list(handles),
    }
    if agg_meta:
        payload["query"]["aggregation"] = agg_meta["aggregation_label"]
    payload.setdefault("metadata", {})
    payload["metadata"]["source_handles"] = list(handles)
    return payload


# ── Tools ─────────────────────────────────────────────────────────────────────


def make_plot_singular(mcp_tools: dict[str, BaseTool]):
    @tool
    async def plot_singular(
        handle: Annotated[
            str,
            Field(description="An obs_/cube_ handle from a retrieval or transform tool."),
        ],
        location: str,
        title: str = "",
        cmap: Optional[str] = "Spectral_r",
        variable: Optional[str] = None,
        dimension: Optional[str] = None,
        dimension_value: Optional[float] = None,
    ) -> str:
        """
        Plot a spatial heatmap of a variable over a single location at one point in time.
        Use when the user asks for a "map", "plot", or "show" for a single snapshot.

        Do NOT use this for time series, trends, or requests involving change over time —
        use conduct_temporal_statistic instead.

        Args:
            handle   : obs_/cube_ handle from a retrieval or transform tool.
            location : Place name e.g. 'New York City', 'California'.
            title    : Plot title. Auto-generated from variable + location if omitted.
            cmap     : Colormap hint for the frontend (default 'Spectral_r').
            variable : Science variable to plot, for a multi-variable file with
                       no variable chosen at retrieval time. See describe_dataset's
                       variable metadata to pick one — required if the tool
                       returns a "variable_choice_required" error listing candidates.
            dimension       : Name of an extra non-spatial, non-time dimension
                               to select a single value from (e.g. a vertical
                               level) — required if the tool returns a
                               "dimension_choice_required" error naming one.
            dimension_value : Coordinate value to select from ``dimension``
                               (nearest match), e.g. a pressure level in hPa.

        Returns:
            JSON string — chart payload for the frontend to render interactively.
        """
        try:
            ds = await open_handle(handle, mcp_tools)
            # Normalize longitude on the whole opened Dataset, before
            # extracting the science DataArray -- so it and its sibling
            # QA-flag variable (still reachable through ds, T25 masking-
            # execution fix) share one coordinate convention. Doing this
            # only on the extracted DataArray would leave ds's flag variable
            # on the original 0..360 convention, and da.where(qf.isin(...))
            # would align on an empty intersection instead of masking.
            ds_lon_coord = find_lon_coord(ds)
            if ds_lon_coord:
                ds = _normalize_longitudes(ds, ds_lon_coord)
            da = _open_dataarray(ds, handle=handle, variable=variable)
        except MCPToolError as e:
            emit_status("Visualization failed while opening data.", stage=STAGE_RENDER)
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            emit_status("Visualization failed while opening data.", stage=STAGE_RENDER)
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        emit_status("Resolving requested location...", stage=STAGE_RENDER)
        region = await _resolver.aresolve_location(location)
        if region is None:
            emit_status("Location lookup failed.", stage=STAGE_RENDER)
            return json.dumps({"error": f"Could not geocode location: '{location}'"})

        emit_status("Generating visualization...", stage=STAGE_RENDER)

        def _mask_aggregate_payload():
            # CPU-bound mask -> aggregate -> payload chain (T16): run off the
            # event loop via asyncio.to_thread below so a large grid doesn't
            # freeze every other concurrent stream for its duration.
            try:
                lat_coord = find_lat_coord(da)
                lon_coord = find_lon_coord(da)
                if lat_coord is None or lon_coord is None:
                    raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")
                masked = _normalize_longitudes(da, lon_coord)
                masked = mask_data_by_geometry(masked, region["geometry"])
                bounds = region["bounds"]  # (minx, miny, maxx, maxy)
                masked = _sel_bounds(masked, lat_coord, lon_coord, bounds)
            except Exception as e:
                return "mask", None, None, f"Masking failed: {e}"

            units = masked.attrs.get("units", "")
            variable_name = masked.name or ""
            col_info = _mask_col_info(masked, ds)
            try:
                aggregation = _aggregation_service.aggregate(
                    masked,
                    variable=variable_name,
                    stat="mean",
                    col_info=col_info,
                    source_ds=ds,
                )
                reduced = next(iter(aggregation.ds.data_vars.values()))
                reduced = _normalize_to_2d(reduced, dim_selector=_build_dim_selector(dimension, dimension_value))
            except MCPToolError as e:
                return "resolve", None, None, e.to_dict()
            agg_meta = aggregation.meta
            is_aggregated = agg_meta["n_granules"] > 1
            if title:
                resolved_title = title
            elif is_aggregated:
                resolved_title = f"{variable_name} {agg_meta['title_suffix']} over {region['name']}"
            else:
                resolved_title = f"{variable_name} over {region['name']}"

            try:
                payload = _da_to_heatmap_payload(reduced, resolved_title, variable_name, units, render_overlay=True)
                payload["cmap"]   = cmap or "Spectral_r"
                payload["bounds"] = list(region["bounds"])  # (minx, miny, maxx, maxy)
                payload["aggregation_meta"] = agg_meta
                payload["is_aggregated"] = is_aggregated
                _attach_reproducibility(
                    payload,
                    [handle],
                    reduced,
                    region["name"],
                    agg_meta["aggregation_label"] if is_aggregated else "single snapshot",
                    {"chart_type": "heatmap", "cmap": payload["cmap"], "location": location},
                    agg_meta,
                    region,
                )
            except Exception as e:
                return "payload", None, None, f"Failed to build chart payload: {e}"

            return None, payload, resolved_title, None

        stage, payload, resolved_title, error_message = await asyncio.to_thread(_mask_aggregate_payload)
        if stage == "mask":
            emit_status("Visualization failed while processing map bounds.", stage=STAGE_RENDER)
            return json.dumps({"error": error_message})
        if stage == "resolve":
            emit_status("Visualization needs a variable or dimension choice.", stage=STAGE_RENDER)
            return json.dumps({"error": error_message})
        if stage == "payload":
            emit_status("Visualization failed while building chart data.", stage=STAGE_RENDER)
            return json.dumps({"error": error_message})

        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(payload, resolved_title)

    return plot_singular


def make_plot_multiple(mcp_tools: dict[str, BaseTool]):
    @tool
    async def plot_multiple(
        handles: Annotated[List[str], Field(description="obs_/cube_ handles, one per location.")],
        locations: List[str],
        title: str = "",
        cmap: Optional[str] = "Spectral_r",
        variable: Optional[str] = None,
        dimension: Optional[str] = None,
        dimension_value: Optional[float] = None,
    ) -> str:
        """
        Plot the same environmental variable across multiple locations side by side.

        IMPORTANT — retrieve a handle for each location first, collecting each
        into a list. Only call this tool once you have a handle for every location.

        Args:
            handles   : obs_/cube_ handles, one per location.
            locations : List of place names matching handles order.
            title     : Overall title (optional).
            cmap      : Colormap hint for the frontend (default 'Spectral_r').
            variable  : Science variable to plot, for a multi-variable file with
                        no variable chosen at retrieval time (applies to every handle).
            dimension       : Name of an extra non-spatial, non-time dimension to
                               select a single value from (e.g. a vertical level).
            dimension_value : Coordinate value to select from ``dimension`` (nearest match).

        Returns:
            JSON string — multi-panel chart payload for the frontend to render.
        """
        emit_status("Generating visualization...", stage=STAGE_RENDER)
        if len(handles) != len(locations):
            emit_status("Visualization failed while matching locations to datasets.", stage=STAGE_RENDER)
            return json.dumps({"error": f"len(handles)={len(handles)} != len(locations)={len(locations)}"})

        panels = []
        variable_name = ""
        for handle, location in zip(handles, locations):
            try:
                ds = await open_handle(handle, mcp_tools)
                # See plot_singular: normalize the whole Dataset's longitude
                # before extraction, so da and its sibling QA-flag variable
                # (still reachable through ds) share one coordinate
                # convention (T25 masking-execution fix).
                ds_lon_coord = find_lon_coord(ds)
                if ds_lon_coord:
                    ds = _normalize_longitudes(ds, ds_lon_coord)
                da = _open_dataarray(ds, handle=handle, variable=variable)
            except MCPToolError as e:
                emit_status("Visualization failed while opening data.", stage=STAGE_RENDER)
                return json.dumps({"error": e.to_dict()})
            except OpenHandleError as e:
                emit_status("Visualization failed while opening data.", stage=STAGE_RENDER)
                return json.dumps({"error": f"Failed to open handle '{handle}' for '{location}': {e}"})

            emit_status("Resolving requested location...", stage=STAGE_RENDER)
            region = await _resolver.aresolve_location(location)
            if region is None:
                emit_status("Location lookup failed.", stage=STAGE_RENDER)
                return json.dumps({"error": f"Could not geocode location: '{location}'"})

            def _mask_aggregate_panel(da=da, ds=ds, region=region, handle=handle, location=location, variable_name=variable_name):
                # CPU-bound mask -> aggregate -> payload chain (T16), run off
                # the event loop via asyncio.to_thread below.
                try:
                    lat_coord = find_lat_coord(da)
                    lon_coord = find_lon_coord(da)
                    if lat_coord is None or lon_coord is None:
                        raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")
                    masked = _normalize_longitudes(da, lon_coord)
                    masked = mask_data_by_geometry(masked, region["geometry"])
                except Exception as e:
                    return "mask", None, None, f"Masking failed for '{location}': {e}"

                bounds = region["bounds"]
                masked = _sel_bounds(masked, lat_coord, lon_coord, bounds)

                resolved_variable_name = masked.name or variable_name
                units = masked.attrs.get("units", "")
                col_info = _mask_col_info(masked, ds)

                try:
                    aggregation = _aggregation_service.aggregate(
                        masked,
                        variable=resolved_variable_name,
                        stat="mean",
                        col_info=col_info,
                        source_ds=ds,
                    )
                    reduced = next(iter(aggregation.ds.data_vars.values()))
                    reduced = _normalize_to_2d(reduced, dim_selector=_build_dim_selector(dimension, dimension_value))
                except MCPToolError as e:
                    return "resolve", None, None, e.to_dict()

                try:
                    agg_meta = aggregation.meta
                    panel = _da_to_heatmap_payload(reduced, region["name"], resolved_variable_name, units, render_overlay=True)
                    panel["cmap"]   = cmap or "Spectral_r"
                    panel["bounds"] = list(region["bounds"])
                    panel["aggregation_meta"] = agg_meta
                    panel["is_aggregated"] = agg_meta["n_granules"] > 1
                    _attach_reproducibility(
                        panel,
                        [handle],
                        reduced,
                        region["name"],
                        agg_meta["aggregation_label"] if agg_meta["n_granules"] > 1 else "single snapshot",
                        {"chart_type": "heatmap", "cmap": panel["cmap"], "location": location},
                        agg_meta,
                        region,
                    )
                except Exception as e:
                    return "payload", None, None, f"Failed to build panel for '{location}': {e}"

                return None, panel, resolved_variable_name, None

            stage, panel, resolved_variable_name, error_message = await asyncio.to_thread(_mask_aggregate_panel)
            if stage == "mask":
                emit_status("Visualization failed while processing map bounds.", stage=STAGE_RENDER)
                return json.dumps({"error": error_message})
            if stage == "resolve":
                emit_status("Visualization needs a variable or dimension choice.", stage=STAGE_RENDER)
                return json.dumps({"error": error_message})
            if stage == "payload":
                emit_status("Visualization failed while building chart data.", stage=STAGE_RENDER)
                return json.dumps({"error": error_message})

            variable_name = resolved_variable_name
            panels.append(panel)

        multi_payload = {"type": "heatmap_multi", "title": title or f"{variable_name} Comparison", "panels": panels}
        if panels:
            multi_payload["provenance"] = {
                **panels[0].get("provenance", {}),
                "region_name": ", ".join(panel.get("provenance", {}).get("region_name", "") for panel in panels),
                "aggregation": "single snapshot comparison",
            }
            multi_payload["query"] = {
                "dataset": variable_name,
                "aggregation": "single snapshot comparison",
                "panels": [panel.get("query", {}) for panel in panels],
                "chart_parameters": {"chart_type": "heatmap_multi", "cmap": cmap or "Spectral_r"},
            }
            multi_payload["export"] = {
                "type": "heatmap_multi",
                "variable": variable_name,
                "units": panels[0].get("units", ""),
                "aggregation": "single snapshot comparison",
                "chart_parameters": {"chart_type": "heatmap_multi", "cmap": cmap or "Spectral_r"},
                "panels": [panel.get("export", {}) for panel in panels],
                "source_handles": list(handles),
            }
            multi_payload["metadata"] = {"source_handles": list(handles)}
        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(multi_payload, title or f"{variable_name}_comparison")

    return plot_multiple


def make_conduct_temporal_statistic(mcp_tools: dict[str, BaseTool]):
    @tool
    async def conduct_temporal_statistic(
        handle: Annotated[str, Field(description="An obs_/cube_ handle from a retrieval or transform tool.")],
        location: str,
        stat: str = "mean",
        variable: Optional[str] = None,
        dimension: Optional[str] = None,
        dimension_value: Optional[float] = None,
    ) -> str:
        """
        Produce a time-series line chart showing how a variable changes over time.

        Use this tool when the user asks for a "time series", "trend", "how X changed over time",
        "monthly values", or anything involving change across multiple time steps.
        Do NOT use plot_singular for these requests — plot_singular only shows a single snapshot.

        Args:
            handle:   obs_/cube_ handle covering a multi-day or multi-month range
                      with multiple granules.
            location: place name to spatially mask before computing e.g. 'New Jersey'
            stat:     statistic to compute at each time step.
                      One of: 'mean', 'median', 'max', 'min', 'std'  (default: 'mean')
            variable  : Science variable to use, for a multi-variable file with no
                        variable chosen at retrieval time.
            dimension       : Name of an extra non-spatial, non-time dimension to
                               select a single value from (e.g. a vertical level).
            dimension_value : Coordinate value to select from ``dimension`` (nearest match).

        Returns:
            JSON string — time-series chart payload for the frontend to render interactively.
        """
        import pandas as pd
        from utils.geo_utils import identify_time

        try:
            ds = await open_handle(handle, mcp_tools)
            da = _open_dataarray(ds, handle=handle, variable=variable)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        except OpenHandleError as e:
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        time_dim = identify_time(da)
        if time_dim is None or time_dim not in da.dims:
            return json.dumps({"error": f"No time dimension found. dims={list(da.dims)}"})

        emit_status("Resolving requested location...", stage=STAGE_RENDER)
        region = await _resolver.aresolve_location(location)
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        emit_status("Computing time series...", stage=STAGE_RENDER)

        def _mask_aggregate_timeseries():
            # CPU-bound mask -> per-timestep aggregate -> payload chain
            # (T16), run off the event loop via asyncio.to_thread below.
            masked = mask_data_by_geometry(da, region["geometry"])

            lat_coord = find_lat_coord(masked)
            lon_coord = find_lon_coord(masked)
            if lat_coord is None or lon_coord is None:
                return "error", f"Cannot find lat/lon coords. Available: {list(masked.coords)}"
            bounds = region["bounds"]
            masked = _sel_bounds(masked, lat_coord, lon_coord, bounds)

            variable_name = masked.name or ""
            if stat not in AggregationService._STAT_FUNCS:
                return "error", f"Unknown stat '{stat}'. Use: mean, median, max, min, std"

            dim_selector = _build_dim_selector(dimension, dimension_value)
            if dim_selector:
                for dim_name, value in dim_selector.items():
                    if dim_name in masked.dims:
                        masked = masked.sel({dim_name: value}, method="nearest")
            extra_dims = [d for d in masked.dims if d not in (lat_coord, lon_coord, time_dim)]
            if extra_dims:
                from utils.plotting import _dimension_choice_error

                return "dimension_choice_required", _dimension_choice_error(masked, extra_dims[0]).to_dict()

            # T25 masking-execution fix: route through the same shared
            # masking-resolution path aggregate() uses (collections.yaml ->
            # UMM-Var -> CF-attrs precedence, plus the three-tier QA-flag
            # doctrine) instead of a hand-rolled apply_quality_mask(ds=None)
            # call -- this path now actually applies QA-flag masking (via
            # source_ds=ds) and discloses it honestly, the same as
            # plot/stat, rather than a parallel copy that always claimed
            # "not applied".
            col_info = _mask_col_info(masked, ds)
            masked, masking_provenance = _aggregation_service.resolve_and_mask(
                masked, variable=variable_name, col_info=col_info, source_ds=ds,
            )

            times, values = [], []
            for i in range(masked.sizes[time_dim]):
                slice_2d = masked.isel({time_dim: i}).values
                try:
                    value = _aggregation_service.compute_values_stat(slice_2d, stat)
                except ValueError:
                    continue
                raw_time = masked[time_dim].values[i]
                timestamp = pd.Timestamp(raw_time).isoformat()
                times.append(timestamp)
                values.append(round(float(value), 6))

            if not times:
                return "error", f"No valid data found for '{location}' across any time step."

            # Sort by time
            paired = sorted(zip(times, values))
            sorted_times, sorted_values = zip(*paired)

            ts_payload = {
                "type":     "timeseries",
                "title":    f"{variable_name} {stat} over {location}",
                "variable": variable_name,
                "units":    masked.attrs.get("units", ""),
                "stat":     stat,
                "times":    list(sorted_times),
                "values":   list(sorted_values),
            }
            ts_payload["masking"] = masking_provenance
            _attach_reproducibility(
                ts_payload,
                [handle],
                masked,
                region["name"],
                stat,
                {"chart_type": "timeseries", "location": location},
                region=region,
            )
            return None, (ts_payload, variable_name)

        status, result = await asyncio.to_thread(_mask_aggregate_timeseries)
        if status in ("error", "dimension_choice_required"):
            return json.dumps({"error": result})
        ts_payload, variable_name = result
        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(ts_payload, f"{variable_name}_{stat}_{location}")

    return conduct_temporal_statistic
