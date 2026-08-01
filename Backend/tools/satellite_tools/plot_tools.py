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
from datasets.mask_info import col_info_for_short_name, resolve_mask_info, short_name_from_attrs
from datasets.variable_roles import classify_inventory, related_variables
from earthdata_mcp.results import MCPToolError
from services import scope_registry
from services.artifact_registry import build_artifact_reference
from services.open_handle import OpenHandleError, open_handle
from utils.geo_utils import find_lat_coord, find_lon_coord
from utils.colormaps import resolve as resolve_colormap
from utils.overlay_render import render_overlay_png
from utils.phase_timing import phase_timer
from utils.plotting import (
    _normalize_to_2d,
    apply_mask_region_type,
    geometry_mask,
    half_cell as _half_cell,
    mask_data_by_geometry,
    RegionResolver,
    sel_bounds as _sel_bounds,
)
from utils.streaming import emit_chart, emit_status
from preprocessing.aggregation_service import (
    VARIABLE_RESOLUTION_ATTR,
    AggregationService,
    VariableChoiceRequired,
    area_weighted_mean,
    fill_match,
)
from preprocessing.variable_choice_builder import emit_variable_choice_payload

logger = logging.getLogger(__name__)

_RENDER_TYPE_TO_ARTIFACT_PREFIX = {"heatmap": "map", "heatmap_multi": "cmp", "timeseries": "ts"}

_resolver = RegionResolver()
_aggregation_service = AggregationService()


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
    scale_disclosure: dict | None = None,
) -> dict:
    # T51: the overlay PNG rasterization and grid downsampling are pure CPU on
    # the plot path, and are where a "the chart took forever" turn actually
    # spends its time once the data is in memory. Timed at this one seam so
    # every caller (singular/multiple/compare panels) is covered.
    with phase_timer(
        "render",
        cells_in=int(getattr(da, "size", 0)),
        overlay=render_overlay,
    ):
        return _build_heatmap_payload(
            da, title, variable, units,
            diverging=diverging, render_overlay=render_overlay,
            value_range=value_range, scale_disclosure=scale_disclosure,
        )


def _build_heatmap_payload(
    da, title: str, variable: str, units: str, *,
    diverging: bool = False, render_overlay: bool = False, value_range: tuple[float, float] | None = None,
    scale_disclosure: dict | None = None,
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
    # Disclose how the color scale was set so the legend can be honest about
    # saturation (T43): a percentile clip saturates the extreme tails, which a
    # reader must not mistake for the data's true range. A caller imposing its
    # own range (comparison's shared/diverging scale) says how that range was
    # derived via ``scale_disclosure`` -- a shared 2/98 clip or a diverging
    # 98th-percentile-magnitude clip is still a clip, just computed across
    # panels, so the same saturation warning must fire. Absent a disclosure,
    # an imposed range is a true fixed choice (nothing to clip).
    if value_range is not None:
        scale = scale_disclosure if scale_disclosure is not None else {"method": "explicit"}
    else:
        scale = {"method": "percentile", "p": [2, 98]}

    lats_out = da[lat_coord].values
    lons_out = da[lon_coord].values
    points = _points_from_grid(lats_out, lons_out, arr)

    # Full-native-resolution extent, captured before downsampling, for the
    # server-rendered overlay PNG (T23) — visual fidelity and interaction
    # resolution are deliberately decoupled. The bounds are pixel-EDGE, not
    # pixel-center: render_overlay_png rasterizes edge-to-edge (left =
    # lons[0] - res/2, …), so reporting center min/max here would pin an
    # edge-to-edge PNG onto center-to-center bounds and misregister every
    # pixel by up to half a cell (visible on coarse grids like GPM/MERRA-2).
    lon_half = _half_cell(lons_out)
    lat_half = _half_cell(lats_out)
    overlay_bounds = [
        float(np.nanmin(lons_out)) - lon_half, float(np.nanmin(lats_out)) - lat_half,
        float(np.nanmax(lons_out)) + lon_half, float(np.nanmax(lats_out)) + lat_half,
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
        "scale": scale,
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


def _time_range(da, agg_meta: dict | None = None) -> tuple[str, str]:
    """Temporal range of ``da`` -- from its time coordinate when it still has
    one, else from the aggregation meta. The fallback matters twice over: the
    array reaching provenance/query builders is the *reduced* one (time dim
    already collapsed away by the aggregation), and some granules (monthly L3
    means) never had a time coordinate at all -- their coverage lives in global
    attrs, which aggregation meta now captures (``_build_meta``)."""
    if "time" in da.coords:
        times = sorted(str(t) for t in np.atleast_1d(da["time"].values))
        if times:
            return times[0], times[-1]
    if agg_meta:
        start = agg_meta.get("start_date") or ""
        end = agg_meta.get("end_date") or ""
        dates = agg_meta.get("granule_dates") or []
        return start or (dates[0] if dates else ""), end or (dates[-1] if dates else "")
    return "", ""


def _query_definition(da, region: dict | None, aggregation: str, chart_parameters: dict | None = None, agg_meta: dict | None = None) -> dict:
    start_date, end_date = _time_range(da, agg_meta)
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


def _dataset_facts(col_info: dict | None) -> dict:
    """Registry facts about the *collection* (T32) -- distinct from
    ``provenance["variable"]``, which names the science variable plotted,
    not the dataset it came from. ``col_info`` is whatever ``_mask_col_info``
    already resolved for the masking pipeline (collections.yaml, matched by
    the opened granule's short_name attr), so this is a read of data already
    in hand, never a new lookup."""
    col_info = col_info or {}
    provider = col_info.get("provider") or ""
    instrument = col_info.get("instrument") or ""
    return {
        "dataset": col_info.get("short_name") or "",
        "dataset_description": col_info.get("description") or "",
        "dataset_version": col_info.get("version") or "",
        "collection_id": col_info.get("collection_id") or "",
        "provider": provider,
        "instrument": instrument,
        "source": " — ".join(part for part in (provider, instrument) if part),
    }


def _variable_definition(da, col_info: dict | None) -> dict:
    """long_name/valid-range/mask facts for the plotted variable (T32
    Details -> Variable Definition), sourced from the registry col_info
    already resolved for masking and the CF ``long_name`` attribute the
    opened DataArray already carries -- no new MCP round trip.
    ``advisory_notes`` stays empty here: those only exist in describe_dataset's
    UMM-Var response, which no chart tool calls today."""
    col_info = col_info or {}
    valid_min = col_info.get("valid_min")
    valid_max = col_info.get("valid_max")
    fill_value = col_info.get("fill_value")
    has_range = valid_min is not None or valid_max is not None
    has_fill = fill_value is not None
    if has_fill and has_range:
        mask_note = "fill values and a valid range are defined"
    elif has_fill:
        mask_note = "fill values are defined, no valid range"
    elif has_range:
        mask_note = "a valid range is defined, no fill values"
    else:
        mask_note = "no fill/range metadata"
    return {
        "long_name": da.attrs.get("long_name", ""),
        "units": da.attrs.get("units", ""),
        "advisory_notes": [],
        "valid_ranges": {"min": valid_min, "max": valid_max} if has_range else {},
        "fill_value": fill_value,
        "mask_note": mask_note,
    }


def _qa_methodology(col_info: dict | None) -> dict:
    """The pinned collections.yaml QA rule as general methodology (T32
    Details -> Provenance) -- distinct from ``masking``, which discloses
    what was actually applied to *this* request."""
    col_info = col_info or {}
    methodology = {
        "quality_flag_var": col_info.get("quality_flag_var"),
        "qa_good_values": col_info.get("qa_good_values"),
        "qa_bad_values": col_info.get("qa_bad_values"),
    }
    return {k: v for k, v in methodology.items() if v is not None}


def _inventory_records(ds) -> list[dict]:
    """The opened Dataset's bands as ``classify_inventory`` records. Names
    here are bare leaves (open_handle merges groups without prefixing), so
    each record carries the ``group_path`` attr open_handle stamped — the
    only way the classifier's group priors can fire post-open."""
    return [
        {
            "name": name,
            "group": var.attrs.get("group_path"),
            "standard_name": var.attrs.get("standard_name"),
            "long_name": var.attrs.get("long_name"),
            "units": var.attrs.get("units"),
        }
        for name, var in ds.data_vars.items()
    ]


def _related_variables(da, col_info: dict | None, ds=None) -> dict:
    """A lightweight related-variables view for the chart page (PRD T35): the
    plotted variable's role plus its QA/uncertainty/context siblings. Built
    from the opened Dataset's actual bands when it travels — the SAME source
    ``_evidence`` reads, so the related-variables panel and the evidence facts
    can never contradict each other (the registry's curated ``variables``
    subset can be far narrower than the retrieved file). Falls back to that
    curated subset when no source Dataset is available. The plotted variable
    is the opened DataArray's (leaf) name."""
    col_info = col_info or {}
    variables = col_info.get("variables")
    if ds is not None and getattr(ds, "data_vars", None):
        variables = _inventory_records(ds)
    return related_variables(
        variables,
        groups=col_info.get("groups"),
        primary_var=col_info.get("primary_var"),
        quality_flag_var=col_info.get("quality_flag_var"),
        plotted_variable=da.name or "",
    )


def _evi_leaf(name) -> str:
    """The bare, lowercased leaf name of a (possibly group-qualified) variable
    -- for comparing a classified inventory entry against the plotted science
    variable / QA-flag variable without importing variable_roles' internals."""
    return str(name or "").rsplit("/", 1)[-1].lower()


def _crop_band_to_region(band, region):
    """Crop a companion band to the plotted science variable's region footprint
    -- the same geometry mask + ``_sel_bounds`` crop the science variable
    received, so the band is co-located pixel-for-pixel (companions share the
    science grid, so this is xarray's inner-join alignment, no reprojection).
    Returns ``(cropped_band, in_region_cell_count)`` where the count is the
    region footprint measured off the geometry mask broadcast over an all-ones
    twin -- independent of the band's own fill/NaN, so ``coverage`` counts
    fill cells against the total rather than dropping them (the valid-pct
    trap). The geometry is rasterized ONCE (``geometry_mask``): the mask
    depends only on grid and geometry, so crop and footprint both derive from
    it. Returns ``(None, 0)`` when the band has no usable lat/lon grid to
    co-locate on."""
    import xarray as xr

    lon_coord = find_lon_coord(band)
    if lon_coord:
        band = _normalize_longitudes(band, lon_coord)  # keeps the coord's name
    lat_coord = find_lat_coord(band)
    if lat_coord is None or lon_coord is None:
        return None, 0

    mask = geometry_mask(band, region["geometry"])
    cropped = _sel_bounds(band.where(mask), lat_coord, lon_coord, region["bounds"])

    footprint = _sel_bounds(xr.ones_like(band).where(mask), lat_coord, lon_coord, region["bounds"])
    in_region = int(np.isfinite(np.asarray(footprint.values)).sum())
    return cropped, in_region


def _band_time_mean(band, resolved):
    """Mask the band's own fill/out-of-range cells, then collapse its time
    dimension to a per-pixel mean, so an evidence fact describes the SAME
    time-reduced field the science variable is plotted as -- not a
    space-time-mean that, for uncertainty, would be divided by the science
    *time-mean* (two incommensurable aggregations; T36 evidence honesty).

    Fill/range masking happens BEFORE the time reduction so a fill sentinel
    (e.g. -9999) is never averaged into the per-pixel mean; the time collapse
    is skipna, matching the science field's own ``reduce(mean, skipna)`` over
    valid timesteps. A no-op time collapse when the band carries no time
    dimension (every current L3-snapshot fixture)."""
    from utils.geo_utils import identify_time

    fill = resolved.get("fill_value")
    valid_min = resolved.get("valid_min")
    valid_max = resolved.get("valid_max")
    if fill is not None:
        try:
            band = band.where(~fill_match(band, fill))
        except (TypeError, ValueError):
            pass
    if valid_min is not None:
        band = band.where(band >= valid_min)
    if valid_max is not None:
        band = band.where(band <= valid_max)
    time_dim = identify_time(band)
    if time_dim is not None and time_dim in band.dims:
        band = band.mean(dim=time_dim, skipna=True)
    return band


def _band_mean_fact(band, leaf, role, region, *, pct_of_science=None):
    """A deterministic mean-over-valid-pixels evidence fact for a context or
    uncertainty band, in the band's own units, carrying an honest coverage
    valid-fraction. ``pct_of_science`` (the masked science mean) adds the
    uncertainty as a fraction of the science value when available.

    The band is masked to its own valid cells and collapsed to a per-pixel
    time-mean (``_band_time_mean``) before the region crop, so the fact
    summarizes the same time-reduced field the science variable is plotted as
    -- coverage and the pct-of-science ratio then compare like with like."""
    resolved, _ = resolve_mask_info(cf_attrs=dict(band.attrs))
    band = _band_time_mean(band, resolved)
    cropped, in_region = _crop_band_to_region(band, region)
    if cropped is None or in_region == 0:
        return None
    vals = np.asarray(cropped.values, dtype="float64")
    valid = np.isfinite(vals)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return None
    # Cos(latitude) area-weighted, the SAME regional-mean definition as the
    # headline science mean (Finding #13) -- an unweighted grid-cell mean
    # over-weights poleward pixels, so context/uncertainty evidence over a
    # continental region would silently disagree with the value it qualifies.
    # Falls back to the unweighted mean when the band has no latitude dim.
    mean_val = area_weighted_mean(cropped)
    fact = {
        "name": leaf,
        "role": role,
        "stat": "mean",
        "value": round(mean_val, 6),
        "units": band.attrs.get("units", "") or "",
        "coverage": round(valid_count / in_region, 4),
    }
    # A 0.0 science mean (a legitimate value for anomaly-style fields) makes
    # the ratio undefined — omit the key rather than divide by zero, which
    # would lose the whole fact to the caller's best-effort except.
    if pct_of_science is not None and pct_of_science != 0:
        fact["pct_of_science"] = round(abs(mean_val / pct_of_science), 4)
    return fact


def _evidence(ds, da, col_info: dict | None, region: dict | None) -> list[dict]:
    """Deterministic companion-evidence facts (PRD T36 Phase 2): the quality
    and context bands sitting unused beside the plotted science variable in the
    same opened Dataset, summarized as co-located facts a scientist can use to
    judge *this* measurement -- retrieval uncertainty, cloud fraction, aerosol
    index -- each with honest coverage. No LLM, no narrative; this is the facts
    layer P3 will later explain. The QA pass rate is deliberately NOT here
    (T55): it is counted where the mask is applied and reported once as masking
    provenance, so it cannot disagree with the plotted data.

    Driven entirely by T35's ``classify_inventory`` over ``ds.data_vars`` (the
    High-confidence CF ``standard_name`` tier is reachable here, post-open) so
    an evidence band is never invented -- a product with no context siblings
    (e.g. MODIS AOD) yields ``[]``. Only bands already present in ``ds`` are
    read; fetching a companion the file lacks is out of scope (Phase 3+).

    Returns a list of ``{name, role, stat, value, units, coverage}`` facts
    (uncertainty facts may add ``pct_of_science``). Best-effort: a band that
    fails to co-locate is skipped, and any failure yields ``[]`` rather than
    disturbing the chart's provenance -- evidence is purely additive.
    """
    if ds is None or not hasattr(ds, "data_vars") or region is None:
        return []
    col_info = col_info or {}
    facts: list[dict] = []
    science_leaf = _evi_leaf(da.name)

    # The masked science mean, for uncertainty-as-percent-of-science. Cos-lat
    # area-weighted (Finding #13) so the pct-of-science ratio divides a
    # weighted band mean by a weighted science mean -- like by like -- rather
    # than mixing a weighted numerator with an unweighted denominator.
    try:
        finite = np.asarray(da.values, dtype="float64")
        finite = finite[np.isfinite(finite)]
        science_mean = area_weighted_mean(da) if finite.size else None
    except Exception:
        science_mean = None

    # Locate the QA-flag variable purely to EXCLUDE it from the context/
    # uncertainty loop below. Its pass rate is not an evidence fact: T55 counts
    # it where the mask is actually applied (aggregation_service) and reports it
    # once as masking provenance, so there is no second computation here that
    # could legitimately disagree with the mask the chart was drawn from.
    qf_leaf = None
    try:
        qf_var, _ = _aggregation_service._resolve_qa_flag_var(ds, da, col_info)
        if qf_var and qf_var in ds.data_vars:
            qf_leaf = _evi_leaf(qf_var)
    except Exception:
        pass

    # Context / uncertainty means -- classify the opened Dataset's bands and
    # keep only High/Medium-confidence quality (uncertainty) and context bands,
    # never the plotted science var, its science siblings, the QA flag (handled
    # above), or unclassified bands.
    try:
        classified = classify_inventory(
            _inventory_records(ds),
            primary_var=col_info.get("primary_var"),
            quality_flag_var=col_info.get("quality_flag_var"),
        )
    except Exception:
        return facts

    for entry in classified:
        role, confidence, name = entry["role"], entry["confidence"], entry["name"]
        leaf, norm = entry["leaf"], _evi_leaf(entry["leaf"])
        if confidence not in ("high", "medium"):
            continue
        if role not in ("quality", "context"):
            continue
        if norm == science_leaf or norm == qf_leaf:
            continue
        if name not in ds.data_vars:
            continue
        is_uncertainty = "uncertainty" in norm
        # Non-uncertainty quality bands (precision, std, sample counts) have no
        # defined summary -- omit rather than invent one.
        if role == "quality" and not is_uncertainty:
            continue
        try:
            fact = _band_mean_fact(
                ds[name], leaf, role, region,
                pct_of_science=science_mean if is_uncertainty else None,
            )
        except Exception:
            fact = None
        if fact:
            facts.append(fact)

    return facts


def _merged_multi_provenance(panels: list[dict]) -> dict:
    """Top-level provenance for a heatmap_multi payload: panel 0's provenance
    with the region names joined across panels — the pre-existing merge shape
    — minus the per-panel ``evidence``/``related_variables`` sections. Those
    are region-specific facts (QA pass rate, cloud fraction); presenting one
    panel's as the whole comparison's would mislead exactly the trust
    judgment they exist to support (T36). Each panel keeps its own."""
    merged = dict(panels[0].get("provenance", {}))
    merged.pop("evidence", None)
    merged.pop("related_variables", None)
    merged["region_name"] = ", ".join(
        panel.get("provenance", {}).get("region_name", "") for panel in panels
    )
    merged["aggregation"] = "single snapshot comparison"
    return merged


def _delivered_scope(region_name: str, start_date: str, end_date: str, agg_meta: dict | None) -> dict:
    """The scope the retrieval actually delivered — region and the data's own
    date span and cadence. Compared against the recorded requested scope by
    the T46 disclosure template."""
    return {
        "region_name": region_name,
        "start_date": start_date,
        "end_date": end_date,
        "cadence": (agg_meta or {}).get("cadence"),
    }


def _requested_scope(handles: list[str]) -> dict | None:
    """The requested scope a composite recorded for any of this chart's source
    handles (T46), or None if none was recorded (a plot over a handle minted
    outside safe_retrieve/point_timeseries — nothing to disclose against)."""
    for handle in handles:
        recorded = scope_registry.get(handle)
        if recorded:
            return recorded
    return None


def _provenance(
    handles: list[str], da, region_name: str, aggregation: str,
    agg_meta: dict | None = None, col_info: dict | None = None,
    ds=None, region: dict | None = None,
) -> dict:
    start_date, end_date = _time_range(da, agg_meta)
    provenance = {
        "variable": da.name or "",
        "start_date": start_date,
        "end_date": end_date,
        "region_name": region_name,
        # T46 silent scope substitution: the delivered scope (always) and the
        # requested scope the composite recorded for this handle (when one was
        # recorded) travel together, so the dispatch layer can disclose a
        # single-day request served by a monthly mean, or a clamped range, in
        # the chat answer -- not just the Metadata tab's fine print.
        "delivered_scope": _delivered_scope(region_name, start_date, end_date, agg_meta),
        "requested_scope": _requested_scope(handles),
        # T42 region fidelity: what kind of region was masked, and the
        # display_name it resolved to -- so a bounding-box "US" or a
        # wrong-place geocode is checkable in the answer, not just the title.
        "region_type": (region or {}).get("region_type"),
        "display_name": (region or {}).get("display_name") or region_name,
        "aggregation": aggregation,
        "units": da.attrs.get("units", ""),
        "source_handles": list(handles),
        **_dataset_facts(col_info),
        "variable_definition": _variable_definition(da, col_info),
        "qa_methodology": _qa_methodology(col_info),
        "related_variables": _related_variables(da, col_info, ds=ds),
        "evidence": _evidence(ds, da, col_info, region),
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
        if agg_meta.get("variable_resolution"):
            # T48: which variable the resolver auto-picked (and its disclosure)
            # rides into provenance, so the dispatch layer can append the
            # deterministic note naming the chosen product + alternatives.
            provenance["variable_resolution"] = agg_meta["variable_resolution"]
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
    col_info: dict | None = None,
    ds=None,
) -> dict:
    aggregation_label = agg_meta["aggregation_label"] if agg_meta else aggregation
    payload["provenance"] = _provenance(
        handles, da, region_name, aggregation_label, agg_meta, col_info, ds=ds, region=region,
    )
    payload["query"] = _query_definition(da, region, aggregation_label, chart_parameters, agg_meta)
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
        except VariableChoiceRequired as e:
            # T49: the file is genuinely ambiguous. Hand the choice to the
            # researcher as a deterministic, uncapped picker (out-of-band), and
            # return the compact P1-bounded refusal to the model as this call's
            # terminal result — the model never sees or re-feeds the full list.
            emit_variable_choice_payload(e.resolution, ds)
            emit_status("Waiting for a variable choice.", stage=STAGE_RENDER)
            return json.dumps({"error": e.mcp_error.to_dict()})
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
                apply_mask_region_type(masked, region)  # T42: disclose boundary_cells self-heal
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
            # T48: the variable-resolution disclosure was stashed on ``da`` by
            # to_dataarray, but masking's ``.where`` above stripped it before
            # aggregate() saw ``masked`` -- so carry it across from the
            # pre-mask array into the meta the provenance is built from.
            if da.attrs.get(VARIABLE_RESOLUTION_ATTR):
                agg_meta["variable_resolution"] = da.attrs[VARIABLE_RESOLUTION_ATTR]
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
                    col_info,
                    ds=ds,
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
            except VariableChoiceRequired as e:
                emit_variable_choice_payload(e.resolution, ds)
                emit_status("Waiting for a variable choice.", stage=STAGE_RENDER)
                return json.dumps({"error": e.mcp_error.to_dict()})
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
                    apply_mask_region_type(masked, region)  # T42: disclose boundary_cells self-heal
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
                    if da.attrs.get(VARIABLE_RESOLUTION_ATTR):
                        agg_meta["variable_resolution"] = da.attrs[VARIABLE_RESOLUTION_ATTR]
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
                        col_info,
                        ds=ds,
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
            multi_payload["provenance"] = _merged_multi_provenance(panels)
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
            # Normalize longitude on the whole opened Dataset before extraction
            # -- the plot_singular/stat convention. A 0..360 global product
            # otherwise rasterizes a western-hemisphere region entirely outside
            # the grid ("No valid data found ... across any time step." for data
            # that plots fine), and normalizing only the extracted array would
            # leave ds's sibling QA-flag variable on 0..360 so QA alignment
            # would hit an empty intersection.
            ds_lon_coord = find_lon_coord(ds)
            if ds_lon_coord:
                ds = _normalize_longitudes(ds, ds_lon_coord)
            da = _open_dataarray(ds, handle=handle, variable=variable)
        except VariableChoiceRequired as e:
            emit_variable_choice_payload(e.resolution, ds)
            emit_status("Waiting for a variable choice.", stage=STAGE_RENDER)
            return json.dumps({"error": e.mcp_error.to_dict()})
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
            apply_mask_region_type(masked, region)  # T42: disclose boundary_cells self-heal

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
                # Same selection seam as _normalize_to_2d: nearest-match on a
                # real coordinate, positional on a coordinate-less dim, and a
                # structured refusal (never an xarray internals crash) on a
                # bad value.
                from utils.plotting import _select_dim_nearest

                for dim_name, value in dim_selector.items():
                    if dim_name in masked.dims:
                        try:
                            masked = _select_dim_nearest(masked, dim_name, value)
                        except MCPToolError as e:
                            return "dimension_choice_required", e.to_dict()
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

            times, values, valid_time_indices = [], [], []
            for i in range(masked.sizes[time_dim]):
                slice_da = masked.isel({time_dim: i})
                try:
                    if stat == "mean":
                        # The regional mean is the cos(latitude) area-weighted
                        # mean (area_weighted_mean) -- the SAME definition the
                        # stats tool uses -- so the trend line and the single-
                        # value stats mean for the identical region can never
                        # disagree. A plain np.nanmean over grid cells over-
                        # weights high latitudes (cells shrink by cos(lat)
                        # toward the poles), biasing continental/global trends.
                        # median/std/max/min stay per-cell (compute_values_stat).
                        value = area_weighted_mean(slice_da)
                    else:
                        value = _aggregation_service.compute_values_stat(slice_da.values, stat)
                except ValueError:
                    continue
                raw_time = masked[time_dim].values[i]
                timestamp = pd.Timestamp(raw_time).isoformat()
                times.append(timestamp)
                values.append(round(float(value), 6))
                valid_time_indices.append(i)

            if not times:
                return "error", f"No valid data found for '{location}' across any time step."

            # Sort by time -- valid_time_indices travels with the same sort
            # key so aggregation_meta's granule_dates/date-range (built from
            # it below) agree with the chart's actual plotted order, even
            # when source timesteps arrive non-chronologically.
            paired = sorted(zip(times, values, valid_time_indices))
            sorted_times, sorted_values, sorted_valid_time_indices = zip(*paired)

            # T32: same aggregation_label/granule_dates/n_granules/cadence
            # summary the heatmap/comparison paths get from aggregate() --
            # conduct_temporal_statistic keeps every timestep instead of
            # reducing over time, so it builds this from its own
            # valid_time_indices rather than calling aggregate().
            agg_meta = _aggregation_service.timeseries_aggregation_meta(
                masked, list(sorted_valid_time_indices), stat, time_dim, col_info=col_info,
            )
            agg_meta["masking"] = masking_provenance

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
            ts_payload["aggregation_meta"] = agg_meta
            _attach_reproducibility(
                ts_payload,
                [handle],
                masked,
                region["name"],
                stat,
                {"chart_type": "timeseries", "location": location},
                agg_meta,
                region,
                col_info,
                ds=ds,
            )
            return None, (ts_payload, variable_name)

        status, result = await asyncio.to_thread(_mask_aggregate_timeseries)
        if status in ("error", "dimension_choice_required"):
            return json.dumps({"error": result})
        ts_payload, variable_name = result
        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(ts_payload, f"{variable_name}_{stat}_{location}")

    return conduct_temporal_statistic
