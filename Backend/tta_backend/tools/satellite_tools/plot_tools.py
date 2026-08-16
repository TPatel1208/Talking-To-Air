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

from tta_backend.config.settings import get_settings
from tta_backend.config.workflow_stages import STAGE_RENDER
from tta_backend.datasets.mask_info import col_info_for_variable, resolve_mask_info
from tta_backend.datasets.variable_roles import classify_inventory, related_variables
from tta_backend.earthdata_mcp.results import (
    CATEGORY_DIMENSION_CHOICE_REQUIRED,
    CATEGORY_TOO_LARGE,
    MCPToolError,
)
from tta_backend.services import scope_registry
from tta_backend.services.artifact_registry import build_artifact_reference
from tta_backend.services.open_handle import (
    OPEN_PIPELINE_VERSION,
    OpenHandleError,
    open_handle,
)
from tta_backend.utils.geo_utils import find_lat_coord, find_lon_coord, vertical_axis_kind
from tta_backend.utils.colormaps import resolve as resolve_colormap
from tta_backend.utils.overlay_render import render_overlay_png
from tta_backend.utils.phase_timing import phase_timer
from tta_backend.utils.plotting import (
    _normalize_to_2d,
    apply_mask_region_type,
    geometry_mask,
    half_cell as _half_cell,
    mask_data_by_geometry,
    RegionResolver,
    sel_bounds as _sel_bounds,
)
from tta_backend.utils.streaming import emit_chart, emit_status
from tta_backend.preprocessing.aggregation_service import (
    VARIABLE_RESOLUTION_ATTR,
    AggregationService,
    VariableChoiceRequired,
    area_weighted_mean,
    fill_match,
)
from tta_backend.preprocessing.variable_choice_builder import emit_variable_choice_payload

logger = logging.getLogger(__name__)

_RENDER_TYPE_TO_ARTIFACT_PREFIX = {
    "heatmap": "map", "heatmap_multi": "cmp", "timeseries": "ts", "profile": "prof",
}

_resolver = RegionResolver()
_aggregation_service = AggregationService()


def overlay_store_dir() -> str:
    """Where rendered overlay PNGs are persisted.

    Overlay PNGs live outside the public output dir on purpose: that one is
    mounted unauthenticated at /outputs (api.py), and overlays must only be
    reachable through the authenticated /chart/{id}/overlay.png route (T23).

    Resolved per call from settings, and *not* a module constant with an
    ``os.makedirs`` beside it, as it was until this became a setting. A constant
    can only be computed at import time, and the value available then was
    ``APP_ROOT``-relative — so merely importing this module created
    ``Backend/overlay_store/`` inside the checkout, and the test suite wrote
    into the developer's own store. Gitignored, so that state survived branch
    switches and stayed invisible to ``git status``. The directory is now
    created at write time by the one function that writes.

    The module's ``OUTPUT_DIR`` constant is gone entirely rather than converted:
    nothing here ever read it, and it created ``Backend/outputs/`` at import for
    no one's benefit. ``api.py`` owns that path now.
    """
    return get_settings().overlay_store_dir

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


# Ceiling on the cells serialized into a payload's `values` grid. Not a display
# limit -- the map draws the full-native-resolution overlay PNG, and this grid
# is what buildCanvasFallbackFrame rasterizes when that PNG is unavailable. So
# it is a payload-size budget, and 8,000 cells is roughly the point past which
# the JSON costs more to ship and parse than the fallback render gains in
# detail. Anything a reader is *told* (statistics, vmin/vmax) is computed on
# the full field before this thinning, so raising or lowering it changes
# fallback sharpness and payload size only -- never a reported number.
#
# It used to be documented as matching a frontend MAX_POINTS constant. That
# constant went away with the per-cell map readout on 2026-08-05; nothing on
# either side has to agree with this number any more.
_MAX_GRID_CELLS = 8_000


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


def _field_statistics(arr: np.ndarray, lats: np.ndarray) -> dict:
    """Summary statistics for the analyzed region, computed on the FULL
    resolution field -- deliberately before ``_downsample_grid`` thins it, for
    the same reason the overlay PNG is rendered before thinning: what the
    reader is told and what fits in the payload are different concerns.

    Computing these from the thinned grid understates the extremes badly, since
    a stride steps over exactly the small-area features that produce them
    (measured on a real TEMPO NO2 L3 scene: true max 9.2418e+17 reported as
    6.9003e+16). ``valid_fraction`` is the one figure that survives thinning
    intact, which is why the discrepancy stayed invisible for so long.
    """
    finite = np.isfinite(arr)
    count = int(finite.sum())
    if count == 0:
        return {"count": 0, "valid_fraction": 0.0}
    values = arr[finite]
    return {
        "count": count,
        "valid_fraction": round(float(count) / float(arr.size), 6),
        "mean": float(f"{_area_weighted_mean(arr, lats, finite):.6e}"),
        "min": float(f"{values.min():.6e}"),
        "max": float(f"{values.max():.6e}"),
    }


def _area_weighted_mean(arr: np.ndarray, lats: np.ndarray, finite: np.ndarray) -> float:
    """cos(latitude)-weighted mean over the finite cells of a (lat, lon) grid.

    Cells shrink toward the poles, so a plain cell average overweights
    high-latitude cells -- on a 0-80 degree grid that is a 31% error. The stats
    and trend tools already weight this way; a map reporting its own mean has
    to agree with them, or one scene answers two different numbers.
    """
    lat_vals = np.asarray(lats, dtype=float)
    if lat_vals.ndim != 1 or lat_vals.size != arr.shape[0]:
        # not a (lat, lon) grid we can weight
        return float(arr[finite].mean(dtype=np.float64))
    weights = np.clip(np.cos(np.deg2rad(lat_vals)), 0.0, None)
    weights = np.where(np.isfinite(weights), weights, 0.0)

    # Reduce row by row against the 1-D weights instead of broadcasting them
    # to the full grid. Every cell shares its row's weight, so the row sum is
    # the same arithmetic -- but the broadcast spelling materialized three
    # more full-size arrays (the expanded weights, the re-indexed values, and
    # their product), and it did so while the caller's copy of the field was
    # still alive. That was the largest single block on the render path.
    #
    # ``dtype=np.float64`` is load-bearing, not decoration: the field is
    # carried at float32 (see _build_heatmap_payload) and this sum runs over
    # millions of same-signed values, where a float32 accumulator drifts
    # inside the six significant digits the payload publishes. This mean is
    # required to agree with the stats and trend tools, so it accumulates in
    # double regardless of how the field is stored.
    contribution = np.where(finite, arr, 0.0).sum(axis=1, dtype=np.float64)
    counted = finite.sum(axis=1, dtype=np.float64)

    total = float((counted * weights).sum())
    if total <= 0.0:
        return float(arr[finite].mean(dtype=np.float64))
    return float((contribution * weights).sum() / total)


def _render_and_store_overlay(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray, lut: list, vmin: float, vmax: float) -> str | None:
    """Render the full-native-resolution overlay PNG and persist it to the
    overlay store. Returns the stored path, or None on failure -- a
    failed render must degrade the chart (no overlay.url; the frontend
    falls back to canvas-from-arrays), never fail the whole tool call.

    The store directory is created here rather than at import: this is the only
    function that writes into it, so nothing else has a reason to bring it into
    existence. A failed mkdir degrades exactly like a failed render, which is
    the same behaviour the read-only-mount case already had."""
    try:
        png_bytes = render_overlay_png(lats, lons, arr, lut, vmin, vmax)
        store = overlay_store_dir()
        os.makedirs(store, exist_ok=True)
        path = os.path.join(store, f"{uuid.uuid4().hex}.png")
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

    # float32, not float64. This is the one line that decides the working size
    # of everything below it -- the percentile bounds, the overlay
    # rasterization and the statistics all copy whatever arrives here, so a
    # needless upcast is paid for several times over. A
    # native-resolution full-day TEMPO field upcast to float64 is what put the
    # backend under the OOM killer on 2026-08-05. A retrieval does not carry
    # float64 precision to begin with, the map is drawn in 8-bit color, and
    # the payload reports six significant digits; the mantissa being dropped
    # here was never anything a reader could see. ``copy=False`` because the
    # common case already arrives as float32 and needs no copy at all.
    #
    # Precision that *is* load-bearing -- the area-weighted mean, which has to
    # agree with the stats and trend tools -- is protected where it is
    # computed, by accumulating in float64 (_area_weighted_mean).
    arr = da.values.astype(np.float32, copy=False)
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

    statistics = _field_statistics(arr, lats_out)

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
        "statistics": statistics,
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
    """Grid dimensions and value range for the compact model-facing summary —
    enough for the agent to describe the chart (T13 story #4) without
    re-reading the raw grid."""
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

    if render_type in ("timeseries", "profile"):
        axis = payload.get("times") if render_type == "timeseries" else payload.get("layers")
        axis = axis or []
        values = [v for v in (payload.get("values") or []) if isinstance(v, (int, float))]
        dims = [len(axis)] if axis else None
        vmin = min(values) if values else None
        vmax = max(values) if values else None
        return dims, vmin, vmax

    return None, payload.get("vmin"), payload.get("vmax")


def _chart_model_summary(payload: dict) -> dict:
    """The compact, model-facing view of a chart payload (T13): render type,
    title, variable, units, dimensions, value range, artifact id, and source
    handles — everything the agent needs to describe the chart and cite it,
    never the raw grid the frontend renders from ``emit_chart``."""
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
    summary.update(_frames_summary(payload))
    return summary


def _frames_summary(payload: dict) -> dict:
    """What the model is told about the chart's time axis (T59 D7/D3).

    Compact, in T13's posture -- the axis itself is up to 60 labeled intervals
    with coverage and QA rates on each, and the agent needs none of that to say
    "you can scrub this". What it does need is the two facts it cannot see any
    other way: that a scrubber exists, and, when one does not, WHICH refusal
    was hit. A disclosure that never leaves the payload is a disclosure nobody
    is told, because the agent only ever sees this summary.
    """
    frames = payload.get("frames")
    if isinstance(frames, dict) and frames.get("frames"):
        block = {
            "n_frames": len(frames["frames"]),
            "cadence": frames.get("cadence"),
            "tier": frames.get("tier"),
            # D6a decision 7's toggle. Compact in T13's posture -- names, not
            # the per-plane disclosure blocks, because the agent needs to know
            # a max exists and not what its ``extent_overstatement`` was.
            "statistics": _scrubbable_statistics(frames),
        }
        planes_unavailable = frames.get("planes_unavailable")
        if planes_unavailable:
            # Only the machine-readable reason: which limit was hit is what
            # lets the agent say "narrow the region", and the sentence itself
            # is already in the payload for the reader.
            block["planes_unavailable"] = planes_unavailable.get("reason")
        return {"frames": block}
    unavailable = payload.get("frames_unavailable")
    return {"frames_unavailable": unavailable} if unavailable else {}


def _scrubbable_statistics(frames: dict) -> list[str]:
    """Which statistics this chart can actually be scrubbed as.

    Derived from the keys that LANDED, never from what was asked for: a plane
    whose write failed, or one evicted since, has a block entry and no ``_key``,
    and an agent that offered it would be promising a toggle the researcher
    watches 404. The same rule the urls are minted under, read off the same
    field.

    ``"mean"`` leads the list although it is never a ``planes`` key -- it is the
    chart's own top-level entry (D6a decision 5), and a list that named only the
    extras would read as though the default were not among them. Ordered by
    ``PLANE_STATISTICS`` rather than by dict order so the agent sees one stable
    vocabulary.
    """
    from tta_backend.preprocessing.frame_stack import PLANE_STATISTICS

    planes = frames.get("planes") or {}
    landed = {
        name for name, plane in planes.items()
        if isinstance(plane, dict) and plane.get("_key")
    }
    if frames.get("_key"):
        landed.add("mean")
    return [name for name in PLANE_STATISTICS if name in landed]


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
    payload (grid/provenance/query/export) is emitted via
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
    _wire_frames_url(payload)

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
    not the dataset it came from. ``col_info`` is whatever
    ``col_info_for_variable`` already resolved for the masking pipeline
    (collections.yaml, matched by the opened granule's short_name attr), so
    this is a read of data already in hand, never a new lookup."""
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
        # T57: where the product sits in its provider's validation lifecycle,
        # and that provider's own caveat. Rides in provenance so it reaches the
        # chart disclosure and, from there, methods.md -- a "don't publish on
        # this" warning is worthless if it only ever appears in chat.
        "maturity": col_info.get("maturity") or "unknown",
        "maturity_note": col_info.get("maturity_note") or "",
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
    # Timed alongside "evidence" because both read the opened Dataset and only
    # their split says which one owns the provenance cost -- this pass builds
    # an inventory and classifies it, which touches metadata rather than
    # values, so a large share here would be a different bug than a large
    # share there.
    with phase_timer("related_variables", from_dataset=ds is not None) as timing:
        if ds is not None and getattr(ds, "data_vars", None):
            variables = _inventory_records(ds)
        timing["variables"] = len(variables or [])
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
    from tta_backend.utils.geo_utils import identify_time

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
    with phase_timer("evidence", science_cells=int(getattr(da, "size", 0))) as _timing:
        facts = _evidence_facts(ds, da, col_info, region, _timing)
    return facts


class _DeferredScienceMean:
    """The plotted variable's area-weighted mean, computed on first use.

    Only ``pct_of_science`` on an *uncertainty* band reads this, and most
    products carry no uncertainty band at all. It used to be computed up front
    for every chart: a live trace on 2026-08-07 (TEMPO NO2, 36 granules)
    classified 2 bands, read 0, produced 0 facts -- and spent **226 s**, 26%
    of an 870 s turn, on this one value that nothing then consumed.

    It is that expensive because the aggregation result stays **dask-backed**
    (verified: ``AggregationService.aggregate`` returns a lazy array), so every
    pass over it re-runs the whole graph back to the source granules. The old
    code made three: ``da.values`` for a finiteness pre-check, then
    ``area_weighted_mean``'s own ``np.asarray(da.values)`` and its weighted
    reduction. Two of those are removed here --

    * the pre-check is redundant: ``area_weighted_mean`` already raises when
      nothing is finite, and ``None`` on that is exactly what this returns;
    * the array is materialized **once** up front, so the two passes inside
      ``area_weighted_mean`` run against memory rather than the dask graph.

    Materializing is bounded and safe: ``da`` here is the *reduced* 2-D field
    the chart is drawn from (~17 M cells, ~68 MB at float32), not the hundreds
    of millions of cells behind it.

    ``None`` means "no finite data" -- the contract the previous code
    expressed by testing ``finite.size`` first, and preserved here by treating
    ``area_weighted_mean``'s ValueError the same way. Evidence is purely
    additive, so a failure here must cost the chart nothing.
    """

    def __init__(self, da):
        self._da = da
        self._value: float | None = None
        self.computed = False

    def __call__(self) -> float | None:
        if not self.computed:
            self.computed = True
            self._value = self._compute()
        return self._value

    def _compute(self) -> float | None:
        try:
            da = self._da
            if getattr(da, "chunks", None) is not None:
                da = da.compute()
            return area_weighted_mean(da)
        except Exception:
            return None


def _evidence_facts(ds, da, col_info: dict | None, region: dict | None, timing: dict) -> list[dict]:
    """The body of :func:`_evidence`, split out only so the timer above can
    wrap it and still record the band counts learned partway through."""
    col_info = col_info or {}
    facts: list[dict] = []
    science_leaf = _evi_leaf(da.name)

    # The masked science mean, for uncertainty-as-percent-of-science. Cos-lat
    # area-weighted (Finding #13) so the pct-of-science ratio divides a
    # weighted band mean by a weighted science mean -- like by like -- rather
    # than mixing a weighted numerator with an unweighted denominator.
    #
    # Deferred (see _DeferredScienceMean): only an *uncertainty* band reads it,
    # and most products have none.
    science_mean = _DeferredScienceMean(da)

    # Locate the QA-flag variable purely to EXCLUDE it from the context/
    # uncertainty loop below. Its pass rate is not an evidence fact: T55 counts
    # it where the mask is actually applied (aggregation_service) and reports it
    # once as masking provenance, so there is no second computation here that
    # could legitimately disagree with the mask the chart was drawn from.
    qf_var = _aggregation_service.qa_flag_variable(ds, da, col_info)
    qf_leaf = _evi_leaf(qf_var) if qf_var else None

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

    # Counted, not just timed: the per-band pass below reads the *unaggregated*
    # Dataset, so "how many bands" is what turns this phase's duration into a
    # per-band cost -- and a per-band cost is the number that says whether the
    # fix is to read fewer bands or to read each one over less data.
    timing["bands_classified"] = len(classified)
    bands_read = 0
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
        bands_read += 1
        timing["band_cells"] = timing.get("band_cells", 0) + int(getattr(ds[name], "size", 0))
        try:
            fact = _band_mean_fact(
                ds[name], leaf, role, region,
                pct_of_science=science_mean() if is_uncertainty else None,
            )
        except Exception:
            fact = None
        if fact:
            facts.append(fact)

    timing["bands_read"] = bands_read
    timing["facts"] = len(facts)
    # Whether the deferred mean was actually forced. The live trace that
    # motivated this recorded bands_read=0 -- so a "true" here alongside a
    # zero band count would mean the deferral had regressed.
    timing["science_mean_computed"] = science_mean.computed
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
        # T60 D10a: region_type is the *rasterization* fact and
        # apply_mask_region_type overwrites it on a self-heal. The shape's
        # own provenance -- "this was a construction, not a named place" --
        # travels beside it or it is lost before the researcher sees it.
        "region_origin": (region or {}).get("region_origin"),
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
        if agg_meta.get("level_resolution"):
            # T58 D5: which layer a physical level resolved to, how much of the
            # analyzed region agrees with that layer, and how far the layer
            # actually sits from what was asked. Travels with the result because
            # "nearest available layer to 300 hPa" and "a 300 hPa map" are
            # different claims, and only the disclosure distinguishes them.
            provenance["level_resolution"] = agg_meta["level_resolution"]
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
    # The outer span. It runs *after* the "render" timer closes and before the
    # tool returns, which is precisely the window that read as a hole in the
    # 2026-08-07 traces (107s on 18 granules, 303s on 36) -- charged to no
    # phase because every phase then was either data work or the model.
    with phase_timer("provenance", science_cells=int(getattr(da, "size", 0))):
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


# ── Frame stack (T59 Phase 5) ─────────────────────────────────────────────────

#: Which gate refusals a reader is told about, and which pass in silence.
#:
#: The split is not squeamishness about noise. These three are requests that
#: COULD have been scrubbed and were not, and they are three different facts: a
#: span past the backstop, a product whose cadence is not registered, and a
#: region too large for the reduction to survive. Someone who asked for a week
#: of TEMPO and got no slider deserves to know which one they hit, because only
#: one of them is fixed by narrowing the map.
#:
#: The rest -- a single granule, everything inside one interval, a field still
#: carrying a vertical axis -- withhold nothing the request implied. There is
#: no time axis to browse, and saying so on every single-granule map would be
#: noise about a feature nobody asked for.
_DISCLOSED_FRAME_REFUSALS = ("cadence_unknown", "span_too_long", "extent_too_large")


def _attach_frames(payload: dict, result, masked, agg_meta: dict) -> None:
    """D7's auto-upgrade: give ``payload`` a browsable time axis, or say why not.

    Additive and optional, always (D15). ``type`` stays ``"heatmap"``, the
    aggregate stays exactly what it was, and every unupgraded consumer --
    ``CHART_TABS``, ``_RENDER_TYPE_TO_ARTIFACT_TYPE``, export's ``heatmap_multi``
    branches -- keeps reading the aggregate and stays correct. That is the
    fallback, not a bug.

    ``result`` is the ``AggregatedResult``, which carries the masked field the
    map was reduced from and the QA counters the mask recorded on the way.
    ``masked`` is the pre-QA geometry-masked array, the only place
    ``region_area`` exists. Both are handed over rather than re-derived: D5a
    binds here, and a number the caller forgets to pass is a second full I/O
    pass over a lazily-opened bundle with nothing saying so.

    Never raises. A frame stack is a rendering convenience, and nothing about
    building one may cost a researcher the chart they asked for -- the same
    posture ``_render_and_store_overlay`` takes toward a failed PNG render.
    """
    from tta_backend.preprocessing import frame_stack as frame_stack_module
    from tta_backend.preprocessing.frame_stack import (
        FRAME_CELL_CEILING,
        MAX_FRAMES,
        PLANE_STATISTICS,
        frame_gate,
        plane_gate,
    )
    from tta_backend.services import frame_store
    from tta_backend.utils.geo_utils import identify_time

    field = getattr(result, "masked", None)
    if field is None:
        return
    cadence = agg_meta.get("cadence") or "unknown"
    time_dim = identify_time(field.data)

    refusal = frame_gate(field.data, time_dim=time_dim, cadence=cadence)
    if refusal is not None:
        if refusal.reason in _DISCLOSED_FRAME_REFUSALS:
            disclosure = {"reason": refusal.reason, "detail": refusal.detail}
            payload["frames_unavailable"] = disclosure
            # Into the recipe as well, not only the render payload: a jsonb row
            # that simply lacks a frame block cannot say whether frames were
            # refused or never attempted.
            payload.setdefault("export", {})["frames"] = {"unavailable": disclosure}
        return

    # D6a's extra planes, behind their own extent limit. A chart above it is
    # NOT refused -- it keeps exactly the mean scrubber it has always had, and
    # only the toggle is withheld -- because Phase 14 measured the three-
    # statistic build being OOM-killed at extents the mean build completes at,
    # and paying for the new tier with the old one is not a trade anyone asked
    # for.
    plane_refusal = plane_gate(field.data, time_dim=time_dim)
    statistics = ("mean",) if plane_refusal is not None else PLANE_STATISTICS

    try:
        with phase_timer("frames", cells_in=int(getattr(field.data, "size", 0))):
            stack = frame_stack_module.build_frame_stack(
                field.data,
                time_dim=time_dim,
                cadence=cadence,
                statistics=statistics,
                # Every scientific quantity off the NATIVE field (D5a). The
                # region's own cos(latitude)-weighted footprint denominates
                # coverage -- without it a frame is denominated on the BOUNDING
                # BOX, and a complete retrieval over the continental US reads
                # 60% covered because the Atlantic counts as missing
                # observations. The counters carry the per-timestep areas
                # finding 12's roll-up needs, and the value bracket the pooled
                # 2-98 histogram would otherwise buy with a second full pass.
                region_area=masked.attrs.get("region_area"),
                qa_counts=field.counts,
            )
        block = frame_store.store_frame_stack(
            stack, pipeline_version=OPEN_PIPELINE_VERSION,
        )
    except Exception:  # noqa: BLE001 — degrade to a chart with no scrubber
        logger.warning("frame_stack_failed", exc_info=True)
        return

    if plane_refusal is not None:
        # Beside the axis rather than beside ``frames_unavailable``, because
        # this chart HAS a scrubber -- what it lacks is the toggle, and a
        # reader who finds no toggle and no reason is in exactly the position
        # ``_DISCLOSED_FRAME_REFUSALS`` exists to keep them out of. Additive,
        # and absent entirely when every plane was built (D15).
        block["planes_unavailable"] = {
            "reason": plane_refusal.reason, "detail": plane_refusal.detail,
        }

    payload["frames"] = block
    payload.setdefault("export", {})["frames"] = {
        # The recipe, beside source_handles/variable/region_name/
        # aggregation_meta, so the jsonb row reproduces the stack rather than
        # merely holding it.
        "spec": {
            "cadence": stack.cadence,
            "tier": stack.tier,
            "n_frames": len(stack.frames),
            "buckets_per_frame": stack.buckets_per_frame,
            "max_frames": MAX_FRAMES,
            "target_cells": FRAME_CELL_CEILING,
            "cells_per_frame": stack.cells_per_frame,
            "coarsen_k": [int(k) for k in stack.coarsen_k],
            "boundary": "pad",
            # What the EXPORT is, unchanged and unchanging (D12). The exported
            # thing is still the period aggregate of the mean, so this field
            # still says so.
            "statistic": "mean",
            # What the SCRUBBER offers, which is a different fact about the
            # same recipe and therefore a different key. Renaming or
            # repurposing the one above would change the meaning of a field
            # already on the wire -- the one thing D15 exists to prevent -- and
            # leave every archived row ambiguous about which sense it meant.
            # Derived from what LANDED, so the row reproduces the stack it
            # holds rather than the one that was asked for.
            "statistics": _scrubbable_statistics(block),
            "dtype": "float32",
            "span": [stack.frames[0].t_start, stack.frames[-1].t_end],
            "pipeline_version": OPEN_PIPELINE_VERSION,
        },
        # D12, said in the row rather than only in JSX: export is the period
        # aggregate and is unchanged by any of this. The frame grid is a
        # rendering resolution -- a 20,000-cell block mean that has already
        # lost 16% of its own p98 at k=8 -- and exporting it would put a
        # downsampled field into someone's paper.
        "exports": "period aggregate",
    }


def _wire_frames_url(payload: dict) -> None:
    """Turn a stored stack's internal ``_key`` into a servable url, now that
    ``_save_chart`` has minted ``chart_id`` -- ``_wire_overlay_urls``' rule, for
    the same reason: the stack is built inside ``asyncio.to_thread``, before a
    ``chart_id`` exists, and the key addresses the blob store directly rather
    than being anything the frontend may hold.

    No ``_key`` means the axis is drawn and the values did not land, which is a
    state D8 already requires the frontend to handle: a labeled, unscrubbable
    axis. It is also exactly what an evicted chart looks like later.

    Each of D6a's extra planes gets the same treatment under the same rule, at
    a path of its own (Phase 13 decision 1). Applied PER PLANE rather than to
    the block as a whole: ``store_frame_stack`` degrades one statistic at a
    time, so an absent url here means that statistic is unavailable and never
    that the chart is broken. The mean's url is untouched by any of it.
    """
    frames = payload.get("frames")
    chart_id = payload.get("chart_id")
    if not chart_id or not isinstance(frames, dict):
        return
    if frames.get("_key"):
        frames["url"] = f"/chart/{chart_id}/frames.f32.gz"
    for statistic, plane in (frames.get("planes") or {}).items():
        if isinstance(plane, dict) and plane.get("_key"):
            plane["url"] = f"/chart/{chart_id}/frames.{statistic}.f32.gz"


# ── Vertical profile (T56) ────────────────────────────────────────────────────

# Post-narrowing ceiling on the cells a profile reduces over (T56 D11). The
# reduction is a mean that dask streams at num_workers=2, so peak memory stays
# chunk-bounded no matter how large this gets -- what runs away is wall clock. A
# regional profile is nowhere near it (a New Jersey box is ~38x38x24 per
# granule); a full-domain one is 137M cells per variable per scan, which would
# spend minutes producing 24 numbers. Refused as the same structured too_large
# error every other size gate raises, so the agent relays "narrow the request"
# instead of the researcher watching a turn time out.
_MAX_PROFILE_CELLS = 40_000_000


def _vertical_dim(da, time_dim: str | None) -> tuple[str | None, list[str]]:
    """The dimension a profile is plotted against: the one left after latitude,
    longitude and time. Returns ``(dim, all_candidates)`` so a caller can refuse
    an ambiguous file by naming what it found rather than picking one."""
    lat_coord = find_lat_coord(da)
    lon_coord = find_lon_coord(da)
    spatial = {name for name in (lat_coord, lon_coord, time_dim) if name}
    candidates = [str(d) for d in da.dims if d not in spatial]
    return (candidates[0] if len(candidates) == 1 else None), candidates


def _vertical_axis_candidates(narrowed, ds, vertical_dim: str, region: dict) -> dict[str, object]:
    """The physical vertical axes available for ``vertical_dim``, keyed
    ``"pressure"``/``"altitude"``, each already narrowed to ``region``.

    The narrowed science array is searched FIRST. That is not a fallback
    ordering, it is where these actually live: the granule declares them as CF
    auxiliary coordinates, so they arrive co-located pixel-for-pixel with the
    science variable, with no second alignment to get wrong.

    They ride the CROP, though, not the ``.where``: xarray's ``.where`` masks
    the data and leaves auxiliary coordinates alone, so an axis taken straight
    off the narrowed array still spans the whole bounding box. This function's
    docstring used to claim otherwise, and the profile's per-layer ``spread``
    was a bounding-box max-minus-min because of it. Callers must restrict to the
    cells the science variable actually kept -- see ``_profile_axis_block``.

    A product that publishes them as ordinary data variables is still
    supported, through the opened Dataset -- but that copy is on the FULL
    granule grid, so it is narrowed here with the same geometry mask and crop
    (``_crop_band_to_region``, the seam the companion-evidence bands already
    use). Reading it straight off ``ds`` would report a "regional" axis
    averaged over a continent.
    """
    found: dict[str, object] = {}
    for name, var in narrowed.coords.items():
        if vertical_dim not in getattr(var, "dims", ()):
            continue
        kind = vertical_axis_kind(var)
        if kind and kind not in found:
            found[kind] = var
    if ds is None or getattr(ds, "data_vars", None) is None:
        return found
    for name, var in ds.data_vars.items():
        if vertical_dim not in getattr(var, "dims", ()) or name == narrowed.name:
            continue
        kind = vertical_axis_kind(var)
        if not kind or kind in found:
            continue
        cropped, in_region = _crop_band_to_region(var, region)
        if cropped is not None and in_region:
            found[kind] = cropped
    return found


def _layer_order(axis_values: list, kind: str) -> str:
    """Whether index 0 of the vertical axis is the TOP of the atmosphere or the
    bottom -- MEASURED off the axis, never assumed.

    TEMPO_O3PROF orders its layers top-down (layer 0 at ~0.175 hPa / 60 km,
    layer 23 at the surface), so an index-increasing-upward plot renders the
    atmosphere inverted. Deriving this from the physical axis instead of pinning
    it means a product ordered the other way is drawn correctly too, and a chart
    reading this key can be right without knowing which product it came from.
    """
    finite = [v for v in axis_values if v is not None]
    if len(finite) < 2 or finite[0] == finite[-1]:
        return "unknown"
    # Pressure falls with height; altitude rises. Both agree on the answer.
    rising = finite[-1] > finite[0]
    if kind == "pressure":
        return "top_down" if rising else "bottom_up"
    return "bottom_up" if rising else "top_down"


def _rounded(values) -> list:
    return [None if not np.isfinite(v) else float(f"{float(v):.6g}") for v in np.asarray(values).ravel()]


def _profile_axis_block(axis_da, vertical_dim: str, kind: str, region_mask=None) -> dict:
    """One physical vertical axis reduced to the analyzed region, with the
    per-layer spread that says how much of an approximation that is.

    Finding 4: the grid is fixed aloft and terrain-following only near the
    surface, so a regional-mean axis is *exact* for the upper layers and
    approximate in the boundary layer. The size of that approximation depends
    entirely on the region -- a box over the Rockies looks nothing like one over
    New Jersey -- so it is measured per request and disclosed, never assumed.
    """
    from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

    # Restricted to the cells the science variable actually kept. The axis rode
    # the crop but not the geometry ``.where`` (see _vertical_axis_candidates),
    # so without this the spread is a BOUNDING-BOX max-minus-min and reports
    # variation from pixels the profile is not about.
    if region_mask is not None:
        axis_da = axis_da.where(region_mask.broadcast_like(axis_da))

    keep = (vertical_dim,)
    mean = reduce_keeping_axes(axis_da, keep=keep, stat="mean")
    highest = reduce_keeping_axes(axis_da, keep=keep, stat="max")
    lowest = reduce_keeping_axes(axis_da, keep=keep, stat="min")
    spread = np.asarray(highest.values, dtype="float64") - np.asarray(lowest.values, dtype="float64")
    return {
        "kind": kind,
        "units": axis_da.attrs.get("units", "") or "",
        "values": _rounded(mean.values),
        "spread": _rounded(spread),
        "layer_order": _layer_order(_rounded(mean.values), kind),
    }


def _per_layer_valid_fraction(masked, vertical_dim: str, region_cells: int | None = None) -> list:
    """What fraction of the analyzed region's cells actually held a value at
    each layer. A profile drawn from one surviving pixel at 60 km and ten
    thousand at the surface is two different measurements sharing an axis, and
    nothing about the line itself says so.

    The denominator is the REGION's cell count (``region_cells``, recorded by
    ``mask_data_by_geometry`` where the rasterized mask exists), times however
    many timesteps were reduced -- not the cropped array's size, which is the
    bounding box. For any region that isn't a rectangle those differ a lot: a
    complete retrieval over the continental US measured 60% by the bounding
    box, because the Atlantic counted as missing observations. Falls back to
    the array's own size when no footprint was recorded, which is exact for a
    box-shaped region and is what the caller had before.
    """
    finite = np.isfinite(masked)
    collapsed = [d for d in masked.dims if d != vertical_dim]
    counts = finite.sum(collapsed).values
    spatial_cells = region_cells if region_cells else None
    if spatial_cells is None:
        spatial_cells = 1
        for dim in collapsed:
            spatial_cells *= int(masked.sizes[dim])
        total = spatial_cells
    else:
        lat_coord, lon_coord = find_lat_coord(masked), find_lon_coord(masked)
        total = spatial_cells
        for dim in collapsed:
            if dim not in (lat_coord, lon_coord):
                total *= int(masked.sizes[dim])
    if total <= 0:
        return [0.0] * int(masked.sizes[vertical_dim])
    # Clamped: the recorded footprint is measured on the pre-QA grid, so a
    # rounding difference must never publish a fraction above 1.
    return [min(1.0, round(float(c) / total, 6)) for c in np.atleast_1d(counts)]


def _resolve_level_selector(masked, level: str, dimension, dimension_value, ds=None):
    """Turn a physical ``level`` request into the ``(dim, value)`` pair the
    existing selection seam takes, plus the disclosure that travels with it.

    Raises the same structured :class:`MCPToolError` every other refusal in this
    module raises, so a level that cannot honestly be resolved reaches the
    researcher as a specific answer rather than as a plausible wrong map.
    """
    from dataclasses import asdict

    from tta_backend.preprocessing.level_resolver import resolve_level
    from tta_backend.utils.geo_utils import identify_time

    time_dim = identify_time(masked)
    vertical_dim, candidates = _vertical_dim(masked, time_dim)
    # Only a selector aimed at the SAME dimension conflicts. Refusing any
    # dimension_value at all locked every product with a vertical axis plus a
    # second non-spatial dimension (an ensemble member, a retrieval attempt, a
    # wavelength) out of `level` entirely -- and the multi-dimension refusal
    # below then told the caller to pass dimension_value first, which this
    # refusal rejected. No call satisfied both.
    if dimension_value is not None and (dimension is None or dimension == vertical_dim):
        raise MCPToolError(
            CATEGORY_DIMENSION_CHOICE_REQUIRED,
            f"Both 'level' ({level!r}) and 'dimension_value' ({dimension_value!r}) were "
            "given for the same vertical dimension. They are different requests -- 'level' "
            "names a physical level, 'dimension_value' names a coordinate value or an index "
            "-- and guessing which was meant is what this parameter exists to avoid.",
            suggestion="Pass 'level' for a physical level, or 'dimension'/'dimension_value' for a layer.",
        )
    if vertical_dim is None and dimension is not None and dimension_value is not None:
        # The other dimensions have a selector, so the vertical one is whatever
        # is left once they are applied.
        remaining = [d for d in candidates if d != dimension]
        vertical_dim = remaining[0] if len(remaining) == 1 else None
        candidates = remaining
    if vertical_dim is None:
        # Deliberately two different refusals. Nothing to select from is a
        # different problem to too many things to select from, and telling a
        # researcher to "narrow the region" when the file has no vertical axis
        # at all would send them somewhere that cannot help.
        if not candidates:
            raise MCPToolError(
                CATEGORY_DIMENSION_CHOICE_REQUIRED,
                f"This variable has no vertical dimension, so {level!r} cannot be "
                "resolved against it.",
                suggestion="Drop the 'level' parameter, or plot a layered product.",
            )
        raise MCPToolError(
            CATEGORY_DIMENSION_CHOICE_REQUIRED,
            f"This variable has more than one non-spatial dimension ({', '.join(candidates)}), "
            f"so {level!r} does not say which one to resolve against.",
            suggestion=f"Select the others with 'dimension'/'dimension_value' first: {', '.join(candidates)}.",
        )

    resolution = resolve_level(masked, vertical_dim, level, dataset=ds)
    return vertical_dim, resolution.selector_value, asdict(resolution)


def _profile_scale_guard(narrowed) -> str | None:
    cells = int(getattr(narrowed, "size", 0))
    if cells <= _MAX_PROFILE_CELLS:
        return None
    return (
        f"This profile would reduce over ~{cells:,} cells, above the "
        f"{_MAX_PROFILE_CELLS:,}-cell limit for a single request."
    )


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
        level: Optional[str] = None,
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
                               On a dimension with no coordinate values this is
                               an integer INDEX, not a physical value — use
                               ``level`` instead to name a physical level.
            level    : A physical vertical level, WITH its units, e.g.
                       "500 hPa", "26 km", "50000 Pa". Use this when the user
                       names an altitude or pressure ("ozone at 26 km"). The
                       units are required: they are what says whether pressure
                       or altitude was meant, so a bare number is refused.
                       Resolves to the nearest available layer and discloses how
                       close it is. Do not combine with ``dimension_value``.

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
        # T60 D14: a composite that cannot be built raises the taxonomy's
        # error naming the offending token -- a ``None`` return could never
        # carry which token failed. Same shape as the open_handle catch.
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
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
            col_info = col_info_for_variable(masked, ds)
            # T58 D7 -- resolve early, select late. The vertical axes ride the
            # time dimension, so aggregate() destroys them; a physical level has
            # to be turned into an index HERE, on the narrowed pre-aggregation
            # array. The index then goes through the SAME selection seam a
            # direct dimension/dimension_value request uses, which is what keeps
            # the two ways of asking for one layer agreeing about n_granules,
            # date range and provenance (Architectural Constraint).
            selected_dim, selected_value = dimension, dimension_value
            level_disclosure = None
            try:
                if level is not None:
                    selected_dim, selected_value, level_disclosure = _resolve_level_selector(
                        masked, level, dimension, dimension_value, ds=ds,
                    )
                aggregation = _aggregation_service.aggregate(
                    masked,
                    variable=variable_name,
                    stat="mean",
                    col_info=col_info,
                    source_ds=ds,
                )
                reduced = next(iter(aggregation.ds.data_vars.values()))
                reduced = _normalize_to_2d(
                    reduced, dim_selector=_build_dim_selector(selected_dim, selected_value),
                )
            except MCPToolError as e:
                return "resolve", None, None, e.to_dict()
            agg_meta = aggregation.meta
            if level_disclosure is not None:
                agg_meta["level_resolution"] = level_disclosure
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

            # T59 D7: the time axis this map collapsed, kept and browsable --
            # after the reproducibility block, because the frame spec lands
            # beside it in ``payload["export"]``.
            _attach_frames(payload, aggregation, masked, agg_meta)

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
            # T60 D14: a composite that cannot be built raises the taxonomy's
            # error naming the offending token -- a ``None`` return could never
            # carry which token failed. Same shape as the open_handle catch.
            try:
                region = await _resolver.aresolve_location(location)
            except MCPToolError as e:
                return json.dumps({"error": e.to_dict()})
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
                col_info = col_info_for_variable(masked, ds)

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
        from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes
        from tta_backend.utils.geo_utils import identify_time

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
        # T60 D14: a composite that cannot be built raises the taxonomy's
        # error naming the offending token -- a ``None`` return could never
        # carry which token failed. Same shape as the open_handle catch.
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
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
                from tta_backend.utils.plotting import _select_dim_nearest

                for dim_name, value in dim_selector.items():
                    if dim_name in masked.dims:
                        try:
                            masked = _select_dim_nearest(masked, dim_name, value)
                        except MCPToolError as e:
                            return "dimension_choice_required", e.to_dict()
            extra_dims = [d for d in masked.dims if d not in (lat_coord, lon_coord, time_dim)]
            if extra_dims:
                from tta_backend.utils.plotting import _dimension_choice_error

                return "dimension_choice_required", _dimension_choice_error(masked, extra_dims[0]).to_dict()

            # T25 masking-execution fix: route through the same shared
            # masking-resolution path aggregate() uses (collections.yaml ->
            # UMM-Var -> CF-attrs precedence, plus the three-tier QA-flag
            # doctrine) instead of a hand-rolled apply_quality_mask(ds=None)
            # call -- this path now actually applies QA-flag masking (via
            # source_ds=ds) and discloses it honestly, the same as
            # plot/stat, rather than a parallel copy that always claimed
            # "not applied".
            col_info = col_info_for_variable(masked, ds)
            masked, masking_provenance = _aggregation_service.resolve_and_mask(
                masked, variable=variable_name, col_info=col_info, source_ds=ds,
            )

            # One reduction, retaining time (T56 Phase 2). The regional mean is
            # the cos(latitude) area-weighted mean -- the SAME definition the
            # stats tool and the vertical profile use -- so a trend line, a
            # single-value stats mean and a profile over the identical region
            # can never disagree. A plain per-cell mean over-weights high
            # latitudes (cells shrink by cos(lat) toward the poles), biasing
            # continental trends; median/std/max/min stay per-cell.
            #
            # This replaced a per-timestep Python loop. The arithmetic is the
            # same, but the loop forced one walk of the lazily-opened bundle's
            # dask graph PER GRANULE -- 36 walks on an ordinary TEMPO day, for
            # 36 numbers.
            reduced = reduce_keeping_axes(masked, keep=(time_dim,), stat=stat)

            times, values, valid_time_indices = [], [], []
            for i, value in enumerate(np.atleast_1d(np.asarray(reduced.values))):
                # NaN is "this timestep had nothing left after masking" -- it
                # drops off the line rather than plotting as a very clean zero.
                # (The companion QA series still covers it: T55 reports every
                # timestep, including the ones the chart cannot show.)
                if not np.isfinite(value):
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


def make_plot_vertical_profile(mcp_tools: dict[str, BaseTool]):
    @tool
    async def plot_vertical_profile(
        handle: Annotated[str, Field(description="An obs_/cube_ handle from a retrieval or transform tool.")],
        location: str,
        title: str = "",
        variable: Optional[str] = None,
    ) -> str:
        """
        Show how a variable is distributed with ALTITUDE — a vertical profile
        line chart, one value per atmospheric layer, averaged over a region and
        a period.

        Use this for a layered product (e.g. an ozone profile) when the user
        asks about the shape of the profile, where in the atmosphere something
        is, the stratospheric maximum, boundary-layer amounts, or "at what
        altitude/pressure". It is also the answer when another chart tool
        refuses with a "dimension_choice_required" error naming a vertical
        dimension: that error means the variable HAS a vertical axis, and this
        tool shows it instead of picking one level out of it.

        Do NOT use it for a single map (plot_singular) or for change over time
        (conduct_temporal_statistic). Latitude, longitude and time are all
        reduced away here — what survives is the vertical axis.

        Args:
            handle   : obs_/cube_ handle from a retrieval or transform tool.
            location : Place name to average over, e.g. 'New Jersey'.
            title    : Chart title. Auto-generated from variable + location if omitted.
            variable : Science variable to profile, for a multi-variable file
                        with no variable chosen at retrieval time.

        Returns:
            JSON string — profile chart payload for the frontend to render.
        """
        from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes
        from tta_backend.utils.geo_utils import identify_time
        from tta_backend.utils.plotting import _dimension_choice_error

        try:
            ds = await open_handle(handle, mcp_tools)
            # Same convention as every other tool: normalize the whole opened
            # Dataset's longitude before extraction, so the science variable
            # and anything still reached through ``ds`` share one coordinate
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
            return json.dumps({"error": f"Failed to open handle '{handle}': {e}"})

        time_dim = identify_time(da)
        vertical_dim, candidates = _vertical_dim(da, time_dim)
        if vertical_dim is None:
            if not candidates:
                return json.dumps({"error": (
                    f"'{da.name}' has no vertical dimension to profile (dims: "
                    f"{list(da.dims)}). Use plot_singular for a map, or "
                    "conduct_temporal_statistic for a time series."
                )})
            # More than one non-spatial, non-time axis: which one is "vertical"
            # is a scientific choice, and the T25 doctrine is to refuse rather
            # than guess. The existing structured error already lists the
            # candidate's values, which is what a caller needs to choose.
            return json.dumps({"error": _dimension_choice_error(da, candidates[0]).to_dict()})

        emit_status("Resolving requested location...", stage=STAGE_RENDER)
        # T60 D14: a composite that cannot be built raises the taxonomy's
        # error naming the offending token -- a ``None`` return could never
        # carry which token failed. Same shape as the open_handle catch.
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if region is None:
            emit_status("Location lookup failed.", stage=STAGE_RENDER)
            return json.dumps({"error": f"Could not geocode location: '{location}'"})

        emit_status("Computing vertical profile...", stage=STAGE_RENDER)

        def _narrow_mask_reduce():
            # CPU-bound narrow -> mask -> reduce chain (T16), run off the event
            # loop via asyncio.to_thread below.
            lat_coord = find_lat_coord(da)
            lon_coord = find_lon_coord(da)
            if lat_coord is None or lon_coord is None:
                return "error", f"Cannot find lat/lon coords. Available: {list(da.coords)}"

            # Narrow BEFORE masking (the standing order), so everything counted
            # during masking describes the region the answer is about. The
            # vertical-axis coordinates ride along on ``narrowed`` -- they share
            # its lat/lon dims, so the same ``.where`` and crop narrow them
            # identically, with no second alignment to get wrong.
            narrowed = mask_data_by_geometry(da, region["geometry"])
            apply_mask_region_type(narrowed, region)  # T42
            # The region's own footprint on this grid, captured before the bbox
            # crop and before masking strips attrs -- the honest denominator for
            # per-layer coverage (see _per_layer_valid_fraction).
            region_cells = narrowed.attrs.get("region_cells")
            narrowed = _sel_bounds(narrowed, lat_coord, lon_coord, region["bounds"])

            # T56 D11: bounded AFTER narrowing, because narrowing is what makes
            # the ordinary regional case small. Peak memory is chunk-bounded
            # either way; what this protects is the researcher's wall clock.
            oversized = _profile_scale_guard(narrowed)
            if oversized:
                return "too_large", MCPToolError(
                    CATEGORY_TOO_LARGE, oversized,
                    suggestion="Narrow the region or the time period and try again.",
                ).to_dict()

            variable_name = narrowed.name or ""
            col_info = col_info_for_variable(narrowed, ds)
            masked, masking_provenance = _aggregation_service.resolve_and_mask(
                narrowed, variable=variable_name, col_info=col_info, source_ds=ds,
            )

            # D4 space-then-time: one area-weighted mean per (timestep, layer)
            # first, then the cadence-bucket-weighted mean over time. The
            # intermediate matrix is deliberately not kept -- that is the
            # curtain view, and nothing here reads it.
            keep = tuple(d for d in (time_dim, vertical_dim) if d and d in masked.dims)
            per_slice = reduce_keeping_axes(masked, keep=keep, stat="mean")

            cadence = _aggregation_service.cadence_for(masked, col_info=col_info)
            if time_dim and time_dim in per_slice.dims:
                spatial_of_matrix = [d for d in per_slice.dims if d != time_dim]
                valid_indices = [
                    i for i, ok in enumerate(
                        np.atleast_1d(np.isfinite(per_slice).any(spatial_of_matrix).values)
                    ) if bool(ok)
                ]
                if not valid_indices:
                    return "error", f"No valid data found for '{location}' at any layer."
                per_slice = per_slice.isel({time_dim: valid_indices})
                profile = _aggregation_service.temporal_mean(per_slice, time_dim, cadence)
            else:
                valid_indices = [0]
                profile = per_slice

            values = _rounded(profile.values)
            if not any(v is not None for v in values):
                return "error", f"No valid data found for '{location}' at any layer."

            axes = _vertical_axis_candidates(masked, ds, vertical_dim, region)
            vertical = {
                kind: _profile_axis_block(axis_da, vertical_dim, kind, region_mask=np.isfinite(masked))
                for kind, axis_da in axes.items()
            }
            # Pressure is the default because it is the axis with the smallest
            # derived-ness caveat (finding 4: exactly constant across the region
            # for the upper half of the layers) and because it puts the
            # tropopause where a reader expects it. Altitude is a frontend
            # toggle over data already in this payload.
            default_axis = "pressure" if "pressure" in vertical else next(iter(vertical), None)

            agg_meta = _aggregation_service.timeseries_aggregation_meta(
                masked, valid_indices, "mean", time_dim, col_info=col_info,
            )
            agg_meta["masking"] = masking_provenance
            if da.attrs.get(VARIABLE_RESOLUTION_ATTR):
                agg_meta["variable_resolution"] = da.attrs[VARIABLE_RESOLUTION_ATTR]

            resolved_title = title or f"{variable_name} vertical profile over {region['name']}"
            payload = {
                "type": "profile",
                "title": resolved_title,
                "variable": variable_name,
                "units": masked.attrs.get("units", ""),
                "stat": "mean",
                "vertical_dim": vertical_dim,
                "layers": list(range(int(masked.sizes[vertical_dim]))),
                "values": values,
                "vertical": vertical,
                "default_axis": default_axis,
                "layer_order": (vertical.get(default_axis) or {}).get("layer_order", "unknown"),
                "valid_fraction": _per_layer_valid_fraction(masked, vertical_dim, region_cells),
                "masking": masking_provenance,
                "aggregation_meta": agg_meta,
            }
            _attach_reproducibility(
                payload,
                [handle],
                masked,
                region["name"],
                "mean",
                {"chart_type": "profile", "location": location},
                agg_meta,
                region,
                col_info,
                ds=ds,
            )
            # A profile's export is self-contained: 24 numbers per axis IS the
            # full resolution, so the arrays travel in the export block rather
            # than being re-derived from the source granule the way a heatmap's
            # are. That also keeps the CSV/PNG working after the handle is
            # evicted -- the one chart type for which that is true.
            payload["export"].update({
                "layers": payload["layers"],
                "values": payload["values"],
                "valid_fraction": payload["valid_fraction"],
                "vertical": vertical,
                "default_axis": default_axis,
                "layer_order": payload["layer_order"],
            })
            return None, (payload, resolved_title)

        status, result = await asyncio.to_thread(_narrow_mask_reduce)
        if status in ("error", "too_large"):
            emit_status("Vertical profile failed.", stage=STAGE_RENDER)
            return json.dumps({"error": result})
        payload, resolved_title = result
        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(payload, resolved_title)

    return plot_vertical_profile
