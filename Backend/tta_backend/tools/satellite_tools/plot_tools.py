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
import json
import logging
import uuid
import numpy as np
from langchain.tools import tool
from langchain_core.tools import BaseTool
from typing import Annotated, Any, List, Optional
import xarray as xr
from pydantic import Field

from tta_backend.services import admission
from tta_backend.services import overlay_store
from tta_backend.infrastructure.settings import get_settings
from tta_backend.config.workflow_stages import STAGE_RENDER
from tta_backend.datasets.mask_info import col_info_for_variable
from tta_backend.earthdata_mcp.results import (
    CATEGORY_DIMENSION_CHOICE_REQUIRED,
    CATEGORY_NO_DATA,
    CATEGORY_TOO_LARGE,
    MCPToolError,
)
from tta_backend.services import scope_registry
from tta_backend.services import variable_choice_registry
from tta_backend.services.artifact_registry import build_artifact_reference
from tta_backend.services.open_handle import (
    OPEN_PIPELINE_VERSION,
    OpenHandleError,
    open_handle,
)
from tta_backend.domain.geo import find_lat_coord, find_lon_coord
from tta_backend.utils.colormaps import resolve as resolve_colormap
from tta_backend.utils.overlay_render import render_overlay
from tta_backend.infrastructure.phase_timing import phase_timer
from tta_backend.domain.masking.dimensions import _normalize_to_2d
from tta_backend.domain.masking.geometry import (
    apply_mask_region_type,
    half_cell,
    mask_data_by_geometry,
    sel_bounds as _sel_bounds,
)
from tta_backend.utils.plotting import RegionResolver
from tta_backend.utils.streaming import emit_chart, emit_status
from tta_backend.domain.aggregation.aggregation_service import (
    VARIABLE_RESOLUTION_ATTR,
    AggregatedResult,
    AggregationService,
    VariableChoiceRequired,
)
from tta_backend.domain.earthdata.opening import normalize_longitudes
from tta_backend.domain.charts.heatmap import heatmap_payload
from tta_backend.domain.charts.profile import (
    profile_export,
    profile_payload,
    profile_scale_guard as _profile_scale_guard,
    reduce_profile,
    vertical_dim as _vertical_dim,
)
from tta_backend.domain.charts.provenance import attach_reproducibility, provenance
from tta_backend.domain.charts.summary import (
    chart_model_summary as _chart_model_summary,
    scrubbable_statistics as _scrubbable_statistics,
)
from tta_backend.domain.charts.timeseries import timeseries_payload, timeseries_points
from tta_backend.domain.charts import heatmap as _heatmap
from tta_backend.domain.charts import profile as _profile
from tta_backend.domain.charts import provenance as _provenance_module
from tta_backend.domain.charts import summary as _summary
from tta_backend.preprocessing.variable_choice_builder import emit_variable_choice_payload

logger = logging.getLogger(__name__)

# Imported from here by name by other modules and tests.
_half_cell = half_cell
_normalize_longitudes = normalize_longitudes
_MAX_GRID_CELLS = _heatmap.MAX_GRID_CELLS
_area_weighted_mean = _heatmap.area_weighted_mean
_downsample_grid = _heatmap.downsample_grid
_field_statistics = _heatmap.field_statistics
_percentile_bounds = _heatmap.percentile_bounds
_MAX_PROFILE_CELLS = _profile.MAX_PROFILE_CELLS
_layer_order = _profile.layer_order
_per_layer_valid_fraction = _profile.per_layer_valid_fraction
_profile_axis_block = _profile.profile_axis_block
_rounded = _profile.rounded
_vertical_axis_candidates = _profile.vertical_axis_candidates
_DeferredScienceMean = _provenance_module.DeferredScienceMean
_band_mean_fact = _provenance_module.band_mean_fact
_band_time_mean = _provenance_module.band_time_mean
_crop_band_to_region = _provenance_module.crop_band_to_region
_dataset_facts = _provenance_module.dataset_facts
_delivered_scope = _provenance_module.delivered_scope
_evi_leaf = _provenance_module.evi_leaf
_evidence = _provenance_module.evidence
_evidence_facts = _provenance_module.evidence_facts
_inventory_records = _provenance_module.inventory_records
_merged_multi_provenance = _provenance_module.merged_multi_provenance
_qa_methodology = _provenance_module.qa_methodology
_query_definition = _provenance_module.query_definition
_related_variables = _provenance_module.related_variables_view
_time_range = _provenance_module.time_range
_variable_definition = _provenance_module.variable_definition
_frames_summary = _summary.frames_summary
_heatmap_dims = _summary.heatmap_dims
_summary_dims_and_range = _summary.summary_dims_and_range

_RENDER_TYPE_TO_ARTIFACT_PREFIX = {
    "heatmap": "map", "heatmap_multi": "cmp", "timeseries": "ts", "profile": "prof",
}

_resolver = RegionResolver()
_aggregation_service = AggregationService()


def _resolve_failure_status(error: Any) -> str:
    """The render status for a typed refusal from the resolve/aggregate stage:
    an area outside the retrieved data is not a variable choice."""
    if isinstance(error, dict) and error.get("category") == CATEGORY_NO_DATA:
        return "No data in the requested area."
    return "Visualization needs a variable or dimension choice."


def overlay_store_dir() -> str:
    """Where rendered overlay PNGs are persisted.

    Overlays must only be reachable through the authenticated
    /chart/{id}/overlay.png route, so nothing serves this directory
    statically.

    Resolved per call from settings (not at import time), so importing this
    module never creates a directory. The writer creates it on demand.
    """
    return get_settings().overlay_store_dir

def _render_and_store_overlay(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray, lut: list[Any], vmin: float, vmax: float) -> dict[str, Any]:
    """Render the full-native-resolution overlay PNG and persist it, with the
    value raster it was drawn from, to the overlay store. Returns the overlay
    fields to record: ``_path`` for the PNG and ``_values`` (``_path``, ``lo``,
    ``hi``) for the raster the route recolors at a requested scale.

    Returns ``{}`` on failure -- a failed render must degrade the chart (no
    overlay.url; the frontend falls back to canvas-from-arrays), never fail
    the whole tool call. A raster that fails to store only drops ``_values``:
    the chart keeps its overlay and loses the full-resolution recolor.

    The store creates its directory on first write and evicts older overlays
    to stay under its byte cap (services/overlay_store.py). A failed write
    degrades exactly like a failed render."""
    try:
        rendered = render_overlay(lats, lons, arr, lut, vmin, vmax)
        fields: dict[str, Any] = {"_path": overlay_store.write_overlay(rendered.png)}
    except Exception:
        logger.warning("overlay_render_failed", exc_info=True)
        return {}
    raster = rendered.value_raster
    if raster is not None:
        try:
            raster_path = overlay_store.write_overlay(raster.png)
        except Exception:
            logger.warning("overlay_value_raster_store_failed", exc_info=True)
        else:
            fields["_values"] = {"_path": raster_path, "lo": raster.lo, "hi": raster.hi}
    return fields


def _da_to_heatmap_payload(
    da: xr.DataArray, title: str, variable: str, units: str, *,
    diverging: bool = False, render_overlay: bool = False, value_range: tuple[float, float] | None = None,
    scale_disclosure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Rasterization and downsampling are where a slow plot spends its CPU
    # time. Timed at this one seam so every caller is covered.
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
    da: xr.DataArray, title: str, variable: str, units: str, *,
    diverging: bool = False, render_overlay: bool = False, value_range: tuple[float, float] | None = None,
    scale_disclosure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # The overlay PNG is rendered and stored here, through the overlay store;
    # the payload itself is built in domain.charts.heatmap.
    return heatmap_payload(
        da, title, variable, units,
        diverging=diverging, value_range=value_range, scale_disclosure=scale_disclosure,
        colormap=resolve_colormap(variable, diverging=diverging),
        overlay_renderer=_render_and_store_overlay if render_overlay else None,
    )



def _wire_overlay_url(overlay: dict[str, Any] | None, url: str) -> None:
    if isinstance(overlay, dict) and overlay.get("_path"):
        overlay["url"] = url
        # The same url takes ?vmin=&vmax=&cmap= to recolor the stored value
        # raster; a chart drawn before the raster existed has none.
        if isinstance(overlay.get("_values"), dict) and overlay["_values"].get("_path"):
            overlay["rescalable"] = True


def _wire_overlay_urls(payload: dict[str, Any]) -> None:
    """Turn each rendered overlay's internal `_path` into a servable `url`,
    now that `_save_chart` has minted `chart_id` -- the render happens
    earlier (admission.run_heavy, before chart_id exists) with a `_path` that
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


def _save_chart(payload: dict[str, Any], name: str) -> str:
    """Emit the full chart payload out-of-band (frontend chart/artifact
    pipeline) and return a compact model-facing summary.

    Mints a stable artifact id for render types the artifact vocabulary
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

    prefix = _RENDER_TYPE_TO_ARTIFACT_PREFIX.get(payload.get("type") or "")
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


def _open_dataarray(ds: xr.Dataset, handle: str | None = None, variable: str | None = None) -> xr.DataArray:
    """Pick the science variable off an opened Dataset, unmasked.

    Resolution: explicit ``variable`` -> the choice recorded for
    ``handle`` at retrieval time -> the file's only data variable -> the
    collection's pinned ``primary_var`` (via short_name attr) -> a structured
    candidate-listing error (AggregationService.to_dataarray).
    """
    return _aggregation_service.to_dataarray(
        ds, recorded_variable=variable_choice_registry.get(handle), variable=variable,
    )


def _build_dim_selector(dimension: str | None, dimension_value: float | None) -> dict[str, Any] | None:
    """A single-entry {dim_name: value} selector from a tool's optional
    ``dimension``/``dimension_value`` params, or None when no dimension was
    named -- the shape domain.masking.dimensions._normalize_to_2d's dim_selector expects."""
    if dimension is None or dimension_value is None:
        return None
    return {dimension: dimension_value}


def _requested_scope(handles: list[str]) -> dict[str, Any] | None:
    """The requested scope a composite recorded for any of this chart's source
    handles, or None if none was recorded (a plot over a handle minted
    outside safe_retrieve/point_timeseries — nothing to disclose against)."""
    for handle in handles:
        recorded = scope_registry.get(handle)
        if recorded:
            return recorded
    return None


def _provenance(
    handles: list[str], da: xr.DataArray, region_name: str, aggregation: str,
    agg_meta: dict[str, Any] | None = None, col_info: dict[str, Any] | None = None,
    ds: xr.Dataset | None = None, region: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return provenance(
        handles, da, region_name, aggregation, agg_meta, col_info, ds=ds, region=region,
        requested_scope=_requested_scope(handles),
    )


def _attach_reproducibility(
    payload: dict[str, Any],
    handles: list[str],
    da: xr.DataArray,
    region_name: str,
    aggregation: str,
    chart_parameters: dict[str, Any] | None = None,
    agg_meta: dict[str, Any] | None = None,
    region: dict[str, Any] | None = None,
    col_info: dict[str, Any] | None = None,
    ds: xr.Dataset | None = None,
) -> dict[str, Any]:
    return attach_reproducibility(
        payload, handles, da, region_name, aggregation, chart_parameters, agg_meta, region, col_info, ds,
        requested_scope=_requested_scope(handles),
    )



# ── Frame stack ───────────────────────────────────────────────────────────────

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


def _attach_frames(payload: dict[str, Any], result: AggregatedResult, masked: xr.DataArray, agg_meta: dict[str, Any]) -> None:
    """Give ``payload`` a browsable time axis, or say why not.

    Purely additive: ``type`` stays ``"heatmap"`` and the aggregate is
    unchanged, so consumers that ignore frames stay correct.

    ``result`` is the ``AggregatedResult``, which carries the masked field the
    map was reduced from and the QA counters the mask recorded on the way.
    ``masked`` is the pre-QA geometry-masked array, the only place
    ``region_area`` exists. Both are passed in because re-deriving them would
    be a second full read of a lazily-opened bundle.

    Never raises: a failed frame stack must never cost the user the chart
    (same as ``_render_and_store_overlay``).
    """
    from tta_backend.domain.aggregation import frame_stack as frame_stack_module
    from tta_backend.domain.aggregation.frame_stack import (
        FRAME_CELL_CEILING,
        MAX_FRAMES,
        PLANE_STATISTICS,
        frame_gate,
        plane_gate,
    )
    from tta_backend.services import frame_store
    from tta_backend.domain.geo import identify_time

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
    # frame_gate refuses a missing time axis, so past it there always is one.
    assert time_dim is not None

    # Max/min planes have their own extent limit, because building all three
    # statistics can OOM where the mean alone fits. Above it the chart keeps
    # its mean scrubber; only the toggle is withheld.
    plane_refusal = plane_gate(field.data, time_dim=time_dim)
    statistics = ("mean",) if plane_refusal is not None else PLANE_STATISTICS

    try:
        with phase_timer("frames", cells_in=int(getattr(field.data, "size", 0))):
            stack = frame_stack_module.build_frame_stack(
                field.data,
                time_dim=time_dim,
                cadence=cadence,
                statistics=statistics,
                # Every quantity comes from the native field. Coverage uses the
                # region's own footprint, not the bounding box (otherwise a full
                # US retrieval reads 60% covered because the Atlantic counts as
                # missing). The counters also save a second pass for the 2-98
                # histogram.
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
        # Next to the axis, not ``frames_unavailable``: the chart has a
        # scrubber, it just lacks the toggle, and that should come with a
        # reason. Absent when every plane was built.
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
            # What the EXPORT is, unchanged and unchanging. The exported
            # thing is still the period aggregate of the mean, so this field
            # still says so.
            "statistic": "mean",
            # What the SCRUBBER offers: a separate key, so the field above
            # keeps its meaning on archived rows. Derived from what landed,
            # not what was asked for.
            "statistics": _scrubbable_statistics(block),
            "dtype": "float32",
            "span": [stack.frames[0].t_start, stack.frames[-1].t_end],
            "pipeline_version": OPEN_PIPELINE_VERSION,
        },
        # Recorded in the row, not just the UI: exports are the period
        # aggregate. The frame grid is a downsampled rendering and must never
        # end up in someone's paper.
        "exports": "period aggregate",
    }


def _wire_frames_url(payload: dict[str, Any]) -> None:
    """Turn a stored stack's internal ``_key`` into a servable url, now that
    ``_save_chart`` has minted ``chart_id`` -- ``_wire_overlay_urls``' rule, for
    the same reason: the stack is built inside ``admission.run_heavy``, before a
    ``chart_id`` exists, and the key addresses the blob store directly rather
    than being anything the frontend may hold.

    No ``_key`` means the values didn't land: the frontend shows a labeled
    axis that can't be scrubbed (same as an evicted chart).

    Each extra plane gets its own url under the same rule, per plane, so a
    missing url means only that statistic is unavailable.
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


# ── Vertical profile ──────────────────────────────────────────────────────────

def _resolve_level_selector(
    masked: xr.DataArray, level: str, dimension: str | None, dimension_value: float | None, ds: xr.Dataset | None = None,
) -> tuple[str | None, float, dict[str, Any]]:
    """Turn a physical ``level`` request into the ``(dim, value)`` pair the
    existing selection seam takes, plus the disclosure that travels with it.

    Raises the same structured :class:`MCPToolError` every other refusal in this
    module raises, so a level that cannot honestly be resolved reaches the
    researcher as a specific answer rather than as a plausible wrong map.
    """
    from dataclasses import asdict

    from tta_backend.domain.aggregation.level_resolver import resolve_level
    from tta_backend.domain.geo import identify_time

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


# ── Tools ─────────────────────────────────────────────────────────────────────


def make_plot_singular(mcp_tools: dict[str, BaseTool]) -> BaseTool:
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
            # QA-flag variable (still reachable through ds) share one
            # coordinate convention. Doing this
            # only on the extracted DataArray would leave ds's flag variable
            # on the original 0..360 convention, and da.where(qf.isin(...))
            # would align on an empty intersection instead of masking.
            ds_lon_coord = find_lon_coord(ds)
            if ds_lon_coord:
                ds = _normalize_longitudes(ds, ds_lon_coord)
            da = _open_dataarray(ds, handle=handle, variable=variable)
        except VariableChoiceRequired as e:
            # Genuinely ambiguous file: send the researcher a variable picker
            # out-of-band and return a compact refusal to the model, which
            # never sees the full list.
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
        # A composite region that can't be built raises an error naming the
        # bad token (a None return couldn't say which one).
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if region is None:
            emit_status("Location lookup failed.", stage=STAGE_RENDER)
            return json.dumps({"error": f"Could not geocode location: '{location}'"})

        emit_status("Generating visualization...", stage=STAGE_RENDER)

        def _mask_aggregate_payload() -> tuple[str | None, dict[str, Any] | None, str | None, Any]:
            # CPU-bound mask -> aggregate -> payload chain: run off the
            # event loop via admission.run_heavy below -- which also bounds how
            # many may run at once -- so a large grid doesn't
            # freeze every other concurrent stream for its duration.
            try:
                lat_coord = find_lat_coord(da)
                lon_coord = find_lon_coord(da)
                if lat_coord is None or lon_coord is None:
                    raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")
                masked = _normalize_longitudes(da, lon_coord)
                masked = mask_data_by_geometry(masked, region["geometry"])
                apply_mask_region_type(masked, region)  # discloses a boundary_cells self-heal
                bounds = region["bounds"]  # (minx, miny, maxx, maxy)
                masked = _sel_bounds(masked, lat_coord, lon_coord, bounds)
            except Exception as e:
                return "mask", None, None, f"Masking failed: {e}"

            units = masked.attrs.get("units", "")
            variable_name = str(masked.name or "")
            col_info = col_info_for_variable(masked, ds)
            # Resolve early, select late: aggregate() destroys the vertical
            # axes, so turn a physical level into an index here. It then uses
            # the same selection path as a direct dimension request, so both
            # ways of asking for a layer agree on n_granules, dates and
            # provenance.
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
            # The variable-resolution disclosure was stashed on ``da`` by
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

            # The time axis this map collapsed, kept and browsable --
            # after the reproducibility block, because the frame spec lands
            # beside it in ``payload["export"]``.
            _attach_frames(payload, aggregation, masked, agg_meta)

            return None, payload, resolved_title, None

        stage, payload, resolved_title, error_message = await admission.run_heavy(_mask_aggregate_payload)
        if stage == "mask":
            emit_status("Visualization failed while processing map bounds.", stage=STAGE_RENDER)
            return json.dumps({"error": error_message})
        if stage == "resolve":
            emit_status(_resolve_failure_status(error_message), stage=STAGE_RENDER)
            return json.dumps({"error": error_message})
        if stage == "payload":
            emit_status("Visualization failed while building chart data.", stage=STAGE_RENDER)
            return json.dumps({"error": error_message})

        # Every failing stage returned above; a None stage carries both.
        assert payload is not None and resolved_title is not None
        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(payload, resolved_title)

    return plot_singular


def make_plot_multiple(mcp_tools: dict[str, BaseTool]) -> BaseTool:
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
        for handle, location in zip(handles, locations, strict=True):
            try:
                ds = await open_handle(handle, mcp_tools)
                # See plot_singular: normalize the whole Dataset's longitude
                # before extraction, so da and its sibling QA-flag variable
                # (still reachable through ds) share one coordinate
                # convention.
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
            # A composite region that can't be built raises an error naming the
            # bad token (a None return couldn't say which one).
            try:
                region = await _resolver.aresolve_location(location)
            except MCPToolError as e:
                return json.dumps({"error": e.to_dict()})
            if region is None:
                emit_status("Location lookup failed.", stage=STAGE_RENDER)
                return json.dumps({"error": f"Could not geocode location: '{location}'"})

            def _mask_aggregate_panel(
                da: xr.DataArray = da, ds: xr.Dataset | None = ds, region: dict[str, Any] = region, handle: str = handle,
                location: str = location, variable_name: str = variable_name,
            ) -> tuple[str | None, dict[str, Any] | None, str | None, Any]:
                # CPU-bound mask -> aggregate -> payload chain, run off
                # the event loop via admission.run_heavy below, which also bounds
                # how many such reductions may hold memory at once.
                try:
                    lat_coord = find_lat_coord(da)
                    lon_coord = find_lon_coord(da)
                    if lat_coord is None or lon_coord is None:
                        raise ValueError(f"Cannot find lat/lon coords. Available: {list(da.coords)}")
                    masked = _normalize_longitudes(da, lon_coord)
                    masked = mask_data_by_geometry(masked, region["geometry"])
                    apply_mask_region_type(masked, region)  # discloses a boundary_cells self-heal
                except Exception as e:
                    return "mask", None, None, f"Masking failed for '{location}': {e}"

                bounds = region["bounds"]
                masked = _sel_bounds(masked, lat_coord, lon_coord, bounds)

                resolved_variable_name = str(masked.name or variable_name)
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

            stage, panel, resolved_variable_name, error_message = await admission.run_heavy(_mask_aggregate_panel)
            if stage == "mask":
                emit_status("Visualization failed while processing map bounds.", stage=STAGE_RENDER)
                return json.dumps({"error": error_message})
            if stage == "resolve":
                emit_status(_resolve_failure_status(error_message), stage=STAGE_RENDER)
                return json.dumps({"error": error_message})
            if stage == "payload":
                emit_status("Visualization failed while building chart data.", stage=STAGE_RENDER)
                return json.dumps({"error": error_message})

            # Every failing stage returned above; a None stage carries both.
            assert panel is not None and resolved_variable_name is not None
            variable_name = resolved_variable_name
            panels.append(panel)

        multi_payload: dict[str, Any] = {"type": "heatmap_multi", "title": title or f"{variable_name} Comparison", "panels": panels}
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


def make_conduct_temporal_statistic(mcp_tools: dict[str, BaseTool]) -> BaseTool:
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
        from tta_backend.domain.aggregation.regional_reduction import reduce_keeping_axes
        from tta_backend.domain.geo import identify_time

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
        # A composite region that can't be built raises an error naming the
        # bad token (a None return couldn't say which one).
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if region is None:
            return json.dumps({"error": f"Could not resolve location: '{location}'"})

        emit_status("Computing time series...", stage=STAGE_RENDER)

        def _mask_aggregate_timeseries() -> tuple[str | None, Any]:
            # CPU-bound mask -> per-timestep aggregate -> payload chain
            # run off the event loop via admission.run_heavy below, which also
            # bounds how many such reductions may hold memory at once.
            masked = mask_data_by_geometry(da, region["geometry"])
            apply_mask_region_type(masked, region)  # discloses a boundary_cells self-heal

            lat_coord = find_lat_coord(masked)
            lon_coord = find_lon_coord(masked)
            if lat_coord is None or lon_coord is None:
                return "error", f"Cannot find lat/lon coords. Available: {list(masked.coords)}"
            bounds = region["bounds"]
            masked = _sel_bounds(masked, lat_coord, lon_coord, bounds)

            variable_name = str(masked.name or "")
            if stat not in AggregationService._STAT_FUNCS:
                return "error", f"Unknown stat '{stat}'. Use: mean, median, max, min, std"

            dim_selector = _build_dim_selector(dimension, dimension_value)
            if dim_selector:
                # Same selection seam as _normalize_to_2d: nearest-match on a
                # real coordinate, positional on a coordinate-less dim, and a
                # structured refusal (never an xarray internals crash) on a
                # bad value.
                from tta_backend.domain.masking.dimensions import _select_dim_nearest

                for dim_name, value in dim_selector.items():
                    if dim_name in masked.dims:
                        try:
                            masked = _select_dim_nearest(masked, dim_name, value)
                        except MCPToolError as e:
                            return "dimension_choice_required", e.to_dict()
            extra_dims = [d for d in masked.dims if d not in (lat_coord, lon_coord, time_dim)]
            if extra_dims:
                from tta_backend.domain.masking.dimensions import _dimension_choice_error

                return "dimension_choice_required", _dimension_choice_error(masked, str(extra_dims[0])).to_dict()

            # Use the same masking path as aggregate() (collections.yaml ->
            # UMM-Var -> CF attrs), so QA masking is actually applied (via
            # source_ds=ds) and disclosed the same way as plot/stat.
            col_info = col_info_for_variable(masked, ds)
            masked, masking_provenance = _aggregation_service.resolve_and_mask(
                masked, variable=variable_name, col_info=col_info, source_ds=ds,
            )

            # One reduction, retaining time. The regional mean is
            # the cos(latitude) area-weighted mean -- the SAME definition the
            # stats tool and the vertical profile use -- so a trend line, a
            # single-value stats mean and a profile over the identical region
            # can never disagree. A plain per-cell mean over-weights high
            # latitudes (cells shrink by cos(lat) toward the poles), biasing
            # continental trends; median/std/max/min stay per-cell.
            #
            # Vectorized: a per-timestep loop would walk the lazy dask graph
            # once per granule.
            reduced = reduce_keeping_axes(masked, keep=(time_dim,), stat=stat)

            # Finite timesteps only, sorted by time with their indices, so the
            # aggregation meta's dates follow the plotted order.
            sorted_times, sorted_values, sorted_valid_time_indices = timeseries_points(
                reduced.values, masked[time_dim].values,
            )
            if not sorted_times:
                return "error", f"No valid data found for '{location}' across any time step."

            # Built from the kept timesteps, since this chart does not reduce over time.
            agg_meta = _aggregation_service.timeseries_aggregation_meta(
                masked, list(sorted_valid_time_indices), stat, time_dim, col_info=col_info,
            )
            agg_meta["masking"] = masking_provenance

            ts_payload = timeseries_payload(
                f"{variable_name} {stat} over {location}", variable_name,
                masked.attrs.get("units", ""), stat, sorted_times, sorted_values,
            )
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

        status, result = await admission.run_heavy(_mask_aggregate_timeseries)
        if status in ("error", "dimension_choice_required"):
            return json.dumps({"error": result})
        ts_payload, variable_name = result
        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(ts_payload, f"{variable_name}_{stat}_{location}")

    return conduct_temporal_statistic


def make_plot_vertical_profile(mcp_tools: dict[str, BaseTool]) -> BaseTool:
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
        from tta_backend.domain.geo import identify_time
        from tta_backend.domain.masking.dimensions import _dimension_choice_error

        try:
            ds = await open_handle(handle, mcp_tools)
            # Same convention as every other tool: normalize the whole opened
            # Dataset's longitude before extraction, so the science variable
            # and anything still reached through ``ds`` share one coordinate
            # convention.
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
            # is a scientific choice, so refuse rather than guess. The existing structured error already lists the
            # candidate's values, which is what a caller needs to choose.
            return json.dumps({"error": _dimension_choice_error(da, candidates[0]).to_dict()})

        emit_status("Resolving requested location...", stage=STAGE_RENDER)
        # A composite region that can't be built raises an error naming the
        # bad token (a None return couldn't say which one).
        try:
            region = await _resolver.aresolve_location(location)
        except MCPToolError as e:
            return json.dumps({"error": e.to_dict()})
        if region is None:
            emit_status("Location lookup failed.", stage=STAGE_RENDER)
            return json.dumps({"error": f"Could not geocode location: '{location}'"})

        emit_status("Computing vertical profile...", stage=STAGE_RENDER)

        def _narrow_mask_reduce() -> tuple[str | None, Any]:
            # CPU-bound narrow -> mask -> reduce chain, run off the event
            # loop via admission.run_heavy below, which also bounds how many
            # such reductions may hold memory at once.
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
            apply_mask_region_type(narrowed, region)
            # The region's own footprint on this grid, captured before the bbox
            # crop and before masking strips attrs -- the honest denominator for
            # per-layer coverage (see _per_layer_valid_fraction).
            region_cells = narrowed.attrs.get("region_cells")
            narrowed = _sel_bounds(narrowed, lat_coord, lon_coord, region["bounds"])

            # Bounded AFTER narrowing, because narrowing is what makes
            # the ordinary regional case small. Peak memory is chunk-bounded
            # either way; what this protects is the researcher's wall clock.
            oversized = _profile_scale_guard(narrowed)
            if oversized:
                return "too_large", MCPToolError(
                    CATEGORY_TOO_LARGE, oversized,
                    suggestion="Narrow the region or the time period and try again.",
                ).to_dict()

            variable_name = str(narrowed.name or "")
            col_info = col_info_for_variable(narrowed, ds)
            masked, masking_provenance = _aggregation_service.resolve_and_mask(
                narrowed, variable=variable_name, col_info=col_info, source_ds=ds,
            )

            # Space then time (domain.charts.profile.reduce_profile): one
            # area-weighted mean per (timestep, layer), then the cadence-weighted
            # mean over the timesteps that held data.
            cadence = _aggregation_service.cadence_for(masked, col_info=col_info)
            profile = reduce_profile(
                masked, vertical_dim=vertical_dim, time_dim=time_dim, cadence=cadence,
                ds=ds, region=region, region_cells=region_cells,
            )
            if profile is None:
                return "error", f"No valid data found for '{location}' at any layer."

            agg_meta = _aggregation_service.timeseries_aggregation_meta(
                masked, profile.valid_indices, "mean", time_dim, col_info=col_info,
            )
            agg_meta["masking"] = masking_provenance
            if da.attrs.get(VARIABLE_RESOLUTION_ATTR):
                agg_meta["variable_resolution"] = da.attrs[VARIABLE_RESOLUTION_ATTR]

            resolved_title = title or f"{variable_name} vertical profile over {region['name']}"
            payload = profile_payload(
                profile, title=resolved_title, variable=variable_name, units=masked.attrs.get("units", ""),
                masking=masking_provenance, agg_meta=agg_meta,
            )
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
            # The arrays travel in the export block: they are the full resolution,
            # so the export keeps working after the handle is evicted.
            payload["export"].update(profile_export(payload))
            return None, (payload, resolved_title)

        status, result = await admission.run_heavy(_narrow_mask_reduce)
        if status in ("error", "too_large"):
            emit_status("Vertical profile failed.", stage=STAGE_RENDER)
            return json.dumps({"error": result})
        payload, resolved_title = result
        # Every failing stage returned above; a None stage carries both.
        assert payload is not None and resolved_title is not None
        emit_status("Preparing response...", stage=STAGE_RENDER)
        return _save_chart(payload, resolved_title)

    return plot_vertical_profile
