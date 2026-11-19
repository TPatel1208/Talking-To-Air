"""A map's stored media: its frame stack (the time scrubber) and its overlay PNG.

The values live in host stores, not in the payload. A store hands back the
fields the payload records: the frame block with ``_key`` per statistic whose
values landed, and the overlay's ``_path`` (and ``_values`` for its value
raster). Neither is servable as is; once the chart has an id,
``wire_media_urls`` adds the ``url`` the frontend fetches, under
``/chart/{chart_id}/frames*.f32.gz`` and ``/chart/{chart_id}/overlay.png``.

Nothing here imports xarray at module scope: the frame store, which the API
imports at startup, builds its blocks with ``axis_block``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from tta_backend.infrastructure.phase_timing import phase_timer

logger = logging.getLogger(__name__)

# The period mean is plane 0 of a stored stack; the frames follow in axis order.
PERIOD_INDEX = 0

# The gate refusals a reader is told about, as ``frames_unavailable``. Each is
# a request that could have been scrubbed and was not: a span past the limit,
# a product with no registered cadence, a region too large to reduce. The
# others (a single granule, one interval, a field with a vertical axis) withhold
# nothing the request implied, so they pass in silence.
DISCLOSED_FRAME_REFUSALS = ("cadence_unknown", "span_too_long", "extent_too_large")

# Stores a stack's values and returns its block (``axis_block`` plus ``_key``s).
FrameStore = Callable[[Any], dict[str, Any]]


def stored_shape(source: Any) -> list[int]:
    """The stored layout ``(1 + n_frames, ny, nx)`` of a stack or one of its planes."""
    n_frames, ny, nx = (int(v) for v in source.values.shape)
    return [n_frames + 1, ny, nx]


def axis_block(stack: Any, *, pipeline_version: str) -> dict[str, Any]:
    """Everything about a frame stack except its values, as plain JSON values.

    ``planes`` is present only when the stack carries extra statistics.
    """
    from tta_backend.domain.aggregation.frame_stack import POOLED_SCALE_BASIS

    block: dict[str, Any] = {
        "frames": [
            {
                "t_start": frame.t_start,
                "t_end": frame.t_end,
                "n_granules": int(frame.n_granules),
                "valid_fraction": float(frame.valid_fraction),
                # None, never zero, when QA was not counted.
                "qa_pass_rate": None if frame.qa_pass_rate is None else float(frame.qa_pass_rate),
                "statistics": {k: float(v) if k != "count" else int(v) for k, v in frame.statistics.items()},
            }
            for frame in stack.frames
        ],
        # The realized cell count, never the ceiling that was asked for.
        "shape": stored_shape(stack),
        "dtype": "float32",
        "period_index": PERIOD_INDEX,
        "lats": [round(float(v), 6) for v in stack.lats],
        "lons": [round(float(v), 6) for v in stack.lons],
        "cells_per_frame": int(stack.cells_per_frame),
        "cadence": stack.cadence,
        "tier": stack.tier,
        "buckets_per_frame": int(stack.buckets_per_frame),
        "coarsen_k": [int(k) for k in stack.coarsen_k],
        "delta": stack.delta,
        # Whether the planes in the blob average to the period plane; neither
        # delta bounds the other.
        "frame_grid_delta": stack.frame_grid_delta,
        "value_range": None if stack.value_range is None else [float(v) for v in stack.value_range],
        # The scrubber's scale only, never the Map tab's.
        "scale_basis": POOLED_SCALE_BASIS,
        # Frames are never served across a version change.
        "pipeline_version": pipeline_version,
    }
    planes = {name: plane_block(plane) for name, plane in (getattr(stack, "planes", None) or {}).items()}
    if planes:
        block["planes"] = planes
    return block


def plane_block(plane: Any) -> dict[str, Any]:
    """One extra statistic's own fields; the axis and grid are the stack's."""
    return {
        # The plane's own 2-98 range: a max plane on the mean's scale saturates.
        "value_range": None if plane.value_range is None else [float(v) for v in plane.value_range],
        "frame_grid_delta": plane.frame_grid_delta,
        # Max plane only; None elsewhere.
        "extent_overstatement": plane.extent_overstatement,
    }


def attach_frames(
    payload: dict[str, Any],
    field: Any,
    *,
    cadence: str,
    store: FrameStore,
    region_area: float | None = None,
    qa_counts: dict[str, Any] | None = None,
) -> None:
    """Give ``payload`` a browsable time axis built from ``field``, or say why not.

    ``field`` is the masked field the map's time mean was taken over, every
    timestep included. ``region_area`` is the cos(latitude)-weighted footprint
    of the analysed region (without it coverage is measured against the
    bounding box); ``qa_counts`` the quality mask's counters (without them
    each frame's ``qa_pass_rate`` is None).

    Additive: ``type`` stays ``heatmap`` and the map is unchanged. A disclosed
    gate refusal sets ``frames_unavailable`` and ``export.frames.unavailable``;
    another refusal, or a failed build, leaves the payload without frames.
    A block whose values did not land carries no ``_key``. Never raises.
    """
    from tta_backend.domain.aggregation import frame_stack as frame_stack_module
    from tta_backend.domain.charts.summary import scrubbable_statistics
    from tta_backend.domain.geo import identify_time

    time_dim = identify_time(field)
    refusal = frame_stack_module.frame_gate(field, time_dim=time_dim, cadence=cadence)
    if refusal is not None:
        if refusal.reason in DISCLOSED_FRAME_REFUSALS:
            disclosure = {"reason": refusal.reason, "detail": refusal.detail}
            payload["frames_unavailable"] = disclosure
            # In the recipe too: a stored row that lacks a frame block cannot
            # otherwise say whether frames were refused or never attempted.
            payload.setdefault("export", {})["frames"] = {"unavailable": disclosure}
        return
    # frame_gate refuses a missing time axis.
    assert time_dim is not None

    # Above the plane limit the chart keeps its mean scrubber without the
    # max/min toggle.
    plane_refusal = frame_stack_module.plane_gate(field, time_dim=time_dim)
    statistics = ("mean",) if plane_refusal is not None else frame_stack_module.PLANE_STATISTICS

    try:
        with phase_timer("frames", cells_in=int(getattr(field, "size", 0))):
            stack = frame_stack_module.build_frame_stack(
                field, time_dim=time_dim, cadence=cadence, statistics=statistics,
                region_area=region_area, qa_counts=qa_counts,
            )
        block = store(stack)
    except Exception:  # noqa: BLE001 -- degrade to a chart with no scrubber
        logger.warning("frame_stack_failed", exc_info=True)
        return

    if plane_refusal is not None:
        # Beside the axis: the chart has a scrubber and lacks only the toggle.
        block["planes_unavailable"] = {"reason": plane_refusal.reason, "detail": plane_refusal.detail}

    payload["frames"] = block
    payload.setdefault("export", {})["frames"] = {
        # The recipe, so the stored row reproduces the stack rather than only holding it.
        "spec": {
            "cadence": stack.cadence,
            "tier": stack.tier,
            "n_frames": len(stack.frames),
            "buckets_per_frame": stack.buckets_per_frame,
            "max_frames": frame_stack_module.MAX_FRAMES,
            "target_cells": frame_stack_module.FRAME_CELL_CEILING,
            "cells_per_frame": stack.cells_per_frame,
            "coarsen_k": [int(k) for k in stack.coarsen_k],
            "boundary": "pad",
            # What the export is: the period aggregate of the mean.
            "statistic": "mean",
            # What the scrubber offers, from what landed.
            "statistics": scrubbable_statistics(block),
            "dtype": "float32",
            "span": [stack.frames[0].t_start, stack.frames[-1].t_end],
            "pipeline_version": block.get("pipeline_version"),
        },
        # Exports are the period aggregate; the frame grid is a downsampled rendering.
        "exports": "period aggregate",
    }


def _wire_overlay_url(overlay: dict[str, Any] | None, url: str) -> None:
    if isinstance(overlay, dict) and overlay.get("_path"):
        overlay["url"] = url
        # The same url takes ?vmin=&vmax=&cmap= to recolour the stored value raster.
        if isinstance(overlay.get("_values"), dict) and overlay["_values"].get("_path"):
            overlay["rescalable"] = True


def wire_overlay_urls(payload: dict[str, Any]) -> None:
    """Give each stored overlay (``_path``) of a chart with a ``chart_id`` its servable ``url``."""
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


def wire_frames_url(payload: dict[str, Any]) -> None:
    """Give a stored frame block (``_key``) of a chart with a ``chart_id`` its
    servable ``url``, and each landed plane its own. No ``_key`` means the
    values did not land: the frontend shows a labelled axis that cannot be scrubbed."""
    frames = payload.get("frames")
    chart_id = payload.get("chart_id")
    if not chart_id or not isinstance(frames, dict):
        return
    if frames.get("_key"):
        frames["url"] = f"/chart/{chart_id}/frames.f32.gz"
    for statistic, plane in (frames.get("planes") or {}).items():
        if isinstance(plane, dict) and plane.get("_key"):
            plane["url"] = f"/chart/{chart_id}/frames.{statistic}.f32.gz"


def wire_media_urls(payload: dict[str, Any]) -> None:
    """Both of the above. Writes into the ``frames`` and ``overlay`` blocks in place."""
    wire_overlay_urls(payload)
    wire_frames_url(payload)


def carries_media(payload: dict[str, Any]) -> bool:
    """Whether the payload holds stored media that needs a url once the chart has an id."""
    frames = payload.get("frames")
    overlay = payload.get("overlay")
    stored_frames = isinstance(frames, dict) and (
        bool(frames.get("_key"))
        or any(isinstance(p, dict) and p.get("_key") for p in (frames.get("planes") or {}).values())
    )
    return stored_frames or (isinstance(overlay, dict) and bool(overlay.get("_path")))
