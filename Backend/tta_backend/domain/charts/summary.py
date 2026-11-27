"""The compact, model-facing view of a chart payload: never the grid itself."""
from __future__ import annotations

from typing import Any

from tta_backend.domain.aggregation.frame_stack import PLANE_STATISTICS


def heatmap_dims(payload: dict[str, Any] | None) -> list[int] | None:
    if not payload:
        return None
    lats, lons = payload.get("lats"), payload.get("lons")
    if isinstance(lats, list) and isinstance(lons, list):
        return [len(lats), len(lons)]
    return None


def summary_dims_and_range(payload: dict[str, Any], render_type: str | None) -> tuple[Any, Any, Any]:
    """Grid dimensions and value range of a payload, by render type."""
    if render_type == "heatmap":
        return heatmap_dims(payload), payload.get("vmin"), payload.get("vmax")

    if render_type == "heatmap_multi":
        if payload.get("mode") == "difference" and isinstance(payload.get("difference"), dict):
            diff = payload["difference"]
            return heatmap_dims(diff), diff.get("vmin"), diff.get("vmax")
        panels = [p for p in (payload.get("panels") or []) if isinstance(p, dict)]
        first = next((p for p in panels if p.get("lats")), None)
        dims = heatmap_dims(first)
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


def scrubbable_statistics(frames: dict[str, Any]) -> list[str]:
    """The statistics a frame block can be scrubbed as: those whose values
    landed (a ``_key``), ``mean`` for the block's own key, in PLANE_STATISTICS order."""
    planes = frames.get("planes") or {}
    landed = {
        name for name, plane in planes.items()
        if isinstance(plane, dict) and plane.get("_key")
    }
    if frames.get("_key"):
        landed.add("mean")
    return [name for name in PLANE_STATISTICS if name in landed]


def frames_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """What the model is told about a chart's time axis: that a scrubber
    exists, or which refusal withheld it."""
    frames = payload.get("frames")
    if isinstance(frames, dict) and frames.get("frames"):
        block = {
            "n_frames": len(frames["frames"]),
            "cadence": frames.get("cadence"),
            "tier": frames.get("tier"),
            "statistics": scrubbable_statistics(frames),
        }
        planes_unavailable = frames.get("planes_unavailable")
        if planes_unavailable:
            block["planes_unavailable"] = planes_unavailable.get("reason")
        return {"frames": block}
    unavailable = payload.get("frames_unavailable")
    return {"frames_unavailable": unavailable} if unavailable else {}


def chart_model_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Render type, title, variable, units, dimensions, value range, chart id
    and source handles of a payload, leaving out what is unset."""
    render_type = payload.get("type")
    grid_dims, vmin, vmax = summary_dims_and_range(payload, render_type)
    raw_metadata = payload.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
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
    summary.update(frames_summary(payload))
    return summary
