from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from tta_backend.models.artifact import (
    ArtifactReference,
    ComparisonArtifactMetadata,
    MapArtifactMetadata,
    ProfileArtifactMetadata,
    TimeseriesArtifactMetadata,
)

# Chart payloads carry an internal render "type" (used by Plotly on the
# frontend and by export_service's PNG/CSV dispatch) that predates the T06
# artifact vocabulary. This maps that render type to the artifact type shown
# in the gallery — the render type itself is left untouched.
_RENDER_TYPE_TO_ARTIFACT_TYPE = {
    "heatmap": "map",
    "heatmap_multi": "comparison",
    "timeseries": "timeseries",
    "profile": "profile",
}


def build_artifact_reference(payload: dict[str, Any]) -> ArtifactReference | None:
    """Build a typed, validated ArtifactReference from a chart-style payload.

    Returns None if the payload's render type has no T06 artifact mapping
    (e.g. a plain table). Raises pydantic.ValidationError if the payload is
    missing fields its artifact type requires.

    Built with ``model_validate`` over the payload's raw values rather than
    keyword constructors: the values are unchecked until pydantic checks them,
    and that check (not a cast) is what rejects a payload missing a field.
    """
    artifact_type = _RENDER_TYPE_TO_ARTIFACT_TYPE.get(payload.get("type") or "")
    if artifact_type is None:
        return None

    metadata = _build_metadata(artifact_type, payload)
    return ArtifactReference.model_validate({
        "id": payload.get("chart_id"),
        "type": artifact_type,
        "title": payload.get("title"),
        "metadata": metadata.model_dump(),
    })


def _build_metadata(artifact_type: str, payload: dict[str, Any]) -> BaseModel:
    source_handles = (payload.get("metadata") or {}).get("source_handles", [])
    if artifact_type == "map":
        return MapArtifactMetadata.model_validate({
            "bbox": payload.get("bounds"),
            "variable": payload.get("variable"),
            "units": payload.get("units"),
            "colorbar": {"vmin": payload.get("vmin"), "vmax": payload.get("vmax")},
            "source_handles": source_handles,
        })
    if artifact_type == "comparison":
        panels = [
            {
                "handle": (panel.get("metadata") or {}).get("source_handles", [None])[0],
                "title": panel.get("title"),
            }
            for panel in payload.get("panels", [])
        ]
        return ComparisonArtifactMetadata.model_validate({
            "mode": payload.get("mode", "n-panel"),
            "panels": panels,
            "source_handles": source_handles,
        })
    if artifact_type == "profile":
        default_axis = payload.get("default_axis") or ""
        axis = (payload.get("vertical") or {}).get(default_axis) or {}
        return ProfileArtifactMetadata.model_validate({
            "variable": payload.get("variable"),
            "units": payload.get("units"),
            "layer_count": len(payload.get("layers") or []),
            "vertical_axis": default_axis,
            "vertical_units": axis.get("units", ""),
            "layer_order": payload.get("layer_order", "unknown"),
            "source_handles": source_handles,
            "masking": payload.get("masking"),
        })
    series = (payload.get("metadata") or {}).get("series") or [{
        "label": payload.get("title"),
        "source_kind": "satellite",
    }]
    return TimeseriesArtifactMetadata.model_validate({
        "series": series,
        "source_handles": source_handles,
        "stats": payload.get("stats"),
        "coverage": payload.get("coverage"),
        "exceedance_dates": payload.get("exceedance_dates"),
        "masking": payload.get("masking"),
    })
