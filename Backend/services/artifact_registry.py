from __future__ import annotations

from typing import Any

from models.artifact import (
    ArtifactReference,
    ComparisonArtifactMetadata,
    MapArtifactMetadata,
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
}


def build_artifact_reference(payload: dict[str, Any]) -> ArtifactReference | None:
    """Build a typed, validated ArtifactReference from a chart-style payload.

    Returns None if the payload's render type has no T06 artifact mapping
    (e.g. a plain table). Raises pydantic.ValidationError if the payload is
    missing fields its artifact type requires.
    """
    artifact_type = _RENDER_TYPE_TO_ARTIFACT_TYPE.get(payload.get("type"))
    if artifact_type is None:
        return None

    metadata = _build_metadata(artifact_type, payload)
    return ArtifactReference(
        id=payload.get("chart_id"),
        type=artifact_type,
        title=payload.get("title"),
        metadata=metadata.model_dump(),
    )


def _build_metadata(artifact_type: str, payload: dict[str, Any]):
    source_handles = (payload.get("metadata") or {}).get("source_handles", [])
    if artifact_type == "map":
        return MapArtifactMetadata(
            bbox=payload.get("bounds"),
            variable=payload.get("variable"),
            units=payload.get("units"),
            colorbar={"vmin": payload.get("vmin"), "vmax": payload.get("vmax")},
            source_handles=source_handles,
        )
    if artifact_type == "comparison":
        panels = [
            {
                "handle": (panel.get("metadata") or {}).get("source_handles", [None])[0],
                "title": panel.get("title"),
            }
            for panel in payload.get("panels", [])
        ]
        return ComparisonArtifactMetadata(
            mode=payload.get("mode", "n-panel"),
            panels=panels,
            source_handles=source_handles,
        )
    series = (payload.get("metadata") or {}).get("series")
    if series:
        return TimeseriesArtifactMetadata(series=series, source_handles=source_handles)
    return TimeseriesArtifactMetadata(
        series=[{
            "label": payload.get("title"),
            "source_kind": "satellite",
        }],
        source_handles=source_handles,
    )
