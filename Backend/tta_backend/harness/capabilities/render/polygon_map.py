"""``render.polygon_map``: polygons such as smoke extent drawn on a map with no grid.

Stored as a ``polygon_map`` chart. ``polygons`` holds one GeoJSON
FeatureCollection per input handle (``handle``, ``geojson``), ``bounds`` is
``[west, south, east, north]`` over every feature so the frontend can frame
the map (None when there are no features), and ``overlays.points`` carries an
optional points handle the way ``render.map`` does. No polygons at all is a
real answer, so it draws an empty map and says so.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, JsonValue

from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.render.common import chart_summary, point_overlay, sources, store_and_emit
from tta_backend.harness.capabilities.render.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import polygons_collection

__all__ = ["CAPABILITIES", "GROUP", "render_polygon_map"]

MAX_POLYGON_HANDLES = 20

NO_POLYGONS = "There are no polygons to show, so the map is empty."


class PolygonMapParams(Params):
    polygons: Annotated[list[str], handle_param("polygons")] = Field(
        min_length=1, max_length=MAX_POLYGON_HANDLES,
        description="One or more polygons handles, such as smoke extent, drawn on the same map.",
    )
    points: Annotated[str | None, handle_param("points")] = Field(
        default=None, description="Optional points handle, such as monitor locations, carried as an overlay.",
    )
    title: str | None = Field(default=None, max_length=200, description="Map title; made from the polygons when omitted.")


def _positions(coordinates: Any) -> list[tuple[float, float]]:
    """Every (lon, lat) position nested in a GeoJSON coordinates value."""
    if not isinstance(coordinates, list | tuple) or not coordinates:
        return []
    first = coordinates[0]
    if isinstance(first, int | float) and len(coordinates) >= 2:
        return [(float(coordinates[0]), float(coordinates[1]))]
    found: list[tuple[float, float]] = []
    for part in coordinates:
        found.extend(_positions(part))
    return found


def _geometries(geometry: Any) -> list[Any]:
    if not isinstance(geometry, dict):
        return []
    if geometry.get("type") == "GeometryCollection":
        return [g for part in geometry.get("geometries") or [] for g in _geometries(part)]
    return [geometry.get("coordinates")]


def _bounds(collections: list[dict[str, Any]]) -> list[float] | None:
    positions = [
        position
        for collection in collections
        for feature in collection["features"]
        if isinstance(feature, dict)
        for coordinates in _geometries(feature.get("geometry"))
        for position in _positions(coordinates)
    ]
    if not positions:
        return None
    lons = [lon for lon, _ in positions]
    lats = [lat for _, lat in positions]
    return [min(lons), min(lats), max(lons), max(lats)]


async def _render_polygon_map(call: Call[PolygonMapParams]) -> CapabilityResult:
    p = call.params
    records = call.handles("polygons")
    collections = [polygons_collection(c) for c in await call.open_each("polygons")]
    count = sum(len(c["features"]) for c in collections)
    title = p.title or "Polygon map"
    handles = list(call.input_ids())

    payload: dict[str, Any] = {
        "type": "polygon_map",
        "title": title,
        "polygons": [
            {"handle": handle, "geojson": collection} for handle, collection in zip(p.polygons, collections, strict=True)
        ],
        "bounds": _bounds(collections),
        "feature_count": count,
    }
    disclosures: list[str] = []
    if count == 0:
        disclosures.append(NO_POLYGONS)
    else:
        disclosures.append(f"The chart data holds {count} polygon feature(s) (polygons).")
    if p.points is not None:
        points, total = point_overlay(await call.open("points"))
        payload["overlays"] = {"points": {"handle": p.points, "points": points}}
        held = f"the first {len(points)} of {total}" if total > len(points) else f"all {total}"
        disclosures.append(f"The chart data carries {held} points as an overlay (overlays.points).")

    records_used = (*records, *call.handles("points"))
    payload["provenance"] = {"source_handles": handles, "sources": sources(records_used)}
    payload["export"] = {"type": "polygon_map", "chart_parameters": {"chart_type": "polygon_map"}, "source_handles": handles}
    payload["metadata"] = {"source_handles": handles}

    ref = await store_and_emit(call, payload)
    values: dict[str, JsonValue] = {"features": count, "handles": len(collections)}
    if "overlays" in payload:
        values["overlay_points"] = len(payload["overlays"]["points"]["points"])
    return call.done(charts=(ref.id,), summary=chart_summary(payload), values=values, disclosures=tuple(disclosures))


render_polygon_map = Capability(
    name="render.polygon_map",
    description="Draw polygons, such as smoke extent, on a map when there is no grid; no polygons gives an empty map.",
    params=PolygonMapParams,
    output_kind=None,
    label="Drawing a polygon map",
    method="The polygons were drawn as a map holding {features} feature(s).",
    retrieves=False,
    retry_safe=True,
    run=_render_polygon_map,
)

CAPABILITIES = (render_polygon_map,)
