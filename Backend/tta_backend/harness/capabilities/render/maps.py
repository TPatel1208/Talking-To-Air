"""``render.map``: a grid as a ``heatmap`` chart, with points and polygons carried as overlays.

A grid with a time axis is shown as its cadence-weighted time mean over the
timesteps that hold data; ``is_aggregated`` and ``aggregation_meta`` say so,
as the current map does. Overlays are additive payload fields:
``overlays.points`` (``id``, ``name``, ``lat``, ``lon``, ``value`` when the
points carry one; at most MAX_OVERLAY_POINTS) and ``overlays.polygons`` (one
GeoJSON FeatureCollection per handle).

The overlay PNG and, for a grid with a browsable time axis, the frame stack
the scrubber plays are kept in the host's stores (``Artifacts.store_overlay``
and ``store_frames``); the payload records what they return, and the host
adds the urls as it stores the chart. The stack is built from the same
masked grid the time mean is taken over, at the same cadence. A grid the
frame gate refuses for its span, cadence or extent carries
``frames_unavailable``; a failed store leaves the map without that medium.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import xarray as xr
from pydantic import Field, JsonValue

from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.charts import media
from tta_backend.domain.charts.heatmap import (
    DIVERGING_SCALE_DISCLOSURE,
    diverging_bounds,
    heatmap_payload,
    time_mean_field,
)
from tta_backend.domain.charts.provenance import attach_reproducibility
from tta_backend.domain.geo import identify_time
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import col_info_for, region_name
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.grids import open_grid_param
from tta_backend.harness.capabilities.render.common import (
    chart_summary,
    point_overlay,
    refuse,
    region_of,
    sources,
    store_and_emit,
)
from tta_backend.harness.capabilities.render.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import polygons_collection
from tta_backend.harness.run_context import Artifacts

__all__ = ["CAPABILITIES", "GROUP", "render_map"]

# The colormap hint the current map payload carries.
CMAP = "Spectral_r"

NO_VALID = "The grid has no valid values after masking, so there is no map to draw."


class MapParams(Params):
    grid: Annotated[str, handle_param("grid")] = Field(description="The grid handle to draw.")
    points: Annotated[str | None, handle_param("points")] = Field(
        default=None, description="Optional points handle, such as monitor locations, carried as an overlay.",
    )
    polygons: Annotated[list[str] | None, handle_param("polygons")] = Field(
        default=None, description="Optional polygons handles, such as smoke extent, carried as an overlay.",
    )
    style: Literal["sequential", "diverging"] = Field(
        default="sequential", description="diverging centres the colour scale on zero, for a difference.",
    )
    title: str | None = Field(default=None, max_length=200, description="Chart title; made from the variable and area when omitted.")


def _build(
    data: xr.DataArray, *, style: str, title: str | None, place: str, col_info: dict[str, Any] | None,
    masking: dict[str, Any], handles: list[str], region: dict[str, Any] | None,
    artifacts: Artifacts, region_area: float | None,
) -> dict[str, Any] | None:
    """The heatmap payload with its overlay and frame stack stored, or None when no cell holds a value."""
    service = AggregationService()
    time_dim = identify_time(data)
    if time_dim is not None and time_dim in data.dims:
        cadence = service.cadence_for(data, col_info=col_info)
        field, valid = time_mean_field(data, time_dim, cadence)
        agg_meta = service.timeseries_aggregation_meta(data, valid, "mean", time_dim, col_info=col_info)
    else:
        field, valid = data, [0]
        agg_meta = service.timeseries_aggregation_meta(data, valid, "mean", None, col_info=col_info)
    if not valid or not bool(np.isfinite(np.asarray(field.values, dtype="float64")).any()):
        return None
    agg_meta["masking"] = masking

    variable = str(data.name or "")
    units = str(data.attrs.get("units", "") or "")
    field = field.rename(variable or None)
    if units:
        field = field.assign_attrs(units=units)
    is_aggregated = agg_meta["n_granules"] > 1
    if title:
        resolved_title = title
    elif is_aggregated:
        resolved_title = f"{variable} {agg_meta['title_suffix']} over {place}"
    else:
        resolved_title = f"{variable} over {place}"

    diverging = style == "diverging"
    payload = heatmap_payload(
        field, resolved_title, variable, units,
        diverging=diverging,
        value_range=diverging_bounds(np.asarray(field.values)) if diverging else None,
        scale_disclosure=DIVERGING_SCALE_DISCLOSURE if diverging else None,
        overlay_renderer=artifacts.store_overlay,
    )
    payload["cmap"] = CMAP
    if region is not None:
        payload["bounds"] = list(region["bounds"])
    payload["aggregation_meta"] = agg_meta
    payload["is_aggregated"] = is_aggregated
    attach_reproducibility(
        payload, handles, field, place,
        agg_meta["aggregation_label"] if is_aggregated else "single snapshot",
        {"chart_type": "heatmap", "style": style, "cmap": CMAP},
        agg_meta, region, col_info,
    )
    # After the reproducibility block: the frame recipe lands beside it in ``export``.
    media.attach_frames(
        payload, data, cadence=agg_meta.get("cadence") or "unknown", store=artifacts.store_frames,
        region_area=region_area,
    )
    return payload


async def _render_map(call: Call[MapParams]) -> CapabilityResult:
    p = call.params
    record = call.handle("grid")
    grid = await open_grid_param(call, "grid")
    extra = [str(d) for d in grid.data.dims if d not in ("time", "lat", "lon")]
    if extra:
        raise refuse(call, "wrong_dims", param="grid", dims=", ".join(str(d) for d in grid.data.dims))

    payload = await call.compute(
        _build, grid.data, style=p.style, title=p.title, place=region_name(record),
        col_info=col_info_for(record), masking=dict(grid.masking), handles=list(call.input_ids()),
        region=region_of(record), artifacts=call.env.context.artifacts, region_area=grid.region_area,
    )
    if payload is None:
        return CapabilityResult(status="insufficient_evidence", reason=NO_VALID, values={"situation": "too_little_valid"})

    disclosures: list[str] = []
    meta = payload["aggregation_meta"]
    if payload["is_aggregated"]:
        disclosures.append(f"The map shows the time mean of the grid ({meta['aggregation_label']}).")
    overlays: dict[str, Any] = {}
    if p.points is not None:
        points, total = point_overlay(await call.open("points"))
        overlays["points"] = {"handle": p.points, "points": points}
        held = f"the first {len(points)} of {total}" if total > len(points) else f"all {total}"
        disclosures.append(f"The chart data carries {held} points as an overlay (overlays.points).")
    if p.polygons:
        collections = [polygons_collection(c) for c in await call.open_each("polygons")]
        overlays["polygons"] = [
            {"handle": handle, "geojson": collection} for handle, collection in zip(p.polygons, collections, strict=True)
        ]
        features = sum(len(c["features"]) for c in collections)
        disclosures.append(f"The chart data carries {features} polygon feature(s) as an overlay (overlays.polygons).")
    if overlays:
        payload["overlays"] = overlays
    payload["provenance"]["sources"] = sources((record, *call.handles("points"), *call.handles("polygons")))

    ref = await store_and_emit(call, payload)
    values: dict[str, JsonValue] = {"style": p.style, "time_mean": bool(payload["is_aggregated"])}
    if "points" in overlays:
        values["overlay_points"] = len(overlays["points"]["points"])
    if "polygons" in overlays:
        values["overlay_polygons"] = sum(len(o["geojson"]["features"]) for o in overlays["polygons"])
    return call.done(charts=(ref.id,), summary=chart_summary(payload), values=values, disclosures=tuple(disclosures))


render_map = Capability(
    name="render.map",
    description="Draw a grid as a map, optionally carrying points and polygons as overlays; a grid with a time axis is shown as its time mean.",
    params=MapParams,
    output_kind=None,
    label="Drawing a map",
    method="The grid was drawn as a map ({summary.title}).",
    retrieves=False,
    retry_safe=True,
    run=_render_map,
)

CAPABILITIES = (render_map,)
