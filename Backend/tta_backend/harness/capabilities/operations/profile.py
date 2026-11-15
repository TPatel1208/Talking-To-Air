"""``operations.vertical_profile``: a layered grid as a table of per-layer regional means.

The reduction is the current vertical profile's (``domain.charts.profile``):
one area-weighted mean per timestep and layer, then the cadence-weighted mean
over the timesteps that held data. The table has ``layer``, ``value`` and
``valid_fraction``, then for each physical axis the grid carries ``{kind}``
(its regional mean), ``{kind}_spread`` (max minus min across the region) and
``{kind}_unit``; ``render.profile`` draws it.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import Field, JsonValue

from tta_backend.domain.aggregation.aggregation_service import AggregationService
from tta_backend.domain.charts.profile import profile_scale_guard, profile_table, reduce_profile, vertical_dim
from tta_backend.domain.geo import identify_time
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import col_info_for, region_name
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.grids import open_grid_param
from tta_backend.harness.capabilities.operations.common import derived, insufficient, refuse
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.trace import FailureCategory, StepFailure

__all__ = ["CAPABILITIES", "vertical_profile"]

NAME = "operations.vertical_profile"

NO_VALID = "Too little valid data: the grid has no valid values at any layer after masking."
AXIS_NOTE = (
    "The {kind} axis is the regional mean of each layer's {kind}; {kind}_spread is how much it varies across the area."
)


class VerticalProfileParams(Params):
    grid: Annotated[str, handle_param("grid")] = Field(description="A grid with one vertical dimension besides time, latitude and longitude.")


async def _vertical_profile(call: Call[VerticalProfileParams]) -> CapabilityResult:
    record = call.handle("grid")
    grid = await open_grid_param(call, "grid")
    da = grid.data
    time_dim = identify_time(da)
    vdim, candidates = vertical_dim(da, time_dim)
    dims = ", ".join(str(d) for d in da.dims)
    if vdim is None:
        raise refuse(call, "ambiguous_vertical" if candidates else "no_vertical_dim", param="grid", dims=dims)
    oversized = profile_scale_guard(da)
    if oversized:
        raise StepFailure(FailureCategory.REQUEST_TOO_LARGE, f"{NAME}: {oversized} Narrow the area or the dates.")

    col_info = col_info_for(record)
    service = AggregationService()
    cadence = service.cadence_for(da, col_info=col_info)
    profile = await call.compute(
        reduce_profile, da, vertical_dim=vdim, time_dim=time_dim, cadence=cadence,
        region_cells=grid.region_cells or None,
    )
    if profile is None:
        return insufficient("the grid", reason=NO_VALID)
    meta = service.timeseries_aggregation_meta(da, profile.valid_indices, "mean", time_dim, col_info=col_info)

    frame = profile_table(profile)
    variable = str(da.name or record.descriptor.quantity or "value")
    masking: dict[str, JsonValue] = dict(grid.masking)
    masking["region_cells"] = grid.region_cells
    descriptor = derived(
        record,
        f"Vertical profile of {variable} over {region_name(record)}, {len(profile.layers)} layers ({meta['aggregation_label']})",
        unit=record.descriptor.unit or (str(da.attrs.get("units")) if da.attrs.get("units") else None),
        masking=masking,
    )
    table = await call.mint("table", frame, descriptor)
    summary: dict[str, JsonValue] = {
        "layers": len(profile.layers),
        "vertical_dim": vdim,
        "axes": list(profile.vertical),
        "default_axis": profile.default_axis,
        "layer_order": profile.layer_order,
        "timesteps": len(profile.valid_indices) if time_dim else 1,
        "aggregation": meta["aggregation_label"],
    }
    disclosures = [AXIS_NOTE.format(kind=kind) for kind in profile.vertical]
    if not profile.vertical:
        disclosures.append("The grid carries no pressure or altitude for its layers, so the profile is against the layer index.")
    if time_dim and len(profile.valid_indices) > 1:
        disclosures.append("Each layer is the mean over time, weighting each cadence interval equally.")
    return call.done(table, summary=summary, disclosures=tuple(disclosures))


vertical_profile = Capability(
    name=NAME,
    description="Reduce a layered grid to one area-weighted mean per vertical layer, against pressure or altitude, as a table.",
    params=VerticalProfileParams,
    output_kind="table",
    label="Computing a vertical profile",
    method="The grid was reduced to a vertical profile of {summary.layers} layers ({summary.aggregation}).",
    retrieves=False,
    retry_safe=True,
    run=_vertical_profile,
)

CAPABILITIES = (vertical_profile,)
