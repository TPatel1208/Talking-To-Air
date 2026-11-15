"""The one grid opener: how every capability reads a ``grid`` handle.

No capability may read a retrieval grid any other way: ``Call.open`` and
``Call.open_each`` refuse a grid handle. A retrieval grid is
the earthdata service's raw file; only this module turns it into data fit to
analyse, so masking is applied identically wherever a grid is read.

A retrieval grid is opened through the handle resolver, then, off the event
loop: its longitudes are normalised, the science variable recorded on the
handle is selected, it is masked to its area's geometry, and the registry's
fill, valid-range and quality-flag masking is applied exactly as the current
system applies it (``domain.earthdata.opening.mask_retrieval``).

A derived grid was masked when it was made, so it is opened as stored, with
the masking provenance its descriptor recorded.

Retrieval grid handles record what the opener needs in their descriptor:
``masking.variable`` (the science variable) and ``region.area`` (the area
handle whose geometry masks it), with ``region.bounds`` and ``product``.
"""
from __future__ import annotations

import math
from typing import Any

from tta_backend.datasets.registry import load_registry
from tta_backend.domain.earthdata.opening import MaskedGrid, mask_retrieval
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.shapes import grid_variable
from tta_backend.harness.capabilities.sources.areas import area_geometry_of
from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.trace import FailureCategory, StepFailure

__all__ = ["MaskedGrid", "open_grid", "open_grid_param"]


BAD_PLAN = "{name}: the retrieval grid {handle} does not record a usable {what}."


def _bad_plan(call: Call[Any], record: HandleRecord, what: str) -> StepFailure:
    wording = BAD_PLAN.format(name=call.capability.name, handle=record.id, what=what)
    return StepFailure(FailureCategory.INTERNAL_ERROR, wording)


def _retrieval_plan(call: Call[Any], record: HandleRecord) -> tuple[str, str, list[float]]:
    """The science variable, area id and bounds a retrieval grid records; refuses anything else."""
    masking, region = record.descriptor.masking, record.descriptor.region
    variable = masking.get("variable") if isinstance(masking, dict) else None
    area = region.get("area") if isinstance(region, dict) else None
    bounds = region.get("bounds") if isinstance(region, dict) else None
    if not isinstance(variable, str) or not variable:
        raise _bad_plan(call, record, "science variable")
    if not isinstance(area, str) or not area:
        raise _bad_plan(call, record, "area")
    if not (
        isinstance(bounds, list) and len(bounds) == 4
        and all(isinstance(b, int | float) and not isinstance(b, bool) and math.isfinite(b) for b in bounds)
    ):
        raise _bad_plan(call, record, "bounds")
    return variable, area, [float(b) for b in bounds]  # type: ignore[arg-type]


async def open_grid(call: Call[Any], record: HandleRecord) -> MaskedGrid:
    """The masked grid behind ``record`` and its masking provenance.

    Raises InputEvicted when a derived grid's data, or the area a retrieval
    grid is masked to, is no longer stored.
    """
    if record.kind != "grid":
        raise TypeError(f"{record.id} is a {record.kind}, not a grid")
    resolver, user = call.env.resolver, call.user_id
    if record.origin == "derived":
        stored = await resolver.open(user, record.id, "grid")
        masking = record.descriptor.masking if isinstance(record.descriptor.masking, dict) else {}
        cells = masking.get("region_cells")
        return MaskedGrid(
            data=grid_variable(stored), masking=dict(masking),
            region_cells=cells if isinstance(cells, int) else 0,
        )

    variable, area_id, bounds = _retrieval_plan(call, record)
    _, geometry = await area_geometry_of(call, area_id)
    product = load_registry().get(record.descriptor.product or "")
    opened = await resolver.open(user, record.id, "grid")
    return await call.compute(
        mask_retrieval, opened, variable=variable, geometry=geometry, bounds=bounds, product=product,
    )


async def open_grid_param(call: Call[Any], name: str) -> MaskedGrid:
    """The masked grid behind a single grid-handle parameter."""
    return await open_grid(call, call.handle(name))
