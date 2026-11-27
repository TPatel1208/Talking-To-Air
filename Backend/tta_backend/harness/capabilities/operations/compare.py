"""The difference of two grids: B minus A, absolute or percent, cell by cell.

The grids must line up (``checks.grid_mismatch``): nothing is regridded, so
grids of different resolution or offset from each other are refused. The
result reports the fraction of cells valid in either grid that are valid in
both, and below ``MIN_PAIRED_FRACTION`` it is ``insufficient_evidence``.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import xarray as xr
from pydantic import Field, JsonValue

from tta_backend.domain.analysis.checks import (
    disjoint_periods,
    grid_mismatch,
    time_range,
    units_mismatch,
    variable_mismatch,
)
from tta_backend.domain.analysis.comparison import (
    anomaly_stats,
    difference as difference_of,
    field_summary,
    pair_cells,
    percent_change_field,
    shared_cells,
)
from tta_backend.domain.analysis.reduction import reduce_over_time
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import period_json, plain
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.operations.common import (
    GridInput,
    derived,
    grid_dataset,
    insufficient,
    open_grid_input,
    refuse,
    refuse_mismatch,
)
from tta_backend.harness.capabilities.operations.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.handles.model import Period

__all__ = ["CAPABILITIES", "GROUP", "MIN_PAIRED_FRACTION", "difference"]

# The least fraction of the cells valid in either grid that must be valid in both.
MIN_PAIRED_FRACTION = 0.5

BASELINE_TOO_SMALL = (
    "Percent change withheld everywhere: the baseline (a) is zero or too small, relative to its own range, "
    "in every paired cell. Use change=absolute."
)
TOO_FEW_PAIRED = (
    "Too little valid data: only {paired} of the {either} cells valid in either grid are valid in both "
    "({fraction:.0%}); at least {floor:.0%} is needed to compare them."
)
TWO_PRODUCTS = (
    "Two products are compared: a is {a} and b is {b}. The difference includes any difference between the "
    "products, not only a change in what they measure."
)


class DifferenceParams(Params):
    a: Annotated[str, handle_param("grid"), Field(description="The baseline grid; the result is b minus a.")]
    b: Annotated[str, handle_param("grid"), Field(description="The grid compared with the baseline.")]
    change: Literal["absolute", "percent"] = "absolute"
    over: Literal["two_periods", "one_period"] = Field(
        default="two_periods",
        description="one_period refuses grids whose dates do not overlap; two_periods compares any two periods.",
    )
    threshold: float | None = Field(default=None, gt=0, description="A change magnitude; the summary reports the area reaching it.")


def _period_range(grid: GridInput) -> tuple[str | None, str | None]:
    start, end = time_range(grid.data)
    if start is not None:
        return start, end
    period = grid.record.descriptor.period
    return (period.start, period.end) if period is not None else (None, None)


def _quantities(a: GridInput, b: GridInput) -> tuple[str | None, str | None]:
    return a.record.descriptor.quantity, b.record.descriptor.quantity


def _check(call: Call[DifferenceParams], a: GridInput, b: GridInput) -> None:
    for name_a, name_b in (_quantities(a, b), (a.data.name, b.data.name)):
        mismatch = variable_mismatch(name_a, name_b)
        if mismatch is not None:
            raise refuse_mismatch(call, mismatch)
    mismatch = units_mismatch(a.unit, b.unit)
    if mismatch is not None:
        raise refuse_mismatch(call, mismatch)
    mismatch = grid_mismatch(a.data["lat"].values, a.data["lon"].values, b.data["lat"].values, b.data["lon"].values)
    if mismatch is not None:
        raise refuse_mismatch(call, mismatch)
    if call.params.over == "one_period":
        mismatch = disjoint_periods(_period_range(a), _period_range(b))
        if mismatch is not None:
            raise refuse_mismatch(call, mismatch)


def _time_mean(grid: GridInput) -> xr.DataArray:
    return reduce_over_time(grid.data, "mean", grid.cadence)[0].transpose("lat", "lon")


def _agreed(a: Any, b: Any) -> Any:
    return a if a == b else None


def _masking(a: GridInput, b: GridInput) -> dict[str, JsonValue]:
    """The keys both inputs agree on, A's region cells, and each input's full provenance."""
    agreed: dict[str, JsonValue] = {k: v for k, v in a.masking.items() if k != "region_cells" and b.masking.get(k) == v}
    return {**agreed, "region_cells": a.masking.get("region_cells"), "inputs": [a.masking, b.masking]}


def _period(a: GridInput, b: GridInput) -> Period | None:
    return _agreed(a.record.descriptor.period, b.record.descriptor.period)  # type: ignore[no-any-return]


def _period_text(period: Period | None) -> str:
    return f"{period.start} to {period.end}" if period is not None else "an unrecorded period"


def _periods_note(a: GridInput, b: GridInput) -> str:
    """The summary line's statement of the two periods, empty when they agree."""
    pa, pb = a.record.descriptor.period, b.record.descriptor.period
    return "" if pa == pb else f" (a covers {_period_text(pa)}; b covers {_period_text(pb)})"


def _valid_cells(da: xr.DataArray) -> int:
    return int(np.isfinite(np.asarray(da.values, dtype=float)).sum())


def _compute(a: xr.DataArray, b: xr.DataArray, change: str, threshold: float | None) -> dict[str, Any]:
    shared = shared_cells(a, b)
    if not shared:
        return {"shared": 0}
    a_paired, b_paired = pair_cells(a, b)
    diff = difference_of(a_paired, b_paired)
    stats = anomaly_stats(a_paired, b_paired, diff, threshold)
    # The grids line up, so each paired cell is one cell of each grid.
    either = _valid_cells(a) + _valid_cells(b) - stats["n_cells"]
    withheld = 0
    field = diff
    if change == "percent":
        field, withheld = percent_change_field(a_paired, b_paired)
    field.name = a.name
    return {
        "shared": shared, "field": field, "stats": stats, "withheld": withheld, "either": either,
        "summary": field_summary(field),
    }


async def _difference(call: Call[DifferenceParams]) -> CapabilityResult:
    p = call.params
    a = await open_grid_input(call, "a")
    b = await open_grid_input(call, "b")
    _check(call, a, b)
    mean_a = await call.compute(_time_mean, a)
    mean_b = await call.compute(_time_mean, b)
    out = await call.compute(_compute, mean_a, mean_b, p.change, p.threshold)
    if not out["shared"]:
        raise refuse(call, "empty_overlap")
    stats = out["stats"]
    if not stats["n_cells"]:
        return insufficient("the cells the two grids share")
    paired_fraction = stats["n_cells"] / out["either"]
    if paired_fraction < MIN_PAIRED_FRACTION:
        return insufficient("the paired cells", reason=TOO_FEW_PAIRED.format(
            paired=stats["n_cells"], either=out["either"], fraction=paired_fraction, floor=MIN_PAIRED_FRACTION,
        ))
    if out["summary"] is None:
        return insufficient("the percent change", situation="baseline_too_small", reason=BASELINE_TOO_SMALL)

    da, db = a.record.descriptor, b.record.descriptor
    disclosures: list[str] = []
    if da.product != db.product:
        disclosures.append(TWO_PRODUCTS.format(a=da.product or "an unnamed product", b=db.product or "an unnamed product"))
    unit = "%" if p.change == "percent" else a.unit
    what = "percent change" if p.change == "percent" else "difference"
    quantity = f"{da.quantity} {what}" if da.quantity else what
    region = da.region if da.region == db.region else {"a": da.region, "b": db.region}
    # Descriptor.period holds one period, so two different ones are stated in the summary line and parameters.
    parameters: dict[str, JsonValue] = {**call.recorded}
    if da.period != db.period:
        parameters["periods"] = {"a": period_json(da.period), "b": period_json(db.period)}
    record = await call.mint(
        "grid", grid_dataset(out["field"], unit),
        derived(
            a.record, f"{what}{_periods_note(a, b)} of {db.summary} minus {da.summary}",
            quantity=quantity, unit=unit, product=_agreed(da.product, db.product),
            region=region, period=_period(a, b), masking=_masking(a, b),
        ),
        parameters=parameters,
    )
    summary: dict[str, JsonValue] = {
        "change": p.change,
        **{k: plain(v) for k, v in out["summary"].items() if k != "cells"},
        "paired_cells": stats["n_cells"],
        "cells_valid_in_either": out["either"],
        "paired_fraction": round(paired_fraction, 3),
        "mean_difference": plain(stats["mean_difference"]),
        "percent_change": plain(stats["percent_change"]),
    }
    if "percent_change_note" in stats:
        summary["percent_change_note"] = stats["percent_change_note"]
    if p.change == "percent":
        summary["withheld_cells"] = out["withheld"]
    if "area_exceeding_threshold" in stats:
        summary["area_exceeding_threshold"] = plain(stats["area_exceeding_threshold"])
    if da.product != db.product:
        summary["products"] = {"a": da.product, "b": db.product}
    values: dict[str, JsonValue] = {
        "period_a": plain(period_json(da.period)),
        "period_b": plain(period_json(db.period)),
        "paired_percent": f"{paired_fraction:.0%}",
    }
    return call.done(record, summary=summary, values=values, disclosures=tuple(disclosures))


difference = Capability(
    name="operations.difference",
    description="Grid b minus grid a, cell by cell, absolute or percent, after averaging each over time; refuses mismatched units, variables or grids.",
    params=DifferenceParams,
    output_kind="grid",
    label="Differencing two grids",
    method=(
        "Each grid was averaged over time, then a was subtracted from b cell by cell ({change} change) "
        "over {summary.paired_cells} cells valid on both, {paired_percent} of the {summary.cells_valid_in_either} "
        "cells valid in either; cells missing on either side were left out."
    ),
    retrieves=False,
    retry_safe=True,
    run=_difference,
)


CAPABILITIES: tuple[Capability[Any], ...] = (difference,)
