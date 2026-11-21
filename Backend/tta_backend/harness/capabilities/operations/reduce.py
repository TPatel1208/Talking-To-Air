"""Reductions: a grid over time, a grid over its area at each time, and one statistic."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, JsonValue, model_validator

from tta_backend.datasets.cadence import CADENCES
from tta_backend.domain.analysis.comparison import field_summary
from tta_backend.domain.analysis.reduction import reduce_over_time, regional_series
from tta_backend.domain.analysis.statistics import grid_statistic, series_statistic
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import number, plain, region_name
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.operations.common import (
    derived,
    grid_dataset,
    insufficient,
    open_grid_input,
    open_series_input,
    refuse,
)
from tta_backend.harness.capabilities.operations.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import series_frame
from tta_backend.harness.handles.payloads import ScalarInput

__all__ = ["CAPABILITIES", "GROUP", "aggregate_time", "regional_series_capability", "statistic"]

ReductionStat = Literal["mean", "median", "max", "min", "std"]
Statistic = Literal["mean", "median", "max", "min", "std", "percentile"]


# aggregate_time


class AggregateTimeParams(Params):
    data: Annotated[str, handle_param("grid"), Field(description="A grid handle with a time axis.")]
    stat: ReductionStat = "mean"


def _time_basis(stat: str, cadence: str) -> str:
    if stat != "mean":
        return "per cell over every valid time step"
    if cadence in CADENCES:
        return f"each {cadence} period weighted equally"
    return "each time step weighted equally"


async def _aggregate_time(call: Call[AggregateTimeParams]) -> CapabilityResult:
    p = call.params
    grid = await open_grid_input(call, "data")
    if not grid.has_time:
        raise refuse(call, "no_time_axis", param="data")
    reduced, steps = await call.compute(reduce_over_time, grid.data, p.stat, grid.cadence)
    summary_stats = await call.compute(field_summary, reduced) if steps else None
    if summary_stats is None:
        return insufficient(grid.record.descriptor.summary)
    source = grid.record
    record = await call.mint(
        "grid", grid_dataset(reduced, grid.unit),
        derived(
            source, f"{p.stat} over time of {source.descriptor.summary}",
            unit=grid.unit, masking=grid.masking,
        ),
    )
    summary: dict[str, JsonValue] = {
        "stat": p.stat,
        "time_steps": len(steps),
        "time_steps_total": int(grid.data.sizes["time"]),
        **{k: plain(v) for k, v in summary_stats.items()},
    }
    return call.done(record, summary=summary, values={"basis": _time_basis(p.stat, grid.cadence)})


aggregate_time = Capability(
    name="operations.aggregate_time",
    description="Reduce a grid with a time axis to one map: the mean, median, max, min or std of each cell over time.",
    params=AggregateTimeParams,
    output_kind="grid",
    label="Reducing over time",
    method="Each grid cell was reduced to its {stat} over {summary.time_steps} valid time steps ({basis}).",
    retrieves=False,
    retry_safe=True,
    run=_aggregate_time,
)


# regional_series


class RegionalSeriesParams(Params):
    data: Annotated[str, handle_param("grid"), Field(description="A grid handle with a time axis.")]
    stat: ReductionStat = "mean"


async def _regional_series(call: Call[RegionalSeriesParams]) -> CapabilityResult:
    p = call.params
    grid = await open_grid_input(call, "data")
    if not grid.has_time:
        raise refuse(call, "no_time_axis", param="data")
    frame = await call.compute(regional_series, grid.data, p.stat)
    if frame.empty:
        return insufficient(grid.record.descriptor.summary)
    source = grid.record
    name = region_name(source)
    frame["series"] = name
    frame = series_frame(frame)
    record = await call.mint(
        "series", frame,
        derived(source, f"{p.stat} over {name} at each time step of {source.descriptor.summary}", unit=grid.unit, masking=grid.masking),
    )
    values = frame["value"].to_numpy(dtype=float)
    summary: dict[str, JsonValue] = {
        "stat": p.stat,
        "points": len(frame),
        "time_steps_total": int(grid.data.sizes["time"]),
        "first": frame["time"].min().isoformat(),
        "last": frame["time"].max().isoformat(),
        "min": number(np.min(values)),
        "max": number(np.max(values)),
    }
    weighting = "cos(latitude) area-weighted" if p.stat == "mean" else "per cell, unweighted"
    return call.done(record, summary=summary, values={"area_name": name, "weighting": weighting})


regional_series_capability = Capability(
    name="operations.regional_series",
    description="Turn a grid with a time axis into a series of the area's mean (area-weighted), median, max, min or std at each time step.",
    params=RegionalSeriesParams,
    output_kind="series",
    label="Computing a regional series",
    method="The {stat} over {area_name} was taken at each of {summary.points} time steps ({weighting}).",
    retrieves=False,
    retry_safe=True,
    run=_regional_series,
)


# statistic


class StatisticParams(Params):
    data: Annotated[str, handle_param("grid", "series"), Field(description="A grid or series handle.")]
    stat: Statistic = "mean"
    percentile: float | None = Field(default=None, ge=0, le=100, description="The rank, 0 to 100, when stat is percentile.")
    line: str | None = Field(default=None, description="For a series with several lines, the one line to use.")

    @model_validator(mode="after")
    def _percentile_with_its_rank(self) -> Self:
        if (self.stat == "percentile") != (self.percentile is not None):
            raise ValueError("percentile is given with stat percentile, and only then")
        return self


async def _statistic(call: Call[StatisticParams]) -> CapabilityResult:
    p = call.params
    source = call.handle("data")
    result: dict[str, Any] | None
    masking: JsonValue
    if source.kind == "grid":
        if p.line is not None:
            raise refuse(call, "line_not_found", param="data", line=p.line)
        grid = await open_grid_input(call, "data")
        unit, masking = grid.unit, grid.masking
        result = await call.compute(grid_statistic, grid.data, p.stat, grid.cadence, p.percentile)
    else:
        series = await open_series_input(call, "data", p.line)
        unit, masking = series.unit, source.descriptor.masking
        result = await call.compute(series_statistic, series.frame["value"].to_numpy(dtype=float), p.stat, p.percentile)
    if result is None:
        return insufficient(source.descriptor.summary)
    value, spread, count = number(result["value"]), number(result["spread"]), int(result["count"])
    spread_basis = str(result["spread_basis"])
    label = f"{p.percentile:g}th percentile" if p.stat == "percentile" and p.percentile is not None else p.stat
    spread_text = "none" if spread is None else f"{spread:.4g}"
    record = await call.mint(
        "scalar", ScalarInput(value=value),
        derived(
            source, f"{label} of {source.descriptor.summary}; spread {spread_text} ({spread_basis}), not an uncertainty",
            unit=unit, masking=masking,
        ),
        parameters={**call.recorded, "spread": {"value": spread, "basis": spread_basis, "count": count}},
    )
    summary: dict[str, JsonValue] = {
        "value": value, "stat": p.stat, "basis": str(result["basis"]),
        "spread": spread, "spread_basis": spread_basis, "count": count,
    }
    if p.percentile is not None:
        summary["percentile"] = p.percentile
    return call.done(record, summary=summary)


statistic = Capability(
    name="operations.statistic",
    description="One number for a grid or series: mean, median, max, min, std or a percentile, with the spread of the values and their count.",
    params=StatisticParams,
    output_kind="scalar",
    label="Computing a statistic",
    method=(
        "The {stat} was computed as the {summary.basis}, from {summary.count} valid values; "
        "their spread ({summary.spread_basis}) is reported beside it and is not an uncertainty of the {stat}."
    ),
    retrieves=False,
    retry_safe=True,
    run=_statistic,
)


CAPABILITIES: tuple[Capability[Any], ...] = (aggregate_time, regional_series_capability, statistic)
