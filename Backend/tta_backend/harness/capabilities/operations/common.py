"""What the operations share: opening inputs, their refusal wordings, and describing outputs.

A refusal is a failure with fixed wording and ``{"refusal": reason}`` on its
trace entry (``capabilities.common.refusal``). Too little valid data is an
``insufficient_evidence`` result with ``values.situation``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import xarray as xr
from pydantic import JsonValue

from tta_backend.datasets.registry import load_registry
from tta_backend.domain.analysis.checks import Mismatch
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import fact, plain, refusal, short
from tta_backend.harness.capabilities.grids import open_grid_param
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import series_frame
from tta_backend.harness.handles.model import Descriptor, HandleRecord
from tta_backend.harness.trace import FailureCategory, StepFailure

GRID_DIMS = ("time", "lat", "lon")
MAX_LISTED = 31

REFUSALS: dict[str, str] = {
    "unit_mismatch": "{name}: the inputs are in different units ({a} and {b}); units are not converted, so they cannot be compared.",
    "unit_missing": "{name}: one input declares units ({declared}) and input {missing_side} declares none, so they cannot be confirmed comparable.",
    "variable_mismatch": "{name}: the inputs hold different variables ({a} and {b}), so they cannot be compared.",
    "disjoint_periods": "{name}: the inputs cover periods that do not overlap ({start_a} to {end_a} and {start_b} to {end_b}).",
    "empty_overlap": "{name}: the two grids share no cells, so there is nothing to pair.",
    "no_time_axis": "{name}: {param} has no time axis to work over.",
    "unsupported_dims": "{name}: {param} has dimensions other than time, latitude and longitude ({dims}); select a single level first.",
    "line_not_found": "{name}: {param} has no series line named {line}.",
    "unknown_standard": (
        "{name}: the standards file has no {pollutant} standard for {averaging_period}; "
        "call it again with averaging_period one of: {periods}."
    ),
    "standard_unit_mismatch": "{name}: the data is in {a} and the {pollutant} standard in {b}; units are not converted, so they cannot be compared.",
    "grid_mismatch": (
        "{name}: the two grids' cells do not line up (a has {spacing_a} cells, b has {spacing_b}); grids are not "
        "regridded, so they cannot be compared cell by cell."
    ),
    "no_vertical_dim": "{name}: {param} has no vertical dimension to profile (dimensions: {dims}).",
    "ambiguous_vertical": (
        "{name}: {param} has more than one dimension besides time, latitude and longitude ({dims}), so which is "
        "vertical is not known; select the others first."
    ),
}

# The category each refusal is recorded under; the rest are bad input.
_CATEGORY = {
    "unsupported_dims": FailureCategory.UNSUPPORTED_SHAPE,
    "no_vertical_dim": FailureCategory.UNSUPPORTED_SHAPE,
    "ambiguous_vertical": FailureCategory.UNSUPPORTED_SHAPE,
}

INSUFFICIENT = "Too little valid data: {what} has no valid values after masking."


def refuse(call: Call[Any], reason: str, **facts: Any) -> StepFailure:
    """The failure for ``reason``, worded from observed facts only."""
    return refusal(call, REFUSALS[reason], reason, category=_CATEGORY.get(reason, FailureCategory.BAD_INPUT), **facts)


def refuse_mismatch(call: Call[Any], mismatch: Mismatch) -> StepFailure:
    return refuse(call, mismatch.reason, **mismatch.facts)


def insufficient(what: str, *, situation: str = "too_little_valid", reason: str | None = None) -> CapabilityResult:
    """An ``insufficient_evidence`` result naming what had no valid values."""
    return CapabilityResult(
        status="insufficient_evidence",
        reason=reason or INSUFFICIENT.format(what=fact(what)),
        values={"situation": situation},
    )


@dataclass(frozen=True)
class GridInput:
    """An opened grid parameter: its record, the loaded masked field, its unit and masking provenance."""

    record: HandleRecord
    data: xr.DataArray
    unit: str | None
    masking: dict[str, JsonValue]

    @property
    def has_time(self) -> bool:
        return "time" in self.data.dims

    @property
    def cadence(self) -> str:
        cfg = load_registry().get(self.record.descriptor.product or "")
        return cfg.cadence if cfg is not None else "unknown"


def _load(da: xr.DataArray) -> xr.DataArray:
    return da.load()


async def open_grid_input(call: Call[Any], param: str) -> GridInput:
    """The masked grid behind ``param``, loaded, refused when it has dimensions besides time, lat and lon."""
    grid = await open_grid_param(call, param)
    data = await call.compute(_load, grid.data)
    extra = [str(d) for d in data.dims if d not in GRID_DIMS]
    if extra:
        raise refuse(call, "unsupported_dims", param=param, dims=", ".join(extra))
    record = call.handle(param)
    unit = data.attrs.get("units") or record.descriptor.unit
    masking: dict[str, JsonValue] = {str(k): plain(v) for k, v in grid.masking.items()}
    masking["region_cells"] = grid.region_cells
    return GridInput(record=record, data=data, unit=str(unit) if unit else None, masking=masking)


@dataclass(frozen=True)
class SeriesInput:
    record: HandleRecord
    frame: pd.DataFrame
    unit: str | None


async def open_series_input(call: Call[Any], param: str, line: str | None) -> SeriesInput:
    """The series behind ``param``, narrowed to ``line`` when one is named."""
    frame = series_frame(await call.open(param))
    if line is not None:
        if line not in set(frame["series"].astype(str)):
            raise refuse(call, "line_not_found", param=param, line=line)
        frame = frame[frame["series"].astype(str) == line].reset_index(drop=True)
    record = call.handle(param)
    return SeriesInput(record=record, frame=frame, unit=record.descriptor.unit)


def masking_of(record: HandleRecord) -> JsonValue:
    """A series or table input's recorded masking provenance."""
    return record.descriptor.masking


def derived(source: HandleRecord, summary: str, **fields: Any) -> Descriptor:
    """A descriptor that carries ``source``'s quantity, unit, product, region,
    period and masking unless ``fields`` replaces them."""
    d = source.descriptor
    base: dict[str, Any] = {
        "quantity": d.quantity, "unit": d.unit, "product": d.product,
        "region": d.region, "period": d.period, "masking": d.masking,
    }
    return Descriptor(summary=short(summary), **{**base, **fields})


def grid_dataset(da: xr.DataArray, unit: str | None) -> xr.Dataset:
    """A field in the grid layout: one data variable with its unit."""
    da = da.copy()
    da.attrs = {"units": unit} if unit else {}
    return da.to_dataset(name=str(da.name or "value"))
