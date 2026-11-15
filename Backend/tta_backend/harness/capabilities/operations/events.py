"""Days that stand out: each day's peak, and what is above a threshold or a standard."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import Field, JsonValue, model_validator

from tta_backend.domain.analysis.checks import units_mismatch
from tta_backend.domain.analysis.peaks import daily_peaks_grid, daily_peaks_series
from tta_backend.domain.analysis.thresholds import (
    UnknownStandard,
    grid_exceedances,
    naaqs_for,
    reporting_decimals,
    series_exceedances,
)
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import number, plain
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.operations.common import (
    MAX_LISTED,
    derived,
    insufficient,
    open_grid_input,
    open_series_input,
    refuse,
)
from tta_backend.harness.capabilities.operations.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.handles.model import HandleRecord

__all__ = ["CAPABILITIES", "GROUP", "daily_peak", "threshold"]

Pollutant = Literal["o3", "pm25", "pm10", "no2", "so2", "co"]


def _dates(frame: pd.DataFrame, column: str = "date") -> list[JsonValue]:
    return [str(d) for d in list(dict.fromkeys(frame[column].dropna().astype(str)))[:MAX_LISTED]]


async def _opened(call: Call[Any], line: str | None) -> tuple[HandleRecord, Any, str | None, JsonValue]:
    """The input's record, its field or frame, its unit and its masking provenance."""
    source = call.handle("data")
    if source.kind == "grid":
        if line is not None:
            raise refuse(call, "line_not_found", param="data", line=line)
        grid = await open_grid_input(call, "data")
        return source, grid, grid.unit, grid.masking
    series = await open_series_input(call, "data", line)
    return source, series, series.unit, source.descriptor.masking


# daily_peak


class DailyPeakParams(Params):
    data: Annotated[str, handle_param("grid", "series"), Field(description="A grid with a time axis, or a series.")]
    line: str | None = Field(default=None, description="For a series with several lines, the one line to use.")


async def _daily_peak(call: Call[DailyPeakParams]) -> CapabilityResult:
    source, opened, unit, masking = await _opened(call, call.params.line)
    if source.kind == "grid":
        if not opened.has_time:
            raise refuse(call, "no_time_axis", param="data")
        table = await call.compute(daily_peaks_grid, opened.data)
        where = "grid cell"
    else:
        table = await call.compute(daily_peaks_series, opened.frame)
        where = "reading"
    if table.empty:
        return insufficient(source.descriptor.summary)
    record = await call.mint(
        "table", table,
        derived(source, f"Daily peaks of {source.descriptor.summary}", unit=unit, masking=masking),
    )
    top = table.loc[table["value"].idxmax()]
    location: JsonValue = (
        {"lat": float(top["lat"]), "lon": float(top["lon"])} if source.kind == "grid" else str(top["series"])
    )
    summary: dict[str, JsonValue] = {
        "days": len(table),
        "max": number(top["value"]),
        "max_date": str(top["date"]),
        "max_time": str(top["time"]),
        "max_location": location,
    }
    values: dict[str, JsonValue] = {"where": where, "dates": _dates(table), "dates_truncated": len(table) > MAX_LISTED}
    return call.done(record, summary=summary, values=values)


daily_peak = Capability(
    name="operations.daily_peak",
    description="Each day's highest value of a grid with a time axis or a series, with when and where it occurred, as a table.",
    params=DailyPeakParams,
    output_kind="table",
    label="Finding daily peaks",
    method="Each day's highest value was found over every valid {where} and time step, for {summary.days} days.",
    retrieves=False,
    retry_safe=True,
    run=_daily_peak,
)


# threshold


class ThresholdParams(Params):
    data: Annotated[str, handle_param("grid", "series"), Field(description="A grid or a series.")]
    level: float | None = Field(default=None, description="The threshold, in the data's unit.")
    pollutant: Pollutant | None = Field(default=None, description="Use this pollutant's NAAQS level instead of a level.")
    averaging_period: str | None = Field(default=None, description="Which of the pollutant's standards, such as 8-hour or 24-hour.")
    line: str | None = Field(default=None, description="For a series with several lines, the one line to use.")

    @model_validator(mode="after")
    def _one_threshold(self) -> Self:
        if (self.level is None) == (self.pollutant is None):
            raise ValueError("give a level or a pollutant, not both")
        if self.averaging_period is not None and self.pollutant is None:
            raise ValueError("averaging_period goes with pollutant")
        return self


async def _threshold(call: Call[ThresholdParams]) -> CapabilityResult:
    p = call.params
    source, opened, unit, masking = await _opened(call, p.line)
    disclosures: list[str] = []
    level_unit: str | None
    if p.pollutant is not None:
        try:
            standard = naaqs_for(p.pollutant, p.averaging_period)
        except UnknownStandard:
            raise refuse(call, "unknown_standard", pollutant=p.pollutant, averaging_period=p.averaging_period) from None
        if isinstance(standard, list):
            return call.ask(f"Which {p.pollutant} standard should be applied?", standard)
        mismatch = units_mismatch(unit, standard.unit)
        if mismatch is not None:
            facts = mismatch.facts
            raise refuse(
                call, "standard_unit_mismatch", pollutant=p.pollutant,
                a=facts.get("a") or "no declared unit", b=facts.get("b") or standard.unit,
            )
        level, level_unit = standard.level, standard.unit
        name = f"the {p.pollutant} {standard.averaging_period} NAAQS level of {level:g} {level_unit}"
        truncate_for = p.pollutant
        decimals = reporting_decimals(p.pollutant)
        comparison = f"truncated to {decimals} decimal(s) ({level_unit}) before comparing, as for the AQI category"
        disclosures.append(
            f"Values were compared with {name}; the standard's averaging and form "
            f"({standard.form}) were not computed, so a value above it is not a violation."
        )
        disclosures.append(f"Each value was {comparison}, so a value counts as above the level only once truncated.")
    else:
        assert p.level is not None
        level, level_unit = p.level, unit
        name = f"a threshold of {level:g}" + (f" {level_unit}" if level_unit else "")
        truncate_for = None
        comparison = "compared as given, without truncation"

    if source.kind == "grid":
        table, overall = await call.compute(grid_exceedances, opened.data, level, truncate_for=truncate_for)
        days = table[table["cells_above"] > 0]
        what = "days"
    else:
        table, overall = await call.compute(series_exceedances, opened.frame, level, truncate_for=truncate_for)
        days = table
        what = "values"
    if not overall["days_with_data"]:
        return insufficient(source.descriptor.summary)
    record = await call.mint(
        "table", table,
        derived(source, f"{what.capitalize()} above {name} in {source.descriptor.summary}", unit=level_unit, masking=masking),
    )
    summary: dict[str, JsonValue] = {
        "level": level, "unit": level_unit, "truncated": truncate_for is not None,
        **{k: plain(v) for k, v in overall.items()},
    }
    values: dict[str, JsonValue] = {
        "standard": name,
        "comparison": comparison,
        "days": _dates(days),
        "days_truncated": int(days["date"].nunique()) > MAX_LISTED,
    }
    return call.done(record, summary=summary, values=values, disclosures=tuple(disclosures))


threshold = Capability(
    name="operations.threshold",
    description="What in a grid or series is above a level, given or taken from a pollutant's NAAQS, as a table; a grid reports the area above.",
    params=ThresholdParams,
    output_kind="table",
    label="Checking against a threshold",
    method=(
        "Values were compared with {standard} ({comparison}), a value exceeding it when above it; "
        "{summary.days_above} of {summary.days_with_data} days with data had a value above it."
    ),
    retrieves=False,
    retry_safe=True,
    run=_threshold,
)


CAPABILITIES: tuple[Capability[Any], ...] = (daily_peak, threshold)
