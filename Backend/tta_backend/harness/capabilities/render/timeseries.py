"""``render.timeseries``: one or several series handles on one ``timeseries`` chart.

One line per (handle, ``series`` value), labelled with the line's ``series``
value and the handle's product. Lines in different units are assigned
separate y axes: the first unit seen on ``y``, the second on ``y2``; a third
unit is refused (``too_many_units``), since a chart has two sides. Units are
never converted. The payload keeps the first line in ``times``/``values``,
as the one-line chart has them, and adds ``series`` and ``axes``; the
disclosures describe what the payload holds, not what a viewer draws.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, JsonValue

from tta_backend.domain.charts.timeseries import MAX_AXES, Line, axis_units, line_points, multi_series_payload
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.common import region_name
from tta_backend.harness.capabilities.render.common import chart_summary, refuse, sources, store_and_emit
from tta_backend.harness.capabilities.render.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import series_frame
from tta_backend.harness.handles.model import HandleRecord

__all__ = ["CAPABILITIES", "GROUP", "render_timeseries"]

MAX_LINES = 20

NO_VALUES = "The series hold no valid values, so there is no time series to draw."


class TimeseriesParams(Params):
    series: Annotated[list[str], handle_param("series")] = Field(
        min_length=1, max_length=MAX_LINES, description="One or more series handles, drawn on the same chart.",
    )
    title: str | None = Field(default=None, max_length=200, description="Chart title; made from the series when omitted.")


def _label(record: HandleRecord, line: str) -> str:
    product = record.descriptor.product
    return f"{line} · {product}" if product and product != line else line


def _lines(records: tuple[HandleRecord, ...], frames: list[Any]) -> tuple[list[Line], list[str]]:
    """Every line with a value, and the labels of those without one."""
    lines: list[Line] = []
    empty: list[str] = []
    for record, raw in zip(records, frames, strict=True):
        frame = series_frame(raw)
        for name, rows in frame.groupby("series", sort=False):
            label = _label(record, str(name))
            times, values = line_points(rows)
            if not times:
                empty.append(label)
                continue
            lines.append(Line(
                label=label, units=record.descriptor.unit, times=times, values=values,
                source={"handle": record.id, "series": str(name)},
            ))
    return lines, empty


def _variable(records: tuple[HandleRecord, ...]) -> str:
    quantities = list(dict.fromkeys(r.descriptor.quantity or "" for r in records))
    return ", ".join(q for q in quantities if q)


async def _render_timeseries(call: Call[TimeseriesParams]) -> CapabilityResult:
    p = call.params
    records = call.handles("series")
    lines, empty = await call.compute(_lines, records, await call.open_each("series"))
    if len(lines) > MAX_LINES:
        raise refuse(call, "too_many_lines", count=len(lines), limit=MAX_LINES)
    units = axis_units(lines)
    if len(units) > MAX_AXES:
        raise refuse(call, "too_many_units", count=len(units), units=", ".join(u or "no unit" for u in units), limit=MAX_AXES)
    if not lines:
        return CapabilityResult(status="insufficient_evidence", reason=NO_VALUES, values={"situation": "too_little_valid"})

    variable = _variable(records)
    title = p.title or (f"{variable} over {region_name(records[0])}" if variable else f"Time series over {region_name(records[0])}")
    payload = multi_series_payload(lines, title=title, variable=variable, stat="value")
    handles = list(call.input_ids())
    payload["provenance"] = {
        "variable": variable,
        "units": payload["units"],
        "region_name": region_name(records[0]),
        "start_date": min(line.times[0] for line in lines),
        "end_date": max(line.times[-1] for line in lines),
        "source_handles": handles,
        "sources": sources(records),
    }
    payload["export"] = {
        "type": "timeseries",
        "variable": variable,
        "units": payload["units"],
        "chart_parameters": {"chart_type": "timeseries", "lines": len(lines)},
        "source_handles": handles,
    }
    payload["metadata"] = {"source_handles": handles}

    ref = await store_and_emit(call, payload)
    disclosures: list[str] = []
    if len(lines) > 1:
        disclosures.append(
            f"The chart data holds {len(lines)} lines (series); its times and values fields hold the first, "
            f"{lines[0].label}."
        )
    if len(units) > 1:
        disclosures.append(
            f"The lines are in two units ({', '.join(u or 'no unit' for u in units)}); the chart data assigns them "
            "separate y axes (axes), and their heights are not comparable."
        )
    if empty:
        disclosures.append(f"{len(empty)} line(s) had no valid values and are left out of the chart data: {', '.join(empty[:5])}.")
    values: dict[str, JsonValue] = {"lines": len(lines), "axes": len(units)}
    return call.done(charts=(ref.id,), summary=chart_summary(payload), values=values, disclosures=tuple(disclosures))


render_timeseries = Capability(
    name="render.timeseries",
    description="Draw one or several series handles as lines of one time-series chart, with a second y axis for a second unit.",
    params=TimeseriesParams,
    output_kind=None,
    label="Drawing a time series",
    method="The time-series chart data holds {lines} line(s).",
    retrieves=False,
    retry_safe=True,
    run=_render_timeseries,
)

CAPABILITIES = (render_timeseries,)
