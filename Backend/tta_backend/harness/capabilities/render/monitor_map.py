"""``render.monitor_map``: monitors that carry a value, each given its AQI category and colour.

Takes a ``points`` handle with a ``value`` column, or a ``table`` handle with
``site_id``, ``name``, ``lat``, ``lon``, ``value`` and ``unit`` (the monitor
summary and exceedance tables), whose rows are reduced to one per site with
``stat`` (max or mean). Stored as the ground-monitor table artifact the
frontend maps (rows with ``latitude``/``longitude``,
``metadata.has_coordinates``), through ``Artifacts.store_table``; it is a
table artifact, not a chart payload, so nothing is emitted as a chart. Each
row adds ``value``, ``units``, ``aqi_category``, ``aqi_low``/``aqi_high``,
``color`` and, where the NAAQS has the same averaging period,
``standard_level`` and ``above_standard_level``, compared after truncating to
the reporting precision as the AQI category is. ``metadata.colouring`` holds
the legend and how each site's value was taken. A value outside the
breakpoint table is labelled as such, never given the nearest category. The
disclosures describe what the map data holds, not what a viewer draws.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import Field, JsonValue

from tta_backend.datasets.standards import load_standards
from tta_backend.domain.analysis.thresholds import reporting_decimals
from tta_backend.domain.charts.monitor_map import (
    BEYOND_LABEL,
    SITE_TABLE_COLUMNS,
    averaging_period,
    category_counts,
    legend,
    missing_site_columns,
    monitor_rows,
    one_point_per_site,
    pollutant_for,
    site_units,
    standard_for,
)
from tta_backend.domain.monitors.pollutants import convert
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import region_name
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.render.common import refuse, sources
from tta_backend.harness.capabilities.render.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import points_frame
from tta_backend.harness.handles.model import HandleRecord

__all__ = ["CAPABILITIES", "GROUP", "render_monitor_map"]

NO_POINTS = "The input holds no monitors, so there is no monitor map to draw."
EXCEEDANCE_NOTE = (
    "A value above the NAAQS level is not a violation of the standard, which is judged on its form over several years."
)


class MonitorMapParams(Params):
    data: Annotated[str, handle_param("points", "table")] = Field(
        description="A points handle whose points carry a value, or a monitor table with site_id, name, lat, lon, value and unit.",
    )
    stat: Literal["max", "mean"] = Field(
        default="max", description="For a table, how each site's rows are reduced to the one value mapped.",
    )
    pollutant: str | None = Field(
        default=None, description="Pollutant key (o3, pm25, pm10, no2, so2, co); taken from the input's quantity when omitted.",
    )
    averaging_period: str | None = Field(
        default=None, description="Averaging period of the values, such as 8-hour or 24-hour; the pollutant's default when omitted.",
    )
    title: str | None = Field(default=None, max_length=200, description="Map title; made from the pollutant and area when omitted.")


async def _sites(call: Call[MonitorMapParams], record: HandleRecord) -> tuple[pd.DataFrame, str | None, dict[str, Any]]:
    """One point per monitor with its value, the values' unit, and how each value was taken."""
    raw = await call.open("data")
    if record.kind == "points":
        frame = points_frame(raw)
        if "value" not in frame.columns:
            raise refuse(call, "no_values", param="data")
        return frame, record.descriptor.unit, {"source": "points", "stat": None, "max_rows_per_site": 1}
    missing = missing_site_columns(raw)
    if missing:
        raise refuse(call, "missing_columns", param="data", missing=", ".join(missing), columns=", ".join(SITE_TABLE_COLUMNS))
    units = site_units(raw)
    if len(units) > 1:
        raise refuse(call, "mixed_units", param="data", count=len(units), units=", ".join(units))
    frame = points_frame(await call.compute(one_point_per_site, raw, call.params.stat))
    unit = units[0] if units else record.descriptor.unit
    most = int(frame["rows"].max()) if not frame.empty else 0
    return frame, unit, {"source": "table", "stat": call.params.stat, "max_rows_per_site": most}


async def _render_monitor_map(call: Call[MonitorMapParams]) -> CapabilityResult:
    p = call.params
    record = call.handle("data")
    standards = load_standards()
    pollutant = p.pollutant if p.pollutant is not None else pollutant_for(record.descriptor.quantity)
    if pollutant not in standards:
        raise refuse(call, "unknown_pollutant", param="data", choices=", ".join(sorted(standards)))
    assert pollutant is not None
    try:
        period = averaging_period(pollutant, p.averaging_period)
    except ValueError:
        raise refuse(call, "no_breakpoints", pollutant=pollutant, averaging_period=p.averaging_period) from None
    info = standards[pollutant]

    frame, unit, site_value = await _sites(call, record)
    if frame.empty:
        return CapabilityResult(status="insufficient_evidence", reason=NO_POINTS, values={"situation": "no_points"})

    if unit != info.unit:
        if convert(1.0, unit, info.unit) is None:
            raise refuse(call, "unit_mismatch", param="data", unit=unit or "no unit", pollutant=info.name, breakpoint_unit=info.unit)
        frame = frame.assign(value=[
            None if pd.isna(v) else convert(float(v), unit, info.unit) for v in frame["value"]
        ])

    rows = monitor_rows(frame, pollutant=pollutant, period=period, unit=info.unit)
    counts = category_counts(rows)
    standard = standard_for(pollutant, period)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    per_site = f" (each site's {site_value['stat']})" if site_value["stat"] else ""
    title = p.title or f"{info.name} {period}{per_site} by AQI category, {region_name(record)}"
    decimals = reporting_decimals(pollutant)
    metadata: dict[str, Any] = {
        "dataset": record.descriptor.product or "",
        "has_coordinates": True,
        "source_handles": [record.id],
        "sources": sources((record,)),
        "colouring": {
            "method": "aqi_category",
            "pollutant": pollutant,
            "averaging_period": period,
            "unit": info.unit,
            "legend": legend(pollutant, period),
            "standard": None if standard is None else standard.model_dump(),
            "standard_comparison": f"truncated to {decimals} decimal(s) before comparing",
            "site_value": site_value,
        },
    }
    ref = await call.env.context.artifacts.store_table(title, columns, rows, metadata)

    disclosures: list[str] = [
        f"The map data gives each monitor its {info.name} {period} AQI category and that category's colour "
        "(row fields aqi_category and color), not a colour from its place among the values shown.",
    ]
    if site_value["stat"]:
        disclosures.append(
            f"Each monitor's value is the {site_value['stat']} of its rows in the table "
            f"(up to {site_value['max_rows_per_site']} per site)."
        )
    if unit != info.unit:
        disclosures.append(f"Values were converted from {unit} to {info.unit} for the AQI breakpoints.")
    if counts.get(BEYOND_LABEL):
        disclosures.append(f"{counts[BEYOND_LABEL]} value(s) are above the AQI table's highest breakpoint, where the index is not defined for this period.")
    if standard is not None:
        disclosures.append(
            f"Values were truncated to {decimals} decimal(s) ({info.unit}) before comparing with the NAAQS level of "
            f"{standard.level:g} {standard.unit}, as for the AQI category."
        )
        if any(row.get("above_standard_level") for row in rows):
            disclosures.append(EXCEEDANCE_NOTE)
    summary: dict[str, JsonValue] = {
        "render_type": "monitor_map",
        "title": title,
        "pollutant": pollutant,
        "averaging_period": period,
        "unit": info.unit,
        "points": len(rows),
        "site_value": site_value["stat"] or "as given",
        "by_category": dict(counts),
        "source_handles": [record.id],
    }
    return call.done(charts=(ref.id,), summary=summary, disclosures=tuple(disclosures))


render_monitor_map = Capability(
    name="render.monitor_map",
    description="Map monitors that carry a value (points, or a monitor table reduced to one value per site), each given its pollutant's AQI category and colour.",
    params=MonitorMapParams,
    output_kind=None,
    label="Drawing a monitor map",
    method=(
        "{summary.points} monitor(s) were given their {summary.pollutant} {summary.averaging_period} AQI category "
        "and its colour in the map data (site value: {summary.site_value})."
    ),
    retrieves=False,
    retry_safe=True,
    run=_render_monitor_map,
)

CAPABILITIES = (render_monitor_map,)
