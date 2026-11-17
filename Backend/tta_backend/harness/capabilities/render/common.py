"""What the render capabilities share: their refusal wordings, a handle's region, and storing and emitting a chart.

A refusal is a failure with fixed wording and ``{"refusal": reason}`` on its
trace entry (``capabilities.common.refusal``). A chart is stored, then emitted
with its artifact id as ``chart_id``. Its ``metadata.source_handles`` are the
input handle ids and its ``metadata.run_id`` the run that drew it; a stored
table's metadata carries the same two.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import JsonValue

from tta_backend.domain.charts.summary import chart_model_summary
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.common import period_json, refusal
from tta_backend.harness.capabilities.shapes import points_frame
from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.run_context import ArtifactRef
from tta_backend.harness.trace import FailureCategory, StepFailure

REFUSALS: dict[str, str] = {
    "wrong_dims": "{name}: {param} has dimensions other than time, latitude and longitude ({dims}); select a single level first, or use operations.vertical_profile for a layered grid.",
    "no_values": "{name}: {param} has no value column, so its points cannot be coloured.",
    "missing_columns": "{name}: {param} lacks the columns {missing}; a table for a monitor map needs {columns}.",
    "mixed_units": "{name}: {param} holds values in {count} units ({units}); a monitor map takes values in one unit.",
    "unknown_pollutant": "{name}: could not tell which pollutant {param} measures; pass pollutant as one of {choices}.",
    "no_breakpoints": "{name}: the standards file has no {pollutant} AQI breakpoints for {averaging_period}.",
    "unit_mismatch": "{name}: {param} is in {unit} and the {pollutant} AQI breakpoints are in {breakpoint_unit}, which it cannot be converted to.",
    "too_many_units": "{name}: the series are in {count} different units ({units}); a chart has at most {limit} y axes.",
    "too_many_lines": "{name}: the series hold {count} lines; a chart shows at most {limit}.",
    "not_a_profile": "{name}: {param} is not a vertical profile table; it needs the columns {columns}.",
}

_CATEGORY = {"wrong_dims": FailureCategory.UNSUPPORTED_SHAPE}


def refuse(call: Call[Any], reason: str, **facts: Any) -> StepFailure:
    """The failure for ``reason``, worded from observed facts only."""
    return refusal(call, REFUSALS[reason], reason, category=_CATEGORY.get(reason, FailureCategory.BAD_INPUT), **facts)


def region_of(record: HandleRecord) -> dict[str, Any] | None:
    """A handle's recorded region as the payload builders read it (name, bounds, region_type), or None."""
    region = record.descriptor.region
    if not isinstance(region, dict):
        return None
    bounds = region.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4 or not all(isinstance(b, int | float) for b in bounds):
        return None
    found: dict[str, Any] = {"bounds": [float(b) for b in bounds]}  # type: ignore[arg-type]
    if isinstance(region.get("name"), str):
        found["name"] = region["name"]
    if isinstance(region.get("region_type"), str):
        found["region_type"] = region["region_type"]
    return found


def sources(records: tuple[HandleRecord, ...]) -> list[dict[str, Any]]:
    """Each input handle with what its descriptor says it holds."""
    return [
        {
            key: value for key, value in {
                "handle": record.id,
                "kind": record.kind,
                "product": record.descriptor.product,
                "quantity": record.descriptor.quantity,
                "unit": record.descriptor.unit,
                "summary": record.descriptor.summary,
                "period": period_json(record.descriptor.period),
            }.items() if value is not None
        }
        for record in records
    ]


def chart_summary(payload: dict[str, Any]) -> dict[str, JsonValue]:
    """The model-facing summary of a stored chart."""
    summary: dict[str, JsonValue] = chart_model_summary(payload)
    return summary


def with_run(call: Call[Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    """``metadata`` with the run that drew the chart or table."""
    return {**(metadata or {}), "run_id": call.env.run_id}


async def store_and_emit(call: Call[Any], payload: dict[str, Any]) -> ArtifactRef:
    """Store the chart, then emit it with its artifact id."""
    payload = {**payload, "metadata": with_run(call, payload.get("metadata"))}
    context = call.env.context
    # The store assigns the id as it stores (the app's chart repository writes it into the stored payload), so only the emitted copy is given it here.
    ref = await context.artifacts.store_chart(payload)
    context.events.chart({**payload, "chart_id": ref.id})
    return ref


MAX_OVERLAY_POINTS = 2_000


def point_overlay(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    """The first MAX_OVERLAY_POINTS points as overlay entries, and how many there were."""
    frame = points_frame(frame)
    total = len(frame)
    kept = frame.head(MAX_OVERLAY_POINTS)
    points: list[dict[str, Any]] = []
    for record in kept.to_dict(orient="records"):
        point: dict[str, Any] = {
            "id": str(record["id"]),
            "name": None if pd.isna(record.get("name")) else str(record["name"]),
            "lat": float(record["lat"]),
            "lon": float(record["lon"]),
        }
        if "value" in record:
            value = record["value"]
            point["value"] = float(value) if value is not None and np.isfinite(float(value)) else None
        points.append(point)
    return points, total
