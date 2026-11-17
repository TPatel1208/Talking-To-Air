"""``render.table``: a table handle as the table artifact, cut to ``max_rows``.

Stored through ``Artifacts.store_table`` in the existing table artifact's
shape; ``metadata`` adds ``row_count`` (rows in the handle), ``rows_shown``
and ``truncated``. A table artifact is not a chart payload, so nothing is
emitted as a chart.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, JsonValue

from tta_backend.domain.charts.table import table_rows
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.render.common import sources, with_run
from tta_backend.harness.capabilities.render.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult

__all__ = ["CAPABILITIES", "GROUP", "render_table"]

DEFAULT_ROWS = 200
MAX_ROWS = 5_000


class TableParams(Params):
    table: Annotated[str, handle_param("table")] = Field(description="The table handle to show.")
    max_rows: int = Field(default=DEFAULT_ROWS, ge=1, le=MAX_ROWS, description="Most rows to show; the rest are left out with a disclosure.")
    title: str | None = Field(default=None, max_length=200, description="Table title; the handle's summary when omitted.")


def _has_coordinates(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("latitude") is not None and row.get("longitude") is not None for row in rows)


async def _render_table(call: Call[TableParams]) -> CapabilityResult:
    p = call.params
    record = call.handle("table")
    frame = await call.open("table")
    total = len(frame)
    columns, rows = await call.compute(table_rows, frame, p.max_rows)
    truncated = total > len(rows)
    title = p.title or record.descriptor.summary
    metadata: dict[str, Any] = {
        "dataset": record.descriptor.product or "",
        "has_coordinates": _has_coordinates(rows),
        "row_count": total,
        "rows_shown": len(rows),
        "truncated": truncated,
        "unit": record.descriptor.unit,
        "source_handles": [record.id],
        "sources": sources((record,)),
    }
    ref = await call.env.context.artifacts.store_table(title, columns, rows, with_run(call, metadata))
    disclosures = (f"The table shows the first {len(rows)} of {total} rows.",) if truncated else ()
    summary: dict[str, JsonValue] = {
        "render_type": "table",
        "title": title,
        "columns": list(columns),
        "rows": total,
        "rows_shown": len(rows),
        "source_handles": [record.id],
    }
    return call.done(charts=(ref.id,), summary=summary, disclosures=disclosures)


render_table = Capability(
    name="render.table",
    description="Show a table handle as a table, cut to max_rows with a disclosure when rows are left out.",
    params=TableParams,
    output_kind=None,
    label="Preparing a table",
    method="The table was shown, {summary.rows_shown} of {summary.rows} rows.",
    retrieves=False,
    retry_safe=True,
    run=_render_table,
)

CAPABILITIES = (render_table,)
