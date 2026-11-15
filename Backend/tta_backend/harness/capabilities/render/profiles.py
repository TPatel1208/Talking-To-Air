"""``render.profile``: a vertical profile table as the ``profile`` chart.

The table is the one ``operations.vertical_profile`` makes. The payload has
the current profile chart's fields; ``vertical_dim`` is ``layer`` and
``aggregation_meta`` is left out, since the table does not carry them.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import Field, JsonValue

from tta_backend.domain.charts.profile import PROFILE_COLUMNS, profile_export, profile_from_table, profile_payload
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.common import region_name
from tta_backend.harness.capabilities.render.common import chart_summary, refuse, sources, store_and_emit
from tta_backend.harness.capabilities.render.group import GROUP
from tta_backend.harness.capabilities.result import CapabilityResult

__all__ = ["CAPABILITIES", "GROUP", "render_profile"]

NO_VALUES = "The profile table has no valid values, so there is no profile to draw."


class ProfileParams(Params):
    profile: Annotated[str, handle_param("table")] = Field(description="A table from operations.vertical_profile.")
    title: str | None = Field(default=None, max_length=200, description="Chart title; made from the variable and area when omitted.")


async def _render_profile(call: Call[ProfileParams]) -> CapabilityResult:
    p = call.params
    record = call.handle("profile")
    frame = await call.open("profile")
    try:
        profile = profile_from_table(frame)
    except ValueError:
        raise refuse(call, "not_a_profile", param="profile", columns=", ".join(PROFILE_COLUMNS)) from None
    if not any(v is not None for v in profile.values):
        return CapabilityResult(status="insufficient_evidence", reason=NO_VALUES, values={"situation": "too_little_valid"})

    variable = record.descriptor.quantity or ""
    place = region_name(record)
    title = p.title or f"{variable or 'Value'} vertical profile over {place}"
    payload = profile_payload(
        profile, title=title, variable=variable, units=record.descriptor.unit or "",
        masking=record.descriptor.masking,
    )
    handles = list(call.input_ids())
    period = record.descriptor.period
    payload["provenance"] = {
        "variable": variable,
        "units": payload["units"],
        "region_name": place,
        "start_date": period.start if period else "",
        "end_date": period.end if period else "",
        "aggregation": "mean",
        "source_handles": handles,
        "sources": sources((record,)),
    }
    payload["export"] = {
        "type": "profile",
        "variable": variable,
        "units": payload["units"],
        "region_name": place,
        "chart_parameters": {"chart_type": "profile"},
        "source_handles": handles,
        **profile_export(payload),
    }
    payload["metadata"] = {"source_handles": handles}

    ref = await store_and_emit(call, payload)
    values: dict[str, JsonValue] = {"layers": len(profile.layers), "default_axis": profile.default_axis}
    return call.done(charts=(ref.id,), summary=chart_summary(payload), values=values)


render_profile = Capability(
    name="render.profile",
    description="Draw a vertical profile table (from operations.vertical_profile) as a profile chart against pressure or altitude.",
    params=ProfileParams,
    output_kind=None,
    label="Drawing a vertical profile",
    method="A vertical profile of {layers} layers was drawn.",
    retrieves=False,
    retry_safe=True,
    run=_render_profile,
)

CAPABILITIES = (render_profile,)
