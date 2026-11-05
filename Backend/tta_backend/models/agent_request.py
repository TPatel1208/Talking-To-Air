"""
agent_request.py
----------------
T64: the typed requests the supervisor hands its sub-agents through the
``dispatch`` tool, in place of a free-text task string.
"""
from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, Field


# The intent enums' docstrings are the descriptions the model reads in the
# dispatch schema ($defs), one line per value: keep them model-facing. They
# sit on the enum, not the field, so each intent field stays a bare $ref --
# the shape the T64 Phase 0 gate verified.
class SatelliteIntent(StrEnum):
    """What the satellite request is for.
    visualize: a map, time series or vertical profile of the variable, also when ground readings are wanted alongside; those are a separate ground request.
    statistic: numbers about the variable over the area and dates (mean, max, peak location), no ground data.
    compare: two periods, regions or variables; or checking the satellite data against ground monitors, only when the researcher asks for that check.
    explain_reliability: how far to trust a chart already plotted (quality flags, coverage).
    availability: whether data exists for the variable, area and dates.
    discover: which datasets or variables exist for a topic.
    """

    VISUALIZE = "visualize"
    STATISTIC = "statistic"
    COMPARE = "compare"
    EXPLAIN_RELIABILITY = "explain_reliability"
    AVAILABILITY = "availability"
    DISCOVER = "discover"


class FigureType(StrEnum):
    """The researcher's requested output. Each value matches one chart
    ``type`` the plot tools emit: map=heatmap, map_multi=heatmap_multi,
    timeseries=timeseries, vertical_profile=profile."""

    MAP = "map"
    MAP_MULTI = "map_multi"
    TIMESERIES = "timeseries"
    VERTICAL_PROFILE = "vertical_profile"
    NONE = "none"


class GroundIntent(StrEnum):
    """What the ground request is for.
    nearest_monitor: the monitor(s) closest to a place.
    site_info: details of a named monitor or station.
    readings: daily or hourly values, or the AQI, for a place and dates.
    summary: quarterly or annual summaries.
    exceedances: days a monitor exceeded the pollutant's regulatory standard.
    """

    NEAREST_MONITOR = "nearest_monitor"
    SITE_INFO = "site_info"
    READINGS = "readings"
    SUMMARY = "summary"
    EXCEEDANCES = "exceedances"


Pollutant = Literal["NO2", "PM2.5", "O3", "SO2", "CO"]

_LOCATION_DOC = (
    "The place exactly as the researcher wrote it, character for character "
    "(e.g. 'the OTC', 'NJ + NY'). Do not expand acronyms, add words, or describe it."
)
_PRIOR_FINDINGS_DOC = (
    "Facts from an earlier result this task depends on, e.g. the peak lat/lon "
    "a satellite result reported. Omit when there are none."
)


class SatelliteRequest(BaseModel, extra="forbid"):
    """A task for the satellite agent: any regularly-gridded NASA Earthdata
    collection, global, any environmental domain."""

    agent: Literal["satellite"] = "satellite"
    intent: SatelliteIntent
    figure_type: FigureType = FigureType.NONE
    variable: str = Field(description="The variable as the researcher named it, e.g. 'NO2', 'soil moisture'.")
    dataset: str | None = Field(default=None, description="Instrument or product, only if the researcher named one.")
    location: str = Field(description=_LOCATION_DOC)
    start_date: date | None = None
    end_date: date | None = None
    prior_findings: str | None = Field(default=None, max_length=500, description=_PRIOR_FINDINGS_DOC)


class GroundRequest(BaseModel, extra="forbid"):
    """A task for the ground agent: EPA AQS monitors, US only, the five
    air-quality pollutants only."""

    agent: Literal["ground"] = "ground"
    intent: GroundIntent
    pollutant: Pollutant
    location: str = Field(description=_LOCATION_DOC)
    start_date: date | None = None
    end_date: date | None = None
    prior_findings: str | None = Field(default=None, max_length=500, description=_PRIOR_FINDINGS_DOC)


AgentRequest = Annotated[SatelliteRequest | GroundRequest, Field(discriminator="agent")]

# A schema constant rather than a setting: it is part of the tool declaration
# the model sees. The per-turn budget and concurrency caps are runtime settings.
MAX_REQUESTS_PER_DISPATCH = 6


class DispatchArgs(BaseModel):
    """The dispatch tool's arguments: the tasks to run this round."""

    requests: list[AgentRequest] = Field(min_length=1, max_length=MAX_REQUESTS_PER_DISPATCH)


def canonical_key(request: AgentRequest) -> str:
    """The request's identity for dedupe within a turn: every typed field,
    serialized with sorted keys. ``prior_findings`` is context, not part of
    the task, so a model rewording it still lands on the same key."""
    return json.dumps(request.model_dump(mode="json", exclude={"prior_findings"}), sort_keys=True)


def render_task(request: AgentRequest, user_message: str | None) -> str:
    """The task string a sub-agent receives: the typed fields as labeled
    lines, which are its task, then the researcher's own message (bound
    server-side, never written by the model) as context. The message comes
    last and is labeled context because one question often spans several
    requests, and a sub-agent that takes it as its task widens into the
    parts other agents were sent. A pure template, so the same request
    always renders to the same text."""
    lines = [f"Intent: {request.intent}"]
    if isinstance(request, SatelliteRequest):
        lines.append(f"Figure: {request.figure_type}")
        lines.append(f"Variable: {request.variable}")
        if request.dataset:
            lines.append(f"Dataset: {request.dataset}")
    else:
        lines.append(f"Pollutant: {request.pollutant}")
    lines.append(f"Location (use exactly as written): {request.location}")
    dates = _render_dates(request.start_date, request.end_date)
    if dates:
        lines.append(dates)
    if request.prior_findings:
        lines.append(f"Prior findings: {request.prior_findings}")
    if user_message:
        lines.append(f"Researcher's full message (context only): {user_message}")
    return "\n".join(lines)


def _render_dates(start: date | None, end: date | None) -> str | None:
    if start and end:
        return f"Dates: {start} to {end}"
    if start:
        return f"Date: {start}"
    if end:
        return f"Dates: up to {end}"
    return None


def dispatch_json_schema() -> dict[str, Any]:
    """DispatchArgs as the JSON Schema the model is given.

    This is the shape the T64 Phase 0 gate verified on every vetted Gemini
    model: the union as a plain ``anyOf``, with the list caps. The
    discriminator (``oneOf`` + ``discriminator``) and ``extra="forbid"``
    (``additionalProperties``) were not part of that gate, so they are
    removed here and enforced by validating against DispatchArgs instead.

    Each branch's ``agent`` tag is marked required. Validation picks the
    branch by it, but the field has a default, so the schema called it
    optional once the discriminator was gone, and a model that left it out
    had its first call rejected."""
    schema = cast(dict[str, Any], _gated_shape(DispatchArgs.model_json_schema()))
    for branch in schema.get("$defs", {}).values():
        agent = branch.get("properties", {}).get("agent")
        if isinstance(agent, dict) and "const" in agent:
            agent.pop("default", None)
            branch["required"] = ["agent", *(name for name in branch.get("required", []) if name != "agent")]
    return schema


def _gated_shape(node: Any) -> Any:
    if isinstance(node, list):
        return [_gated_shape(value) for value in node]
    if not isinstance(node, dict):
        return node
    shaped = {}
    for key, value in node.items():
        if key in ("discriminator", "additionalProperties"):
            continue
        shaped["anyOf" if key == "oneOf" else key] = _gated_shape(value)
    return shaped
