"""T64 Phase 0 gate: does every vetted supervisor model accept the typed
``dispatch(requests: list[AgentRequest])`` tool, and fill it correctly?

Four questions per vetted model (settings._VETTED_MODELS["google"]):

1. Schema accepted -- binding the discriminated-union schema and calling the
   model raises no provider error.
2. Arguments validate -- the tool-call args round-trip through the pydantic
   classes the backend would actually execute.
3. Fields correct -- per-prompt checks (agent, intent, figure_type, verbatim
   location, dates, item count).
4. Legacy history -- a thread holding a synthetic ``ask_earthdata_agent``
   call (the shape the fast-path write-back stores today) for a function no
   longer declared is accepted, and the model answers with ``dispatch``.

``--schema flat`` runs the D3 fallback (one flat class, combined intent enum)
instead of the union.

The request classes here are a draft for the gate only; Phase 1 builds the
real ones in tta_backend/models/agent_request.py test-first.

Network-touching and never a test. Results are transcribed into
docs/prds/prd-t64-phase0-gate-verdict.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402

from tta_backend.config.model_factory import build_chat_model  # noqa: E402
from tta_backend.config.settings import _VETTED_MODELS, get_settings  # noqa: E402

# ── Draft schema (union) ──────────────────────────────────────────────────────


class SatelliteIntent(StrEnum):
    VISUALIZE = "visualize"
    STATISTIC = "statistic"
    COMPARE = "compare"
    EXPLAIN_RELIABILITY = "explain_reliability"
    AVAILABILITY = "availability"
    DISCOVER = "discover"


class FigureType(StrEnum):
    MAP = "map"
    MAP_MULTI = "map_multi"
    TIMESERIES = "timeseries"
    VERTICAL_PROFILE = "vertical_profile"
    NONE = "none"


class GroundIntent(StrEnum):
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


class SatelliteRequest(BaseModel):
    """A task for the satellite agent (any gridded NASA Earthdata product, global)."""

    agent: Literal["satellite"] = "satellite"
    intent: SatelliteIntent
    figure_type: FigureType = FigureType.NONE
    variable: str = Field(description="The variable as the researcher named it, e.g. 'NO2', 'soil moisture'.")
    dataset: str | None = Field(default=None, description="Instrument/product only if the researcher named one.")
    location: str = Field(description=_LOCATION_DOC)
    start_date: date | None = None
    end_date: date | None = None
    prior_findings: str | None = Field(default=None, max_length=500)


class GroundRequest(BaseModel):
    """A task for the ground agent (EPA AQS monitors, US only, air-quality pollutants only)."""

    agent: Literal["ground"] = "ground"
    intent: GroundIntent
    pollutant: Pollutant
    location: str = Field(description=_LOCATION_DOC)
    start_date: date | None = None
    end_date: date | None = None
    prior_findings: str | None = Field(default=None, max_length=500)


AgentRequest = Annotated[SatelliteRequest | GroundRequest, Field(discriminator="agent")]


class UnionDispatchArgs(BaseModel):
    requests: list[AgentRequest] = Field(min_length=1, max_length=6)


class PlainUnionDispatchArgs(BaseModel):
    """Same union without the discriminator: pydantic emits ``anyOf`` here
    rather than ``oneOf`` + ``discriminator``, the subset Gemini's JSON-schema
    parameters document. Used only to produce the native declaration;
    validation still goes through UnionDispatchArgs."""

    requests: list[SatelliteRequest | GroundRequest] = Field(min_length=1, max_length=6)


# ── Draft schema (flat fallback, D3) ──────────────────────────────────────────

_SATELLITE_INTENTS = {i.value for i in SatelliteIntent}
_GROUND_INTENTS = {i.value for i in GroundIntent}
CombinedIntent = Literal[
    "visualize", "statistic", "compare", "explain_reliability", "availability", "discover",
    "nearest_monitor", "site_info", "readings", "summary", "exceedances",
]


class FlatRequest(BaseModel):
    """One sub-agent task. satellite = gridded NASA Earthdata (global); ground =
    EPA AQS monitors (US, air-quality pollutants only)."""

    agent: Literal["satellite", "ground"]
    intent: CombinedIntent = Field(description=(
        "satellite: visualize|statistic|compare|explain_reliability|availability|discover. "
        "ground: nearest_monitor|site_info|readings|summary|exceedances."
    ))
    figure_type: FigureType = Field(default=FigureType.NONE, description="satellite only.")
    variable: str | None = Field(default=None, description="satellite only: the variable as named.")
    dataset: str | None = Field(default=None, description="satellite only: instrument/product if named.")
    pollutant: Pollutant | None = Field(default=None, description="ground only.")
    location: str = Field(description=_LOCATION_DOC)
    start_date: date | None = None
    end_date: date | None = None
    prior_findings: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _fields_match_agent(self) -> FlatRequest:
        if self.agent == "satellite":
            if self.intent not in _SATELLITE_INTENTS:
                raise ValueError(f"intent {self.intent!r} is not a satellite intent")
            if self.pollutant is not None:
                raise ValueError("pollutant is ground-only")
            if not self.variable:
                raise ValueError("satellite requests need a variable")
        else:
            if self.intent not in _GROUND_INTENTS:
                raise ValueError(f"intent {self.intent!r} is not a ground intent")
            if self.pollutant is None:
                raise ValueError("ground requests need a pollutant")
            if self.figure_type != FigureType.NONE or self.variable or self.dataset:
                raise ValueError("figure_type/variable/dataset are satellite-only")
        return self


class FlatDispatchArgs(BaseModel):
    requests: list[FlatRequest] = Field(min_length=1, max_length=6)


_DISPATCH_DOC = (
    "Dispatch one or more tasks to the sub-agents. Independent tasks go in one "
    "call and run in parallel; a task that needs another's result goes in a "
    "second call after the first returns."
)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _native_dispatch_tool(inline: bool) -> Any:
    """The union handed to Gemini as raw JSON Schema (parameters_json_schema),
    bypassing langchain_google_genai's converter, which reduces a list of
    ``anyOf`` items to a bare OBJECT. bind_tools passes a types.Tool through
    unconverted."""
    from google.genai import types

    schema = PlainUnionDispatchArgs.model_json_schema()
    if inline:
        schema = _inline_refs(schema)
    return types.Tool(function_declarations=[types.FunctionDeclaration(
        name="dispatch", description=_DISPATCH_DOC, parameters_json_schema=schema,
    )])


def _dispatch_tool(args_schema: type[BaseModel]) -> StructuredTool:
    async def _noop(**_: Any) -> str:
        return "{}"

    return StructuredTool.from_function(
        coroutine=_noop, name="dispatch", description=_DISPATCH_DOC, args_schema=args_schema,
    )


# A deliberately minimal supervisor prompt: agent descriptions only. No
# place-name/STOP/AQ-only rules -- the gate measures what the SCHEMA enforces.
_SYSTEM = """You are a supervisor for an environmental data assistant. Today is 2026-10-16.
Agents:
- satellite: any gridded NASA Earthdata collection, global. Maps, time-series,
  vertical profiles, spatial statistics, any environmental domain.
- ground: EPA AQS ground monitors, US only, air-quality pollutants only
  (NO2, PM2.5, O3, SO2, CO). Nearest monitor, site info, readings, summaries,
  exceedances.
Answer every data request by calling the dispatch tool. Resolve relative dates
to ISO dates."""


# ── Prompt cases ──────────────────────────────────────────────────────────────

Check = Callable[[str, list[dict[str, Any]]], list[str]]


def _items(args: dict[str, Any]) -> list[dict[str, Any]]:
    reqs = args.get("requests")
    return [r for r in reqs if isinstance(r, dict)] if isinstance(reqs, list) else []


def _non_object_items(args: dict[str, Any]) -> list[Any]:
    """Items the model sent as something other than an object -- what the
    langchain converter's `items: STRING` declaration invites."""
    reqs = args.get("requests")
    return [r for r in reqs if not isinstance(r, dict)] if isinstance(reqs, list) else [reqs]


def _of(items: list[dict[str, Any]], agent: str) -> list[dict[str, Any]]:
    return [i for i in items if i.get("agent") == agent]


def _verbatim(message: str, item: dict[str, Any], expected: str) -> list[str]:
    loc = str(item.get("location") or "")
    if loc not in message or expected.lower() not in loc.lower():
        return [f"location {loc!r} not verbatim (expected substring {expected!r})"]
    return []


def _expect(item: dict[str, Any], **want: Any) -> list[str]:
    return [f"{k}={item.get(k)!r}, want {v!r}" for k, v in want.items() if item.get(k) != v]


def check_single_ground(msg: str, items: list[dict[str, Any]]) -> list[str]:
    if len(items) != 1 or not _of(items, "ground"):
        return [f"want 1 ground item, got {[i.get('agent') for i in items]}"]
    i = items[0]
    return _expect(i, intent="nearest_monitor", pollutant="NO2") + _verbatim(msg, i, "Tampa")


def check_single_sat_map(msg: str, items: list[dict[str, Any]]) -> list[str]:
    if len(items) != 1 or not _of(items, "satellite"):
        return [f"want 1 satellite item, got {[i.get('agent') for i in items]}"]
    i = items[0]
    errs = _expect(i, intent="visualize", figure_type="map", start_date="2024-01-15")
    if i.get("end_date") not in (None, "2024-01-15"):
        errs.append(f"end_date={i.get('end_date')!r}")
    if "tropomi" not in str(i.get("dataset") or "").lower():
        errs.append(f"dataset={i.get('dataset')!r}, want TROPOMI")
    return errs + _verbatim(msg, i, "New Jersey")


def check_cross(msg: str, items: list[dict[str, Any]]) -> list[str]:
    g, s = _of(items, "ground"), _of(items, "satellite")
    errs = []
    if not g:
        errs.append("no ground item")
    elif g[0].get("intent") != "exceedances" or g[0].get("pollutant") != "PM2.5":
        errs.append(f"ground intent/pollutant {g[0].get('intent')!r}/{g[0].get('pollutant')!r}")
    if not s:
        errs.append("no satellite item (ok only if dispatched in a later round)")
    return errs


def check_two_region(msg: str, items: list[dict[str, Any]]) -> list[str]:
    s = _of(items, "satellite")
    if len(s) != 2:
        return [f"want 2 satellite items, got {len(s)} ({[i.get('figure_type') for i in s]})"]
    errs = []
    for item, place in zip(s, ("New Jersey", "New York"), strict=True):
        errs += _expect(item, figure_type="map") + _verbatim(msg, item, place)
    return errs


def check_otc(msg: str, items: list[dict[str, Any]]) -> list[str]:
    if len(items) != 1 or not _of(items, "satellite"):
        return [f"want 1 satellite item, got {[i.get('agent') for i in items]}"]
    return _verbatim(msg, items[0], "OTC") + _expect(items[0], figure_type="map")


def check_non_aq(msg: str, items: list[dict[str, Any]]) -> list[str]:
    errs = []
    if _of(items, "ground"):
        errs.append(f"emitted a ground item: {_of(items, 'ground')}")
    if not _of(items, "satellite"):
        errs.append("no satellite item")
    return errs


def check_timeseries(msg: str, items: list[dict[str, Any]]) -> list[str]:
    if len(items) != 1 or not _of(items, "satellite"):
        return [f"want 1 satellite item, got {[i.get('agent') for i in items]}"]
    i = items[0]
    return _expect(i, figure_type="timeseries", start_date="2025-08-01", end_date="2025-08-07") + _verbatim(
        msg, i, "Los Angeles"
    )


def check_profile(msg: str, items: list[dict[str, Any]]) -> list[str]:
    if len(items) != 1 or not _of(items, "satellite"):
        return [f"want 1 satellite item, got {[i.get('agent') for i in items]}"]
    return _expect(items[0], figure_type="vertical_profile") + _verbatim(msg, items[0], "Houston")


CASES: list[tuple[str, str, Check]] = [
    ("single_ground", "Find the closest NO2 monitor to Tampa FL", check_single_ground),
    ("single_sat_map", "Plot TROPOMI NO2 over New Jersey for 2024-01-15", check_single_sat_map),
    ("cross_check",
     "Were there PM2.5 exceedances in Phoenix in July 2024, and does satellite aerosol optical "
     "depth show the same event?", check_cross),
    ("two_region", "Plot NO2 over New Jersey and, separately, over New York for 2024-07-01", check_two_region),
    ("otc_verbatim", "Plot ozone over the OTC for 2024-07-15", check_otc),
    ("non_aq_ground",
     "Show soil moisture over Iowa for June 2024 and check it against ground monitors", check_non_aq),
    ("timeseries",
     "Show a time series of TEMPO NO2 over Los Angeles for the first week of August 2025", check_timeseries),
    ("vertical_profile", "Plot the vertical ozone profile from TEMPO over Houston on 2025-06-10", check_profile),
]


# ── Runner ────────────────────────────────────────────────────────────────────


def _validate(args_schema: type[BaseModel], args: dict[str, Any]) -> str | None:
    try:
        args_schema.model_validate(args)
    except ValidationError as exc:
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:4])
    return None


def _dispatch_calls(response: Any) -> list[dict[str, Any]]:
    return [tc for tc in getattr(response, "tool_calls", None) or [] if tc.get("name") == "dispatch"]


_TRANSIENT = ("RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
RETRIES = {"n": 0, "wait": 30.0}


async def _invoke(bound: Any, messages: list[Any]) -> Any:
    """Retry per-minute quota (429) and overload (503) errors: they say
    nothing about the schema, and a cell lost to them is a coverage gap."""
    for attempt in range(RETRIES["n"] + 1):
        try:
            return await bound.ainvoke(messages)
        except Exception as exc:
            if attempt == RETRIES["n"] or not any(t in str(exc) for t in _TRANSIENT):
                raise
            await asyncio.sleep(RETRIES["wait"] * (attempt + 1))
    raise AssertionError("unreachable")


async def run_case(bound: Any, args_schema: type[BaseModel], name: str, message: str, check: Check) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {"case": name}
    try:
        response = await _invoke(bound, [SystemMessage(_SYSTEM), HumanMessage(message)])
    except Exception as exc:
        row.update(provider_error=f"{type(exc).__name__}: {str(exc)[:400]}")
        return row
    row["seconds"] = round(time.monotonic() - started, 1)
    calls = _dispatch_calls(response)
    other = [tc.get("name") for tc in getattr(response, "tool_calls", None) or [] if tc.get("name") != "dispatch"]
    row["dispatch_calls"] = len(calls)
    if other:
        row["other_tool_calls"] = other
    if not calls:
        row["text"] = str(response.content)[:300]
        row["field_errors"] = ["no dispatch call"]
        return row
    # Native parallel tool calls are flattened: the executor would run them
    # concurrently too, so the gate judges the union of their items.
    items: list[dict[str, Any]] = []
    validation = []
    non_objects: list[Any] = []
    for call in calls:
        items.extend(_items(call["args"]))
        non_objects.extend(_non_object_items(call["args"]))
        err = _validate(args_schema, call["args"])
        if err:
            validation.append(err)
    row["items"] = items
    if non_objects:
        row["non_object_items"] = [str(x)[:300] for x in non_objects]
    row["validation_error"] = validation or None
    row["field_errors"] = check(message, items)
    return row


async def run_legacy_history(bound: Any) -> dict[str, Any]:
    history = [
        SystemMessage(_SYSTEM),
        HumanMessage("Plot TROPOMI NO2 over New Jersey for 2024-01-15"),
        AIMessage(content="", tool_calls=[{
            "name": "ask_earthdata_agent",
            "args": {"task": "Plot TROPOMI NO2 over New Jersey for 2024-01-15"},
            "id": "call_legacy_1",
        }]),
        ToolMessage(
            content=json.dumps({"text": "Plotted TROPOMI NO2 over New Jersey for 2024-01-15; peak 8.1e15 near Newark."}),
            tool_call_id="call_legacy_1",
            name="ask_earthdata_agent",
        ),
        AIMessage(content="Agent consulted: SATELLITE\n\nThe map shows NO2 peaking near Newark."),
        HumanMessage("Now find the closest NO2 monitor to Newark NJ"),
    ]
    try:
        response = await _invoke(bound, history)
    except Exception as exc:
        return {"case": "legacy_history", "provider_error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    names = [tc.get("name") for tc in getattr(response, "tool_calls", None) or []]
    return {
        "case": "legacy_history",
        "tool_calls": names,
        "field_errors": [] if names and all(n == "dispatch" for n in names) else [f"tool calls {names}"],
    }


def _declaration_preview(tool: StructuredTool) -> Any:
    """The function declaration langchain_google_genai actually sends, so a
    rejected or silently-flattened union can be read off, not guessed."""
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations

    return [t.model_dump(mode="json", exclude_none=True) for t in convert_to_genai_function_declarations([tool])]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=("union", "native", "flat"), default="union")
    parser.add_argument("--inline-refs", action="store_true", help="native only: inline $defs")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--cases", nargs="*", default=None, help="subset of case names")
    parser.add_argument("--retries", type=int, default=0, help="retries on 429/503, with linear backoff")
    args = parser.parse_args()
    RETRIES["n"] = args.retries

    settings = get_settings()
    args_schema = FlatDispatchArgs if args.schema == "flat" else UnionDispatchArgs
    if args.schema == "native":
        tool = _native_dispatch_tool(args.inline_refs)
        declaration = tool.model_dump(mode="json", exclude_none=True)
    else:
        tool = _dispatch_tool(args_schema)
        declaration = _declaration_preview(tool)
    models = args.models or sorted(_VETTED_MODELS["google"])
    report: dict[str, Any] = {
        "schema": args.schema, "inline_refs": args.inline_refs, "reps": args.reps,
        "declaration": declaration, "models": {},
    }

    for model in models:
        llm = build_chat_model("google", model, settings, agent_type="supervisor")
        try:
            # bind_tools round-trips every tool through a dict and crashes on a
            # declaration whose `parameters` is None (parameters_json_schema
            # set instead); bind() hands the types.Tool to _format_tools as-is.
            bound = llm.bind(tools=[tool]) if args.schema == "native" else llm.bind_tools([tool])
        except Exception as exc:
            report["models"][model] = {"bind_error": f"{type(exc).__name__}: {exc}"}
            print(f"{model}: BIND ERROR {exc}", flush=True)
            continue
        rows = []
        for rep in range(args.reps):
            for name, message, check in CASES:
                if args.cases and name not in args.cases:
                    continue
                row = await run_case(bound, args_schema, name, message, check)
                row["rep"] = rep
                rows.append(row)
                status = (
                    "PROVIDER_ERROR" if row.get("provider_error")
                    else "INVALID" if row.get("validation_error")
                    else "FIELDS" if row.get("field_errors")
                    else "ok"
                )
                print(f"{model} rep{rep} {name}: {status} {row.get('provider_error') or row.get('validation_error') or row.get('field_errors') or ''}", flush=True)
                await asyncio.sleep(args.pause)
        legacy = await run_legacy_history(bound)
        print(f"{model} legacy_history: {legacy}", flush=True)
        report["models"][model] = {"cases": rows, "legacy_history": legacy}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())
