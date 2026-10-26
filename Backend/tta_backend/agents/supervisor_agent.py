"""
Supervisor agent that orchestrates the Ground Sensor and Satellite agents.

Memory model
------------
- Supervisor : stateful — uses a Postgres checkpointer, one thread per session.
- Subagents  : stateless — no checkpointer; each run is a fresh invocation.
               The supervisor names their work as typed requests through its
               one tool, dispatch (T64); each request is rendered into a task
               string under the researcher's own message.
"""
import logging
from typing import Any, cast
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AnyMessage, trim_messages
from langchain_core.tools import StructuredTool
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from collections.abc import Awaitable, Callable

from tta_backend.config.model_factory import NATIVE_JSON_SCHEMA, build_chat_model
from tta_backend.config.settings import get_settings
from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT
from tta_backend.models import (
    AgentRequest,
    AgentResult,
    DispatchArgs,
    DispatchItemResult,
    DispatchResult,
    GroundRequest,
    SatelliteIntent,
    canonical_key,
    dispatch_json_schema,
    parse_agent_result,
    parse_chart_payload,
    parse_dispatch_result,
    render_task,
)
from tta_backend.services.chart_service import ChartService
from tta_backend.services.dispatch_executor import begin_model_step, dispatch_persist_lock, execute_dispatch
from tta_backend.services.subagent_dispatch import persist_dispatch_contexts, run_ground, run_satellite
from tta_backend.utils.db import get_checkpointer
from tta_backend.utils.message_utils import truncate_text
from tta_backend.utils.streaming import (
    current_thread_id,
    current_user_id,
    current_user_message,
    ground_validation_scope,
)

logger = logging.getLogger(__name__)


# ── Build supervisor ──────────────────────────────────────────────────────────

async def build_agent(
    model: str | None = None,
    provider: str | None = None,
    *,
    ground_agent: Any,
    satellite_agent: Any,
    mcp_manager: Any = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Build and return the supervisor agent.

    The supervisor is the only stateful component — it owns the Postgres
    checkpointer and persists the full conversation history under one
    thread_id per user session.

    ``ground_agent``/``satellite_agent`` are the already-built stateless
    sub-agents (see agents/ground_sensor_agent.py, agents/earthdata_agent.py)
    — built once at startup (api.py's lifespan) and shared with the router
    fast path (services/chat_stream_service.py, T14) so both callers invoke
    the identical instances via services/subagent_dispatch.py.

    ``mcp_manager`` (T17), if given, is passed through to
    ``run_satellite`` so a satellite request returns the deterministic
    unavailable answer instead of burning a model call when the
    earthdata-retrieval MCP isn't ready.
    """
    settings = get_settings()
    model = model or settings.llm_model
    provider = provider or settings.supervisor_model_provider
    logger.info(
        "supervisor_model",
        extra={"_event": "supervisor_model", "_model": model, "_provider": provider},
    )
    llm = build_chat_model(
        provider,
        model,
        settings,
        agent_type="supervisor",
        fallback_model=settings.llm_fallback_model,
    )

    # ── Trim middleware — keeps the supervisor's context window bounded ───────

    @wrap_model_call
    async def trim_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        messages = [_compact_model_input_message(msg) for msg in request.state["messages"]]
        trimmed: list[AnyMessage] = cast(list[AnyMessage], trim_messages(
            messages,
            max_tokens=8000,
            strategy="last",
            token_counter="approximate",
            include_system=True,
            allow_partial=False,
            start_on="human",
        ))
        # Gemini rejects an empty contents list, so never let trimming remove
        # the only usable turn from a request. Fall back to the original
        # message list if trimming collapses everything.
        if not trimmed:
            trimmed = messages
        return await handler(request.override(messages=trimmed))

    # ── Dispatch closes once it can do nothing more ───────────────────────────

    @wrap_model_call
    async def dispatch_limit_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # Each model call is a dispatch step. Once both rounds or the run
        # budget are used, or a step's calls started nothing new, the model is
        # forbidden tool calls, so it answers instead of repeating a call
        # until the recursion limit.
        if not begin_model_step():
            request = request.override(tool_choice="none")
        return await handler(request)

    # ── The one sub-agent tool ────────────────────────────────────────────────

    async def dispatch(requests: list[AgentRequest]) -> str:
        user_message = current_user_message()
        thread_id = current_thread_id()
        # canonical_key -> (agent, the context that run handed back). Only the
        # runs this call starts land here; a repeat of an earlier run was
        # already saved by the call that started it.
        captured: dict[str, tuple[str, dict[str, str]]] = {}

        async def run_one(request: AgentRequest) -> AgentResult:
            sink: dict[str, str] = {}
            captured[canonical_key(request)] = (request.agent, sink)
            task = render_task(request, user_message)
            if isinstance(request, GroundRequest):
                return await run_ground(ground_agent, task, thread_id, context_sink=sink)
            # The satellite agent's own ground tools are for a comparison only;
            # any other ground data is a ground request the supervisor makes.
            with ground_validation_scope(request.intent == SatelliteIntent.COMPARE):
                return await run_satellite(
                    satellite_agent, task, thread_id, mcp_manager=mcp_manager, context_sink=sink,
                )

        batch = await execute_dispatch(requests, run_one)
        if thread_id is not None and captured:
            in_order = [captured.pop(key) for key in map(canonical_key, requests) if key in captured]
            # Sibling dispatch calls from one model message finish together.
            async with dispatch_persist_lock():
                await persist_dispatch_contexts(
                    thread_id,
                    satellite=[sink for agent, sink in in_order if agent == "satellite"],
                    ground=[sink for agent, sink in in_order if agent == "ground"],
                )
        return await _checkpoint_dispatch_envelope(batch)

    dispatch_tool = StructuredTool.from_function(
        coroutine=dispatch,
        name="dispatch",
        description=_DISPATCH_DESCRIPTION,
        args_schema=DispatchArgs,
        # Sent to the model verbatim: langchain's own converter reduces the
        # request union to a bare type (T64 Phase 0 F1).
        metadata={NATIVE_JSON_SCHEMA: dispatch_json_schema()},
    )

    # ── Build supervisor ──────────────────────────────────────────────────────
    checkpointer = await get_checkpointer()
    supervisor = create_agent(
        model=llm,
        tools=[dispatch_tool],
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
        middleware=[trim_middleware, dispatch_limit_middleware],
    )
    return supervisor


# ── Helpers ───────────────────────────────────────────────────────────────────
# Sub-agent invocation, envelope finalization, and monitor-context helpers
# live in services/subagent_dispatch.py — shared by these tool wrappers and
# the router fast path (T14). Only supervisor-model-input compaction (used
# solely by this module's own trim_middleware) stays here.


_DISPATCH_DESCRIPTION = """Run one or more tasks on the sub-agents and return each one's result.

satellite: any regularly-gridded NASA Earthdata collection, global, any
environmental domain -- maps, time series, vertical profiles, spatial
statistics, comparisons, reliability of a plotted measurement, data availability.
ground: EPA AQS monitors, US only, the five air-quality pollutants only --
nearest monitor, site info, daily/hourly readings, summaries, exceedances.

Put every independent task in one call; they run in parallel. A task that needs
another's result (e.g. ground readings at a satellite peak) goes in a second
call, with that result in prior_findings. A request repeated in one turn is
answered from its first run."""


async def _checkpoint_dispatch_envelope(batch: DispatchResult) -> str:
    """The tool's return value, which LangGraph checkpoints as a ToolMessage.

    Charts go in as references to their agent_charts rows. Without a bound
    user and thread there is no row to point at, so the charts stay inline
    rather than being lost."""
    thread_id, user_id = current_thread_id(), current_user_id()
    has_charts = any(item.result is not None and item.result.charts for item in batch.results)
    if not has_charts or thread_id is None or user_id is None:
        return batch.model_dump_json(exclude_none=True)
    return await ChartService().checkpoint_dispatch_envelope(batch, thread_id, user_id)


def _truncate_text(text: str, max_chars: int, agent_name: str, request_id: str | None = None) -> str:
    return truncate_text(text, max_chars, agent_name, request_id)


def _compact_model_input_message(msg: Any) -> Any:
    """Replace bulky chart payloads with concise summaries before LLM calls.
    Only a ToolMessage carries a sub-agent envelope; every other message is
    returned untouched without parsing, since this runs over the whole
    history before every model call."""
    if getattr(msg, "type", None) != "tool":
        return msg
    content = getattr(msg, "content", None)
    compacted = _compact_model_input_content(content)
    if compacted is content:
        return msg
    if hasattr(msg, "model_copy"):
        return msg.model_copy(update={"content": compacted})
    try:
        copied = msg.copy()
        copied.content = compacted
        return copied
    except Exception:
        return msg


def _compact_model_input_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content

    batch = parse_dispatch_result(content)
    if batch is not None:
        return "\n\n".join(_compact_dispatch_item(item) for item in batch.results)

    result = parse_agent_result(content)
    if result is not None and result.charts:
        summaries = [
            _chart_summary(chart) or f"chart {index}"
            for index, chart in enumerate(result.charts, start=1)
        ]
        chart_text = "; ".join(summaries)
        return f"{result.text}\n\nCharts generated: {chart_text}"

    chart = parse_chart_payload(content)
    if chart is not None:
        summary = _chart_summary(chart)
        return f"Chart generated: {summary}" if summary else "Chart generated."

    return content


def _compact_dispatch_item(item: DispatchItemResult) -> str:
    """One dispatch item as the supervisor model sees it: its position,
    agent/intent and status, then the sub-agent's text and a one-line chart
    summary in place of any grid."""
    header = f"[{item.index + 1}] {item.agent}/{item.intent}: {item.status}"
    if item.reason:
        header += f" ({item.reason})"
    lines = [header]
    if item.result is not None:
        if item.result.text:
            lines.append(item.result.text)
        if item.result.charts:
            summaries = [
                _chart_summary(chart) or f"chart {index}"
                for index, chart in enumerate(item.result.charts, start=1)
            ]
            lines.append(f"Charts generated: {'; '.join(summaries)}")
    return "\n".join(lines)


def _chart_summary(chart: Any) -> str:
    payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
    raw_metadata = payload.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    chart_type = str(payload.get("type") or "").strip() or "chart"
    title = str(payload.get("title") or metadata.get("name") or "").strip()
    variable = payload.get("variable")
    units = payload.get("units")
    bits = [f"{chart_type} '{title}'" if title else chart_type]
    if variable:
        bits.append(f"variable={variable}")
    if units:
        bits.append(f"units={units}")
    if payload.get("lats") and payload.get("lons"):
        bits.append(f"grid={len(payload['lats'])}x{len(payload['lons'])}")
    if payload.get("times"):
        bits.append(f"points={len(payload['times'])}")
    return ", ".join(bits)
