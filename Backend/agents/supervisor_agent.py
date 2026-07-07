"""
Supervisor agent that orchestrates the Ground Sensor and Satellite agents.

Memory model
------------
- Supervisor : stateful — uses a Postgres checkpointer, one thread per session.
- Subagents  : stateless — no checkpointer; each tool call is a fresh invocation.
               The supervisor includes all necessary context in the task string
               it passes to each subagent tool.
"""
import logging
from typing import Any
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import trim_messages
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from collections.abc import Awaitable, Callable

from config.model_factory import build_chat_model
from config.settings import get_settings
from config.supervisor_prompt import SUPERVISOR_PROMPT
from models import agent_result_to_json, parse_agent_result, parse_chart_payload
from services.subagent_dispatch import run_ground, run_satellite
from utils.db import get_checkpointer
from utils.message_utils import truncate_text
from utils.streaming import current_thread_id

logger = logging.getLogger(__name__)


# ── Build supervisor ──────────────────────────────────────────────────────────

async def build_agent(
    model: str | None = None,
    provider: str | None = None,
    *,
    ground_agent: Any,
    satellite_agent: Any,
):
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
    """
    settings = get_settings()
    model = model or settings.llm_model
    provider = provider or settings.supervisor_model_provider
    logger.info(
        "supervisor_model",
        extra={"_event": "supervisor_model", "_model": model, "_provider": provider},
    )
    llm = build_chat_model(provider, model, settings)

    # ── Trim middleware — keeps the supervisor's context window bounded ───────

    @wrap_model_call
    async def trim_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        messages = [_compact_model_input_message(msg) for msg in request.state["messages"]]
        trimmed = trim_messages(
            messages,
            max_tokens=8000,
            strategy="last",
            token_counter="approximate",
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        # Gemini rejects an empty contents list, so never let trimming remove
        # the only usable turn from a request. Fall back to the original
        # message list if trimming collapses everything.
        if not trimmed:
            trimmed = messages
        return await handler(request.override(messages=trimmed))

    # ── Wrap subagents as tools ───────────────────────────────────────────────

    @tool
    async def ask_ground_sensor_agent(task: str) -> str:
        """
        Delegate a task to the ground sensor agent which has access to EPA AQS
        air quality monitor data across the United States.

        Use for: finding the closest monitor to a location, retrieving NO2 /
        PM2.5 / ozone / CO / SO2 daily or quarterly readings, identifying days
        that exceeded regulatory thresholds, and fetching hourly concentration
        profiles.

        Input: a natural language task description that includes the location,
               pollutant, and date range (e.g. 'Find the closest NO2 monitor
               to Tampa FL and return exceedance days in Q1 2025').
        Output: text summary including monitor name, site_id, coordinates,
                exceedance dates, and peak concentration values.
        """
        result = await run_ground(ground_agent, task, current_thread_id())
        return agent_result_to_json(result)

    @tool
    async def ask_earthdata_agent(task: str) -> str:
        """
        Delegate a task to the earthdata agent which has access to NASA
        satellite data via the earthdata-retrieval MCP (TROPOMI NO2, aerosol
        optical depth, ozone, HCHO, and other variables).

        Use for: fetching and plotting satellite-derived pollutant maps over a
        region, computing spatial statistics, and visually confirming ground-
        level pollution events from space.

        Input: a natural language task description that includes the variable,
               date or date range (YYYY-MM-DD), and location or bounding box
               (e.g. 'Plot TROPOMI NO2 over New Jersey for 2024-01-15').
        Output: text summary with plot path and spatial statistics.
        """
        result = await run_satellite(satellite_agent, task, current_thread_id())
        return agent_result_to_json(result)

    # ── Build supervisor ──────────────────────────────────────────────────────
    checkpointer = await get_checkpointer()
    supervisor = create_agent(
        model=llm,
        tools=[ask_ground_sensor_agent, ask_earthdata_agent],
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
        middleware=[trim_middleware],
    )
    return supervisor


# ── Helpers ───────────────────────────────────────────────────────────────────
# Sub-agent invocation, envelope finalization, and monitor-context helpers
# live in services/subagent_dispatch.py — shared by these tool wrappers and
# the router fast path (T14). Only supervisor-model-input compaction (used
# solely by this module's own trim_middleware) stays here.


def _truncate_text(text: str, max_chars: int, agent_name: str, request_id: str | None = None) -> str:
    return truncate_text(text, max_chars, agent_name, request_id)


def _compact_model_input_message(msg):
    """Replace bulky chart payloads with concise summaries before LLM calls."""
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


def _compact_model_input_content(content):
    if not isinstance(content, str):
        return content

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


def _chart_summary(chart) -> str:
    payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
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
