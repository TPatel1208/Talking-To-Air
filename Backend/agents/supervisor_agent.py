"""
Supervisor agent that orchestrates the Ground Sensor and Satellite agents.

Memory model
------------
- Supervisor : stateful — uses a Postgres checkpointer, one thread per session.
- Subagents  : stateless — no checkpointer; each tool call is a fresh invocation.
               The supervisor includes all necessary context in the task string
               it passes to each subagent tool.
"""
import os
import asyncio
import calendar
import logging
import re
import sys
import uuid
from collections.abc import Awaitable, Callable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, trim_messages
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from agents.ground_sensor_agent import build_ground_agent
from agents.satellite_agent import build_satellite_agent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import get_settings
from config.supervisor_prompt import SUPERVISOR_PROMPT
from models import AgentResult, agent_result_to_json, parse_agent_result, parse_chart_payload
from repositories.chart_repository import delete_charts_for_session
from utils.db import get_checkpointer, pg_connection
from utils.streaming import stream_response

logger = logging.getLogger(__name__)


# ── Build supervisor ──────────────────────────────────────────────────────────

async def build_agent(
    model: str | None = None,
    ground_agent_model: str | None = None,
    satellite_agent_model: str | None = None,
):
    """
    Build and return the supervisor agent.

    The supervisor is the only stateful component — it owns the Postgres
    checkpointer and persists the full conversation history under one
    thread_id per user session.

    Subagents are stateless: each tool call creates a fresh invocation with
    no checkpointer attached, so they write nothing to the DB and accumulate
    no history of their own.
    """
    settings = get_settings()
    model = model or settings.llm_model
    ground_agent_model = ground_agent_model or settings.ground_agent_model
    satellite_agent_model = satellite_agent_model or settings.satellite_agent_model
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
    )

    # Stateless subagents — no checkpointer passed.
    ground_agent    = build_ground_agent(model=ground_agent_model)
    satellite_agent = build_satellite_agent(model=satellite_agent_model)

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
        result = await ground_agent.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        text = _extract_last_text(
            result,
            "Ground sensor agent returned no response.",
            agent_name="ground_sensor",
        )
        return agent_result_to_json(AgentResult(text=text))

    @tool
    async def ask_satellite_agent(task: str) -> str:
        """
        Delegate a task to the satellite agent which has access to NASA
        satellite data via NASA Harmony (TROPOMI NO2, aerosol optical depth,
        ozone, HCHO, and other variables).

        Use for: fetching and plotting satellite-derived pollutant maps over a
        region, computing spatial statistics, and visually confirming ground-
        level pollution events from space.

        Input: a natural language task description that includes the variable,
               date or date range (YYYY-MM-DD), and location or bounding box
               (e.g. 'Plot TROPOMI NO2 over New Jersey for 2024-01-15').
        Output: text summary with plot path and spatial statistics.
        """
        async def _run_satellite(task_text: str) -> AgentResult:
            charts = []
            text_parts  = []

            try:
                async for event_type, data in stream_response(
                    satellite_agent, task_text, thread_id=str(uuid.uuid4())
                ):
                    if event_type == "tool_result":
                        content = data.get("content", "")
                        chart = parse_chart_payload(content)
                        if chart is not None:
                            charts.append(chart)
                            continue
                        nested = parse_agent_result(content)
                        if nested is not None:
                            text_parts.append(nested.text)
                            charts.extend(nested.charts)
                    elif event_type in ("text", "done"):
                        t = data if isinstance(data, str) else data.get("response", "")
                        if t:
                            text_parts.append(t)
            except Exception as exc:
                text_parts.append(str(exc))

            text = _truncate_text(
                " ".join(text_parts),
                2000,
                agent_name="satellite",
            ) or "Satellite agent returned no response."
            return AgentResult(text=text, charts=charts)

        direct_first = await _try_direct_satellite_plot(task)
        if direct_first is not None:
            return agent_result_to_json(direct_first)

        result = await _run_satellite(task)
        refusal_markers = (
            "necessary tools are not present",
            "don't have access to fetch_environmental_data",
            "do not have access to fetch_environmental_data",
            "failed to call a function",
            "failed_generation",
        )
        if any(marker in result.text.lower() for marker in refusal_markers):
            retry_task = (
                "The satellite tools are registered and available in this runtime: "
                "convert_temporal_range_to_iso, geocode_location, "
                "check_data_availability, fetch_environmental_data, plot_singular, "
                "plot_multiple, compute_statistic_tool, conduct_temporal_statistic, "
                "find_daily_peak. Retry the task using those tools exactly as needed. "
                f"Task: {task}"
            )
            result = await _run_satellite(retry_task)
            if (
                any(marker in result.text.lower() for marker in refusal_markers)
                or not result.charts
            ):
                fallback = await _try_direct_satellite_plot(task)
                if fallback is not None:
                    result = fallback

        return agent_result_to_json(result)

    # ── Build supervisor ──────────────────────────────────────────────────────
    checkpointer = await get_checkpointer()
    supervisor = create_agent(
        model=llm,
        tools=[ask_ground_sensor_agent, ask_satellite_agent],
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
        middleware=[trim_middleware],
    )
    return supervisor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_last_text(
    result: dict,
    fallback: str,
    max_chars: int = 2000,
    agent_name: str = "unknown",
    request_id: str | None = None,
) -> str:
    """Return the last non-empty text content from an agent invoke() result.

    Capped at max_chars to prevent large subagent responses from bloating the
    supervisor's checkpoint and inflating token counts on every subsequent turn.
    """
    for msg in reversed(result.get("messages", [])):
        if not (hasattr(msg, "content") and msg.content):
            continue
        content = msg.content
        if isinstance(content, str):
            return _truncate_text(content, max_chars, agent_name, request_id)
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in content
                if (isinstance(b, dict) and b.get("type") == "text")
                or hasattr(b, "text")
            )
            if text:
                return _truncate_text(text, max_chars, agent_name, request_id)
    return fallback


def _truncate_text(text: str, max_chars: int, agent_name: str, request_id: str | None = None) -> str:
    if len(text) <= max_chars:
        return text
    logger.warning(
        "response_truncated",
        extra={
            "_agent_name": agent_name,
            "_original_length": len(text),
            "_final_length": max_chars,
            "_request_id": request_id,
        },
    )
    return text[:max_chars]


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
        summaries = [_chart_summary(chart) for chart in result.charts]
        chart_text = "; ".join(summary for summary in summaries if summary)
        if chart_text:
            return f"{result.text}\n\nCharts generated: {chart_text}"
        return result.text

    chart = parse_chart_payload(content)
    if chart is not None:
        summary = _chart_summary(chart)
        return f"Chart generated: {summary}" if summary else "Chart generated."

    return content


def _chart_summary(chart) -> str:
    payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
    chart_type = payload.get("type", "chart")
    title = payload.get("title") or payload.get("metadata", {}).get("name") or "Untitled chart"
    variable = payload.get("variable")
    units = payload.get("units")
    bits = [f"{chart_type} '{title}'"]
    if variable:
        bits.append(f"variable={variable}")
    if units:
        bits.append(f"units={units}")
    if payload.get("lats") and payload.get("lons"):
        bits.append(f"grid={len(payload['lats'])}x{len(payload['lons'])}")
    if payload.get("times"):
        bits.append(f"points={len(payload['times'])}")
    return ", ".join(bits)


async def _try_direct_satellite_plot(task: str) -> AgentResult | None:
    """
    Deterministic fallback for simple one-location satellite plot requests.

    This is deliberately narrow: it only handles requests that name a known
    satellite collection, a location, and either a YYYY-MM-DD date or
    Month YYYY range.
    """
    parsed = _parse_simple_satellite_plot_task(task)
    if parsed is None:
        return None

    variable, location, start_date, end_date = parsed
    logger.info(
        "Direct satellite plot fallback: variable=%s location=%s temporal=%s..%s",
        variable,
        location,
        start_date,
        end_date,
    )

    try:
        from tools.satellite_tools.harmony_api import (
            check_data_availability,
            fetch_environmental_data,
            geocode_location,
        )
        from tools.satellite_tools.plot_tools import (
            plot_singular,
        )

        geocoded = await geocode_location.ainvoke({"location_name": location})
        if isinstance(geocoded, dict) and geocoded.get("error"):
            return AgentResult(text=geocoded["error"])
        bbox = geocoded["bbox"]
        logger.info("Direct satellite plot fallback: geocoded bbox=%s", bbox)

        availability = await check_data_availability.ainvoke({
            "variable": variable,
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
        })
        if isinstance(availability, dict) and availability.get("error"):
            logger.warning(
                "Direct satellite plot fallback: availability check failed; continuing to Harmony fetch: %s",
                availability["error"],
            )
        elif isinstance(availability, dict) and availability.get("num_granules", 0) == 0:
            return AgentResult(text=(
                f"No {variable} granules were found for {location} between "
                f"{start_date} and {end_date}."
            ))
        else:
            logger.info(
                "Direct satellite plot fallback: availability num_granules=%s",
                availability.get("num_granules") if isinstance(availability, dict) else None,
            )

        logger.info("Direct satellite plot fallback: calling fetch_environmental_data")
        data = await fetch_environmental_data.ainvoke({
            "variable": variable,
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "max_results": 1 if variable == "TROPOMI_NO2" else 10,
        })
        if isinstance(data, dict) and data.get("error"):
            return AgentResult(text=f"Fetch failed: {data['error']}")
        logger.info("Direct satellite plot fallback: fetch_environmental_data returned data")
        plot_data = data.model_dump() if hasattr(data, "model_dump") else data

        title = f"{variable} over {location}"
        chart_result = await plot_singular.ainvoke({
            "data_dict": plot_data,
            "variable": variable,
            "location": location,
            "title": title,
        })
        chart = parse_chart_payload(chart_result)
        if chart is not None:
            logger.info("Direct satellite plot fallback: chart created for %s", title)
            return AgentResult(text=f"Created {title}.", charts=[chart])
        return AgentResult(text=str(chart_result))
    except Exception as exc:
        return AgentResult(text=f"Satellite fallback failed: {exc}")


def _parse_simple_satellite_plot_task(task: str):
    text = " ".join(task.strip().split())
    lower = text.lower()

    variable_aliases = {
        "TROPOMI_NO2": ("tropomi no2", "tropomi_no2"),
        "OMI_NO2": ("omi no2", "omi_no2"),
        "TEMPO_NO2": ("tempo no2", "tempo_no2"),
        "TEMPO_O3TOT": ("tempo o3tot", "tempo ozone", "tempo_o3tot"),
        "OMI_O3": ("omi o3", "omi ozone", "omi_o3"),
        "TEMPO_HCHO": ("tempo hcho", "tempo_hcho"),
        "TEMPO_HCHO_V03": ("tempo hcho v03", "tempo_hcho_v03"),
        "OMI_HCHO": ("omi hcho", "omi_hcho"),
        "MODIS_AOD_TERRA": ("modis aod terra", "modis_aod_terra"),
        "MODIS_AOD_AQUA": ("modis aod aqua", "modis_aod_aqua"),
    }

    variable = next(
        (
            key
            for key, aliases in variable_aliases.items()
            if any(alias in lower for alias in aliases)
        ),
        None,
    )
    if variable is None:
        return None

    location_match = re.search(
        r"\b(?:over|in|for)\s+(.+?)\s+\b(?:for|on|during)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not location_match:
        return None
    location = location_match.group(1).strip(" .")

    iso_day = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_day:
        day = iso_day.group(1)
        return variable, location, f"{day}T00:00:00Z", f"{day}T23:59:59Z"

    month_names = "|".join(calendar.month_name[1:])
    month_match = re.search(
        rf"\b({month_names})\s+(\d{{4}})\b",
        text,
        flags=re.IGNORECASE,
    )
    if month_match:
        month = list(calendar.month_name).index(month_match.group(1).title())
        year = int(month_match.group(2))
        last_day = calendar.monthrange(year, month)[1]
        return (
            variable,
            location,
            f"{year:04d}-{month:02d}-01T00:00:00Z",
            f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59Z",
        )

    return None


# ── list_sessions, delete_session ─────────────────────────────────────────────

async def list_sessions() -> list[str]:
    """Return all supervisor session thread_ids (subagent threads no longer exist)."""
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        rows = await cursor.fetchall()
    return [r[0] for r in rows]


async def delete_session(thread_id: str):
    """Delete a supervisor session from the checkpoint tables."""
    await delete_charts_for_session(thread_id)
    async with pg_connection() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await conn.execute(
                f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
            )
        await conn.commit()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = asyncio.run(build_agent())

    sessions = asyncio.run(list_sessions())
    print("Existing sessions:", sessions or "none")

    thread_id = input("Enter session ID to resume (or press Enter for new): ").strip()
    if not thread_id:
        thread_id = str(uuid.uuid4())
        print(f"New session: {thread_id[:8]}...")
    else:
        print(f"Resuming session: {thread_id[:8]}...")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            break

        for event_type, data in stream_response(agent, user_input, thread_id):
            if event_type == "tool_call":
                print(f"\n⚙ Calling: {data['name']} | args: {data['args']}")
            elif event_type == "tool_result":
                print(f"[{data['name']}]: {data['content']}")
            elif event_type == "text":
                print(f"\n{data}")
        print()
