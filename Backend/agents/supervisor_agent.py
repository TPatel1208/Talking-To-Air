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
import sys
import uuid
from typing import Callable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, trim_messages
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from agents.ground_sensor_agent import build_ground_agent
from agents.satellite_agent import build_satellite_agent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.supervisor_prompt import SUPERVISOR_PROMPT
from utils.db import get_checkpointer, pg_connect
from utils.streaming import stream_response


# ── Build supervisor ──────────────────────────────────────────────────────────

def build_agent(
    model: str = "llama-3.1-8b-instant",
    ground_agent_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    satellite_agent_model: str = "openai/gpt-oss-20b",
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
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    ground_agent_model   = os.getenv("GROUND_AGENT_MODEL",   ground_agent_model)
    satellite_agent_model = os.getenv("SATELLITE_AGENT_MODEL", satellite_agent_model)

    # Stateless subagents — no checkpointer passed.
    ground_agent    = build_ground_agent(model=ground_agent_model)
    satellite_agent = build_satellite_agent(model=satellite_agent_model)

    # ── Trim middleware — keeps the supervisor's context window bounded ───────

    @wrap_model_call
    def trim_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        trimmed = trim_messages(
            request.state["messages"],
            max_tokens=8000,
            strategy="last",
            token_counter="approximate",
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        return handler(request.override(messages=trimmed))

    # ── Wrap subagents as tools ───────────────────────────────────────────────

    @tool
    def ask_ground_sensor_agent(task: str) -> str:
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
        result = ground_agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        return _extract_last_text(result, "Ground sensor agent returned no response.")

    @tool
    def ask_satellite_agent(task: str) -> str:
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
        chart_paths = []
        text_parts  = []

        for event_type, data in stream_response(
            satellite_agent, task, thread_id=str(uuid.uuid4())
        ):
            if event_type == "tool_result":
                content = data.get("content", "")
                if isinstance(content, str) and content.strip().endswith(".chart.json"):
                    p = content.strip()
                    # Fix #4: resolve relative paths the same way api.py does so
                    # the file check succeeds regardless of the working directory.
                    if not os.path.isabs(p):
                        from tools.satellite_tools.plot_tools import OUTPUT_DIR as _PLOT_OUTPUT_DIR
                        p = os.path.join(_PLOT_OUTPUT_DIR, os.path.basename(p))
                    if os.path.isfile(p):
                        chart_paths.append(p)
            elif event_type in ("text", "done"):
                t = data if isinstance(data, str) else data.get("response", "")
                if t:
                    text_parts.append(t)

        summary = " ".join(text_parts)[:2000] or "Satellite agent returned no response."
        if chart_paths:
            summary += "\nCHART_PATHS: " + " ".join(chart_paths)
        return summary

    # ── Build supervisor ──────────────────────────────────────────────────────
    checkpointer = get_checkpointer()
    supervisor = create_agent(
        model=llm,
        tools=[ask_ground_sensor_agent, ask_satellite_agent],
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
        middleware=[trim_middleware],
    )
    return supervisor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_last_text(result: dict, fallback: str, max_chars: int = 2000) -> str:
    """Return the last non-empty text content from an agent invoke() result.

    Capped at max_chars to prevent large subagent responses from bloating the
    supervisor's checkpoint and inflating token counts on every subsequent turn.
    """
    for msg in reversed(result.get("messages", [])):
        if not (hasattr(msg, "content") and msg.content):
            continue
        content = msg.content
        if isinstance(content, str):
            return content[:max_chars]
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in content
                if (isinstance(b, dict) and b.get("type") == "text")
                or hasattr(b, "text")
            )
            if text:
                return text[:max_chars]
    return fallback


# ── list_sessions, delete_session ─────────────────────────────────────────────

def list_sessions() -> list[str]:
    """Return all supervisor session thread_ids (subagent threads no longer exist)."""
    with pg_connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    return [r[0] for r in rows]


def delete_session(thread_id: str):
    """Delete a supervisor session from the checkpoint tables."""
    with pg_connect() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            conn.execute(
                f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,)
            )
        conn.commit()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = build_agent()

    sessions = list_sessions()
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
