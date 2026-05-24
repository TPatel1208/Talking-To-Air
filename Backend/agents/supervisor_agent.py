"""
Supervisor agent that orchestrates the Ground Sensor and Satellite agents.
"""
import psycopg
import os
import sys
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from agents.ground_sensor_agent import build_ground_agent
from agents.satellite_agent import build_satellite_agent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.supervisor_prompt import SUPERVISOR_PROMPT
from utils.streaming import stream_response

def _pg_connect(autocommit: bool = False):
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "talking_to_air_memory"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        autocommit=autocommit,
    )


def get_checkpointer():
    conn = _pg_connect(autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer


# ── Build supervisor ──────────────────────────────────────────────────────────

def build_agent(model: str = "llama-3.1-8b-instant", ground_agent_model: str = "meta-llama/llama-4-scout-17b-16e-instruct", satellite_agent_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Build and return the supervisor agent.

    Uses a SINGLE shared checkpointer for the supervisor and both subagents
    to avoid multiple psycopg connections racing on the same Postgres tables.

    Sub-thread IDs are derived from the active supervisor thread_id so they
    are stable across the conversation (same thread = same sub-agent memory)
    but isolated across sessions.

    Returns (agent, thread_ref) where thread_ref is a dict with key "id"
    that must be updated to the current thread_id before each stream call.
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    ground_agent_model = os.getenv("GROUND_AGENT_MODEL", ground_agent_model)
    satellite_agent_model = os.getenv("SATELLITE_AGENT_MODEL", satellite_agent_model)
    # ONE checkpointer shared across supervisor + both subagents.
    # This eliminates the race condition caused by three separate
    # autocommit=True connections hitting the same checkpoint tables.
    checkpointer = get_checkpointer()

    ground_agent    = build_ground_agent(model=ground_agent_model,    checkpointer=checkpointer)
    satellite_agent = build_satellite_agent(model=satellite_agent_model, checkpointer=checkpointer)

    # Mutable container so the tool closures can always read the current
    # supervisor thread_id even though the tools are defined once at build time.
    _thread_ref = {"id": "default"}

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
        # Derive sub-thread from the active supervisor thread so the ground
        # agent retains context within a session but is isolated across sessions.
        # Using a fixed suffix (not uuid4) means the ground agent remembers
        # prior tool calls made during this same conversation.
        sub_thread = f"ground-{_thread_ref['id']}"
        result = ground_agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": sub_thread}},
        )
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                        for b in content
                        if (isinstance(b, dict) and b.get("type") == "text")
                        or hasattr(b, "text")
                    )
        return "Ground sensor agent returned no response."

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
        sub_thread = f"satellite-{_thread_ref['id']}"
        result = satellite_agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": sub_thread}},
        )
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                        for b in content
                        if (isinstance(b, dict) and b.get("type") == "text")
                        or hasattr(b, "text")
                    )
        return "Satellite agent returned no response."

    # ── Build supervisor ──────────────────────────────────────────────────────
    supervisor = create_agent(
        model=llm,
        tools=[ask_ground_sensor_agent, ask_satellite_agent],
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=checkpointer,
    )
    return supervisor, _thread_ref




# ── list_sessions, delete_session ─────────────────────────────────────────────

def list_sessions() -> list[str]:
    with _pg_connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    # Filter out sub-agent threads so the sidebar only shows supervisor sessions.
    return [r[0] for r in rows if not r[0].startswith(("ground-", "satellite-"))]


def delete_session(thread_id: str):
    threads = [thread_id, f"ground-{thread_id}", f"satellite-{thread_id}"]
    with _pg_connect() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            conn.execute(
                f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (threads,)
            )
        conn.commit()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent, thread_ref = build_agent()

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

        for event_type, data in stream_response(agent, user_input, thread_id, thread_ref):
            if event_type == "tool_call":
                print(f"\n⚙ Calling: {data['name']} | args: {data['args']}")
            elif event_type == "tool_result":
                print(f"[{data['name']}]: {data['content']}")
            elif event_type == "text":
                print(f"\n{data}")
        print()