import sys
import uuid
import os
import psycopg
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.satellite_agent_prompt import SATELLITE_AGENT_PROMPT
from tools import SATELLITE_TOOLS
from utils.streaming import stream_response
print("Tools imported:", [tool.name for tool in SATELLITE_TOOLS])


def _pg_connect(autocommit: bool = False):
    """Return a psycopg connection using individual env vars (avoids URL encoding issues)."""
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "talking_to_air_memory"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        autocommit=autocommit,
    )


def get_checkpointer():
    """Returns a persistent PostgreSQL checkpointer."""
    conn = _pg_connect(autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # creates tables on first run, no-op after
    return checkpointer


def build_satellite_agent(model: str = "llama-3.1-8b-instant", checkpointer=None):
    """
    Build and return a satellite agent.

    Parameters
    ----------
    model : str
        GROQ model identifier.
    checkpointer : optional
        A shared PostgresSaver instance. If None, a new one is created.
        Pass the supervisor's checkpointer to avoid multiple DB connections
        racing against each other.
    """
    llm = ChatGroq(
        model=model,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    if checkpointer is None:
        checkpointer = get_checkpointer()
    agent = create_agent(
        model=llm,
        tools=SATELLITE_TOOLS,
        system_prompt=SATELLITE_AGENT_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def list_sessions() -> list[str]:
    """Return all known thread_ids from the DB."""
    with _pg_connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    return [r[0] for r in rows]


def delete_session(thread_id: str):
    threads = [thread_id, f"ground-{thread_id}", f"satellite-{thread_id}"]
    with _pg_connect() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            conn.execute(
                f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (threads,)
            )
        conn.commit()


if __name__ == "__main__":
    agent = build_satellite_agent()

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