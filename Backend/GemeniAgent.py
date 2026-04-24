import uuid
import os
import psycopg
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent  
from langgraph.checkpoint.postgres import PostgresSaver
from config.system_prompt import SYSTEM_PROMPT
from tools import ALL_TOOLS

# Load env vars first — use explicit path so it works regardless of cwd
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

print("Tools imported:", [tool.name for tool in ALL_TOOLS])


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


def build_agent(model: str = "gemini-3.1-flash-lite-preview"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    checkpointer = get_checkpointer()
    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def stream_response(agent, user_input: str, thread_id: str):
    """Stream one turn, yield (event_type, data) tuples."""
    config = {"configurable": {"thread_id": thread_id}}

    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if stream_mode == "updates":
            for node, data in chunk.items():
                for msg in data.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield ("tool_call", {"name": tc["name"], "args": tc["args"]})
                    elif hasattr(msg, "name") and msg.name:
                        yield ("tool_result", {"name": msg.name, "content": str(msg.content)[:300]})
                    elif hasattr(msg, "content") and msg.content:
                        yield ("text", msg.content)


def list_sessions() -> list[str]:
    """Return all known thread_ids from the DB."""
    with _pg_connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    return [r[0] for r in rows]


def delete_session(thread_id: str):
    """Delete all checkpoints for a given thread."""
    with _pg_connect() as conn:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        conn.commit()


if __name__ == "__main__":
    agent = build_agent("gemma-4-31b-it")

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