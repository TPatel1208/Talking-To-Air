import sys
import uuid
import os
from typing import Callable
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import trim_messages
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.satellite_agent_prompt import SATELLITE_AGENT_PROMPT
from tools import SATELLITE_TOOLS
from utils.db import get_checkpointer
from utils.streaming import stream_response
print("Tools imported:", [tool.name for tool in SATELLITE_TOOLS])


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

    
    
    @wrap_model_call
    def trim_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        trimmed = trim_messages(
            request.state["messages"],
            max_tokens=5000,
            strategy="last",
            token_counter="approximate",
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        return handler(request.override(messages=trimmed))

    agent = create_agent(
        model=llm,
        tools=SATELLITE_TOOLS,
        system_prompt=SATELLITE_AGENT_PROMPT,
        checkpointer=checkpointer,
        middleware=[trim_middleware],
    )
        
    return agent


def list_sessions() -> list[str]:
    """Return all known thread_ids from the DB."""
    from utils.db import pg_connect
    with pg_connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    return [r[0] for r in rows]


def delete_session(thread_id: str):
    from utils.db import pg_connect
    threads = [thread_id, f"ground-{thread_id}", f"satellite-{thread_id}"]
    with pg_connect() as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            conn.execute(
                f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (threads,)
            )
        conn.commit()


if __name__ == "__main__":
    # Standalone REPL — satellite agent only, no supervisor.
    # stream_response is called without thread_ref because thread_ref is
    # supervisor-only; passing None (the default) is correct here.
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

        # thread_ref=None is correct — this agent is not a supervisor subagent
        for event_type, data in stream_response(agent, user_input, thread_id, thread_ref=None):
            if event_type == "tool_call":
                print(f"\n⚙ Calling: {data['name']} | args: {data['args']}")
            elif event_type == "tool_result":
                print(f"[{data['name']}]: {data['content']}")
            elif event_type == "text":
                print(f"\n{data}")
        print()