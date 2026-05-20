"""
ground_sensor_agent.py
----------------------
LangGraph agent wrapping EPA AQS ground sensor tools.
Uses the same build pattern as GemeniAgent.py so both agents
can be composed by the supervisor.
"""
import sys
import os
import psycopg
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.ground_sensor_agent_promp import GROUND_SYSTEM_PROMPT
from tools import GROUND_TOOLS


def _pg_connect(autocommit: bool = False):
    """Return a psycopg connection using individual env vars."""
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
    checkpointer.setup()
    return checkpointer


def build_ground_agent(model: str = "gemma-4-31b-it", checkpointer=None):
    """
    Build and return a ground sensor agent.

    Parameters
    ----------
    model : str
        Gemini model identifier.
    checkpointer : optional
        A shared PostgresSaver instance. If None, a new one is created.
        Pass the supervisor's checkpointer to avoid multiple DB connections
        racing against each other.
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    if checkpointer is None:
        checkpointer = get_checkpointer()
    agent = create_agent(
        model=llm,
        tools=GROUND_TOOLS,
        system_prompt=GROUND_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def stream_response(agent, user_input: str, thread_id: str):
    """Stream one turn, yield (event_type, data) tuples."""
    import re
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
                        raw_content = str(msg.content)
                        # Extract image path before truncating so it's never lost
                        img_match = re.search(r'[\w\-./]+\.png', raw_content)
                        content_out = img_match.group(0) if img_match else raw_content[:300]
                        yield ("tool_result", {"name": msg.name, "content": content_out})
                    elif hasattr(msg, "content") and msg.content:
                        yield ("text", msg.content)


if __name__ == "__main__":
    import uuid

    agent = build_ground_agent()
    thread_id = str(uuid.uuid4())
    print(f"Ground sensor agent started | session: {thread_id[:8]}...")

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