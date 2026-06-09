"""
satellite_agent.py
------------------
LangGraph agent wrapping NASA Harmony satellite tools.

Stateless by design — no checkpointer, no persistent memory.
Each invocation is a self-contained request/response cycle.
The supervisor is solely responsible for conversation history.
"""
import sys
import uuid
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import get_settings
from config.satellite_agent_prompt import get_satellite_agent_prompt
from tools import SATELLITE_TOOLS
from utils.streaming import stream_response


def build_satellite_agent(model: str | None = None):
    """
    Build and return a stateless satellite agent.

    No checkpointer is attached — the agent holds no memory between calls.
    The supervisor passes all necessary context in the task string and is
    responsible for persisting conversation state.

    Parameters
    ----------
    model : str
        GROQ model identifier.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=model or settings.satellite_agent_model,
        groq_api_key=settings.groq_api_key,
    )

    agent = create_agent(
        model=llm,
        tools=SATELLITE_TOOLS,
        system_prompt=get_satellite_agent_prompt(),
        checkpointer=None,
    )
    return agent


if __name__ == "__main__":
    # Standalone REPL — stateless, so each turn is a fresh invocation.
    agent = build_satellite_agent()
    print("Satellite agent started (stateless REPL)")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            break

        for event_type, data in stream_response(agent, user_input, thread_id=str(uuid.uuid4())):
            if event_type == "tool_call":
                print(f"\n⚙ Calling: {data['name']} | args: {data['args']}")
            elif event_type == "tool_result":
                print(f"[{data['name']}]: {data['content']}")
            elif event_type == "text":
                print(f"\n{data}")
        print()
