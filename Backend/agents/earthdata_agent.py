"""
earthdata_agent.py
-------------------
LangGraph agent wrapping the earthdata-retrieval MCP toolset (handle-based
discovery, retrieval, plot/statistics tools).

Stateless by design — no checkpointer, no persistent memory.
Each invocation is a self-contained request/response cycle.
The supervisor is solely responsible for conversation history.
"""
import uuid
from typing import Any

from langchain_groq import ChatGroq
from langchain.agents import create_agent

from config.settings import get_settings
from config.earthdata_agent_prompt import get_earthdata_agent_prompt
from tools.satellite_tools.factory import build_satellite_tools
from utils.streaming import stream_response


def build_earthdata_agent(model: str | None = None, mcp_tools: dict[str, Any] | None = None):
    """
    Build and return a stateless earthdata agent.

    No checkpointer is attached — the agent holds no memory between calls.
    The supervisor passes all necessary context in the task string and is
    responsible for persisting conversation state.

    Parameters
    ----------
    model : str
        GROQ model identifier.
    mcp_tools : dict[str, BaseTool] | None
        Workspace-bound earthdata-retrieval MCP tools (see
        earthdata_mcp.toolset.load_earthdata_tools), used to build this
        agent's handle-based discovery/retrieval/plot/statistics tools.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=model or settings.earthdata_agent_model,
        groq_api_key=settings.groq_api_key,
    )

    agent = create_agent(
        model=llm,
        tools=build_satellite_tools(mcp_tools or {}),
        system_prompt=get_earthdata_agent_prompt(),
        checkpointer=None,
    )
    return agent


if __name__ == "__main__":
    # Standalone REPL — stateless, so each turn is a fresh invocation.
    agent = build_earthdata_agent()
    print("Earthdata agent started (stateless REPL)")

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
