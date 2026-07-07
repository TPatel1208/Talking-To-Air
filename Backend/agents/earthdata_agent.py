"""
earthdata_agent.py
-------------------
LangGraph agent wrapping the earthdata-retrieval MCP toolset (handle-based
discovery, retrieval, plot/statistics tools).

Stateless by design — no checkpointer, no persistent memory.
Each invocation is a self-contained request/response cycle.
The supervisor is solely responsible for conversation history.
"""
import logging
import uuid
from typing import Any

from langchain.agents import create_agent

from agents.subagent_trim import build_subagent_trim_middleware
from config.model_factory import build_chat_model
from config.settings import get_settings
from config.earthdata_agent_prompt import get_earthdata_agent_prompt
from tools.satellite_tools.factory import build_satellite_tools
from utils.streaming import stream_response

logger = logging.getLogger(__name__)


def build_earthdata_agent(
    model: str | None = None,
    provider: str | None = None,
    mcp_tools: dict[str, Any] | None = None,
):
    """
    Build and return a stateless earthdata agent.

    No checkpointer is attached — the agent holds no memory between calls.
    The supervisor passes all necessary context in the task string and is
    responsible for persisting conversation state.

    Parameters
    ----------
    model : str
        Model identifier for the resolved provider.
    provider : str
        Provider name understood by config.model_factory.build_chat_model.
    mcp_tools : dict[str, BaseTool] | None
        Workspace-bound earthdata-retrieval MCP tools (see
        earthdata_mcp.toolset.load_earthdata_tools), used to build this
        agent's handle-based discovery/retrieval/plot/statistics tools.
    """
    settings = get_settings()
    model = model or settings.earthdata_agent_model
    provider = provider or settings.earthdata_agent_provider
    logger.info(
        "earthdata_agent_model",
        extra={"_event": "earthdata_agent_model", "_model": model, "_provider": provider},
    )
    llm = build_chat_model(provider, model, settings)

    agent = create_agent(
        model=llm,
        tools=build_satellite_tools(mcp_tools or {}),
        system_prompt=get_earthdata_agent_prompt(),
        checkpointer=None,
        middleware=[build_subagent_trim_middleware("earthdata", settings.subagent_trim_token_ceiling)],
    )
    # This agent is stateless (no checkpointer), so subagent_dispatch's T15
    # retry demotion — one structured-output re-prompt instead of a full
    # tool-workflow re-run — has no other way to reach the raw chat model.
    agent.subagent_model = llm
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
