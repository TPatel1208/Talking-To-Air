"""
ground_sensor_agent.py
----------------------
LangGraph agent wrapping EPA AQS ground sensor tools.

Stateless by design — no checkpointer, no persistent memory.
Each invocation is a self-contained request/response cycle.
The supervisor is solely responsible for conversation history.
"""
import logging

from langchain.agents import create_agent

from agents.subagent_trim import build_subagent_trim_middleware
from config.model_factory import build_chat_model
from config.settings import get_settings
from config.ground_sensor_agent_prompt import GROUND_SYSTEM_PROMPT
from tools import GROUND_TOOLS
from utils.streaming import stream_response

logger = logging.getLogger(__name__)


def build_ground_agent(model: str | None = None, provider: str | None = None):
    """
    Build and return a stateless ground sensor agent.

    No checkpointer is attached — the agent holds no memory between calls.
    The supervisor passes all necessary context in the task string and is
    responsible for persisting conversation state.

    Parameters
    ----------
    model : str
        Model identifier for the resolved provider.
    provider : str
        Provider name understood by config.model_factory.build_chat_model.
    """
    settings = get_settings()
    model = model or settings.ground_agent_model
    provider = provider or settings.ground_agent_provider
    logger.info(
        "ground_agent_model",
        extra={"_event": "ground_agent_model", "_model": model, "_provider": provider},
    )
    llm = build_chat_model(provider, model, settings)

    agent = create_agent(
        model=llm,
        tools=GROUND_TOOLS,
        system_prompt=GROUND_SYSTEM_PROMPT,
        checkpointer=None,
        middleware=[build_subagent_trim_middleware("ground_sensor", settings.subagent_trim_token_ceiling)],
    )
    # This agent is stateless (no checkpointer), so subagent_dispatch's T15
    # retry demotion — one structured-output re-prompt instead of a full
    # tool-workflow re-run — has no other way to reach the raw chat model.
    agent.subagent_model = llm
    return agent


if __name__ == "__main__":
    import uuid

    agent = build_ground_agent()
    print("Ground sensor agent started (stateless REPL)")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            break

        # In stateless mode each REPL turn is a fresh invocation.
        for event_type, data in stream_response(agent, user_input, thread_id=str(uuid.uuid4())):
            if event_type == "tool_call":
                print(f"\n⚙ Calling: {data['name']} | args: {data['args']}")
            elif event_type == "tool_result":
                print(f"[{data['name']}]: {data['content']}")
            elif event_type == "text":
                print(f"\n{data}")
        print()
