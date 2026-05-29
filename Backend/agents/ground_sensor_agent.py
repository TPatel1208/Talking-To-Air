"""
ground_sensor_agent.py
----------------------
LangGraph agent wrapping EPA AQS ground sensor tools.

Stateless by design — no checkpointer, no persistent memory.
Each invocation is a self-contained request/response cycle.
The supervisor is solely responsible for conversation history.
"""
import sys
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.ground_sensor_agent_prompt import GROUND_SYSTEM_PROMPT
from tools import GROUND_TOOLS
from utils.streaming import stream_response


def build_ground_agent(model: str = "llama-3.1-8b-instant"):
    """
    Build and return a stateless ground sensor agent.

    No checkpointer is attached — the agent holds no memory between calls.
    The supervisor passes all necessary context in the task string and is
    responsible for persisting conversation state.

    Parameters
    ----------
    model : str
        GROQ model identifier.
    """
    llm = ChatGroq(
        model=model,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    agent = create_agent(
        model=llm,
        tools=GROUND_TOOLS,
        system_prompt=GROUND_SYSTEM_PROMPT,
        checkpointer=None,
    )
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
