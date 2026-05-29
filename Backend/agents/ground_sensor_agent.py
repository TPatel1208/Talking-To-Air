"""
ground_sensor_agent.py
----------------------
LangGraph agent wrapping EPA AQS ground sensor tools.
Uses the same build pattern as GemeniAgent.py so both agents
can be composed by the supervisor.
"""
import sys
import os
from typing import Callable
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import trim_messages
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.ground_sensor_agent_prompt import GROUND_SYSTEM_PROMPT
from tools import GROUND_TOOLS
from utils.db import get_checkpointer
from utils.streaming import stream_response


def build_ground_agent(model: str = "llama-3.1-8b-instant", checkpointer=None):
    """
    Build and return a ground sensor agent.

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
        groq_api_key=os.getenv("GROQ_API_KEY"),
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
        model=llm,                    # bare llm — no pipe
        tools=GROUND_TOOLS,
        system_prompt=GROUND_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[trim_middleware],
    )
    return agent




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