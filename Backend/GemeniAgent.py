import uuid 
from langchain_google_genai import ChatGoogleGenerativeAI  
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from config.system_prompt import SYSTEM_PROMPT
from dotenv import load_dotenv
import os
from tools import ALL_TOOLS

print("Tools imported:", [tool.name for tool in ALL_TOOLS])

SYSTEM_PROMPT = SYSTEM_PROMPT

def build_agent(model: str = "gemini-3.1-flash-lite-preview"):
    
    load_dotenv()  #
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))
    

    checkpointer = MemorySaver()

    agent = create_agent(
        model=llm,              # pass instance directly
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent

print("Building agent...")
agent = build_agent()


def stream_response(agent, user_input: str, thread_id: str):
    """Stream one turn and print output."""
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
                            print(f"\n Calling: {tc['name']} | args: {tc['args']}")
                    elif hasattr(msg, "name") and msg.name:
                        print(f"[{msg.name}]: {str(msg.content)[:300]}")
                    elif hasattr(msg, "content") and msg.content:
                        print(f"\n {msg.content[:500]}")


def main():
    print("Building agent...")
    agent = build_agent()

    thread_id = str(uuid.uuid4())
    print(f"\n Environmental Data Assistant ready (session: {thread_id[:8]}...)")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        stream_response(agent, user_input, thread_id)
        print() 


if __name__ == "__main__":
    main()