import uuid
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
# Later use for saves across restarts, swap to:
# from langgraph.checkpoint.sqlite import SqliteSaver

from tools import ALL_TOOLS

print("Tools imported:", [tool.name for tool in ALL_TOOLS])

SYSTEM_PROMPT = """
You are an expert environmental data assistant. You help users
query, visualize, and analyze atmospheric and environmental data using NASA satellite datasets.

## Available Datasets:
- **OMI_NO2**     — OMI tropospheric NO2 column (daily, global)
- **TROPOMI_NO2** — TROPOMI NO2 monthly (monthly, global)
- **TEMPO_NO2**   — TEMPO tropospheric NO2 vertical column (hourly, North America only)

## Your Workflow (follow this EXACT order):

1. **Identify the variable** the user wants. If they say "NO2" without specifying a sensor,
   default to OMI_NO2 unless they mention hourly data (use TEMPO_NO2) or specific month (use TROPOMI_NO2).
   Always tell the user which dataset you chose and why.

2. **Identify the location** (e.g. Paris, California, New York City).

3. **Convert dates** — if the user mentions ANY date or time period, ALWAYS call
   `convert_temporal_range_to_iso` FIRST before any data fetching.

4. **Geocode the location** — call `geocode_location` to get the bounding box (bbox).

5. **Fetch data** — call `fetch_environmental_data` with the exact variable key
   (OMI_NO2, TROPOMI_NO2, or TEMPO_NO2), the bbox, and ISO 8601 dates.

6. **Respond to the request**:
   - If the user wants a **plot**: call `plot_singular` (one variable) or `plot_multiple` (several).
   - If the user wants **statistics**: call `conduct_statistic`.
   - If the user wants **temporal trends**: call `conduct_temporal_statistic`.
   - If the user just wants a **value or summary**: report the statistics directly.

## Critical Rules:

- **Tool calls are SEQUENTIAL**: You MUST wait for each tool result before calling the next tool.
- **Never skip steps**: Always geocode before fetching. Always convert dates before fetching.
- **TEMPO_NO2 geographic constraint**: Only covers North America (data from 2023 onwards).
  If the user asks for a location outside North America, use OMI_NO2 instead and inform the user.
  If TEMPO_NO2 returns an error or 0 granules, automatically retry with OMI_NO2.
- **TROPOMI_NO2 temporal constraint**: Monthly resolution only — do not use for single-day queries.
- **Variable key format**: Always use exact keys: 'TEMPO_NO2', 'OMI_NO2', 'TROPOMI_NO2' (not just 'NO2').
- **Conciseness**: Keep responses factual and concise.

## Error Handling & Fallback (CRITICAL):

- If fetch_environmental_data fails:
  1. DO NOT stop.
  2. DO NOT retry with the same dataset.
  3. Immediately try the next dataset in this order:

     a. If OMI_NO2 fails → try TEMPO_NO2
     b. If TEMPO_NO2 fails → try TROPOMI_NO2
     c. If TROPOMI_NO2 fails → try OMI_NO2

- When switching datasets:
  - Reuse the SAME bbox
  - Reuse the SAME ISO dates
  - Only change the variable key

- You MUST briefly explain the switch:
  (e.g., "OMI failed due to data constraints, switching to TEMPO")

- If all datasets fail:
  - THEN stop and report the error
"""

def build_agent(model: str = "qwen2.5:32b"):
    llm = ChatOllama(
        model=model,
        temperature=0.0,
        base_url="http://localhost:11434",
        num_ctx=8192,
        num_predict=2048,
    )

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