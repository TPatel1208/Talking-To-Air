from langchain_ollama import ChatOllama
from langchain.agents import  create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


from tools import ALL_TOOLS
print("Tools imported:", [tool.name for tool in ALL_TOOLS])

SYSTEM_PROMPT = """You are an expert environmental data assistant. You help users 
query, visualize, and analyze atmospheric and environmental data (NO2, CO, PM25, O3, CO2, etc.)
using NASA satellite datasets.

## Your Workflow — follow this EXACT order:

1. **Identify** the variable(s) the user wants (e.g. NO2, CO2, PM25).
2. **Identify** the location (e.g. Paris, California).
3. **Convert dates** — if the user mentions any date or time period, ALWAYS call 
   `parse_temporal_range` FIRST before any data fetching.
4. **Geocode the location** — call `geocode_location` to get the bounding box (bbox).
5. **Fetch data** — call `fetch_environmental_data` with the variable, bbox, and ISO dates.
6. **Respond to the request**:
   - If the user wants a **plot**: call `plot_singular` (one variable) or `plot_multiple` (several).
   - If the user wants **statistics**: call `conduct_statistic`.
   - If the user wants **temporal trends**: call `conduct_temporal_statistic`.
   - If the user just wants a **value or summary**: report the statistics directly.

## Rules:
- NEVER skip the date conversion step if a date is mentioned.
- NEVER skip geocoding — always get the bbox before fetching data.
- If a step fails, report the error clearly and stop.
- Always tell the user what variable and location you are using.
- Keep responses concise and factual.

Available variables: NO2, CO, PM25, O3, CO2
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def build_llm(model: str = "qwen3:8b", temperature: float = 0.0) -> ChatOllama:
    """  
      - qwen2.5:14b   (needs ~14GB RAM)
      - qwen2.5:7b    (good, needs ~8GB RAM)
      - llama3.1:8b   (fallback)
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url="http://localhost:11434",
        num_ctx=8192,       
        num_predict=2048,   
    )



def build_agent(model: str = "qwen3:8b"):
    llm   = build_llm(model=model)
    agent = create_agent(model = llm, 
                         tools=ALL_TOOLS, 
                         system_prompt=SYSTEM_PROMPT)

    return agent

print("Building agent...")
agent = build_agent()
"""
response = agent.invoke({
    "messages": [{"role": "user", "content": "Plot NO2 levels on january 1st, 2026 in New York City."}]
})

print(response["messages"][-1].content)
"""
for stream_mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Plot NO2 levels on january 1st, 2026 in New York City."}]},
    stream_mode=["updates", "messages"]
):
    if stream_mode == "updates":
        # Shows each step: model decision, tool calls, tool results
        print(f"\n--- STEP ({list(chunk.keys())[0]}) ---")
        for node, data in chunk.items():
            for msg in data.get("messages", []):
                # Tool call being made
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"Calling tool: {tc['name']} | args: {tc['args']}")
                # Tool result
                elif hasattr(msg, "name") and msg.name:
                    print(f"Tool result [{msg.name}]: {str(msg.content)[:200]}")
                # Model thinking/final response
                elif hasattr(msg, "content") and msg.content:
                    print(f"Model: {msg.content[:300]}")