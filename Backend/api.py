import uuid
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GemeniAgent import build_agent

app = FastAPI(title="Talking to Air API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

print("Initializing agent...")
agent = build_agent()
print("Agent ready.")

sessions: dict[str, str] = {}


class ChatRequest(BaseModel):
    message:   str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id:  str
    response:   str
    image_urls: list[str] = []
    tool_calls: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    sessions[thread_id] = thread_id
    config = {"configurable": {"thread_id": thread_id}}

    response_text = ""
    image_urls    = []
    tool_calls    = []

    try:
        for stream_mode, chunk in agent.stream(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if stream_mode == "updates":
                for node, data in chunk.items():
                    for msg in data.get("messages", []):

                        # Tool calls
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls.append({
                                    "name": tc["name"],
                                    "args": tc["args"],
                                })

                        # Tool results — check for PNG paths
                        elif hasattr(msg, "name") and msg.name:
                            content = str(msg.content)
                            print(f"DEBUG tool result [{msg.name}]: {content[:200]}")
                            if content.strip().endswith(".png"):
                                filename = os.path.basename(content.strip())
                                image_urls.append(f"/outputs/{filename}")

                        # Final text response
                        elif hasattr(msg, "content") and msg.content:
                            print(f"DEBUG msg type: {type(msg.content)}")
                            print(f"DEBUG msg content: {str(msg.content)[:200]}")

                            if isinstance(msg.content, str):
                                response_text = msg.content

                            elif isinstance(msg.content, list):
                                for block in msg.content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        response_text = block.get("text", "")
                                    elif hasattr(block, "text"):
                                        response_text = block.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    print(f"DEBUG final response_text: '{response_text[:200]}'")

    return ChatResponse(
        thread_id=thread_id,
        response=response_text,
        image_urls=image_urls,
        tool_calls=tool_calls,
    )


@app.delete("/session/{thread_id}")
def clear_session(thread_id: str):
    sessions.pop(thread_id, None)
    return {"cleared": thread_id}


@app.get("/sessions")
def list_sessions():
    return {"sessions": list(sessions.keys())}