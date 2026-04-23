import uuid
import os
import sys
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
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


def sse(event: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_chat(message: str, thread_id: str):
    """Generator that streams tool calls, images, and final response as SSE."""
    config = {"configurable": {"thread_id": thread_id}}

    response_text = ""
    image_urls    = []

    try:
        for stream_mode, chunk in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if stream_mode == "updates":
                for node, data in chunk.items():
                    for msg in data.get("messages", []):

                        # Tool calls — emit immediately as they happen
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield sse("tool_call", {
                                    "name": tc["name"],
                                    "args": tc["args"],
                                })

                        # Tool results — check for image paths
                        elif hasattr(msg, "name") and msg.name:
                            content = str(msg.content)
                            if content.strip().endswith(".png"):
                                filename = os.path.basename(content.strip())
                                url = f"/outputs/{filename}"
                                image_urls.append(url)
                                yield sse("image", {"url": url})

                        # Final text response
                        elif hasattr(msg, "content") and msg.content:
                            if isinstance(msg.content, str):
                                response_text = msg.content
                            elif isinstance(msg.content, list):
                                for block in msg.content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        response_text = block.get("text", "")
                                    elif hasattr(block, "text"):
                                        response_text = block.text

        # Final done event with complete response
        yield sse("done", {
            "thread_id":  thread_id,
            "response":   response_text,
            "image_urls": image_urls,
        })

    except Exception as e:
        yield sse("error", {"detail": str(e)})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    sessions[thread_id] = thread_id

    return StreamingResponse(
        stream_chat(req.message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering if proxied
        },
    )


@app.delete("/session/{thread_id}")
def clear_session(thread_id: str):
    sessions.pop(thread_id, None)
    return {"cleared": thread_id}


@app.get("/sessions")
def list_sessions():
    return {"sessions": list(sessions.keys())}