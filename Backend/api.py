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

from GemeniAgent import build_agent, stream_response, list_sessions, delete_session

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


class ChatRequest(BaseModel):
    message:   str
    thread_id: Optional[str] = None


def normalize_image_url(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("/outputs/"):
        return raw
    filename = raw.replace("\\", "/").split("/")[-1]
    return f"/outputs/{filename}"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())

    def generate():
        response_text = ""
        image_urls    = []
        tool_calls    = []

        try:
            for event_type, data in stream_response(agent, req.message, thread_id):
                if event_type == "tool_call":
                    tool_calls.append({"name": data["name"], "args": data["args"]})
                    yield sse("tool_call", {"name": data["name"], "args": data["args"]})

                elif event_type == "tool_result":
                    content = data.get("content", "")
                    if content.strip().endswith(".png"):
                        url = normalize_image_url(content.strip())
                        if url:
                            image_urls.append(url)
                            yield sse("image", {"url": url})

                elif event_type == "text":
                    if isinstance(data, str):
                        response_text = data
                    elif isinstance(data, list):
                        for block in data:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response_text = block.get("text", "")
                            elif hasattr(block, "text"):
                                response_text = block.text

            yield sse("done", {
                "thread_id":  thread_id,
                "response":   response_text,
                "image_urls": image_urls,
                "tool_calls": tool_calls,
            })

        except Exception as e:
            yield sse("error", {"detail": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions")
def get_sessions():
    try:
        return {"sessions": list_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{thread_id}/history")
def get_history(thread_id: str):
    """
    Return the conversation history for a thread as a list of
    {role, content, imageUrls, toolCalls} objects the frontend can render.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state  = agent.get_state(config)
        if not state or not state.values:
            return {"messages": []}

        raw_messages = state.values.get("messages", [])
        result = []

        for msg in raw_messages:
            role = getattr(msg, "type", None)
            # LangGraph message types: "human", "ai", "tool"
            if role == "human":
                result.append({
                    "role":      "user",
                    "content":   msg.content if isinstance(msg.content, str) else "",
                    "toolCalls": [],
                    "imageUrls": [],
                })
            elif role == "ai":
                # Gather any tool calls attached to this AI message
                tool_calls = []
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({"name": tc.get("name", ""), "args": tc.get("args", {})})

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            content = block.get("text", "")
                        elif hasattr(block, "text"):
                            content = block.text

                result.append({
                    "role":      "assistant",
                    "content":   content,
                    "toolCalls": tool_calls,
                    "imageUrls": [],   # images are re-derived below from tool messages
                })
            elif role == "tool":
                # Check if tool result contains an image path
                content = msg.content if isinstance(msg.content, str) else ""
                if content.strip().endswith(".png"):
                    url = normalize_image_url(content.strip())
                    if url and result:
                        # Attach to the last assistant message
                        for m in reversed(result):
                            if m["role"] == "assistant":
                                m["imageUrls"].append(url)
                                break

        return {"messages": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{thread_id}")
def remove_session(thread_id: str):
    try:
        delete_session(thread_id)
        return {"deleted": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))