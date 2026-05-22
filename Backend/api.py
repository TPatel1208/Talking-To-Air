import uuid
import os
import re
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.supervisor_agent import build_agent, stream_response, list_sessions, delete_session

app = FastAPI(title="Talking to Air API")

_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost")
origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


_model = os.getenv("LLM_MODEL", "gemma-4-31b-it")
print(f"Initializing agent with model: {_model}")

# build_agent now returns (agent, thread_ref).
# thread_ref is a mutable dict {"id": ...} that stream_response updates
# before each call so subagent tool closures always use the right thread_id.
agent, _thread_ref = build_agent(_model)
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
            # Pass _thread_ref so stream_response can update it for the subagent closures.
            for event_type, data in stream_response(agent, req.message, thread_id, _thread_ref):
                print(f"EVENT: {event_type!r}  DATA: {repr(data)[:200]}")

                if event_type == "tool_call":
                    tool_calls.append({"name": data["name"], "args": data["args"]})
                    yield sse("tool_call", {"name": data["name"], "args": data["args"]})

                elif event_type == "tool_result":
                    content = data.get("content", "")
                    # Use regex to find a .png path anywhere in the content string.
                    # The old endswith(".png") check missed paths embedded mid-sentence
                    # in longer summaries returned by the satellite agent.
                    png_match = re.search(r'(/outputs/[\w\-./]+\.png|[\w\-./]+\.png)', content)
                    if png_match:
                        url = normalize_image_url(png_match.group(1))
                        if url:
                            image_urls.append(url)
                            yield sse("image", {"url": url})

                elif event_type == "text":
                    if isinstance(data, str):
                        response_text += data
                    elif isinstance(data, list):
                        for block in data:
                            if isinstance(block, str):
                                response_text += block
                            elif isinstance(block, dict):
                                if block.get("type") == "text":
                                    response_text += block.get("text", "")
                            elif hasattr(block, "text"):
                                response_text += block.text

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
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state  = agent.get_state(config)
        if not state or not state.values:
            return {"messages": []}

        raw_messages = state.values.get("messages", [])
        result = []

        for msg in raw_messages:
            role = getattr(msg, "type", None)

            if role == "human":
                result.append({
                    "role":      "user",
                    "content":   msg.content if isinstance(msg.content, str) else "",
                    "toolCalls": [],
                    "imageUrls": [],
                })

            elif role == "ai":
                tool_calls = []
                seen_tool_ids = set()
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tid = tc.get("id", "")
                        seen_tool_ids.add(tid)
                        tool_calls.append({
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        })

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            # plain string block (Gemini stores response text this way)
                            content += block
                        elif isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "text":
                                content += block.get("text", "")
                            elif btype == "thinking":
                                pass  # skip internal chain-of-thought
                            elif btype == "tool_use":
                                # tool dispatch in content list
                                tid = block.get("id", "")
                                if tid not in seen_tool_ids:
                                    seen_tool_ids.add(tid)
                                    tool_calls.append({
                                        "name": block.get("name", ""),
                                        "args": block.get("input", {}),
                                    })
                        elif hasattr(block, "text"):
                            content += block.text

                result.append({
                    "role":      "assistant",
                    "content":   content,
                    "toolCalls": tool_calls,
                    "imageUrls": [],
                })

            elif role == "tool":
                # Flatten tool result content — can be str or list of blocks
                if isinstance(msg.content, str):
                    tool_text = msg.content
                elif isinstance(msg.content, list):
                    parts = []
                    for block in msg.content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict):
                            parts.append(block.get("text", "") or str(block.get("content", "")))
                        elif hasattr(block, "text"):
                            parts.append(block.text)
                    tool_text = " ".join(parts)
                else:
                    tool_text = str(msg.content)

                # Find all .png paths anywhere in the tool result
                for png_match in re.finditer(r'(/outputs/[\w\-./]+\.png|[\w\-./]+\.png)', tool_text):
                    url = normalize_image_url(png_match.group(1))
                    if url:
                        for m in reversed(result):
                            if m["role"] == "assistant":
                                if url not in m["imageUrls"]:
                                    m["imageUrls"].append(url)
                                break

        merged = []
        for msg in result:
            if (
                msg["role"] == "assistant"
                and merged
                and merged[-1]["role"] == "assistant"
            ):
                prev = merged[-1]
                prev["toolCalls"].extend(msg["toolCalls"])
                if msg["content"]:
                    prev["content"] += ("\n\n" if prev["content"] else "") + msg["content"]
                for url in msg["imageUrls"]:
                    if url not in prev["imageUrls"]:
                        prev["imageUrls"].append(url)
            else:
                merged.append(msg)

        return {"messages": merged}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{thread_id}")
def remove_session(thread_id: str):
    try:
        delete_session(thread_id)
        return {"deleted": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/debug/{thread_id}")
def debug_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    raw = state.values.get("messages", [])
    return [
        {
            "type": getattr(m, "type", None),
            "content_type": type(m.content).__name__,
            "content_preview": str(m.content)[:300],
            "has_tool_calls": bool(getattr(m, "tool_calls", None)),
        }
        for m in raw
    ]