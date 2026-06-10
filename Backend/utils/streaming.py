"""
Shared stream_response for supervisor, ground, and satellite agents.

Yields:
    ("tool_call", {"name": str, "args": dict})
    ("status", {"message": str})
    ("image", {"name": str, "path": str})
    ("tool_result", {"name": str, "content": str})
    ("text", str)
"""

import asyncio
import re
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from typing import Callable, Optional

_status_emitter: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "status_emitter",
    default=None,
)
_current_thread_id: ContextVar[Optional[str]] = ContextVar("current_thread_id", default=None)


def emit_status(message: str) -> None:
    """Emit a user-visible progress message for the active SSE stream."""
    emitter = _status_emitter.get()
    if emitter and message:
        emitter(str(message))


def current_thread_id() -> str | None:
    return _current_thread_id.get()


async def stream_response(
    agent,
    user_input: str,
    thread_id: str,
    thread_ref: Optional[dict] = None,
) -> AsyncGenerator[tuple, None]:
    """
    Stream one conversation turn, yielding (event_type, data) tuples.

    Status events can be published from nested tools with emit_status(...).
    """
    if thread_ref is not None:
        thread_ref["id"] = thread_id

    config = {"configurable": {"thread_id": thread_id}}
    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    loop = asyncio.get_running_loop()
    parent_emitter = _status_emitter.get()

    def publish_status(message: str) -> None:
        if parent_emitter:
            parent_emitter(message)
        loop.call_soon_threadsafe(queue.put_nowait, ("status", {"message": message}))

    async def publish(event_type: str, data) -> None:
        await queue.put((event_type, data))

    async def produce() -> None:
        try:
            async for stream_mode, chunk in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode=["updates", "messages"],
            ):
                if stream_mode != "updates":
                    continue

                for _node, data in chunk.items():
                    for msg in data.get("messages", []):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                await publish("tool_call", {"name": tc["name"], "args": tc["args"]})
                        elif hasattr(msg, "name") and msg.name:
                            raw = str(msg.content)
                            img = re.search(r'[\w\-./]+\.png', raw)
                            if img:
                                await publish("image", {"name": msg.name, "path": img.group(0)})
                            await publish("tool_result", {"name": msg.name, "content": raw})
                        elif hasattr(msg, "content") and msg.content:
                            await publish("text", msg.content)
        except Exception as exc:
            await queue.put(("__error__", exc))
        finally:
            await queue.put(done)

    token = _status_emitter.set(publish_status)
    thread_token = _current_thread_id.set(thread_id)
    producer = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            event_type, data = item
            if event_type == "__error__":
                raise data
            yield event_type, data
    finally:
        _status_emitter.reset(token)
        _current_thread_id.reset(thread_token)
        if not producer.done():
            producer.cancel()
