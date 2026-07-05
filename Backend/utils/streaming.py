"""
Shared stream_response for supervisor, ground, and satellite agents.

Yields:
    ("tool_call", {"name": str, "args": dict})
    ("status", {"message": str})
    ("image", {"name": str, "path": str})
    ("tool_result", {"name": str, "content": str})
    ("text", str)
    ("job_progress", {"job_handle": str, "status": str, "progress": Any, "phase": Any, "message": str | None})
"""

import asyncio
import re
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from typing import Callable, Optional

from utils.message_utils import flatten_text_content

_status_emitter: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "status_emitter",
    default=None,
)
_job_progress_emitter: ContextVar[Optional[Callable[[dict], None]]] = ContextVar(
    "job_progress_emitter",
    default=None,
)
_current_thread_id: ContextVar[Optional[str]] = ContextVar("current_thread_id", default=None)


def emit_status(message: str) -> None:
    """Emit a user-visible progress message for the active SSE stream."""
    emitter = _status_emitter.get()
    if emitter and message:
        emitter(str(message))


def emit_job_progress(
    job_handle: str,
    status: str,
    progress=None,
    phase: str | None = None,
    message: str | None = None,
) -> None:
    """Emit a structured retrieval-job progress event for the active SSE stream."""
    emitter = _job_progress_emitter.get()
    if emitter:
        emitter({
            "job_handle": job_handle,
            "status": status,
            "progress": progress,
            "phase": phase,
            "message": message,
        })


def current_thread_id() -> str | None:
    return _current_thread_id.get()


def _message_text_chunk(message) -> str:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return ""
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content:
        return ""
    message_type = getattr(message, "type", "")
    if message_type in {"human", "system", "tool"}:
        return ""
    return content


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
    parent_job_progress_emitter = _job_progress_emitter.get()

    def publish_status(message: str) -> None:
        if parent_emitter:
            parent_emitter(message)
        loop.call_soon_threadsafe(queue.put_nowait, ("status", {"message": message}))

    def publish_job_progress(data: dict) -> None:
        if parent_job_progress_emitter:
            parent_job_progress_emitter(data)
        loop.call_soon_threadsafe(queue.put_nowait, ("job_progress", data))

    async def publish(event_type: str, data) -> None:
        await queue.put((event_type, data))

    async def produce() -> None:
        # Track whether messages stream emitted any text this turn.
        # If it didn't (e.g. model returned a complete message without streaming),
        # the updates fallback below will carry the response instead.
        emitted_message_tokens = False
        try:
            async for stream_mode, chunk in agent.astream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode=["updates", "messages"],
            ):
                if stream_mode == "messages":
                    # messages stream owns all LLM text output.
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        first, second = chunk
                        message = second if hasattr(second, "content") else first
                    else:
                        message = chunk
                    text = _message_text_chunk(message)
                    if text:
                        emitted_message_tokens = True
                        await publish("text", text)
                    continue

                if stream_mode != "updates":
                    continue

                # updates stream owns tool calls, tool results, and images only.
                # AIMessage content is intentionally not published here — the
                # messages stream handles it. Publishing from both paths is what
                # produces duplicate responses.
                for _node, data in chunk.items():
                    for msg in data.get("messages", []):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                await publish("tool_call", {"name": tc["name"], "args": tc["args"]})
                        elif hasattr(msg, "name") and msg.name:
                            raw = flatten_text_content(msg.content)
                            img = re.search(r'[\w\-./]+\.png', raw)
                            if img:
                                await publish("image", {"name": msg.name, "path": img.group(0)})
                            await publish("tool_result", {"name": msg.name, "content": raw})
                        elif hasattr(msg, "content") and msg.content:
                            # Fallback: messages stream emitted nothing this turn,
                            # so this AIMessage is the only copy of the response.
                            # Log it so we can confirm this path is only hit when
                            # the messages stream is genuinely silent (not as a
                            # duplicate). Once logs confirm it's safe, remove this.
                            if not emitted_message_tokens:
                                import logging
                                logging.getLogger(__name__).warning(
                                    "updates_aiMessage_fallback",
                                    extra={
                                        "_node": _node,
                                        "_content_preview": flatten_text_content(msg.content)[:100],
                                        "_thread_id": thread_id,
                                    },
                                )
                                await publish("text", msg.content)
        except Exception as exc:
            await queue.put(("__error__", exc))
        finally:
            await queue.put(done)

    token = _status_emitter.set(publish_status)
    job_progress_token = _job_progress_emitter.set(publish_job_progress)
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
        _job_progress_emitter.reset(job_progress_token)
        _current_thread_id.reset(thread_token)
        if not producer.done():
            producer.cancel()
