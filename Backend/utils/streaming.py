"""
Shared stream_response for supervisor, ground, and satellite agents.

Yields:
    ("tool_call", {"name": str, "args": dict})
    ("status", {"message": str})
    ("tool_result", {"name": str, "content": str})
    ("text", str)
    ("job_progress", {"job_handle": str, "status": str, "progress": Any, "phase": Any, "message": str | None})
    ("chart_payload", dict)  # full render payload — see utils.streaming.emit_chart
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import contextmanager
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
_chart_emitter: ContextVar[Optional[Callable[[dict], None]]] = ContextVar(
    "chart_emitter",
    default=None,
)
_current_thread_id: ContextVar[Optional[str]] = ContextVar("current_thread_id", default=None)
_current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)
_call_budget: ContextVar[Optional[dict]] = ContextVar("call_budget", default=None)


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


def emit_chart(payload: dict) -> None:
    """Emit a full chart render payload for the active SSE stream, out-of-band
    from the tool's model-facing return value (T13 two-audience split: the
    model gets a compact summary, the frontend gets this full payload via the
    existing chart/artifact pipeline)."""
    emitter = _chart_emitter.get()
    if emitter:
        emitter(payload)


def get_call_budget() -> dict:
    """Return the current request's mutable per-agent call-budget counters.

    langgraph.prebuilt.tool_node.ToolNode wraps every tool call in
    asyncio.gather(), which spawns a fresh Task with a *copied* context for
    each call — a plain ContextVar.set() inside that Task never becomes
    visible to a sibling Task spawned later from the same parent context.
    A dict survives that boundary: stream_response sets this ContextVar
    once, early, in the context that becomes every such Task's parent, and
    callers mutate the dict in place rather than re-``set()``ing the
    ContextVar — the mutation is visible through every context copy because
    it is the same object, not a new binding.
    """
    budget = _call_budget.get()
    if budget is None:
        budget = {}
        _call_budget.set(budget)
    return budget


def current_thread_id() -> str | None:
    return _current_thread_id.get()


def current_user_id() -> str | None:
    """The authenticated user for the active request — the workspace_id
    earthdata-retrieval MCP tools bind their calls to (see earthdata_mcp.workspace)."""
    return _current_user_id.get()


@contextmanager
def user_id_context(user_id: str):
    """Bind ``current_user_id()`` for non-chat endpoints (e.g. the jobs
    endpoint) that call workspace-bound MCP tools outside of stream_response,
    which normally sets this for the duration of a chat turn."""
    token = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(token)


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
    user_id: Optional[str] = None,
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
    parent_chart_emitter = _chart_emitter.get()

    def publish_status(message: str) -> None:
        if parent_emitter:
            parent_emitter(message)
        loop.call_soon_threadsafe(queue.put_nowait, ("status", {"message": message}))

    def publish_job_progress(data: dict) -> None:
        if parent_job_progress_emitter:
            parent_job_progress_emitter(data)
        loop.call_soon_threadsafe(queue.put_nowait, ("job_progress", data))

    def publish_chart_payload(data: dict) -> None:
        if parent_chart_emitter:
            parent_chart_emitter(data)
        loop.call_soon_threadsafe(queue.put_nowait, ("chart_payload", data))

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
    chart_token = _chart_emitter.set(publish_chart_payload)
    thread_token = _current_thread_id.set(thread_id)
    user_token = _current_user_id.set(user_id)
    # Established once, before produce()'s Task exists, so every ToolNode
    # gather-Task this agent's own run spawns copies a context that already
    # holds a reference to the same budget dict — see get_call_budget().
    # A nested stream_response call (e.g. run_satellite's inner stream)
    # reuses the outer holder rather than resetting it.
    call_budget_token = None
    if _call_budget.get() is None:
        call_budget_token = _call_budget.set({})
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
        _chart_emitter.reset(chart_token)
        _current_thread_id.reset(thread_token)
        _current_user_id.reset(user_token)
        if call_budget_token is not None:
            _call_budget.reset(call_budget_token)
        if not producer.done():
            producer.cancel()
