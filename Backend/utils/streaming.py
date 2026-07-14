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
import logging
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional

from config.workflow_stages import STAGE_WORKING
from utils.message_utils import flatten_text_content

logger = logging.getLogger(__name__)

# The heartbeat threshold is "order of ten seconds" per the PRD — a watchdog
# alongside stream_response's queue consumer checks idle time on this cadence
# and emits a "still working" status once idle crosses the threshold.
HEARTBEAT_INTERVAL_SECONDS = 10.0
HEARTBEAT_CHECK_SECONDS = 1.0

_status_emitter: ContextVar[Optional[Callable[..., None]]] = ContextVar(
    "status_emitter",
    default=None,
)
_turn_started_at: ContextVar[Optional[float]] = ContextVar("turn_started_at", default=None)
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


def emit_status(message: str, *, stage: str | None = None, detail: Any = None) -> None:
    """Emit a user-visible progress message for the active SSE stream.

    ``stage``/``detail`` (T19) are additive structured fields — a small
    closed vocabulary (config.workflow_stages) driving the frontend's
    workflow strip and the eval's stage-sequence assertions. Omitting them
    keeps producing the bare ``{"message": ...}`` shape earlier callers and
    tests already depend on. A stage emission is also logged (``stage_
    reached``) with the current thread and this turn's elapsed time, so
    per-stage latency is measurable from production traffic.
    """
    emitter = _status_emitter.get()
    if emitter and message:
        emitter(str(message), stage=stage, detail=detail)
    if stage:
        started = _turn_started_at.get()
        elapsed = None
        if started is not None:
            try:
                elapsed = round(asyncio.get_running_loop().time() - started, 3)
            except RuntimeError:
                elapsed = None
        logger.info(
            "stage_reached",
            extra={"_stage": stage, "_thread_id": current_thread_id(), "_elapsed_seconds": elapsed},
        )


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
    # Last-activity clock the heartbeat watchdog polls (below) — touched by
    # every real event this turn publishes, including a bubbled event from a
    # nested stream_response call, so the heartbeat only ever fires during
    # genuine silence, never while a nested sub-agent is itself narrating.
    last_activity = {"t": loop.time()}

    def _touch() -> None:
        last_activity["t"] = loop.time()

    def publish_status(message: str, *, stage: str | None = None, detail: Any = None) -> None:
        _touch()
        if parent_emitter:
            parent_emitter(message, stage=stage, detail=detail)
        payload: dict[str, Any] = {"message": message}
        if stage is not None:
            payload["stage"] = stage
        if detail is not None:
            payload["detail"] = detail
        loop.call_soon_threadsafe(queue.put_nowait, ("status", payload))

    def publish_job_progress(data: dict) -> None:
        _touch()
        if parent_job_progress_emitter:
            parent_job_progress_emitter(data)
        loop.call_soon_threadsafe(queue.put_nowait, ("job_progress", data))

    def publish_chart_payload(data: dict) -> None:
        _touch()
        if parent_chart_emitter:
            parent_chart_emitter(data)
        loop.call_soon_threadsafe(queue.put_nowait, ("chart_payload", data))

    async def publish(event_type: str, data) -> None:
        _touch()
        await queue.put((event_type, data))

    async def watchdog() -> None:
        # Owned by the streaming layer, not the tools (Implementation
        # Decisions): stalls a chat turn currently narrates nothing for
        # minutes at a time (46-55s provider rate-limit sleeps, live-
        # verified 2026-07-07) — this covers that silence with an honest,
        # observation-only "still working" status once idle crosses the
        # threshold, and stops as soon as _touch() sees a real event.
        while True:
            await asyncio.sleep(HEARTBEAT_CHECK_SECONDS)
            idle = loop.time() - last_activity["t"]
            if idle >= HEARTBEAT_INTERVAL_SECONDS:
                publish_status(
                    f"Still working — {int(idle)}s elapsed",
                    stage=STAGE_WORKING,
                    detail=int(idle),
                )

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
                                text = flatten_text_content(msg.content)
                                logging.getLogger(__name__).warning(
                                    "updates_aiMessage_fallback",
                                    extra={
                                        "_node": _node,
                                        "_content_preview": text[:100],
                                        "_thread_id": thread_id,
                                    },
                                )
                                await publish("text", text)
        except Exception as exc:
            await queue.put(("__error__", exc))
        finally:
            await queue.put(done)

    token = _status_emitter.set(publish_status)
    job_progress_token = _job_progress_emitter.set(publish_job_progress)
    chart_token = _chart_emitter.set(publish_chart_payload)
    thread_token = _current_thread_id.set(thread_id)
    # Same "set once, outermost wins" pattern as call_budget/turn_started_at
    # below: a nested stream_response call (run_ground/run_satellite's own
    # inner stream) never passes user_id, so this must not clobber an
    # outer-bound value back to None — the caller that already bound it via
    # user_id_context (T26) stays in effect for the nested call.
    user_token = None
    if user_id is not None and _current_user_id.get() is None:
        user_token = _current_user_id.set(user_id)
    # Established once, before produce()'s Task exists, so every ToolNode
    # gather-Task this agent's own run spawns copies a context that already
    # holds a reference to the same budget dict — see get_call_budget().
    # A nested stream_response call (e.g. run_satellite's inner stream)
    # reuses the outer holder rather than resetting it.
    call_budget_token = None
    if _call_budget.get() is None:
        call_budget_token = _call_budget.set({})
    # Same "set once, outermost wins" pattern as call_budget above — a
    # nested stream_response call (run_satellite's inner stream) reuses the
    # outer turn's start time, so stage_reached's elapsed is measured
    # against the whole request, not just the sub-agent's own slice of it.
    turn_started_token = None
    if _turn_started_at.get() is None:
        turn_started_token = _turn_started_at.set(loop.time())
    producer = asyncio.create_task(produce())
    watchdog_task = asyncio.create_task(watchdog())
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
        if user_token is not None:
            _current_user_id.reset(user_token)
        if call_budget_token is not None:
            _call_budget.reset(call_budget_token)
        if turn_started_token is not None:
            _turn_started_at.reset(turn_started_token)
        if not producer.done():
            producer.cancel()
        if not watchdog_task.done():
            watchdog_task.cancel()
