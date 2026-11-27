"""
Request-scoped state the chat host binds for the helpers it calls: the status
emitter, the authenticated user, the per-turn call budget and the MCP
transport-failure breaker state.
"""

import logging
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# How long a live turn may stay silent before the host writes a "still
# working" status line.
HEARTBEAT_INTERVAL_SECONDS = 10.0

_status_emitter: ContextVar[Optional[Callable[..., None]]] = ContextVar(
    "status_emitter",
    default=None,
)
_current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)
_call_budget: ContextVar[Optional[dict[str, Any]]] = ContextVar("call_budget", default=None)
# Per-turn MCP transport-failure circuit breaker state: a mutable dict
# {"consecutive": int, "tripped": bool}, mutated in place so every task copy
# of the context sees the same object. Unbound (None) means the breaker is
# inert.
_mcp_failure_state: ContextVar[Optional[dict[str, Any]]] = ContextVar("mcp_failure_state", default=None)


def bind_status_emitter(emitter: Callable[..., None]) -> None:
    """Send ``emit_status`` calls made from the current context, and the tasks
    and threads it starts, to ``emitter(message, stage=..., detail=...)``."""
    _status_emitter.set(emitter)


def emit_status(message: str, *, stage: str | None = None, detail: Any = None) -> None:
    """Emit a user-visible progress message for the active stream.

    ``stage``/``detail`` are additive structured fields from a small closed
    vocabulary (config.workflow_stages) that drives the frontend's workflow
    strip. A stage emission is also logged (``stage_reached``)."""
    emitter = _status_emitter.get()
    if emitter and message:
        emitter(str(message), stage=stage, detail=detail)
    if stage:
        logger.info("stage_reached", extra={"_stage": stage})


def get_call_budget() -> dict[str, Any]:
    """Return the current turn's mutable per-turn counters.

    A task started from this context gets a *copy* of it, so a plain
    ContextVar.set() inside one task is invisible to its siblings. A dict
    survives that boundary: it is set once, early, in the context that
    becomes every such task's parent, and callers mutate it in place rather
    than re-``set()``ing the ContextVar."""
    budget = _call_budget.get()
    if budget is None:
        budget = {}
        _call_budget.set(budget)
    return budget


def get_mcp_failure_state() -> dict[str, Any] | None:
    """The active turn's MCP transport-failure circuit-breaker state, or None
    when none is bound.

    ``earthdata_mcp.results.call_tool`` reads this: on a transport/timeout
    failure it increments ``consecutive`` and, once that crosses
    ``settings.mcp_transport_failure_ceiling``, sets ``tripped`` so later MCP
    calls short-circuit instead of opening another failing session. A
    ``None`` return means the breaker is inert.

    Unlike ``get_call_budget`` this never lazily creates the dict: an unbound
    caller must read as inert, not silently start its own breaker in a context
    that will be discarded."""
    return _mcp_failure_state.get()


def current_user_id() -> str | None:
    """The authenticated user for the active request — the workspace_id
    earthdata-retrieval MCP tools bind their calls to (see earthdata_mcp.workspace)."""
    return _current_user_id.get()


@contextmanager
def user_id_context(user_id: str) -> Iterator[None]:
    """Bind ``current_user_id()`` for the duration of the block."""
    token = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(token)


async def iter_with_user_id(user_id: str, chunks: AsyncIterator[Any]) -> AsyncGenerator[Any, None]:
    """Re-bind ``current_user_id()`` around every pull of ``chunks``.

    A StreamingResponse body iterates *after* the endpoint handler has
    returned, so a ``with user_id_context(...)`` inside the handler has
    already been reset by the time later chunks are produced — any
    workspace-bound MCP tool called while generating those chunks would
    refuse with missing_user_context. This wrapper carries the request's
    user binding across that boundary, one pull at a time."""
    while True:
        with user_id_context(user_id):
            try:
                chunk = await chunks.__anext__()
            except StopAsyncIteration:
                return
        yield chunk
