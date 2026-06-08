"""
utils/streaming.py
------------------
Shared stream_response for supervisor, ground, and satellite agents.

Usage
-----
Supervisor (needs thread_ref to keep subagent closures in sync):
    from utils.streaming import stream_response
    for event_type, data in stream_response(agent, user_input, thread_id, thread_ref):
        ...

Subagents (no thread_ref):
    from utils.streaming import stream_response
    for event_type, data in stream_response(agent, user_input, thread_id):
        ...

Yields
------
("tool_call",   {"name": str, "args": dict})         — agent is about to call a tool
("image",       {"name": str, "path": str})          — a PNG path found inside a tool result
("tool_result", {"name": str, "content": str})       — tool returned; full content (truncated at 300 chars)
("text",        str)                                 — final assistant response text
"""

import re
from collections.abc import AsyncGenerator
from typing import Optional


async def stream_response(
    agent,
    user_input: str,
    thread_id: str,
    thread_ref: Optional[dict] = None,
) -> AsyncGenerator[tuple, None]:
    """
    Stream one conversation turn, yielding (event_type, data) tuples.

    Parameters
    ----------
    agent       : LangGraph agent returned by build_*_agent().
    user_input  : The user's message for this turn.
    thread_id   : LangGraph checkpoint thread ID for this session.
    thread_ref  : Supervisor-only. The mutable dict returned by build_agent()
                  whose 'id' key is read by subagent tool closures. Updated
                  here before streaming so closures always see the current
                  thread_id. Pass None (default) for ground/satellite agents.
    """
    if thread_ref is not None:
        thread_ref["id"] = thread_id

    config = {"configurable": {"thread_id": thread_id}}

    async for stream_mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if stream_mode != "updates":
            continue

        for _node, data in chunk.items():
            for msg in data.get("messages", []):

                # ── Tool call being dispatched ────────────────────────────
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield ("tool_call", {"name": tc["name"], "args": tc["args"]})

                # ── Tool result returned ──────────────────────────────────
                elif hasattr(msg, "name") and msg.name:
                    raw = str(msg.content)
                    # If the result contains a PNG path, emit it as a separate
                    # "image" event so callers can render it.  The full content
                    # string is always passed through as "tool_result" so no
                    # text is discarded when a PNG path is present.
                    img = re.search(r'[\w\-./]+\.png', raw)
                    if img:
                        yield ("image", {"name": msg.name, "path": img.group(0)})
                    yield ("tool_result", {"name": msg.name, "content": raw})

                # ── Assistant text response ───────────────────────────────
                elif hasattr(msg, "content") and msg.content:
                    yield ("text", msg.content)
