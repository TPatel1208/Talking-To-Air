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
("tool_call",   {"name": str, "args": dict})   — agent is about to call a tool
("tool_result", {"name": str, "content": str}) — tool returned; PNG path or truncated text
("text",        str)                           — final assistant response text
"""

import re
from typing import Generator, Optional


def stream_response(
    agent,
    user_input: str,
    thread_id: str,
    thread_ref: Optional[dict] = None,
) -> Generator[tuple, None, None]:
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

    for stream_mode, chunk in agent.stream(
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
                    # Always extract a PNG path first so it's never lost
                    # even when buried inside a long result string.
                    img = re.search(r'[\w\-./]+\.png', raw)
                    content_out = img.group(0) if img else raw[:300]
                    yield ("tool_result", {"name": msg.name, "content": content_out})

                # ── Assistant text response ───────────────────────────────
                elif hasattr(msg, "content") and msg.content:
                    yield ("text", msg.content)