"""
agents/subagent_trim.py
=========================
High-ceiling trim safety net for the stateless sub-agents (T13). Reuses the
supervisor's trim_middleware pattern (wrap_model_call + trim_messages) with a
ceiling sized so it never fires in a healthy workflow — compact tool results
(tools/satellite_tools/plot_tools.py::_save_chart,
earthdata_mcp/workspace.py::model_view_describe_dataset) are the first line
of defense; this exists only to convert an unforeseen bloat source into a
degraded-but-alive turn instead of a hard provider rejection.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import trim_messages

from config.settings import get_settings

logger = logging.getLogger(__name__)


def build_subagent_trim_middleware(agent_type: str, max_tokens: int | None = None) -> AgentMiddleware:
    """Build a wrap_model_call middleware trimming ``agent_type``'s message
    history to ``max_tokens`` (default: settings.subagent_trim_token_ceiling),
    keeping the most recent messages. Logs a WARNING-level
    ``subagent_trim_activated`` event whenever trimming actually removes
    messages, so any remaining bloat source is discovered from logs.
    """
    ceiling = max_tokens if max_tokens is not None else get_settings().subagent_trim_token_ceiling

    @wrap_model_call
    async def trim_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        messages = request.state["messages"]
        trimmed = trim_messages(
            messages,
            max_tokens=ceiling,
            strategy="last",
            token_counter="approximate",
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        # Never let trimming remove every usable turn from a request.
        if not trimmed:
            trimmed = messages
        if len(trimmed) < len(messages):
            logger.warning(
                "subagent_trim_activated",
                extra={
                    "_event": "subagent_trim_activated",
                    "_agent_type": agent_type,
                    "_input_message_count": len(messages),
                    "_trimmed_message_count": len(trimmed),
                    "_token_ceiling": ceiling,
                },
            )
        return await handler(request.override(messages=trimmed))

    return trim_middleware
