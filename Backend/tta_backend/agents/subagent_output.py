"""The sub-agents answer with a SubAgentEnvelope tool call, not free text."""
from __future__ import annotations

from typing import Any

from langchain.agents.structured_output import ToolStrategy

from tta_backend.models import SubAgentEnvelope

# Replaces the default reply, which repeats the whole envelope.
_ENVELOPE_RECORDED = "Answer recorded."


def subagent_response_format() -> ToolStrategy[Any]:
    """The response_format both sub-agents are built with."""
    return ToolStrategy(SubAgentEnvelope, tool_message_content=_ENVELOPE_RECORDED)
