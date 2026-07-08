from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .artifact import ArtifactReference


class ChartPayload(BaseModel, extra="allow"):
    type: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    text: str
    charts: list[ChartPayload] = Field(default_factory=list)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    images: list[bytes] = Field(default_factory=list)
    handles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # T22: up to two follow-up questions grounded in this turn's handles/
    # artifacts, propagated straight from a sub-agent's envelope (never
    # synthesized by the backend, story #12). None means the sub-agent
    # offered none — always legitimate (story #7) — and salvage
    # (services/subagent_dispatch.py) never sets this, so a malformed
    # envelope never carries invented suggestions.
    suggested_followups: list[str] | None = Field(default=None, max_length=2)


class SubAgentEnvelope(BaseModel):
    """The strict {summary, artifact_ids, handles} contract a sub-agent's
    final message must satisfy. A missing/invalid envelope is the sub-agent's
    failure to report, not prose to fall back on."""

    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    handles: list[str] = Field(default_factory=list)
    # T22 story #7: optional — omitting it is always a legitimate answer.
    # A sub-agent that names more than two collapses the whole envelope
    # (caught by the max_length constraint), rather than silently truncating
    # a model's over-eager list.
    suggested_followups: list[str] | None = Field(default=None, max_length=2)


def agent_result_to_json(result: AgentResult) -> str:
    return result.model_dump_json(exclude_none=True)


def parse_agent_result(raw: Any) -> AgentResult | None:
    if isinstance(raw, AgentResult):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = AgentResult.model_validate_json(raw)
    except Exception:
        return None
    return parsed


def parse_sub_agent_envelope(raw: Any) -> SubAgentEnvelope | None:
    if isinstance(raw, SubAgentEnvelope):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return SubAgentEnvelope.model_validate_json(raw)
    except Exception:
        return None


def parse_chart_payload(raw: Any) -> ChartPayload | None:
    if isinstance(raw, ChartPayload):
        return raw
    if isinstance(raw, dict):
        try:
            return ChartPayload.model_validate(raw)
        except Exception:
            return None
    if not isinstance(raw, str):
        return None
    try:
        return ChartPayload.model_validate_json(raw)
    except Exception:
        return None
