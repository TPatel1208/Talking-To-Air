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


class SubAgentEnvelope(BaseModel):
    """The strict {summary, artifact_ids, handles} contract a sub-agent's
    final message must satisfy. A missing/invalid envelope is the sub-agent's
    failure to report, not prose to fall back on."""

    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    handles: list[str] = Field(default_factory=list)


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
