from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChartPayload(BaseModel, extra="allow"):
    type: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    text: str
    charts: list[ChartPayload] = Field(default_factory=list)
    images: list[bytes] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
