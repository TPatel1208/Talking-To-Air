from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactReference(BaseModel):
    id: str
    type: str
    title: str | None = None
    row_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableArtifactPayload(BaseModel):
    type: Literal["table"] = "table"
    title: str
    columns: list[str]
    rows: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
