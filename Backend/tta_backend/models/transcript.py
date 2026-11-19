from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tta_backend.harness.results import RUN_ID_PATTERN, Mode, ResultStatus
from tta_backend.models.artifact import ArtifactReference


def _contains_nul(value: Any) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_contains_nul(k) or _contains_nul(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_nul(v) for v in value)
    return False


# The connectors a result can stop for, as the frontend's connectors UI names them.
ConnectorName = Literal["earthdata", "epa_aqs"]


class TranscriptResult(BaseModel):
    """One result of a turn as the user saw it: its run, status and mode, the
    fixed-wording reason when it is not done, and the connector it stopped for.
    Work refused because a source was down has no run and no mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str | None = Field(default=None, pattern=RUN_ID_PATTERN.pattern)
    status: ResultStatus
    mode: Mode | None = None
    reason: str | None = None
    connector: ConnectorName | None = None

    def view(self) -> dict[str, Any]:
        """The shape the ``done`` frame and the history carry."""
        return {
            "run_id": self.run_id, "status": self.status, "mode": None if self.mode is None else self.mode.view(),
            "reason": self.reason, "connector": self.connector,
        }


class TranscriptTurn(BaseModel):
    """One turn as the user saw it. Charts and artifacts are references, not data."""

    thread_id: str
    user_id: str
    # Assigned by the store on append.
    turn_index: int | None = None
    user_text: str
    answer_text: str
    charts: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    label: Mode | None = None
    results: list[TranscriptResult] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    status: Literal["done", "stopped", "error"] = "done"
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _storable(self) -> Self:
        # Postgres text and JSONB cannot hold NUL; reject rather than alter user content.
        fields = (
            self.thread_id, self.user_id, self.user_text, self.answer_text,
            self.sources, self.charts, [a.model_dump() for a in self.artifacts],
            [r.model_dump() for r in self.results], self.suggested_followups,
        )
        if _contains_nul(fields):
            raise ValueError("transcript text cannot contain NUL characters")
        if self.created_at is not None and self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return self

    def to_history_messages(self) -> list[dict[str, Any]]:
        """The user and assistant messages, shaped like HistoryService.build_history."""
        return [
            {"role": "user", "content": self.user_text, "toolCalls": []},
            {
                "role": "assistant",
                "content": self.answer_text,
                "toolCalls": [],
                "charts": list(self.charts),
                "artifacts": [a.model_dump(exclude_none=True) for a in self.artifacts],
            },
        ]
