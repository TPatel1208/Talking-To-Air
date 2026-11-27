"""What a system did for one benchmark case, in terms no system owns.

An adapter fills an Outcome from what crossed the data boundary (calls to the
fake earthdata service and to EPA AQS) and from what the user would see
(charts, artifacts, answer text). Checks read only this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class BoundaryCall:
    """One call that reached the fake earthdata service."""

    tool: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class Outcome:
    boundary_calls: list[BoundaryCall] = field(default_factory=list)
    # (endpoint, params) for each EPA AQS request.
    aqs_calls: list[tuple[str, dict]] = field(default_factory=list)
    # Observation and derived handles the data service returned.
    handles: list[str] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    answer_text: str = ""
    # Progress stage keys in the order they were narrated.
    stages: list[str] = field(default_factory=list)
    # The sources named in the answer's header; diagnostic, never scored.
    sources_header: set[str] | None = None
    # The run's dates at start and end; relative-date checks accept either,
    # so a run that crosses midnight still passes. clock_end defaults to clock.
    clock: date = field(default_factory=date.today)
    clock_end: date | None = None
    elapsed_seconds: float = 0.0
    done: bool = False
    error: str | None = None
    # Why a run that stopped without an answer stopped, e.g. a spent budget; Completed reports it.
    unfinished: str | None = None
    mode: str | None = None
    findings: list = field(default_factory=list)
    # Names of the tool calls the system made itself; diagnostic, never scored.
    tool_calls: list[str] = field(default_factory=list)
    # The questions the user answered through the app's control, as each run asked them
    # (run_id, step, kind, and items or options), and what was sent: run_id, step, value, and
    # calls_before, how many boundary calls had been made when it was sent.
    questions: list[dict[str, Any]] = field(default_factory=list)
    answers: list[dict[str, Any]] = field(default_factory=list)
    # Each run's status at the end of the last turn it took part in.
    run_statuses: dict[str, str] = field(default_factory=dict)
    # Model tokens the provider reported, keyed by (model, agent, kind) where
    # kind is "input", "output", "cache_read" or "cache_creation" (the last two are
    # shares of input). Never scored.
    tokens: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def token_total(self, kind: str) -> int | None:
        """Tokens of ``kind`` across every model call, or None if none were reported."""
        counts = [count for (_, _, counted_kind), count in self.tokens.items() if counted_kind == kind]
        return sum(counts) if counts else None


# Submitting a retrieval is not a result, so retrieve_subset is absent: a
# handle counts only once a status poll says it is ready or a derived product exists.
_HANDLE_TOOLS = ("get_retrieval_status", "rematerialize", "align", "convert_format")
_FAILED_STATUSES = ("failed", "error", "cancelled", "not_found", "expired")


def handles_from_calls(calls: list[BoundaryCall]) -> list[str]:
    """Every observation or derived handle the data service handed back, once each."""
    found: list[str] = []
    for call in calls:
        if call.tool not in _HANDLE_TOOLS or call.result.get("status") in _FAILED_STATUSES:
            continue
        for key in ("obs_handle", "handle"):
            value = call.result.get(key)
            if isinstance(value, str) and value and value not in found:
                found.append(value)
    return found
