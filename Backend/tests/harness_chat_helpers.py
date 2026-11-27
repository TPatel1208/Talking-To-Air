"""Doubles for the chat host: a supervisor that plays back a turn, and in-memory transcript and chart stores.

``PlaybackSupervisor`` stands where the harness supervisor does and drives the
run context it is handed exactly as the harness would: progress and text
through ``events``, charts stored then emitted, tables stored. The host around
it (services/harness_chat.HarnessChat) is the real one.
"""
from __future__ import annotations

import asyncio
import itertools
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator
from unittest.mock import AsyncMock, patch

from tta_backend.harness.results import Result
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.supervisor import Turn
from tta_backend.harness.testing import FakeEarthdata, StaticCredentials, env_factory
from tta_backend.harness.trace import new_run_id
from tta_backend.models.transcript import TranscriptTurn
from tta_backend.repositories import transcript_repository
from tta_backend.services.chart_service import ChartService
from tta_backend.services.harness_chat import HarnessChat, JobReporting, TurnPorts
from tta_backend.utils.streaming import current_user_id


def make_turn(
    text: str,
    *,
    results: tuple[Result, ...] = (),
    status: Any = "answered",
    followups: tuple[str, ...] = (),
    question: dict[str, Any] | None = None,
    variable_choice: dict[str, Any] | None = None,
    unavailable: tuple[str, ...] = (),
) -> Turn:
    return Turn(
        text=text, results=results, run_ids=tuple(r.trace_id for r in results), followups=followups,
        question=question, variable_choice=variable_choice, unavailable=unavailable, status=status, model_calls=1, tool_calls=0,
        input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_creation_tokens=0,
    )


class PlaybackSupervisor:
    """Plays ``steps`` into the run context, then returns ``turn`` (or a turn whose text is the text played).

    A step is ``("text", str)``, ``("status", {"message", "stage"?})``,
    ``("chart", payload)`` (stored, then emitted), ``("emit", payload)``
    (emitted only), ``("table", (title, columns, rows))``, ``("call", (tool, arguments))``
    (an earthdata call through a run's environment), ``("check", (tool, arguments))``
    (a call on the turn's own session, as the waiting-job check makes), ``("raise", exc)``
    or an ``asyncio.Event`` to wait on.
    """

    def __init__(self, *steps: Any, turn: Turn | None = None) -> None:
        self.steps = steps
        self.result = turn
        self.contexts: list[Any] = []
        self.users_seen: list[str | None] = []
        self.messages: list[str] = []
        self.answers: list[Any] = []
        self.resumes: list[Any] = []

    async def turn(
        self, context: Any, build_env: Any, earthdata: Any, message: str, answer: Any = None, resume: Any = None,
    ) -> Turn:
        self.contexts.append(context)
        self.users_seen.append(current_user_id())
        self.messages.append(message)
        self.answers.append(answer)
        self.resumes.append(resume)
        played: list[str] = []
        for step in self.steps:
            if isinstance(step, asyncio.Event):
                await step.wait()
                continue
            kind, data = step
            if kind == "text":
                context.events.text(data)
                played.append(data)
            elif kind == "status":
                context.events.progress(data["message"], stage=data.get("stage"), detail=data.get("detail"))
            elif kind == "chart":
                ref = await context.artifacts.store_chart(data)
                context.events.chart({**data, "chart_id": ref.id})
            elif kind == "emit":
                context.events.chart(data)
            elif kind == "table":
                title, columns, rows = data
                await context.artifacts.store_table(title, columns, rows)
            elif kind == "call":
                tool, arguments = data
                env = await build_env(context, new_run_id())
                await env.earthdata.call(tool, arguments)
            elif kind == "check":
                tool, arguments = data
                await earthdata.call(tool, arguments)
            elif kind == "raise":
                raise data
        return self.result or make_turn("".join(played))


class MemoryTranscripts:
    """The transcript repository in memory. With ``watch`` set to the list a
    test collects frames into, ``written_after`` notes how many frames had
    arrived at each write."""

    def __init__(self) -> None:
        self.turns: list[TranscriptTurn] = []
        self.watch: list[str] | None = None
        self.written_after: list[int] = []

    async def append_turn(self, turn: TranscriptTurn) -> TranscriptTurn:
        index = sum(1 for t in self.turns if (t.user_id, t.thread_id) == (turn.user_id, turn.thread_id))
        stored = turn.model_copy(update={"turn_index": index})
        self.turns.append(stored)
        if self.watch is not None:
            self.written_after.append(len(self.watch))
        return stored

    async def list_turns(self, thread_id: str, user_id: str) -> list[TranscriptTurn]:
        return [t for t in self.turns if (t.thread_id, t.user_id) == (thread_id, user_id)]

    async def delete_transcript(self, thread_id: str, user_id: str) -> None:
        self.turns = [t for t in self.turns if (t.thread_id, t.user_id) != (thread_id, user_id)]


class MemoryCharts:
    """The chart repository in memory, counting writes."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.saves = 0
        self._ids = itertools.count(1)

    async def save_chart(self, thread_id: str, payload: dict[str, Any], user_id: str) -> dict[str, Any]:
        self.saves += 1
        chart_id = payload.get("chart_id") or f"chart-{next(self._ids)}"
        stored = {**payload, "chart_id": chart_id, "thread_id": thread_id, "user_id": user_id}
        self.rows[chart_id] = stored
        return stored

    async def get_chart(self, chart_id: str) -> dict[str, Any] | None:
        return self.rows.get(chart_id)

    async def get_charts(self, chart_ids: list[str], user_id: str) -> dict[str, dict[str, Any]]:
        return {i: self.rows[i] for i in chart_ids if i in self.rows and self.rows[i]["user_id"] == user_id}


@contextmanager
def memory_repositories(transcripts: MemoryTranscripts, charts: MemoryCharts) -> Iterator[None]:
    """Transcript, chart and table persistence in memory for the duration."""
    target = "tta_backend.services.chart_service.chart_repository"
    with ExitStack() as stack:
        for name in ("append_turn", "list_turns", "delete_transcript"):
            stack.enter_context(patch.object(transcript_repository, name, getattr(transcripts, name)))
        for name in ("save_chart", "get_chart", "get_charts"):
            stack.enter_context(patch(f"{target}.{name}", getattr(charts, name)))
        stack.enter_context(patch("tta_backend.services.artifact_store.artifact_repository.save_artifact", AsyncMock()))
        stack.enter_context(
            patch("tta_backend.services.artifact_store.artifact_repository.delete_expired_unclaimed", AsyncMock())
        )
        yield


def make_chat(
    supervisor: Any,
    root: str,
    *,
    stores: Stores | None = None,
    ports: Any = None,
    credentials: dict[str, str] | None = None,
    timeout: float = 60.0,
    clock: Any = None,
    earthdata: FakeEarthdata | None = None,
    reachable: Any = None,
) -> HarnessChat:
    """A chat host on ``supervisor``, memory stores and fake earthdata, its runs' jobs reported as the app's are."""
    session = earthdata or FakeEarthdata()

    def default_ports(user_id: str, report_job: Any) -> TurnPorts:
        return TurnPorts(
            build_env=env_factory(root, earthdata=JobReporting(session, report_job)), earthdata=session,
        )

    return HarnessChat(
        supervisor=supervisor, stores=stores or memory_stores(), ports=ports or default_ports,
        credentials=lambda user_id: StaticCredentials(credentials or {}),
        chart_service=ChartService(), chat_turn_timeout_seconds=timeout, long_request_seconds=999.0, clock=clock,
        **({} if reachable is None else {"reachable": reachable}),
    )
