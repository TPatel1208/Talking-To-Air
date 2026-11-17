"""Chat turns through the harness supervisor, as the SSE frames the frontend reads.

One ``HarnessChat`` is built at startup (``build_harness_chat``) and shared by
every request. ``stream_chat_events`` runs one turn: the supervisor runs in a
task of its own while its events reach the stream through a queue, and the
turn ends with the transcript written, then the closing frame.

Frames, in order:

- ``status`` ``{message, stage?, detail?, run_id?}`` for progress, and a
  heartbeat during silence;
- ``text`` ``{content}`` for each piece of the answer as it streams;
- ``job_progress`` ``{job_handle, status, progress, phase, message, note}``
  for each retrieval a run submits or polls, which the jobs panel lists and
  Stop cancels;
- ``chart`` with the stored chart row, then ``artifact`` with its gallery
  reference when the chart has one; ``artifact`` for each stored table. Each
  chart is stored once and sent once;
- ``done`` ``{thread_id, response, artifacts, tool_calls, results, label,
  suggested_followups?, variable_choice?}``;
- on a failed turn, ``error`` ``{detail}`` before ``done``; on a timeout or
  when admission refuses the turn, ``error`` then a short ``done``; on any
  other exception, ``error`` alone.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from tta_backend.config.error_templates import (
    render_error_answer,
    render_overloaded_answer,
    render_turn_timeout_answer,
)
from tta_backend.config.workflow_stages import STAGE_WORKING
from tta_backend.domain.earthdata.retrieval import annotate_paused
from tta_backend.earthdata_mcp.connection import STATE_READY
from tta_backend.earthdata_mcp.results import CATEGORY_CONTRACT, CATEGORY_PROVIDER_UNAVAILABLE, MCPToolError
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.handles.resolver import OpenRetrieval
from tta_backend.harness.jobs import STATUS_TOOL, SUBMIT_TOOLS
from tta_backend.harness.models import Models
from tta_backend.harness.results import Mode, Result
from tta_backend.harness.run_context import (
    ArtifactRef,
    Clock,
    Compute,
    Credentials,
    EarthdataSession,
    Identity,
    Limits,
    Places,
    RunContext,
    RunEnv,
    run_env,
)
from tta_backend.harness.storage import Stores
from tta_backend.harness.supervisor import Supervisor, Turn
from tta_backend.harness.supervisor.answer import data_sources
from tta_backend.harness.workflows.runner import EnvFactory
from tta_backend.infrastructure.settings import Settings
from tta_backend.models import chart_reference
from tta_backend.models.artifact import ArtifactReference
from tta_backend.models.transcript import TranscriptTurn
from tta_backend.repositories import transcript_repository
from tta_backend.services import admission, cube_cache
from tta_backend.services.artifact_registry import build_artifact_reference
from tta_backend.services.artifact_store import artifact_store
from tta_backend.services.chart_service import ChartService
from tta_backend.services.connector_credential_service import (
    CONNECTOR_TYPE_EARTHDATA,
    CONNECTOR_TYPE_EPA_AQS,
    AqsCredentialInjector,
    EdlCredentialInjector,
)
from tta_backend.services.harness_host import (
    AdmissionCompute,
    McpEarthdataSession,
    ResolverPlaces,
    limits_from_settings,
    open_retrieval_over,
)
from tta_backend.utils import streaming
from tta_backend.utils.streaming import user_id_context

logger = logging.getLogger(__name__)

PACKS_ROOT = Path(__file__).resolve().parents[2] / "packs"
# The data layer's wording when the earthdata connection is not up.
EARTHDATA_UNAVAILABLE = "The satellite data layer is temporarily unavailable."

TranscriptStatus = Literal["done", "stopped", "error"]


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# The run context's parts


class UtcClock:
    def today(self) -> datetime.date:
        return self.now().date()

    def now(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.UTC)


class ConnectorCredentials:
    """The user's own connector secrets, resolved at each call. A connector the
    user has not connected, or whose secret has expired, has none; there is no
    shared fallback."""

    def __init__(self, user_id: str, edl: EdlCredentialInjector, aqs: AqsCredentialInjector) -> None:
        self._user_id = user_id
        self._edl = edl
        self._aqs = aqs

    async def secret(self, connector: str) -> str | None:
        if connector == CONNECTOR_TYPE_EARTHDATA:
            return await self._edl.resolve(self._user_id)
        if connector == CONNECTOR_TYPE_EPA_AQS:
            pair = await self._aqs.resolve_credentials(self._user_id)
            return None if pair is None else json.dumps({"email": pair.email, "key": pair.key})
        return None


Put = Callable[[tuple[str, Any]], None]


def _putter(queue: asyncio.Queue[tuple[str, Any]]) -> Put:
    """Puts on ``queue`` from the loop's thread or any other."""
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    def put(item: tuple[str, Any]) -> None:
        if threading.get_ident() == loop_thread:
            queue.put_nowait(item)
        else:
            loop.call_soon_threadsafe(queue.put_nowait, item)

    return put


class StreamEvents:
    """Progress, text and charts onto the turn's queue. A question needs
    nothing here: the turn carries it to the ``done`` frame."""

    def __init__(self, put: Put) -> None:
        self._put = put

    def progress(self, message: str, *, stage: str | None = None, detail: Any = None) -> None:
        data: dict[str, Any] = {"message": message}
        if stage is not None:
            data["stage"] = stage
        if detail is not None:
            data["detail"] = detail
            run_id = detail.get("run_id") if isinstance(detail, dict) else None
            if isinstance(run_id, str):
                data["run_id"] = run_id
        self._put(("status", data))

    def chart(self, payload: dict[str, Any]) -> None:
        self._put(("chart", payload))

    def question(self, payload: dict[str, Any]) -> None:
        return None

    def text(self, content: str) -> None:
        if content:
            self._put(("text", {"content": content}))


class StoredArtifacts:
    """Charts into the chart repository and tables into the artifact store, owned by the turn's user and thread.

    A stored chart's row is kept for the turn, so sending it needs no second
    read. A stored table is sent as an ``artifact`` at once.
    """

    def __init__(self, chart_service: ChartService, thread_id: str, user_id: str, put: Put) -> None:
        self._charts = chart_service
        self._thread_id = thread_id
        self._user_id = user_id
        self._put = put
        self.stored_charts: dict[str, dict[str, Any]] = {}

    async def store_chart(self, payload: dict[str, Any]) -> ArtifactRef:
        stored = await self._charts.persist_chart_payload(self._thread_id, payload, self._user_id)
        if stored is None or not stored.get("chart_id"):
            raise RuntimeError("the chart was not stored")
        chart_id = str(stored["chart_id"])
        self.stored_charts[chart_id] = stored
        return ArtifactRef(id=chart_id, type="chart", title=payload.get("title"))

    async def store_table(
        self,
        title: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        minted = artifact_store.put_table(title, columns, rows, metadata)
        claimed = await artifact_store.claim(minted.id, self._user_id, self._thread_id)
        self._put(("artifact", claimed.model_dump(exclude_none=True)))
        return ArtifactRef(id=claimed.id, type=claimed.type, title=claimed.title)


# The host's ports for one user's turn


@dataclass(frozen=True)
class TurnPorts:
    """What a turn hands the supervisor besides the run context.

    ``earthdata`` is the session the supervisor checks waiting runs' jobs
    with at the start of a turn. Those checks are not reported as the turn's
    jobs, so a Stop never cancels a retrieval an earlier turn left waiting.
    """

    build_env: EnvFactory
    earthdata: EarthdataSession


JobReport = Callable[[dict[str, Any]], None]
# A ports factory: the user, and where the turn's runs report their retrieval jobs.
PortsFactory = Callable[[str, JobReport], TurnPorts]


def job_progress(tool: str, arguments: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any] | None:
    """The ``job_progress`` payload for a submit or status reply, or None for
    any other call. Only the job handle and the reply's status fields go in;
    the request's arguments never do."""
    if tool in SUBMIT_TOOLS:
        handle = reply.get("job_handle")
        default = "submitted"
    elif tool == STATUS_TOOL:
        handle = reply.get("job_handle") or arguments.get("job_handle")
        default = ""
    else:
        return None
    if not isinstance(handle, str) or not handle:
        return None
    data = annotate_paused(reply)
    return {
        "job_handle": handle,
        "status": data.get("status") or default,
        "progress": data.get("progress"),
        "phase": data.get("phase"),
        "message": data.get("message"),
        "note": data.get("note"),
    }


class JobReporting:
    """An earthdata session that reports each retrieval it submits or polls."""

    def __init__(self, inner: EarthdataSession, report: JobReport) -> None:
        self._inner = inner
        self._report = report

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        reply = await self._inner.call(tool, arguments, idempotency_key=idempotency_key)
        try:
            payload = job_progress(tool, arguments, reply)
            if payload is not None:
                self._report(payload)
        except Exception:
            # The call succeeded; a report that cannot be made must not undo it.
            logger.warning("job_progress_report_failed", extra={"_tool": tool})
        return reply


class ReadyEarthdata:
    """The earthdata session over the app's live MCP tools; while the
    connection is not ready, every call fails as provider_unavailable."""

    def __init__(self, manager: Any, tools: dict[str, BaseTool]) -> None:
        self._manager = manager
        self._session = McpEarthdataSession(tools)

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        if self._manager is None or self._manager.state != STATE_READY:
            raise MCPToolError(CATEGORY_PROVIDER_UNAVAILABLE, EARTHDATA_UNAVAILABLE)
        return await self._session.call(tool, arguments, idempotency_key=idempotency_key)


def env_factory(
    payload_root: str | os.PathLike[str],
    *,
    earthdata: EarthdataSession,
    open_retrieval: OpenRetrieval,
    compute: Compute,
    limits: Limits,
    places: Places,
    models: Models,
) -> EnvFactory:
    """Builds the environment of a run already started, on the app's ports."""

    async def build(context: RunContext, run_id: str) -> RunEnv:
        return run_env(
            context, run_id=run_id, payloads=PayloadStore(payload_root), open_retrieval=open_retrieval,
            earthdata=earthdata, compute=compute, limits=limits, places=places, models=models,
        )

    return build


def payload_root(settings: Settings) -> str:
    """Derived handles' payload files, beside the cube store, which sweeps any directory of its own it cannot read."""
    cubes = Path(settings.cube_store_dir)
    return str(cubes.parent / "harness_payloads")


# The frames a turn ends with


def _mode_view(mode: Mode) -> dict[str, Any]:
    workflow = None if mode.workflow is None else {"name": mode.workflow.name, "version": mode.workflow.version}
    return {"kind": mode.kind, "label": mode.label(), "workflow": workflow}


def results_view(results: Sequence[Result]) -> list[dict[str, Any]]:
    return [{"run_id": r.trace_id, "status": r.status, "mode": _mode_view(r.mode)} for r in results]


def answer_mode(results: Sequence[Result]) -> Mode | None:
    """The mode the answer is badged with: its first result's."""
    return results[0].mode if results else None


def _option_text(option: Any) -> str:
    if isinstance(option, str):
        return option
    if isinstance(option, dict):
        for key in ("label", "name", "value"):
            if isinstance(option.get(key), str):
                return str(option[key])
    return json.dumps(option, sort_keys=True, default=str)


def picker_payload(variable_choice: dict[str, Any]) -> dict[str, Any]:
    """A turn's question with options in the variable picker's shape. Picking
    an option sends its text as the user's reply."""
    options = [_option_text(option) for option in variable_choice.get("options") or []]
    prompt = variable_choice.get("prompt")
    return {
        "message": prompt if isinstance(prompt, str) else "",
        "run_id": variable_choice.get("run_id"),
        "candidates": [
            {"name": text, "category": "distinct", "units": None, "valid_fraction": None, "reasons": [],
             "prompt": text}
            for text in options
        ],
    }


# The chat service


class TurnSupervisor(Protocol):
    async def turn(
        self, context: RunContext, build_env: EnvFactory, earthdata: EarthdataSession, message: str
    ) -> Turn: ...


@dataclass
class _Seen:
    """What a turn has sent so far."""

    text: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    chart_ids: set[str] = field(default_factory=set)


class HarnessChat:
    """Runs each chat message as one supervisor turn and streams it as SSE frames."""

    def __init__(
        self,
        *,
        supervisor: TurnSupervisor,
        stores: Stores,
        ports: PortsFactory,
        credentials: Callable[[str], Credentials],
        chart_service: ChartService,
        chat_turn_timeout_seconds: float,
        long_request_seconds: float,
        clock: Clock | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.stores = stores
        self._ports = ports
        self._credentials = credentials
        self.chart_service = chart_service
        self.chat_turn_timeout_seconds = chat_turn_timeout_seconds
        self.long_request_seconds = long_request_seconds
        self._clock = clock or UtcClock()

    async def stream_chat_events(
        self, message: str, thread_id: str, user_id: str, request_id: str
    ) -> AsyncIterator[str]:
        # The earthdata tools read the user from this context. No cube write
        # may start while a turn runs: it would compete with the turn for
        # dask, the HDF5 lock and memory.
        with user_id_context(user_id), cube_cache.active_turn():
            async for frame in self._turn_frames(message, thread_id, user_id, request_id):
                yield frame

    async def _turn_frames(
        self, message: str, thread_id: str, user_id: str, request_id: str
    ) -> AsyncIterator[str]:
        started = time.monotonic()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        put = _putter(queue)
        artifacts = StoredArtifacts(self.chart_service, thread_id, user_id, put)
        context = RunContext(
            identity=Identity(user_id=user_id, conversation_id=thread_id),
            events=StreamEvents(put),
            artifacts=artifacts,
            credentials=self._credentials(user_id),
            storage=self.stores,
            clock=self._clock,
        )
        ports = self._ports(user_id, lambda payload: put(("job_progress", payload)))
        seen = _Seen()
        loop = asyncio.get_running_loop()
        last_activity = loop.time()

        async def run() -> None:
            # Connector last-used writes are coalesced per turn through this.
            streaming.get_call_budget()
            try:
                put(("__turn__", await self.supervisor.turn(context, ports.build_env, ports.earthdata, message)))
            except Exception as exc:  # noqa: BLE001 -- ends the turn with an error frame below
                put(("__error__", exc))

        task = asyncio.create_task(run())
        turn_timeout = asyncio.timeout(self.chat_turn_timeout_seconds)
        turn: Turn | None = None
        try:
            try:
                async with turn_timeout:
                    while turn is None:
                        quiet = streaming.HEARTBEAT_INTERVAL_SECONDS - (loop.time() - last_activity)
                        try:
                            kind, data = await asyncio.wait_for(queue.get(), timeout=max(quiet, 0.0))
                        except TimeoutError:
                            # Readers take long silence for a replica that died, so silence is filled.
                            idle = loop.time() - last_activity
                            last_activity = loop.time()
                            yield sse("status", {
                                "message": f"Still working — {int(idle)}s elapsed", "stage": STAGE_WORKING,
                                "detail": int(idle),
                            })
                            continue
                        last_activity = loop.time()
                        if kind == "__error__":
                            raise data
                        if kind == "__turn__":
                            turn = data
                            continue
                        for frame in await self._live_frames(kind, data, artifacts, seen, thread_id, user_id):
                            yield frame
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.shield(self._record(message, thread_id, user_id, "".join(seen.text), seen, "stopped"))
                raise
            except Exception as exc:
                task.cancel()
                answer, frames = self._failure(exc, turn_timeout, request_id, thread_id)
                partial = "".join(seen.text).strip()
                text = f"{partial}\n\n{answer}" if partial else answer
                await self._record(message, thread_id, user_id, text, seen, "error")
                for frame in frames:
                    yield frame
                return
        finally:
            if not task.done():
                task.cancel()

        sources = await data_sources(self.stores.handles, user_id, turn.results)
        failed = turn.status == "failed"
        await self._record(
            message, thread_id, user_id, turn.text, seen, "error" if failed else "done",
            sources=sources, label=answer_mode(turn.results),
        )
        if failed:
            yield sse("error", {"detail": render_error_answer(CATEGORY_CONTRACT, "request")})
        yield sse("done", self._done_payload(thread_id, turn, seen))
        self._log_request_complete(request_id, thread_id, started)

    async def _live_frames(
        self, kind: str, data: Any, artifacts: StoredArtifacts, seen: _Seen, thread_id: str, user_id: str,
    ) -> list[str]:
        if kind == "status":
            return [sse("status", data)]
        if kind == "text":
            seen.text.append(data["content"])
            return [sse("text", data)]
        if kind == "job_progress":
            return [sse("job_progress", data)]
        if kind == "artifact":
            if data not in seen.artifacts:
                seen.artifacts.append(data)
            return [sse("artifact", data)]
        if kind == "chart":
            return await self._chart_frames(data, artifacts, seen, thread_id, user_id)
        return []

    async def _chart_frames(
        self, payload: dict[str, Any], artifacts: StoredArtifacts, seen: _Seen, thread_id: str, user_id: str,
    ) -> list[str]:
        chart_id = payload.get("chart_id")
        if isinstance(chart_id, str) and chart_id in seen.chart_ids:
            return []
        stored = artifacts.stored_charts.get(chart_id) if isinstance(chart_id, str) else None
        if stored is None:
            stored = await self.chart_service.persist_chart_payload(thread_id, payload, user_id)
        if stored is None:
            return []
        seen.chart_ids.add(str(stored.get("chart_id") or chart_id))
        try:
            seen.charts.append(chart_reference(stored).model_dump(exclude_none=True))
        except ValidationError:
            logger.warning("chart_reference_invalid", extra={"_chart_id": stored.get("chart_id")})
        frames = [sse("chart", stored)]
        try:
            reference = build_artifact_reference(stored)
        except ValidationError:
            logger.warning("chart_artifact_reference_invalid", extra={"_chart_id": stored.get("chart_id")})
            reference = None
        if reference is not None:
            ref = reference.model_dump(exclude_none=True)
            if ref not in seen.artifacts:
                seen.artifacts.append(ref)
            frames.append(sse("artifact", ref))
        return frames

    def _done_payload(self, thread_id: str, turn: Turn, seen: _Seen) -> dict[str, Any]:
        mode = answer_mode(turn.results)
        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "response": turn.text,
            "artifacts": list(seen.artifacts),
            "tool_calls": [],
            "results": results_view(turn.results),
            "label": None if mode is None else mode.label(),
        }
        if turn.followups:
            payload["suggested_followups"] = list(turn.followups)
        if turn.variable_choice is not None:
            payload["variable_choice"] = picker_payload(turn.variable_choice)
        return payload

    async def _record(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        answer: str,
        seen: _Seen,
        status: TranscriptStatus,
        *,
        sources: Iterable[str] = (),
        label: Mode | None = None,
    ) -> None:
        """Write the turn to the transcript; a failed write is logged, since the turn has already answered."""
        try:
            await transcript_repository.append_turn(TranscriptTurn(
                thread_id=thread_id, user_id=user_id, user_text=message, answer_text=answer,
                charts=list(seen.charts), artifacts=[ArtifactReference.model_validate(a) for a in seen.artifacts],
                sources=list(sources), label=label, status=status,
            ))
        except Exception:
            logger.exception("transcript_write_failed", extra={"_thread_id": thread_id})

    def _failure(
        self, exc: BaseException, turn_timeout: asyncio.Timeout, request_id: str, thread_id: str,
    ) -> tuple[str, list[str]]:
        """The answer and frames of a turn that raised.

        A downstream TimeoutError, such as a provider's, is a plain failure;
        ``turn_timeout.expired()`` tells it from the turn's own deadline.
        Called from inside an ``except`` block, for the logged traceback.
        """
        if isinstance(exc, TimeoutError) and turn_timeout.expired():
            answer = render_turn_timeout_answer([])
            logger.warning("chat_turn_timeout", extra={
                "_event": "chat_turn_timeout", "_request_id": request_id, "_thread_id": thread_id,
                "_in_flight_jobs": [],
            })
            return answer, self._ending(answer, thread_id)
        if isinstance(exc, admission.AdmissionOverloaded):
            answer = render_overloaded_answer()
            logger.warning("chat_turn_shed", extra={
                "_event": "chat_turn_shed", "_request_id": request_id, "_thread_id": thread_id,
                "_in_flight": admission.in_flight(), "_queued": admission.queued(),
            })
            return answer, self._ending(answer, thread_id)
        logger.exception("agent_failure", extra={"_request_id": request_id, "_thread_id": thread_id})
        answer = render_error_answer(CATEGORY_CONTRACT, "request")
        return answer, [sse("error", {"detail": answer})]

    def _ending(self, answer: str, thread_id: str) -> list[str]:
        return [
            sse("error", {"detail": answer}),
            sse("done", {"thread_id": thread_id, "response": answer, "artifacts": [], "tool_calls": []}),
        ]

    def _log_request_complete(self, request_id: str, thread_id: str, started: float) -> None:
        elapsed = time.monotonic() - started
        slow = elapsed >= self.long_request_seconds
        log = logger.warning if slow else logger.info
        log("long_running_request" if slow else "request_completed", extra={
            "_request_id": request_id, "_thread_id": thread_id, "_elapsed_seconds": round(elapsed, 3),
        })


# History


async def transcript_history(thread_id: str, user_id: str, chart_service: ChartService) -> list[dict[str, Any]]:
    """The conversation as the user saw it: a user and an assistant message per
    transcript turn. Charts are looked up in one read; one no longer stored is left out."""
    turns = await transcript_repository.list_turns(thread_id, user_id)
    resolved = iter(await chart_service.resolve_charts([c for t in turns for c in t.charts], user_id))
    messages: list[dict[str, Any]] = []
    for turn in turns:
        user, assistant = turn.to_history_messages()
        charts = [next(resolved) for _ in turn.charts]
        assistant["charts"] = [chart for chart in charts if chart is not None]
        assistant["label"] = None if turn.label is None else turn.label.label()
        messages.extend((user, assistant))
    return messages


# The host built at startup


async def build_harness_chat(
    settings: Settings,
    *,
    mcp_manager: Any,
    live_tools: dict[str, BaseTool],
    chart_service: ChartService,
    edl: EdlCredentialInjector,
    aqs: AqsCredentialInjector,
) -> HarnessChat:
    """The app's chat host on its Postgres, its checkpointer and its earthdata connection.

    ``live_tools`` is the dict the connection manager refreshes in place, so
    a turn reads the tools of the connection current at each call.
    """
    from tta_backend.datasets.registry import load_registry
    from tta_backend.harness.capabilities.catalogue import build_registry
    from tta_backend.harness.knowledge import load_notes
    from tta_backend.harness.storage import ensure_tables, postgres_stores
    from tta_backend.harness.workflows import WorkflowRunner
    from tta_backend.harness.workflows.loader import load_packs
    from tta_backend.services.harness_models import AppModels
    from tta_backend.utils.db import get_checkpointer, pg_connection
    from tta_backend.utils.plotting import RegionResolver

    await ensure_tables(pg_connection)
    await transcript_repository.ensure_transcript_table()
    datasets = load_registry()
    capabilities = build_registry()
    catalogue = load_packs(PACKS_ROOT, capabilities, datasets)
    notes = load_notes(PACKS_ROOT, datasets)
    checkpointer = await get_checkpointer()
    models = AppModels(settings)
    supervisor = Supervisor(
        capabilities=capabilities, catalogue=catalogue,
        runner=WorkflowRunner(capabilities, catalogue, checkpointer, notes=notes),
        notes=notes, checkpointer=checkpointer, models=models,
    )
    root = payload_root(settings)
    compute = AdmissionCompute()
    limits = limits_from_settings(settings)
    places = ResolverPlaces(RegionResolver())

    def ports(user_id: str, report_job: JobReport) -> TurnPorts:
        earthdata = ReadyEarthdata(mcp_manager, live_tools)
        return TurnPorts(
            build_env=env_factory(
                root, earthdata=JobReporting(earthdata, report_job),
                open_retrieval=open_retrieval_over(live_tools, user_id),
                compute=compute, limits=limits, places=places, models=models,
            ),
            earthdata=earthdata,
        )

    def credentials(user_id: str) -> Credentials:
        return ConnectorCredentials(user_id, edl, aqs)

    return HarnessChat(
        supervisor=supervisor, stores=postgres_stores(pg_connection), ports=ports, credentials=credentials,
        chart_service=chart_service, chat_turn_timeout_seconds=settings.chat_turn_timeout_seconds,
        long_request_seconds=settings.long_request_seconds,
    )


__all__ = [
    "ConnectorCredentials",
    "HarnessChat",
    "JobReporting",
    "ReadyEarthdata",
    "StoredArtifacts",
    "StreamEvents",
    "TurnPorts",
    "build_harness_chat",
    "job_progress",
    "picker_payload",
    "results_view",
    "sse",
    "transcript_history",
]
