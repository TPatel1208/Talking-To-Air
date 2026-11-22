"""Chat turns through the harness supervisor, as the SSE frames the frontend reads.

One ``HarnessChat`` is built at startup (``build_harness_chat``) and shared by
every request. ``stream_chat_events`` runs one turn: the supervisor runs in a
task of its own while its events reach the stream through a queue, and the
turn ends with the transcript written, then the closing frame.

Frames, in order:

- ``status`` ``{message, stage?, detail?, run_id?}`` for progress, and a
  heartbeat during silence. ``run_id`` is the run the step belongs to, absent
  only on the turn's own frames. A workflow run opens with ``stage``
  ``workflow.<name>`` and ``detail`` ``{workflow, version, mode, steps,
  labels}``, and its steps' frames carry ``detail.step``; an exploration task
  is shown first with ``stage`` ``explore`` and ``detail`` ``{objective,
  bundle, queued: true}``, then opens with the same stage and ``detail``
  ``{objective, bundle}``. Each workflow run, exploration task and the
  turn's lookup run ends with one frame of ``stage`` ``finished`` and
  ``detail`` ``{status}``. Lines from the older helpers (area, coverage,
  opening retrieved data) arrive as ``status`` frames with their own stages,
  carrying the run id when a run's call or retrieval open raised them;
- ``text`` ``{content}`` for each piece of the answer as it streams;
- ``job_progress`` ``{job_handle, status, progress, phase, message, note}``
  for each retrieval a run submits or polls, which the jobs panel lists and
  Stop cancels;
- ``chart`` with the stored chart row, then ``artifact`` with its gallery
  reference when the chart has one; ``artifact`` for each stored table. Each
  chart is stored once and sent once. A chart or table a run drew carries
  that run's ``run_id`` and ``mode`` ``{kind, label, workflow}``, in the
  transcript and the history too;
- ``done`` ``{thread_id, response, artifacts, tool_calls, results, label,
  suggested_followups?, variable_choice?, question?}``. ``question`` is the
  question a paused run waits on, ``{run_id, step, kind, question,
  options|items}``, which the app answers with ``answer: {run_id, step,
  value}`` on its next message; ``variable_choice`` is the picker's view of
  a question with options, carrying the same fields beside its own. Each of ``results`` is
  ``{run_id, status, mode, reason, connector}``, ``connector`` naming the
  connector a result stopped for when it is missing, expired or rejected.
  Work refused because a connector's service is down adds an entry with
  ``run_id`` and ``mode`` null, ``status`` ``failed``, the fixed wording as
  ``reason``, ``connector`` set and ``unavailable`` true; nothing was started
  for it;
- on a failed turn, ``error`` ``{detail}`` before ``done``; on a timeout or
  when admission refuses the turn, ``error`` then a short ``done``; on any
  other exception, or a cancellation raised inside the turn, ``error`` alone.

A turn that is stopped, or whose reader goes away, is cancelled and given
``STOP_WAIT_SECONDS`` to end; what it sent by then is written to the
transcript as ``stopped``. A transcript write already under way is finished
whatever happens to the stream.
"""
from __future__ import annotations

import asyncio
import contextvars
import datetime
import itertools
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from tta_backend.config.error_templates import (
    render_error_answer,
    render_overloaded_answer,
    render_turn_timeout_answer,
)
from tta_backend.config.workflow_stages import STAGE_WORKING
from tta_backend.domain.charts.media import carries_media, wire_media_urls
from tta_backend.domain.earthdata.retrieval import annotate_paused
from tta_backend.earthdata_mcp.connection import STATE_READY
from tta_backend.earthdata_mcp.results import CATEGORY_CONTRACT, CATEGORY_PROVIDER_UNAVAILABLE, MCPToolError
from tta_backend.harness.capabilities.registry import CapabilityRegistry, UnknownCapability
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.handles.resolver import OpenRetrieval
from tta_backend.harness.jobs import STATUS_TOOL, SUBMIT_TOOLS
from tta_backend.harness.models import Models
from tta_backend.harness.results import RUN_ID_PATTERN, Mode, Result
from tta_backend.harness.run_context import (
    ArtifactRef,
    Clock,
    Compute,
    Credentials,
    EarthdataSession,
    Identity,
    Limits,
    Places,
    Reachable,
    RunContext,
    RunEnv,
    all_reachable,
    run_env,
)
from tta_backend.harness.storage import Stores
from tta_backend.harness.supervisor import Supervisor, Turn
from tta_backend.harness.supervisor.answer import data_sources, unavailable_wording
from tta_backend.harness.supervisor.supervisor import AppAnswer, ResumeRequest
from tta_backend.harness.workflows.answers import answerable
from tta_backend.harness.trace import FailureCategory, RunNotFound, TraceStore
from tta_backend.harness.workflows.loader import WorkflowCatalogue
from tta_backend.harness.workflows.runner import EnvFactory, connect_reason
from tta_backend.infrastructure.settings import Settings
from tta_backend.models.artifact import ArtifactReference
from tta_backend.models.transcript import ConnectorName, TranscriptResult, TranscriptTurn
from tta_backend.repositories import transcript_repository
from tta_backend.services import admission, cube_cache, frame_store, overlay_store
from tta_backend.services.artifact_registry import build_artifact_reference
from tta_backend.services.artifact_store import artifact_store
from tta_backend.services.chart_service import ChartLink, ChartService, chart_reference
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
from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION
from tta_backend.utils import streaming
from tta_backend.utils.streaming import user_id_context

logger = logging.getLogger(__name__)

PACKS_ROOT = Path(__file__).resolve().parents[2] / "packs"
# The data layer's wording when the earthdata connection is not up.
EARTHDATA_UNAVAILABLE = "The satellite data layer is temporarily unavailable."

TranscriptStatus = Literal["done", "stopped", "error"]
# How long a stopped turn is given to end before its transcript row is written without it.
STOP_WAIT_SECONDS = 5.0


class _TurnRaised(Exception):
    """Carries a turn's ending that is not an Exception, such as a cancellation from inside it, to the error path."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(type(error).__name__)
        self.error = error


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
    """Puts on ``queue`` from the loop's thread or any other, in the order they were made.

    A put from another thread reaches the queue through the loop. While one
    is still on its way, a put from the loop's thread is scheduled behind it
    rather than put at once, so nothing put earlier, the turn's end
    included, lands after a later put.
    """
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    lock = threading.Lock()
    in_flight = 0

    def deliver(item: tuple[str, Any]) -> None:
        nonlocal in_flight
        with lock:
            in_flight -= 1
        queue.put_nowait(item)

    def put(item: tuple[str, Any]) -> None:
        nonlocal in_flight
        with lock:
            if threading.get_ident() != loop_thread:
                in_flight += 1
                loop.call_soon_threadsafe(deliver, item)
                return
            if in_flight:
                in_flight += 1
                loop.call_soon(deliver, item)
                return
        queue.put_nowait(item)

    return put


class StreamEvents:
    """Progress, text and charts onto the turn's queue. A question needs
    nothing here: the turn carries it to the ``done`` frame."""

    def __init__(self, put: Put) -> None:
        self._put = put

    def progress(
        self, message: str, *, stage: str | None = None, detail: Any = None, run_id: str | None = None
    ) -> None:
        data: dict[str, Any] = {"message": message}
        if stage is not None:
            data["stage"] = stage
        if detail is not None:
            data["detail"] = detail
            if run_id is None and isinstance(detail, dict) and isinstance(detail.get("run_id"), str):
                run_id = detail["run_id"]
        if run_id is not None:
            data["run_id"] = run_id
        self._put(("status", data))

    def chart(self, payload: dict[str, Any]) -> None:
        self._put(("chart", payload))

    def question(self, payload: dict[str, Any]) -> None:
        return None

    def text(self, content: str) -> None:
        if content:
            self._put(("text", {"content": content}))


# The run whose earthdata call or retrieval open is in progress, for the status
# lines of helpers that know nothing of runs.
_working_run: contextvars.ContextVar[str | None] = contextvars.ContextVar("harness_working_run", default=None)


def status_emitter(events: StreamEvents) -> Callable[..., None]:
    """The emitter ``utils.streaming.emit_status`` calls reach the stream through."""

    def emit(message: str, *, stage: str | None = None, detail: Any = None) -> None:
        events.progress(message, stage=stage, detail=detail, run_id=_working_run.get())

    return emit


class RunBoundEarthdata:
    """An earthdata session whose calls, and the status lines they raise, belong to one run."""

    def __init__(self, inner: EarthdataSession, run_id: str) -> None:
        self._inner = inner
        self._run_id = run_id

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        token = _working_run.set(self._run_id)
        try:
            return await self._inner.call(tool, arguments, idempotency_key=idempotency_key)
        finally:
            _working_run.reset(token)


def run_bound_open(open_retrieval: OpenRetrieval, run_id: str) -> OpenRetrieval:
    """``open_retrieval`` with its status lines belonging to ``run_id``."""

    async def open_bound(owner: str, record: Any) -> Any:
        token = _working_run.set(run_id)
        try:
            return await open_retrieval(owner, record)
        finally:
            _working_run.reset(token)

    return open_bound


def _source_handles(payload: dict[str, Any]) -> tuple[str, ...]:
    metadata = payload.get("metadata")
    handles = metadata.get("source_handles") if isinstance(metadata, dict) else None
    if not isinstance(handles, list):
        return ()
    return tuple(dict.fromkeys(h for h in handles if isinstance(h, str)))


class StoredArtifacts:
    """Charts into the chart repository and tables into the artifact store, owned by the turn's user and thread.

    A stored chart's row is kept for the turn, so sending it needs no second
    read. A stored table is sent as an ``artifact`` at once. A chart or table
    whose metadata names the run that drew it (``metadata.run_id``) is linked
    to that run when the run is the user's own; a chart's link is stored with it.

    A map's frame stack goes into the frame store and its overlay into the
    overlay store. A chart carrying either is given its id before it is
    stored, so the stored row holds the urls the ``/chart/{chart_id}/...``
    routes serve them at; those routes serve only the chart's owner.
    """

    def __init__(
        self, chart_service: ChartService, thread_id: str, user_id: str, put: Put, traces: TraceStore,
    ) -> None:
        self._charts = chart_service
        self._thread_id = thread_id
        self._user_id = user_id
        self._put = put
        self._traces = traces
        self._modes: dict[str, Mode | None] = {}
        self.stored_charts: dict[str, dict[str, Any]] = {}

    async def _mode_of(self, run_id: str) -> Mode | None:
        """The mode of the user's run ``run_id``; None for a run that is not theirs."""
        if run_id not in self._modes:
            try:
                self._modes[run_id] = (await self._traces.get_run(self._user_id, run_id)).mode
            except RunNotFound:
                self._modes[run_id] = None
        return self._modes[run_id]

    async def _run_of(self, metadata: Any) -> tuple[str, Mode] | None:
        run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            return None
        mode = await self._mode_of(run_id)
        return None if mode is None else (run_id, mode)

    async def store_chart(self, payload: dict[str, Any]) -> ArtifactRef:
        run = await self._run_of(payload.get("metadata"))
        handles = _source_handles(payload)
        link = None if run is None else ChartLink(
            run_id=run[0], mode=run[1], handle=handles[0] if handles else None, handles=handles,
        )
        if carries_media(payload) and not payload.get("chart_id"):
            payload = {**payload, "chart_id": str(uuid.uuid4())}
            # Into the payload's own frames and overlay blocks, which the caller's copy shares.
            wire_media_urls(payload)
        stored = await self._charts.persist_chart_payload(self._thread_id, payload, self._user_id, link=link)
        if stored is None or not stored.get("chart_id"):
            raise RuntimeError("the chart was not stored")
        chart_id = str(stored["chart_id"])
        self.stored_charts[chart_id] = stored
        return ArtifactRef(id=chart_id, type="chart", title=payload.get("title"))

    def store_frames(self, stack: Any) -> dict[str, Any]:
        return frame_store.store_frame_stack(stack, pipeline_version=OPEN_PIPELINE_VERSION)

    def store_overlay(
        self, lats: Any, lons: Any, field: Any, lut: list[Any], vmin: float, vmax: float,
    ) -> dict[str, Any]:
        return overlay_store.render_and_store(lats, lons, field, lut, vmin, vmax)

    async def store_table(
        self,
        title: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        minted = artifact_store.put_table(title, columns, rows, metadata)
        claimed = await artifact_store.claim(minted.id, self._user_id, self._thread_id)
        frame = claimed.model_dump(exclude_none=True)
        run = await self._run_of(metadata)
        if run is not None:
            frame.update(run_id=run[0], mode=run[1].view())
        self._put(("artifact", frame))
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


def earthdata_ready(manager: Any) -> bool:
    """Whether the earthdata connection is up."""
    return manager is not None and manager.state == STATE_READY


def connector_reachable(manager: Any) -> Reachable:
    """Which connectors' services are up, read from the connection state at each call.
    Only the earthdata service can be down; EPA AQS is always reachable."""

    def reachable(connector: str) -> bool:
        return connector != CONNECTOR_TYPE_EARTHDATA or earthdata_ready(manager)

    return reachable


class ReadyEarthdata:
    """The earthdata session over the app's live MCP tools; while the
    connection is not ready, every call fails as provider_unavailable."""

    def __init__(self, manager: Any, tools: dict[str, BaseTool]) -> None:
        self._manager = manager
        self._session = McpEarthdataSession(tools)

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        if not earthdata_ready(self._manager):
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
            context, run_id=run_id, payloads=PayloadStore(payload_root),
            open_retrieval=run_bound_open(open_retrieval, run_id),
            earthdata=RunBoundEarthdata(earthdata, run_id), compute=compute, limits=limits, places=places, models=models,
        )

    return build


def payload_root(settings: Settings) -> str:
    """Derived handles' payload files, beside the cube store, which sweeps any directory of its own it cannot read."""
    cubes = Path(settings.cube_store_dir)
    return str(cubes.parent / "harness_payloads")


# The frames a turn ends with


# The connectors the frontend can open its connect flow for.
CONNECT_LINKED: frozenset[str] = frozenset({CONNECTOR_TYPE_EARTHDATA, CONNECTOR_TYPE_EPA_AQS})


class StoppedConnectors:
    """Which connector a result stopped for, if any.

    Read from the run's trace: a failed entry in the credential category on a
    capability that declares a connector, which covers one not connected,
    expired or rejected. A workflow run that did not start because a
    connector is not connected has no such entry; its fixed reason names the
    connectors its spec needs, so that reason is matched instead. Only
    connectors the frontend can open the connect flow for are named.
    """

    def __init__(
        self, capabilities: CapabilityRegistry | None = None, catalogue: WorkflowCatalogue | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._catalogue = catalogue

    async def of(self, traces: TraceStore, user_id: str, result: Result) -> ConnectorName | None:
        if result.status == "done" or self._capabilities is None:
            return None
        try:
            entries = await traces.entries(user_id, result.trace_id)
        except RunNotFound:
            return None
        for entry in entries:
            if entry.status != "failed" or entry.failure_category != FailureCategory.CREDENTIAL_INVALID:
                continue
            try:
                connector = self._capabilities.get(entry.capability or "").connector
            except UnknownCapability:
                continue
            if connector in CONNECT_LINKED:
                return cast(ConnectorName, connector)
        return self._not_started_for(result)

    def _not_started_for(self, result: Result) -> ConnectorName | None:
        workflow = result.mode.workflow
        if workflow is None or self._catalogue is None or self._capabilities is None or result.reason is None:
            return None
        try:
            needed = sorted(self._catalogue.get(workflow.name, workflow.version).connectors(self._capabilities))
        except Exception:
            # A spec no longer in the catalogue, or naming a capability no longer registered, names no connector.
            return None
        for size in range(1, len(needed) + 1):
            for missing in itertools.combinations(needed, size):
                if result.reason == connect_reason(list(missing)):
                    linked = [name for name in missing if name in CONNECT_LINKED]
                    return cast(ConnectorName, linked[0]) if linked else None
        return None


async def result_summaries(
    results: Sequence[Result], traces: TraceStore, user_id: str, connectors: StoppedConnectors,
) -> list[TranscriptResult]:
    """Each result as the ``done`` frame and the transcript carry it."""
    return [
        TranscriptResult(
            run_id=result.trace_id, status=result.status, mode=result.mode,
            reason=None if result.status == "done" else result.reason,
            connector=await connectors.of(traces, user_id, result),
        )
        for result in results
    ]


def unavailable_summaries(connectors: Sequence[str]) -> list[TranscriptResult]:
    """An entry for each connector whose service was down, shaped like a result
    that stopped for it; it names no run, since nothing was started."""
    return [
        TranscriptResult(
            status="failed", reason=unavailable_wording(connector), connector=cast(ConnectorName, connector),
            unavailable=True,
        )
        for connector in connectors if connector in CONNECT_LINKED
    ]


def results_view(summaries: Sequence[TranscriptResult]) -> list[dict[str, Any]]:
    return [summary.view() for summary in summaries]


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
    an option sends its text as the user's reply. The question's own fields
    (``step``, ``kind``, ``question``, ``options``) go beside, for a control
    that answers the run directly."""
    raw = list(variable_choice.get("options") or [])
    options = [_option_text(option) for option in raw]
    prompt = variable_choice.get("prompt")
    payload: dict[str, Any] = {
        "message": prompt if isinstance(prompt, str) else "",
        "run_id": variable_choice.get("run_id"),
        "candidates": [
            {"name": text, "category": "distinct", "units": None, "valid_fraction": None, "reasons": [],
             "prompt": text}
            for text in options
        ],
    }
    for name in ("step", "kind", "question"):
        if name in variable_choice:
            payload[name] = variable_choice[name]
    if "step" in variable_choice:
        payload["options"] = raw
    return payload


# The fields of a question a paused run waits on that the app is sent.
QUESTION_FIELDS = ("run_id", "step", "kind", "question", "options", "items")


def question_payload(question: dict[str, Any] | None) -> dict[str, Any] | None:
    """The question a paused run waits on, as the ``done`` frame carries it; None for any other."""
    if question is None or not answerable(question):
        return None
    return {name: question[name] for name in QUESTION_FIELDS if name in question}


def app_answer(answer: Any) -> AppAnswer | None:
    """An answer the chat request carried, as the supervisor takes it."""
    if answer is None or isinstance(answer, AppAnswer):
        return answer
    data = answer if isinstance(answer, dict) else answer.model_dump()
    return AppAnswer(run_id=str(data["run_id"]), step=str(data["step"]), value=data.get("value"))


def app_resume(resume: Any) -> ResumeRequest | None:
    """A continue request the chat request carried, as the supervisor takes it."""
    if resume is None or isinstance(resume, ResumeRequest):
        return resume
    data = resume if isinstance(resume, dict) else resume.model_dump()
    return ResumeRequest(run_id=str(data["run_id"]))


# The chat service


class TurnSupervisor(Protocol):
    async def turn(
        self, context: RunContext, build_env: EnvFactory, earthdata: EarthdataSession, message: str,
        answer: AppAnswer | None = None,
    ) -> Turn: ...

    # A supervisor that continues runs also takes ``resume: ResumeRequest | None`` on ``turn``.


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
        capabilities: CapabilityRegistry | None = None,
        catalogue: WorkflowCatalogue | None = None,
        payload_root: str | None = None,
        reachable: Reachable = all_reachable,
    ) -> None:
        self.supervisor = supervisor
        self._reachable = reachable
        self.stores = stores
        self._ports = ports
        self._credentials = credentials
        self.chart_service = chart_service
        self.chat_turn_timeout_seconds = chat_turn_timeout_seconds
        self.long_request_seconds = long_request_seconds
        self._clock = clock or UtcClock()
        self.capabilities = capabilities
        self.catalogue = catalogue
        # Where derived handles' payloads are stored, which export reads.
        self.payload_root = payload_root
        self._connectors = StoppedConnectors(capabilities, catalogue)

    async def stream_chat_events(
        self, message: str, thread_id: str, user_id: str, request_id: str, answer: Any = None, resume: Any = None,
    ) -> AsyncIterator[str]:
        """One turn's frames. ``answer`` is the request's ``{run_id, step, value}`` from the app's
        question control; ``message`` then holds the picked option's text, which the transcript keeps.
        ``resume`` is its ``{run_id}`` to continue a waiting run, with ``message`` the fixed text sent."""
        # The earthdata tools read the user from this context. No cube write
        # may start while a turn runs: it would compete with the turn for
        # dask, the HDF5 lock and memory.
        # Closed with this stream, so a reader going away reaches the turn's own handling at once.
        frames = aclosing(
            self._turn_frames(message, thread_id, user_id, request_id, app_answer(answer), app_resume(resume)),
        )
        with user_id_context(user_id), cube_cache.active_turn():
            async with frames as turn_frames:
                async for frame in turn_frames:
                    yield frame

    def quiet_context(self, thread_id: str, user_id: str) -> tuple[RunContext, TurnPorts]:
        """A run context and ports for work outside a turn, such as listing or cancelling paused
        runs: nothing it reports is streamed. The caller binds the user, as a turn does."""

        def put(item: tuple[str, Any]) -> None:
            return None

        context = RunContext(
            identity=Identity(user_id=user_id, conversation_id=thread_id),
            events=StreamEvents(put),
            artifacts=StoredArtifacts(self.chart_service, thread_id, user_id, put, self.stores.traces),
            credentials=self._credentials(user_id),
            storage=self.stores,
            clock=self._clock,
            reachable=self._reachable,
        )
        return context, self._ports(user_id, lambda payload: None)

    async def _turn_frames(
        self, message: str, thread_id: str, user_id: str, request_id: str, answered: AppAnswer | None = None,
        resumed: ResumeRequest | None = None,
    ) -> AsyncGenerator[str, None]:
        started = time.monotonic()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        put = _putter(queue)
        artifacts = StoredArtifacts(self.chart_service, thread_id, user_id, put, self.stores.traces)
        context = RunContext(
            identity=Identity(user_id=user_id, conversation_id=thread_id),
            events=StreamEvents(put),
            artifacts=artifacts,
            credentials=self._credentials(user_id),
            storage=self.stores,
            clock=self._clock,
            reachable=self._reachable,
        )
        ports = self._ports(user_id, lambda payload: put(("job_progress", payload)))
        seen = _Seen()
        loop = asyncio.get_running_loop()
        last_activity = loop.time()

        async def run() -> None:
            # Connector last-used writes are coalesced per turn through this.
            streaming.get_call_budget()
            # Status lines from the helpers the tools call reach the stream through here.
            streaming.bind_status_emitter(status_emitter(cast(StreamEvents, context.events)))
            # Whatever ends the turn, one terminal item reaches the queue, so the frame loop never waits on a turn
            # that is over.
            outcome: tuple[str, Any] = ("__error__", RuntimeError("the turn ended without an outcome"))
            # Passed only when given, so a supervisor that takes no answers still serves plain messages.
            extra: dict[str, Any] = {} if answered is None else {"answer": answered}
            if resumed is not None:
                extra["resume"] = resumed
            try:
                outcome = ("__turn__", await self.supervisor.turn(
                    context, ports.build_env, ports.earthdata, message, **extra,
                ))
            except BaseException as exc:
                outcome = ("__error__", exc)
                if not isinstance(exc, Exception | asyncio.CancelledError):
                    raise
            finally:
                put(outcome)

        task = asyncio.create_task(run())
        turn_timeout = asyncio.timeout(self.chat_turn_timeout_seconds)
        turn: Turn | None = None
        # Set once a transcript row for the turn is written or being written, so a Stop never writes a second.
        recorded = False
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
                            raise data if isinstance(data, Exception) else _TurnRaised(data)
                        if kind == "__turn__":
                            turn = data
                            continue
                        for frame in await self._live_frames(kind, data, artifacts, seen, thread_id, user_id):
                            yield frame
            except Exception as exc:
                task.cancel()
                failure = exc.error if isinstance(exc, _TurnRaised) else exc
                answer, frames = self._failure(failure, turn_timeout, request_id, thread_id)
                partial = "".join(seen.text).strip()
                text = f"{partial}\n\n{answer}" if partial else answer
                recorded = True
                await asyncio.shield(self._record(message, thread_id, user_id, text, seen, "error"))
                for frame in frames:
                    yield frame
                return

            recorded = True
            summaries = await asyncio.shield(self._close_turn(message, thread_id, user_id, turn, seen))
            if turn.status == "failed":
                yield sse("error", {"detail": render_error_answer(CATEGORY_CONTRACT, "request")})
            yield sse("done", self._done_payload(thread_id, turn, seen, summaries))
            self._log_request_complete(request_id, thread_id, started)
        except (asyncio.CancelledError, GeneratorExit):
            # Stopped, or the reader went away, mid-turn: the turn is cancelled and what it sent is kept as stopped.
            if not recorded:
                recorded = True
                await asyncio.shield(
                    self._stopped(task, queue, artifacts, seen, message, thread_id, user_id),
                )
            raise
        finally:
            if not task.done():
                task.cancel()

    async def _close_turn(
        self, message: str, thread_id: str, user_id: str, turn: Turn, seen: _Seen,
    ) -> list[TranscriptResult]:
        """The turn's result summaries, once its transcript row is written."""
        sources = await data_sources(self.stores.handles, user_id, turn.results)
        summaries = await result_summaries(turn.results, self.stores.traces, user_id, self._connectors)
        summaries.extend(unavailable_summaries(turn.unavailable))
        await self._record(
            message, thread_id, user_id, turn.text, seen, "error" if turn.status == "failed" else "done",
            sources=sources, label=answer_mode(turn.results), results=summaries, followups=turn.followups,
        )
        return summaries

    async def _stopped(
        self,
        task: asyncio.Task[None],
        queue: asyncio.Queue[tuple[str, Any]],
        artifacts: StoredArtifacts,
        seen: _Seen,
        message: str,
        thread_id: str,
        user_id: str,
    ) -> None:
        """Cancel the turn, wait a bounded time for it to end, keep what it sent meanwhile, and write it as stopped."""
        task.cancel()
        await asyncio.wait({task}, timeout=STOP_WAIT_SECONDS)
        while not queue.empty():
            kind, data = queue.get_nowait()
            if kind in ("text", "artifact", "chart"):
                try:
                    await self._live_frames(kind, data, artifacts, seen, thread_id, user_id)
                except Exception:
                    logger.warning("stopped_turn_item_lost", extra={"_thread_id": thread_id, "_kind": kind})
        await self._record(message, thread_id, user_id, "".join(seen.text), seen, "stopped")

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
        link = ChartLink.of(stored)
        drawn = {} if link is None else link.view()
        frames = [sse("chart", {**stored, **drawn})]
        try:
            reference = build_artifact_reference(stored)
        except ValidationError:
            logger.warning("chart_artifact_reference_invalid", extra={"_chart_id": stored.get("chart_id")})
            reference = None
        if reference is not None:
            ref = {**reference.model_dump(exclude_none=True), **drawn}
            if ref not in seen.artifacts:
                seen.artifacts.append(ref)
            frames.append(sse("artifact", ref))
        return frames

    def _done_payload(
        self, thread_id: str, turn: Turn, seen: _Seen, summaries: Sequence[TranscriptResult],
    ) -> dict[str, Any]:
        mode = answer_mode(turn.results)
        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "response": turn.text,
            "artifacts": list(seen.artifacts),
            "tool_calls": [],
            "results": results_view(summaries),
            "label": None if mode is None else mode.label(),
        }
        if turn.followups:
            payload["suggested_followups"] = list(turn.followups)
        if turn.variable_choice is not None:
            payload["variable_choice"] = picker_payload(turn.variable_choice)
        question = question_payload(turn.question)
        if question is not None:
            payload["question"] = question
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
        results: Sequence[TranscriptResult] = (),
        followups: Iterable[str] = (),
    ) -> None:
        """Write the turn to the transcript; a failed write is logged, since the turn has already answered."""
        try:
            await transcript_repository.append_turn(TranscriptTurn(
                thread_id=thread_id, user_id=user_id, user_text=message, answer_text=answer,
                charts=list(seen.charts), artifacts=[ArtifactReference.model_validate(a) for a in seen.artifacts],
                sources=list(sources), label=label, results=list(results), suggested_followups=list(followups),
                status=status,
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
    transcript turn, the assistant's with its label, results and suggested
    follow-ups. Charts are looked up in one read; one no longer stored is left
    out. A chart or artifact a run drew carries that run's ``run_id`` and
    ``mode``, as its live frame did."""
    turns = await transcript_repository.list_turns(thread_id, user_id)
    resolved = iter(await chart_service.resolve_charts([c for t in turns for c in t.charts], user_id))
    messages: list[dict[str, Any]] = []
    for turn in turns:
        user, assistant = turn.to_history_messages()
        charts = [next(resolved) for _ in turn.charts]
        assistant["charts"] = [_drawn(chart) for chart in charts if chart is not None]
        assistant["label"] = None if turn.label is None else turn.label.label()
        assistant["results"] = [result.view() for result in turn.results]
        assistant["suggested_followups"] = list(turn.suggested_followups)
        messages.extend((user, assistant))
    return messages


def _drawn(chart: dict[str, Any]) -> dict[str, Any]:
    link = ChartLink.of(chart)
    return chart if link is None else {**chart, **link.view()}


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
        long_request_seconds=settings.long_request_seconds, capabilities=capabilities, catalogue=catalogue,
        payload_root=root, reachable=connector_reachable(mcp_manager),
    )


__all__ = [
    "ConnectorCredentials",
    "HarnessChat",
    "JobReporting",
    "ReadyEarthdata",
    "RunBoundEarthdata",
    "StoppedConnectors",
    "StoredArtifacts",
    "StreamEvents",
    "TurnPorts",
    "app_resume",
    "build_harness_chat",
    "connector_reachable",
    "earthdata_ready",
    "job_progress",
    "picker_payload",
    "question_payload",
    "result_summaries",
    "results_view",
    "run_bound_open",
    "sse",
    "status_emitter",
    "transcript_history",
    "unavailable_summaries",
]
