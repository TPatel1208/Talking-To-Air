from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tta_backend.config.error_templates import (
    render_error_answer,
    render_overloaded_answer,
    render_turn_timeout_answer,
)
from tta_backend.config.settings import get_settings
from tta_backend.config.workflow_stages import STAGE_WORKING
from tta_backend.earthdata_mcp.results import CATEGORY_CONTRACT
from tta_backend.models import (
    AgentResult,
    DispatchItemResult,
    DispatchResult,
    parse_agent_result,
    parse_chart_payload,
    parse_dispatch_result,
    parse_tool_results,
)
from tta_backend.services import admission, cube_cache
from tta_backend.services.artifact_store import artifact_store
from tta_backend.services.chart_service import ChartService
from tta_backend.services.intent_router import route_intent
from tta_backend.services.retrieval_composites import TERMINAL_STATUSES
from tta_backend.services.subagent_dispatch import run_ground, run_satellite
from tta_backend.utils import streaming
from tta_backend.utils.message_utils import flatten_text_content
from tta_backend.services.consulted_header import (
    consulted_sources,
    merge_sources,
    result_sources,
    sources_header,
    with_sources_header,
)
from tta_backend.utils.streaming import stream_response, user_id_context, user_message_context

logger = logging.getLogger(__name__)

# The router intents the fast path answers by itself, and the agent each one runs.
_FAST_PATH_AGENTS: dict[str, Literal["ground", "satellite"]] = {
    "GROUND": "ground",
    "SATELLITE": "satellite",
}

# Intent recorded for fast-pathed turns in history. The regex router picks an
# agent but doesn't classify intent, so we don't invent one. Never validated.
_FAST_PATH_INTENT = "fast_path"


# The envelope's own cap on follow-ups (AgentResult.suggested_followups).
_MAX_FOLLOWUPS = 2


def _merged_followups(results: list[AgentResult]) -> list[str] | None:
    """Follow-ups across a tool message's results, in request order, deduped
    and capped. None when no result offered any (an empty list would claim
    the sub-agents explicitly offered none)."""
    offered = [r.suggested_followups for r in results if r.suggested_followups is not None]
    if not offered:
        return None
    merged: list[str] = []
    for suggestion in (s for group in offered for s in group):
        if suggestion not in merged:
            merged.append(suggestion)
    return merged[:_MAX_FOLLOWUPS]


@dataclass
class _LiveTurn:
    """What a turn accumulates from its live event stream (shared by both routes)."""

    #: Every tool the turn invoked, replayed in the terminal ``done`` payload.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: Last-seen status per job_handle, so a timeout answer can name jobs
    #: still running server-side.
    job_statuses: dict[str, str] = field(default_factory=dict)
    #: A chart can arrive twice (live, then in the final envelope) and the
    #: frontend doesn't dedup, so this is the one place that does.
    emitted_chart_ids: set[str] = field(default_factory=set)
    #: Sources the turn's dispatch calls drew on, for the "Data sources" header.
    consulted: list[str] = field(default_factory=list)
    #: Set by each dispatch result: the next answer text opens with the header.
    header_due: bool = False


class ChatStreamService:
    def __init__(
        self,
        chart_service: ChartService,
        long_request_seconds: float,
        mcp_manager: Any = None,
        chat_turn_timeout_seconds: float | None = None,
    ):
        self.chart_service = chart_service
        self.long_request_seconds = long_request_seconds
        # Lets the fast path answer "unavailable" instead of dispatching when
        # the earthdata MCP isn't ready.
        self.mcp_manager = mcp_manager
        # Whole-turn deadline; defaults to the configured setting.
        self.chat_turn_timeout_seconds = (
            chat_turn_timeout_seconds if chat_turn_timeout_seconds is not None
            else get_settings().chat_turn_timeout_seconds
        )

    async def _live_event_sse(
        self,
        event_type: str,
        data: Any,
        thread_id: str,
        user_id: str,
        turn: _LiveTurn,
    ) -> list[str] | None:
        """SSE for the live events both routes forward identically.

        Returns ``None`` when ``event_type`` isn't one of them, so the caller
        handles it. An empty list means "handled, nothing to send" (e.g. a
        chart already emitted this turn).

        Returns a list rather than yielding so it can signal "not mine".
        """
        if event_type == "tool_call":
            turn.tool_calls.append({"name": data["name"], "args": data["args"]})
            return [self.sse("tool_call", {"name": data["name"], "args": data["args"]})]
        if event_type == "status":
            # Forward the whole payload: stage/detail ride alongside message.
            return [self.sse("status", data)]
        if event_type == "job_progress":
            if data.get("job_handle"):
                turn.job_statuses[data["job_handle"]] = data.get("status", "")
            return [self.sse("job_progress", data)]
        if event_type == "chart_payload":
            chart = parse_chart_payload(data)
            if chart is None:
                return []
            event = await self._emit_chart_once(thread_id, chart, user_id, turn.emitted_chart_ids)
            return [] if event is None else [event]
        return None

    def _failure_events(
        self,
        exc: BaseException,
        turn_timeout: Any,
        request_id: str,
        thread_id: str,
        job_statuses: dict[str, str],
    ) -> list[str]:
        """How either route ends a turn that raised.

        A downstream TimeoutError (e.g. a provider timeout) is a generic
        failure, not our turn timeout; ``turn_timeout.expired()`` tells the
        two apart.

        Call only from inside an ``except`` block — ``logger.exception`` reads
        the active exception for its traceback.
        """
        if isinstance(exc, TimeoutError) and turn_timeout.expired():
            return list(self._turn_timeout_events(request_id, thread_id, job_statuses))
        if isinstance(exc, admission.AdmissionOverloaded):
            return list(self._overloaded_events(request_id, thread_id))
        logger.exception("agent_failure", extra={"_request_id": request_id, "_thread_id": thread_id})
        return [self.sse("error", {"detail": render_error_answer(CATEGORY_CONTRACT, "request")})]

    async def stream_chat_events(
        self,
        agent: Any,
        ground_agent: Any,
        satellite_agent: Any,
        message: str,
        thread_id: str,
        user_id: str,
        request_id: str,
    ) -> AsyncIterator[str]:
        # Bind context here, before the fast path spawns its task, so the task
        # inherits the real user id and the researcher's original message
        # (the fast path skips stream_response's own binding).
        #
        # Invariant: no cube write starts during a turn (active_turn). With
        # one worker, a background write would compete with this turn for
        # dask, the HDF5 lock and memory.
        with user_id_context(user_id), user_message_context(message), cube_cache.active_turn():
            intent = route_intent(message)
            route = "fast_path" if intent in _FAST_PATH_AGENTS else "supervisor"
            logger.info(
                "chat_route_decision",
                extra={"_request_id": request_id, "_thread_id": thread_id, "_intent": intent, "_route": route},
            )
            if route == "fast_path":
                sub_agent = ground_agent if intent == "GROUND" else satellite_agent
                async for event in self._fast_path_events(
                    intent, sub_agent, agent, message, thread_id, user_id, request_id,
                ):
                    yield event
                return

            response_text = ""
            artifacts: list[dict[str, Any]] = []
            turn = _LiveTurn()
            # Latest sub-agent follow-ups and variable picker, passed through
            # untouched. Dicts so the helpers below can update them in place.
            suggestions_box: dict[str, list[str]] = {}
            variable_choice_box: dict[str, dict[str, Any]] = {}
            started = time.monotonic()
            turn_timeout = asyncio.timeout(self.chat_turn_timeout_seconds)
            try:
                async with turn_timeout:
                    async for event_type, data in stream_response(agent, message, thread_id, user_id=user_id):
                        live = await self._live_event_sse(event_type, data, thread_id, user_id, turn)
                        if live is not None:
                            if event_type == "tool_call":
                                # A new tool call starts a fresh answer: only
                                # the text after the last one belongs in `done`.
                                response_text = ""
                            for event in live:
                                yield event
                        elif event_type == "tool_result":
                            # Parsed once here; the parsers below pass a
                            # DispatchResult instance straight through.
                            content = data.get("content", "")
                            batch = parse_dispatch_result(content)
                            ran = consulted_sources(batch)
                            turn.consulted = merge_sources(turn.consulted, ran)
                            turn.header_due = turn.header_due or bool(ran)
                            async for event in self._tool_result_events(
                                batch if batch is not None else content,
                                thread_id,
                                user_id,
                                artifacts,
                                turn.emitted_chart_ids,
                                suggestions_box,
                                variable_choice_box,
                            ):
                                yield event
                        elif event_type == "text":
                            text, events = await self._text_events(
                                data, thread_id, user_id, turn.emitted_chart_ids, suggestions_box,
                                variable_choice_box,
                            )
                            if text and turn.header_due:
                                # The sources header streams ahead of the answer.
                                opening = f"{sources_header(turn.consulted)}\n\n"
                                response_text += opening
                                turn.header_due = False
                                yield self.sse("text", {"content": opening})
                            response_text += text
                            if text:
                                yield self.sse("text", {"content": text})
                            for event in events:
                                yield event

                    done_payload: dict[str, Any] = {
                        "thread_id": thread_id,
                        "response": with_sources_header(response_text, turn.consulted),
                        "artifacts": artifacts,
                        "tool_calls": turn.tool_calls,
                    }
                    if "value" in suggestions_box:
                        done_payload["suggested_followups"] = suggestions_box["value"]
                    if "value" in variable_choice_box:
                        done_payload["variable_choice"] = variable_choice_box["value"]
                    yield self.sse("done", done_payload)
                    self._log_request_complete(request_id, thread_id, started)
            except Exception as exc:
                for event in self._failure_events(
                    exc, turn_timeout, request_id, thread_id, turn.job_statuses,
                ):
                    yield event

    async def _fast_path_events(
        self,
        intent: str,
        sub_agent: Any,
        supervisor_agent: Any,
        message: str,
        thread_id: str,
        user_id: str,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Dispatch directly to the ground or satellite sub-agent, bypassing
        the supervisor's two model calls. Forwards tool_call/status/
        job_progress/chart events live as the sub-agent works, then emits the
        finalized answer under the same "Data sources: ..." header the
        supervisor path produces, and writes the turn back into the
        supervisor's checkpointed thread so follow-ups keep their
        antecedents."""
        artifacts: list[dict[str, Any]] = []
        turn = _LiveTurn()
        started = time.monotonic()

        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # Touched by every item the loop below consumes, including the
        # sub-agent's own bubbled events, so the beat covers genuine silence
        # only and never talks over a turn that is already narrating.
        last_activity = {"t": loop.time()}

        async def heartbeat() -> None:
            # The fast path bypasses stream_response's watchdog, and a GROUND
            # turn emits nothing until it answers. Readers treat long silence
            # as "the replica owning this turn died", so keep beating.
            while True:
                await asyncio.sleep(streaming.HEARTBEAT_CHECK_SECONDS)
                idle = loop.time() - last_activity["t"]
                if idle >= streaming.HEARTBEAT_INTERVAL_SECONDS:
                    await queue.put(("status", {
                        "message": f"Still working — {int(idle)}s elapsed",
                        "stage": STAGE_WORKING,
                        "detail": int(idle),
                    }))

        async def on_event(event_type: str, data: Any) -> None:
            await queue.put((event_type, data))

        async def run() -> None:
            try:
                if intent == "GROUND":
                    result = await run_ground(sub_agent, message, thread_id)
                else:
                    result = await run_satellite(
                        sub_agent, message, thread_id, on_event=on_event, mcp_manager=self.mcp_manager,
                    )
                await queue.put(("__result__", result))
            except Exception as exc:  # noqa: BLE001 — surfaced as an SSE error event below
                await queue.put(("__error__", exc))
            finally:
                await queue.put(("__task_done__", None))

        task = asyncio.create_task(run())
        heartbeat_task = asyncio.create_task(heartbeat())
        result = None
        turn_timeout = asyncio.timeout(self.chat_turn_timeout_seconds)
        try:
            async with turn_timeout:
                while True:
                    event_type, data = await queue.get()
                    last_activity["t"] = loop.time()
                    if event_type == "__task_done__":
                        break
                    if event_type == "__error__":
                        raise data
                    if event_type == "__result__":
                        result = data
                        continue
                    # Drop the sub-agent's tool_result/text/done: the final
                    # envelope below becomes the one answer.
                    live = await self._live_event_sse(event_type, data, thread_id, user_id, turn)
                    for event in live or ():
                        yield event
        except Exception as exc:
            for event in self._failure_events(
                exc, turn_timeout, request_id, thread_id, turn.job_statuses,
            ):
                yield event
            return
        finally:
            if not task.done():
                task.cancel()
            heartbeat_task.cancel()

        # run() puts __result__ before __task_done__ on every non-exception
        # path, and the except block above returns before this point on
        # __error__ — so reaching here guarantees result was set.
        assert result is not None
        for chart in result.charts:
            chart_event = await self._emit_chart_once(thread_id, chart, user_id, turn.emitted_chart_ids)
            if chart_event is not None:
                yield chart_event
        for artifact_ref in result.artifacts:
            payload = await self._resolve_artifact_payload(
                artifact_ref.model_dump(exclude_none=True), user_id, thread_id,
            )
            if payload is None:
                continue
            if payload not in artifacts:
                artifacts.append(payload)
            yield self.sse("artifact", payload)

        final_text = with_sources_header(result.text, result_sources(_FAST_PATH_AGENTS[intent], result))
        yield self.sse("text", {"content": final_text})

        await self._write_back_turn(supervisor_agent, thread_id, user_id, message, final_text, intent, result)

        done_payload = {
            "thread_id": thread_id,
            "response": final_text,
            "artifacts": artifacts,
            "tool_calls": turn.tool_calls,
        }
        if result.suggested_followups is not None:
            done_payload["suggested_followups"] = result.suggested_followups
        if result.variable_choice is not None:
            done_payload["variable_choice"] = result.variable_choice.model_dump(exclude_none=True)
        yield self.sse("done", done_payload)
        self._log_request_complete(request_id, thread_id, started)

    async def _write_back_turn(
        self,
        agent: Any,
        thread_id: str,
        user_id: str,
        user_message: str,
        final_answer: str,
        intent: str,
        result: AgentResult,
    ) -> None:
        """Append the fast-pathed turn to the supervisor's thread so later
        turns see the full conversation. A failed write-back is logged, not
        raised: the turn already answered.

        With charts or artifacts, writes the same tool_call/ToolMessage shape
        the supervisor's dispatch tool produces, because HistoryService only
        rebuilds chart/artifact cards from tool messages. Text-only results
        write a plain Human/AI pair to keep the checkpoint small.

        ``as_node="model"`` is required: without it, LangGraph can't infer the
        node after the first manual update and raises InvalidUpdateError."""
        if result.charts or result.artifacts:
            tool_call_id = str(uuid.uuid4())
            agent_name = _FAST_PATH_AGENTS[intent]
            batch = DispatchResult(results=[DispatchItemResult(
                index=0, agent=agent_name, intent=_FAST_PATH_INTENT, status="ok", result=result,
                sources=result_sources(agent_name, result),
            )])
            messages = [
                HumanMessage(content=user_message),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "dispatch",
                        "args": {"requests": [{"agent": agent_name, "intent": _FAST_PATH_INTENT}]},
                        "id": tool_call_id,
                    }],
                ),
                ToolMessage(
                    content=await self.chart_service.checkpoint_dispatch_envelope(batch, thread_id, user_id),
                    tool_call_id=tool_call_id,
                    name="dispatch",
                ),
                AIMessage(content=final_answer),
            ]
        else:
            messages = [
                HumanMessage(content=user_message),
                AIMessage(content=final_answer),
            ]
        try:
            await agent.aupdate_state(
                {"configurable": {"thread_id": thread_id}},
                {"messages": messages},
                as_node="model",
            )
        except Exception:
            logger.warning(
                "fast_path_writeback_failed",
                exc_info=True,
                extra={"_thread_id": thread_id},
            )

    def sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _turn_timeout_events(
        self, request_id: str, thread_id: str, job_statuses: dict[str, str],
    ) -> list[str]:
        """The whole-turn deadline expired. Logs ``chat_turn_timeout`` and
        returns the ``error`` + ``done`` pair, naming any jobs still running."""
        in_flight = [handle for handle, status in job_statuses.items() if status not in TERMINAL_STATUSES]
        logger.warning(
            "chat_turn_timeout",
            extra={
                "_event": "chat_turn_timeout",
                "_request_id": request_id,
                "_thread_id": thread_id,
                "_in_flight_jobs": in_flight,
            },
        )
        return [
            self.sse("error", {"detail": render_turn_timeout_answer(in_flight)}),
            self.sse("done", {
                "thread_id": thread_id,
                "response": render_turn_timeout_answer(in_flight),
                "artifacts": [],
                "tool_calls": [],
            }),
        ]

    def _overloaded_events(self, request_id: str, thread_id: str) -> list[str]:
        """Admission control refused this turn, so the turn ends here.

        The SSE response is already committed, so a 503 isn't possible; end
        with an ``error``/``done`` pair like the timeout case. Logged as
        ``chat_turn_shed`` so shedding can be counted to tune the limit.
        """
        answer = render_overloaded_answer()
        logger.warning(
            "chat_turn_shed",
            extra={
                "_event": "chat_turn_shed",
                "_request_id": request_id,
                "_thread_id": thread_id,
                "_in_flight": admission.in_flight(),
                "_queued": admission.queued(),
            },
        )
        return [
            self.sse("error", {"detail": answer}),
            self.sse("done", {
                "thread_id": thread_id,
                "response": answer,
                "artifacts": [],
                "tool_calls": [],
            }),
        ]

    async def _tool_result_events(
        self,
        content: Any,
        thread_id: str,
        user_id: str,
        artifacts: list[dict[str, Any]] | None = None,
        emitted_chart_ids: set[str] | None = None,
        suggestions_box: dict[str, list[str]] | None = None,
        variable_choice_box: dict[str, dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        if emitted_chart_ids is None:
            emitted_chart_ids = set()
        # One result for a legacy ask_* message, one per item for a dispatch batch.
        agent_results = parse_tool_results(content)
        if agent_results:
            for agent_result in agent_results:
                for artifact_ref in agent_result.artifacts:
                    payload = await self._resolve_artifact_payload(
                        artifact_ref.model_dump(exclude_none=True), user_id, thread_id,
                    )
                    if payload is None:
                        continue
                    if artifacts is not None and payload not in artifacts:
                        artifacts.append(payload)
                    yield self.sse("artifact", payload)
                for chart in agent_result.charts:
                    event = await self._emit_chart_once(thread_id, chart, user_id, emitted_chart_ids)
                    if event is not None:
                        yield event
            followups = _merged_followups(agent_results)
            if suggestions_box is not None and followups is not None:
                suggestions_box["value"] = followups
            # The UI shows one picker, so the first item's wins.
            picker = next((r.variable_choice for r in agent_results if r.variable_choice is not None), None)
            if variable_choice_box is not None and picker is not None:
                variable_choice_box["value"] = picker.model_dump(exclude_none=True)
            return

        artifact_refs = self._artifact_refs(content)
        emitted_something = False

        # A chart-backed artifact type (map/comparison/timeseries) carries its
        # full render payload (lats/lons/values, panels, series) alongside
        # `_artifact_refs` in the SAME tool-result content — persist and emit
        # it first, so the artifact event that follows always has something
        # durable behind it for PNG/CSV export to read.
        _, charts = self.chart_service.parse_charts(content)
        if charts:
            for chart in charts:
                event = await self._emit_chart_once(thread_id, chart, user_id, emitted_chart_ids)
                if event is not None:
                    yield event
            emitted_something = True

        if artifact_refs:
            for ref in artifact_refs:
                payload = await self._resolve_artifact_payload(ref, user_id, thread_id)
                if payload is None:
                    continue
                if artifacts is not None and payload not in artifacts:
                    artifacts.append(payload)
                emitted_something = True
                yield self.sse("artifact", payload)

        if emitted_something:
            return

        if self._looks_like_chart_payload(content):
            logger.warning(
                "chart_payload_parse_failure",
                extra={
                    "_event": "chart_payload_parse_failure",
                    "_result_preview": str(content)[:200],
                    "_thread_id": thread_id,
                },
            )

    async def _resolve_artifact_payload(
        self,
        ref: dict[str, Any],
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        """Table artifacts are claimed (persisted to Postgres) on first sight.
        Chart-backed types are already persisted with their chart, so they
        pass through as-is."""
        if ref.get("type") != "table":
            return ref
        try:
            claimed = await artifact_store.claim(ref["id"], user_id, thread_id)
        except KeyError:
            logger.warning(
                "artifact_ref_missing",
                extra={"_artifact_id": ref.get("id"), "_thread_id": thread_id},
            )
            return None
        return claimed.model_dump(exclude_none=True)

    async def _text_events(
        self,
        data: Any,
        thread_id: str,
        user_id: str,
        emitted_chart_ids: set[str] | None = None,
        suggestions_box: dict[str, list[str]] | None = None,
        variable_choice_box: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[str, list[str]]:
        if emitted_chart_ids is None:
            emitted_chart_ids = set()
        if isinstance(data, str):
            structured_result = parse_agent_result(data)
            if structured_result is not None:
                events = []
                for chart in structured_result.charts:
                    event = await self._emit_chart_once(thread_id, chart, user_id, emitted_chart_ids)
                    if event is not None:
                        events.append(event)
                for artifact in structured_result.artifacts:
                    payload = await self._resolve_artifact_payload(
                        artifact.model_dump(exclude_none=True), user_id, thread_id,
                    )
                    if payload is None:
                        continue
                    events.append(self.sse("artifact", payload))
                if suggestions_box is not None and structured_result.suggested_followups is not None:
                    suggestions_box["value"] = structured_result.suggested_followups
                if variable_choice_box is not None and structured_result.variable_choice is not None:
                    variable_choice_box["value"] = structured_result.variable_choice.model_dump(exclude_none=True)
                return structured_result.text or "", events

            text, charts = self.chart_service.parse_charts(data)
            if text is not None or charts:
                events = []
                for chart in charts:
                    event = await self._emit_chart_once(thread_id, chart, user_id, emitted_chart_ids)
                    if event is not None:
                        events.append(event)
                return text or "", events
            return data, []
        if isinstance(data, list):
            return flatten_text_content(data), []
        return "", []

    async def _emit_chart_once(
        self, thread_id: str, chart: Any, user_id: str, emitted_chart_ids: set[str],
    ) -> str | None:
        """Persist and build the "chart" SSE event for ``chart``, or return
        None if its chart_id was already emitted this turn or the user
        doesn't own it."""
        payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
        chart_id = payload.get("chart_id")
        if chart_id is not None and chart_id in emitted_chart_ids:
            return None
        stored = await self.chart_service.persist_chart_payload(thread_id, chart, user_id)
        if stored is None:
            return None
        stored_id = stored.get("chart_id") if isinstance(stored, dict) else None
        emitted_id = stored_id or chart_id
        if emitted_id is not None:
            emitted_chart_ids.add(emitted_id)
        return self.sse("chart", stored)

    def _artifact_refs(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, dict):
            refs = content.get("_artifact_refs") or []
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
            except Exception:
                return []
            refs = parsed.get("_artifact_refs") if isinstance(parsed, dict) else []
        else:
            return []
        if not isinstance(refs, list):
            return []
        return [
            ref
            for ref in refs
            if isinstance(ref, dict) and isinstance(ref.get("id"), str) and isinstance(ref.get("type"), str)
        ]

    def _looks_like_chart_payload(self, content: Any) -> bool:
        if not isinstance(content, str):
            return False
        stripped = content.strip()
        if not stripped.startswith(("{", "[")):
            return False
        chart_markers = (
            '"charts"',
            '"type"',
            "'charts'",
            "'type'",
            "ChartPayload",
            "AgentResult",
        )
        return any(marker in stripped for marker in chart_markers)

    def _log_request_complete(self, request_id: str, thread_id: str, started: float) -> None:
        elapsed = time.monotonic() - started
        log = logger.warning if elapsed >= self.long_request_seconds else logger.info
        event = "long_running_request" if elapsed >= self.long_request_seconds else "request_completed"
        log(event, extra={"_request_id": request_id, "_thread_id": thread_id, "_elapsed_seconds": round(elapsed, 3)})
