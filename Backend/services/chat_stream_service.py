from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from models import parse_agent_result, parse_chart_payload
from services.artifact_store import artifact_store
from services.chart_service import ChartService
from services.intent_router import inject_routing_hint
from utils.message_utils import flatten_text_content, normalize_image_url
from utils.streaming import stream_response

logger = logging.getLogger(__name__)


class ChatStreamService:
    def __init__(self, chart_service: ChartService, long_request_seconds: float):
        self.chart_service = chart_service
        self.long_request_seconds = long_request_seconds

    async def stream_chat_events(
        self,
        agent: Any,
        message: str,
        thread_id: str,
        user_id: str,
        request_id: str,
    ) -> AsyncIterator[str]:
        response_text = ""
        image_urls = []
        artifacts = []
        tool_calls = []
        # Charts already emitted this turn, by chart_id — a chart can reach
        # this loop twice (once bubbled in real time via a chart_payload
        # event, again batched inside a sub-agent's final tool_result
        # envelope); the frontend appends every "chart" event with no dedup
        # of its own (Frontend/src/hooks/useChat.js), so this set is the
        # single point that keeps each chart_id rendered exactly once (T13).
        emitted_chart_ids: set[str] = set()
        started = time.monotonic()
        routed_message = inject_routing_hint(message)
        try:
            async for event_type, data in stream_response(agent, routed_message, thread_id, user_id=user_id):
                if event_type == "tool_call":
                    tool_calls.append({"name": data["name"], "args": data["args"]})
                    response_text = ""
                    yield self.sse("tool_call", {"name": data["name"], "args": data["args"]})
                elif event_type == "status":
                    yield self.sse("status", {"message": data.get("message", "")})
                elif event_type == "job_progress":
                    yield self.sse("job_progress", data)
                elif event_type == "chart_payload":
                    chart = parse_chart_payload(data)
                    if chart is not None:
                        event = await self._emit_chart_once(thread_id, chart, user_id, emitted_chart_ids)
                        if event is not None:
                            yield event
                elif event_type == "tool_result":
                    async for event in self._tool_result_events(
                        data.get("content", ""),
                        thread_id,
                        user_id,
                        image_urls,
                        artifacts,
                        emitted_chart_ids,
                    ):
                        yield event
                elif event_type == "image":
                    url = normalize_image_url(data.get("path", ""))
                    if url:
                        image_urls.append(url)
                        yield self.sse("image", {"url": url})
                elif event_type == "text":
                    text, events = await self._text_events(data, thread_id, user_id, emitted_chart_ids)
                    response_text += text
                    if text:
                        yield self.sse("text", {"content": text})
                    for event in events:
                        yield event

            yield self.sse("done", {
                "thread_id": thread_id,
                "response": self._strip_supervisor_preamble(response_text),
                "image_urls": image_urls,
                "artifacts": artifacts,
                "tool_calls": tool_calls,
            })
            self._log_request_complete(request_id, thread_id, started)
        except Exception as e:
            logger.exception("agent_failure", extra={"_request_id": request_id, "_thread_id": thread_id})
            yield self.sse("error", {"detail": str(e)})

    def sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _strip_supervisor_preamble(self, text: str) -> str:
        marker = "Agent consulted:"
        idx = text.find(marker)
        return text[idx:] if idx > 0 else text

    async def _tool_result_events(
        self,
        content: str,
        thread_id: str,
        user_id: str,
        image_urls: list[str],
        artifacts: list[dict[str, Any]] | None = None,
        emitted_chart_ids: set[str] | None = None,
    ) -> AsyncIterator[str]:
        if emitted_chart_ids is None:
            emitted_chart_ids = set()
        agent_result = parse_agent_result(content)
        if agent_result is not None:
            for artifact_ref in agent_result.artifacts:
                payload = self._resolve_artifact_payload(
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
                payload = self._resolve_artifact_payload(ref, user_id, thread_id)
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

    def _resolve_artifact_payload(
        self,
        ref: dict[str, Any],
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        """Table artifacts live in the ephemeral in-memory artifact_store and
        need ownership claimed on first sight. Chart-backed artifact types
        (map/comparison/timeseries) are already fully formed by
        artifact_registry and persisted durably alongside the chart payload
        that carries them — pass them through as-is."""
        if ref.get("type") != "table":
            return ref
        try:
            claimed = artifact_store.claim(ref["id"], user_id, thread_id)
        except KeyError:
            logger.warning(
                "artifact_ref_missing",
                extra={"_artifact_id": ref.get("id"), "_thread_id": thread_id},
            )
            return None
        return claimed.model_dump(exclude_none=True)

    async def _text_events(
        self, data: Any, thread_id: str, user_id: str, emitted_chart_ids: set[str] | None = None,
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
                    payload = self._resolve_artifact_payload(
                        artifact.model_dump(exclude_none=True), user_id, thread_id,
                    )
                    if payload is None:
                        continue
                    events.append(self.sse("artifact", payload))
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
        None if its chart_id was already emitted this turn (T13 dedup — see
        the comment on stream_chat_events' emitted_chart_ids)."""
        payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
        chart_id = payload.get("chart_id")
        if chart_id is not None and chart_id in emitted_chart_ids:
            return None
        stored = await self.chart_service.persist_chart_payload(thread_id, chart, user_id)
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
