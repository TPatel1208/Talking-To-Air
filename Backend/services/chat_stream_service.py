from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from models import parse_agent_result
from services.artifact_store import artifact_store
from services.chart_service import ChartService
from services.intent_router import inject_routing_hint
from utils.message_utils import PNG_PATH_RE, flatten_text_content, normalize_image_url
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
        started = time.monotonic()
        routed_message = inject_routing_hint(message)
        try:
            async for event_type, data in stream_response(agent, routed_message, thread_id):
                if event_type == "tool_call":
                    tool_calls.append({"name": data["name"], "args": data["args"]})
                    response_text = ""
                    yield self.sse("tool_call", {"name": data["name"], "args": data["args"]})
                elif event_type == "status":
                    yield self.sse("status", {"message": data.get("message", "")})
                elif event_type == "tool_result":
                    async for event in self._tool_result_events(
                        data.get("content", ""),
                        thread_id,
                        user_id,
                        image_urls,
                        artifacts,
                    ):
                        yield event
                elif event_type == "image":
                    url = normalize_image_url(data.get("path", ""))
                    if url:
                        image_urls.append(url)
                        yield self.sse("image", {"url": url})
                elif event_type == "text":
                    text, events = await self._text_events(data, thread_id, user_id)
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
    ) -> AsyncIterator[str]:
        agent_result = parse_agent_result(content)
        if agent_result is not None:
            for artifact_ref in agent_result.artifacts:
                try:
                    claimed = artifact_store.claim(artifact_ref.id, user_id, thread_id)
                except KeyError:
                    logger.warning(
                        "artifact_ref_missing",
                        extra={"_artifact_id": artifact_ref.id, "_thread_id": thread_id},
                    )
                    continue
                payload = claimed.model_dump(exclude_none=True)
                if artifacts is not None and payload not in artifacts:
                    artifacts.append(payload)
                yield self.sse("artifact", payload)
            for chart in agent_result.charts:
                yield self.sse("chart", await self.chart_service.persist_chart_payload(thread_id, chart, user_id))
            return

        artifact_refs = self._artifact_refs(content)
        if artifact_refs:
            for ref in artifact_refs:
                try:
                    claimed = artifact_store.claim(ref["id"], user_id, thread_id)
                except KeyError:
                    logger.warning(
                        "artifact_ref_missing",
                        extra={"_artifact_id": ref.get("id"), "_thread_id": thread_id},
                    )
                    continue
                payload = claimed.model_dump(exclude_none=True)
                if artifacts is not None and payload not in artifacts:
                    artifacts.append(payload)
                yield self.sse("artifact", payload)
            return

        _, charts = self.chart_service.parse_charts(content)
        if charts:
            for chart in charts:
                yield self.sse("chart", await self.chart_service.persist_chart_payload(thread_id, chart, user_id))
            return

        png_match = PNG_PATH_RE.search(content)
        if png_match:
            url = normalize_image_url(png_match.group(1))
            if url:
                image_urls.append(url)
                yield self.sse("image", {"url": url})
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

    async def _text_events(self, data: Any, thread_id: str, user_id: str) -> tuple[str, list[str]]:
        if isinstance(data, str):
            structured_result = parse_agent_result(data)
            if structured_result is not None:
                events = []
                for chart in structured_result.charts:
                    events.append(self.sse("chart", await self.chart_service.persist_chart_payload(thread_id, chart, user_id)))
                for artifact in structured_result.artifacts:
                    try:
                        claimed = artifact_store.claim(artifact.id, user_id, thread_id)
                    except KeyError:
                        continue
                    events.append(self.sse("artifact", claimed.model_dump(exclude_none=True)))
                return structured_result.text or "", events

            text, charts = self.chart_service.parse_charts(data)
            if text is not None or charts:
                events = [
                    self.sse("chart", await self.chart_service.persist_chart_payload(thread_id, chart, user_id))
                    for chart in charts
                ]
                return text or "", events
            return data, []
        if isinstance(data, list):
            return flatten_text_content(data), []
        return "", []

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
