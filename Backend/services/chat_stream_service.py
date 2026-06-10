from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from services.chart_service import ChartService
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
        tool_calls = []
        started = time.monotonic()
        try:
            async for event_type, data in stream_response(agent, message, thread_id):
                if event_type == "tool_call":
                    tool_calls.append({"name": data["name"], "args": data["args"]})
                    yield self.sse("tool_call", {"name": data["name"], "args": data["args"]})
                elif event_type == "status":
                    yield self.sse("status", {"message": data.get("message", "")})
                elif event_type == "tool_result":
                    async for event in self._tool_result_events(data.get("content", ""), thread_id, user_id, image_urls):
                        yield event
                elif event_type == "image":
                    url = normalize_image_url(data.get("path", ""))
                    if url:
                        image_urls.append(url)
                        yield self.sse("image", {"url": url})
                elif event_type == "text":
                    text, events = await self._text_events(data, thread_id, user_id)
                    response_text += text
                    for event in events:
                        yield event

            yield self.sse("done", {
                "thread_id": thread_id,
                "response": response_text,
                "image_urls": image_urls,
                "tool_calls": tool_calls,
            })
            self._log_request_complete(request_id, thread_id, started)
        except Exception as e:
            logger.exception("agent_failure", extra={"_request_id": request_id, "_thread_id": thread_id})
            yield self.sse("error", {"detail": str(e)})

    def sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def _tool_result_events(
        self,
        content: str,
        thread_id: str,
        user_id: str,
        image_urls: list[str],
    ) -> AsyncIterator[str]:
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

    async def _text_events(self, data: Any, thread_id: str, user_id: str) -> tuple[str, list[str]]:
        if isinstance(data, str):
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

    def _log_request_complete(self, request_id: str, thread_id: str, started: float) -> None:
        elapsed = time.monotonic() - started
        log = logger.warning if elapsed >= self.long_request_seconds else logger.info
        event = "long_running_request" if elapsed >= self.long_request_seconds else "request_completed"
        log(event, extra={"_request_id": request_id, "_thread_id": thread_id, "_elapsed_seconds": round(elapsed, 3)})
