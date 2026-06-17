from __future__ import annotations

import inspect
import json
from typing import Any

from services.artifact_store import artifact_store
from services.chart_service import ChartService
from utils.message_utils import PNG_PATH_RE, flatten_text_content, normalize_image_url


class HistoryService:
    def __init__(self, chart_service: ChartService):
        self.chart_service = chart_service

    async def build_history(self, agent: Any, thread_id: str, user_id: str) -> list[dict[str, Any]]:
        config = {"configurable": {"thread_id": thread_id}}
        if hasattr(agent, "aget_state"):
            state = await agent.aget_state(config)
        else:
            maybe_state = agent.get_state(config)
            state = await maybe_state if inspect.isawaitable(maybe_state) else maybe_state
        if not state or not state.values:
            return []

        result = []
        for msg in state.values.get("messages", []):
            role = getattr(msg, "type", None)
            if role == "human":
                result.append({
                    "role": "user",
                    "content": msg.content if isinstance(msg.content, str) else "",
                    "toolCalls": [],
                    "imageUrls": [],
                })
            elif role == "ai":
                result.append(self._assistant_message(msg))
            elif role == "tool":
                await self._attach_tool_output(result, msg, thread_id, user_id)
        return self._merge_adjacent_assistant_messages(result)

    def _assistant_message(self, msg: Any) -> dict[str, Any]:
        tool_calls = []
        seen_tool_ids = set()
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tid = tc.get("id", "")
                seen_tool_ids.add(tid)
                tool_calls.append({"name": tc.get("name", ""), "args": tc.get("args", {})})

        content = ""
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, str):
                    content += block
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        content += block.get("text", "")
                    elif btype == "tool_use":
                        tid = block.get("id", "")
                        if tid not in seen_tool_ids:
                            seen_tool_ids.add(tid)
                            tool_calls.append({"name": block.get("name", ""), "args": block.get("input", {})})
                elif hasattr(block, "text"):
                    content += block.text

        return {
            "role": "assistant",
            "content": content,
            "toolCalls": tool_calls,
            "imageUrls": [],
            "charts": [],
            "artifacts": [],
        }

    async def _attach_tool_output(
        self,
        result: list[dict[str, Any]],
        msg: Any,
        thread_id: str,
        user_id: str,
    ) -> None:
        tool_text = flatten_text_content(msg.content)
        for png_match in PNG_PATH_RE.finditer(tool_text):
            url = normalize_image_url(png_match.group(1))
            if url:
                assistant = self._last_assistant(result)
                if assistant and url not in assistant["imageUrls"]:
                    assistant["imageUrls"].append(url)

        _, charts = self.chart_service.parse_charts(tool_text)
        for chart in charts:
            chart_payload = await self.chart_service.persist_chart_payload(thread_id, chart, user_id)
            assistant = self._last_assistant(result)
            if assistant is not None:
                assistant.setdefault("charts", [])
                if chart_payload not in assistant["charts"]:
                    assistant["charts"].append(chart_payload)

        for ref in self._artifact_refs(tool_text):
            try:
                artifact = artifact_store.claim(ref["id"], user_id, thread_id).model_dump(exclude_none=True)
            except KeyError:
                continue
            assistant = self._last_assistant(result)
            if assistant is not None:
                assistant.setdefault("artifacts", [])
                if artifact not in assistant["artifacts"]:
                    assistant["artifacts"].append(artifact)

    def _last_assistant(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        for message in reversed(messages):
            if message["role"] == "assistant":
                return message
        return None

    def _merge_adjacent_assistant_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = []
        for msg in messages:
            if msg["role"] == "assistant" and merged and merged[-1]["role"] == "assistant":
                prev = merged[-1]
                prev["toolCalls"].extend(msg["toolCalls"])
                if msg["content"]:
                    prev["content"] += ("\n\n" if prev["content"] else "") + msg["content"]
                for url in msg["imageUrls"]:
                    if url not in prev["imageUrls"]:
                        prev["imageUrls"].append(url)
                for chart in msg.get("charts", []):
                    prev.setdefault("charts", [])
                    if chart not in prev["charts"]:
                        prev["charts"].append(chart)
                for artifact in msg.get("artifacts", []):
                    prev.setdefault("artifacts", [])
                    if artifact not in prev["artifacts"]:
                        prev["artifacts"].append(artifact)
            else:
                merged.append(msg)
        return merged

    def _artifact_refs(self, content: Any) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except Exception:
            return []
        if not isinstance(parsed, dict):
            return []
        refs = parsed.get("_artifact_refs") or []
        if not isinstance(refs, list):
            return []
        return [
            ref
            for ref in refs
            if isinstance(ref, dict) and isinstance(ref.get("id"), str) and isinstance(ref.get("type"), str)
        ]
