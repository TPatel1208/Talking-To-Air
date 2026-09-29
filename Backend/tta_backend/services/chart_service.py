from __future__ import annotations

from typing import Any

from tta_backend.models import (
    AgentResult,
    agent_result_to_json,
    chart_reference,
    is_chart_reference,
    parse_agent_result,
    parse_chart_payload,
)
from tta_backend.repositories import chart_repository


class ChartService:
    async def persist_chart_payload(self, thread_id: str, chart: Any, user_id: str) -> dict[str, Any] | None:
        """Store ``chart`` and return the stored row.

        A chart reference is only ever looked up: it carries no grid, so
        saving it would replace the real row with a stub. None means the
        reference names no chart this user owns."""
        payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
        if payload.get("chart_id"):
            stored = await chart_repository.get_chart(payload["chart_id"])
            if stored and stored.get("user_id") == user_id:
                return stored
        if is_chart_reference(payload):
            return None
        return await chart_repository.save_chart(thread_id, payload, user_id)

    async def resolve_charts(self, charts: list[Any], user_id: str) -> list[dict[str, Any] | None]:
        """``charts`` as stored rows, in order, from one read and no writes.

        A chart without a row this user owns resolves to None if it is a
        reference, and to its own inline payload otherwise (checkpoints
        written before references carry the whole grid)."""
        payloads = [c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else dict(c) for c in charts]
        chart_ids = [p["chart_id"] for p in payloads if p.get("chart_id")]
        stored = await chart_repository.get_charts(chart_ids, user_id) if chart_ids else {}
        resolved: list[dict[str, Any] | None] = []
        for payload in payloads:
            row = stored.get(payload.get("chart_id", ""))
            if row is not None:
                resolved.append(row)
            elif is_chart_reference(payload):
                resolved.append(None)
            else:
                resolved.append(payload)
        return resolved

    async def checkpoint_envelope(self, result: AgentResult, thread_id: str, user_id: str) -> str:
        """``result`` as the JSON a checkpointed ToolMessage carries.

        Each chart is persisted first and then replaced by a reference to its
        agent_charts row. The checkpoint rewrites a thread's whole message
        list at every step, so a grid kept here would be stored again on
        every later turn."""
        references = []
        for chart in result.charts:
            stored = await self.persist_chart_payload(thread_id, chart, user_id)
            if stored is not None:
                references.append(chart_reference(stored))
        return agent_result_to_json(result.model_copy(update={"charts": references}))

    async def get_chart(self, chart_id: str) -> dict[str, Any] | None:
        return await chart_repository.get_chart(chart_id)

    def parse_charts(self, content: Any) -> tuple[str | None, list[Any]]:
        structured_result = parse_agent_result(content)
        if structured_result is not None:
            return structured_result.text, list(structured_result.charts)
        chart = parse_chart_payload(content)
        if chart is not None:
            return None, [chart]
        return None, []
