from __future__ import annotations

from typing import Any

from models import parse_agent_result, parse_chart_payload
from repositories import chart_repository


class ChartService:
    async def persist_chart_payload(self, thread_id: str, chart: Any, user_id: str) -> dict[str, Any]:
        payload = chart.model_dump(exclude_none=True) if hasattr(chart, "model_dump") else dict(chart)
        if payload.get("chart_id"):
            stored = await chart_repository.get_chart(payload["chart_id"])
            if stored and stored.get("user_id") == user_id:
                return stored
        return await chart_repository.save_chart(thread_id, payload, user_id)

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
