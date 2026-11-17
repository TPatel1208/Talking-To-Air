from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from tta_backend.harness.results import RUN_ID_PATTERN, Mode
from tta_backend.models import (
    AgentResult,
    DispatchResult,
    agent_result_to_json,
    chart_reference,
    is_chart_reference,
    parse_chart_payload,
    parse_tool_results,
)
from tta_backend.repositories import chart_repository


# Where a harness chart's link to its run is kept in the stored payload.
LINK_KEY = "harness"


@dataclass(frozen=True)
class ChartLink:
    """The run that drew a chart, the mode it ran in, and the handle the chart renders.

    Kept in the chart's own row (``metadata.harness``): the row is already
    owned by its user and deleted with its session, and a run's charts are
    found from the other side through the run's trace, so a table of links
    would only repeat both.
    """

    run_id: str
    mode: Mode
    handle: str | None

    def to_json(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "mode": self.mode.model_dump(mode="json"), "handle": self.handle}

    @classmethod
    def of(cls, payload: dict[str, Any]) -> ChartLink | None:
        """The link a stored chart carries; None for a chart no harness run drew."""
        metadata = payload.get("metadata")
        link = metadata.get(LINK_KEY) if isinstance(metadata, dict) else None
        if not isinstance(link, dict):
            return None
        run_id, handle = link.get("run_id"), link.get("handle")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            return None
        try:
            mode = Mode.model_validate(link.get("mode"))
        except ValidationError:
            return None
        return cls(run_id=run_id, mode=mode, handle=handle if isinstance(handle, str) else None)


class ChartService:
    async def persist_chart_payload(
        self, thread_id: str, chart: Any, user_id: str, *, link: ChartLink | None = None,
    ) -> dict[str, Any] | None:
        """Store ``chart`` and return the stored row, with ``link`` in its metadata when given.

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
        if link is not None:
            payload["metadata"] = {**(payload.get("metadata") or {}), LINK_KEY: link.to_json()}
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
        return agent_result_to_json(await self._with_chart_references(result, thread_id, user_id))

    async def checkpoint_dispatch_envelope(self, batch: DispatchResult, thread_id: str, user_id: str) -> str:
        """``batch`` (the dispatch tool's result) as checkpointed JSON,
        each item's charts swapped for references exactly as in
        checkpoint_envelope."""
        items = [
            item.model_copy(update={"result": await self._with_chart_references(item.result, thread_id, user_id)})
            if item.result is not None else item
            for item in batch.results
        ]
        return DispatchResult(results=items).model_dump_json(exclude_none=True)

    async def _with_chart_references(self, result: AgentResult, thread_id: str, user_id: str) -> AgentResult:
        references = []
        for chart in result.charts:
            stored = await self.persist_chart_payload(thread_id, chart, user_id)
            if stored is not None:
                references.append(chart_reference(stored))
        return result.model_copy(update={"charts": references})

    async def get_chart(self, chart_id: str) -> dict[str, Any] | None:
        return await chart_repository.get_chart(chart_id)

    async def get_owned_charts(self, chart_ids: list[str], user_id: str) -> dict[str, dict[str, Any]]:
        """The charts among ``chart_ids`` that ``user_id`` owns, keyed by id."""
        return await chart_repository.get_charts(chart_ids, user_id) if chart_ids else {}

    def parse_charts(self, content: Any) -> tuple[str | None, list[Any]]:
        results = parse_tool_results(content)
        if results:
            return "\n\n".join(r.text for r in results), [chart for r in results for chart in r.charts]
        chart = parse_chart_payload(content)
        if chart is not None:
            return None, [chart]
        return None, []
