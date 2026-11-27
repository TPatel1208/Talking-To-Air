from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tta_backend.harness.results import RUN_ID_PATTERN, Mode
from tta_backend.repositories import chart_repository


# Where a harness chart's link to its run is kept in the stored payload.
LINK_KEY = "harness"


class ChartPayload(BaseModel, extra="allow"):
    type: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# The few scalar fields a chart reference keeps: enough to name the chart and
# to find the stored row again.
_CHART_REFERENCE_KEYS = ("chart_id", "type", "title", "variable", "units")


def chart_reference(stored: dict[str, Any]) -> ChartPayload:
    """A persisted chart reduced to a pointer at its agent_charts row.

    The row is the source of truth for the grid; ``reference=True`` tells a
    reader to look the chart up rather than persist what it was handed."""
    fields = {key: stored[key] for key in _CHART_REFERENCE_KEYS if stored.get(key) is not None}
    metadata = stored.get("metadata")
    name = metadata.get("name") if isinstance(metadata, dict) else None
    # model_validate, not the constructor: `reference` is an extra field
    # (extra="allow"), which a typed constructor call does not declare.
    return ChartPayload.model_validate({
        **fields,
        "metadata": {"name": name} if name else {},
        "reference": True,
    })


def is_chart_reference(payload: dict[str, Any]) -> bool:
    return payload.get("reference") is True and bool(payload.get("chart_id"))


@dataclass(frozen=True)
class ChartLink:
    """The run that drew a chart, the mode it ran in, and the handles the chart renders.

    ``handle`` is the first of ``handles``. Kept in the chart's own row
    (``metadata.harness``): the row is already owned by its user and deleted
    with its session, and a run's charts are found from the other side
    through the run's trace, so a table of links would only repeat both.
    """

    run_id: str
    mode: Mode
    handle: str | None
    handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.handles and self.handle is not None:
            object.__setattr__(self, "handles", (self.handle,))

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "mode": self.mode.model_dump(mode="json"), "handle": self.handle,
            "handles": list(self.handles),
        }

    def view(self) -> dict[str, Any]:
        """The ``run_id`` and ``mode`` a chart or its artifact carries on the wire."""
        return {"run_id": self.run_id, "mode": self.mode.view()}

    @classmethod
    def of(cls, payload: dict[str, Any]) -> ChartLink | None:
        """The link a stored chart carries; None for a chart no harness run drew."""
        metadata = payload.get("metadata")
        link = metadata.get(LINK_KEY) if isinstance(metadata, dict) else None
        if not isinstance(link, dict):
            return None
        run_id, handle, handles = link.get("run_id"), link.get("handle"), link.get("handles")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            return None
        try:
            mode = Mode.model_validate(link.get("mode"))
        except ValidationError:
            return None
        listed = tuple(h for h in handles if isinstance(h, str)) if isinstance(handles, list) else ()
        first = handle if isinstance(handle, str) else (listed[0] if listed else None)
        return cls(run_id=run_id, mode=mode, handle=first, handles=listed)


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

    async def get_chart(self, chart_id: str) -> dict[str, Any] | None:
        return await chart_repository.get_chart(chart_id)

    async def get_owned_charts(self, chart_ids: list[str], user_id: str) -> dict[str, dict[str, Any]]:
        """The charts among ``chart_ids`` that ``user_id`` owns, keyed by id."""
        return await chart_repository.get_charts(chart_ids, user_id) if chart_ids else {}
