"""The session ledger: what a conversation has established, for the supervisor's context.

The ledger has no table of its own. ``build_ledger`` reads the conversation's
runs from the trace, their saved results and the handle registry, all as the
context's user, so it cannot disagree with them and never shows another
user's runs. ``render_ledger`` turns it into compact plain text for a model.

Everything is kept oldest first: runs in the order they started, and within
a run in the order its steps were recorded, so the text changes only when
something new happens. Every line that names a run or a handle carries its
id, since the supervisor passes ids on.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import HandleRecord, is_handle_id
from tta_backend.harness.jobs import JobState
from tta_backend.harness.results import Finding, Result
from tta_backend.harness.run_context import RunContext
from tta_backend.harness.trace import Run, RunStatus, TraceEntry

RESOLVE = "sources.areas.resolve"
FIND_MONITORS = "sources.monitors.find"
RENDER_PREFIX = "render."
# Run statuses the ledger lists as not done; a running run is the work in hand.
NOT_DONE: frozenset[RunStatus] = frozenset(
    {"needs_input", "waiting", "incomplete", "failed", "insufficient_evidence", "cancelled"}
)

EMPTY = "Session ledger: nothing has been resolved, retrieved or run in this conversation yet."
HEADING = "Session ledger (oldest first):"
NO_RESULT = "No result was saved for this run."

_WORDING_CHARS = 120
_OPTION_CHARS = 80


@dataclass(frozen=True, kw_only=True)
class LedgerArea:
    """An area handle and the wording it was resolved from."""

    handle: str
    name: str | None
    wording: str | None
    run_id: str


@dataclass(frozen=True, kw_only=True)
class LedgerProduct:
    """A data product and the rule that chose it, with the runs that used it so."""

    product: str
    rule: str
    run_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class LedgerHandle:
    """A handle a run produced, described from its record; never its data.

    ``known`` is False when the registry holds no record for it.
    """

    id: str
    run_id: str
    known: bool = True
    kind: str | None = None
    summary: str | None = None
    quantity: str | None = None
    product: str | None = None
    unit: str | None = None
    area: str | None = None
    period: str | None = None
    evicted: bool = False


@dataclass(frozen=True, kw_only=True)
class LedgerMonitors:
    """Monitors a search found: the points handle holding them, their ids and where they were searched.

    ``area`` is the area handle searched in, or None for a search near ``point`` (lat, lon).
    """

    handle: str
    monitor_ids: tuple[str, ...]
    pollutant: str | None
    period: str | None
    area: str | None
    point: tuple[float, float] | None
    run_id: str


@dataclass(frozen=True, kw_only=True)
class LedgerChart:
    """A stored chart: its artifact ref, what drew it, and the handles it shows.

    ``kind`` is the render capability without its ``render.`` prefix, such as
    ``map``; None for a chart known only from a result.
    """

    ref: str
    run_id: str
    kind: str | None = None
    title: str | None = None
    handles: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LedgerFindings:
    """One run's findings, with the label of how its result was made."""

    run_id: str
    mode_label: str
    findings: tuple[Finding, ...]


@dataclass(frozen=True, kw_only=True)
class LedgerOpenRun:
    """A run that is not done, with what its result says it is waiting for.

    ``job_state`` is set for a waiting run only. ``workflow`` says whether it
    is a workflow run, which can be resumed, or an exploration, which cannot.
    """

    run_id: str
    status: RunStatus
    mode_label: str
    workflow: bool
    reason: str | None
    question: dict[str, Any] | None
    job_state: JobState | None = None
    has_result: bool = True


@dataclass(frozen=True, kw_only=True)
class Ledger:
    conversation_id: str
    areas: tuple[LedgerArea, ...] = ()
    products: tuple[LedgerProduct, ...] = ()
    monitors: tuple[LedgerMonitors, ...] = ()
    handles: tuple[LedgerHandle, ...] = ()
    charts: tuple[LedgerChart, ...] = ()
    findings: tuple[LedgerFindings, ...] = ()
    open_runs: tuple[LedgerOpenRun, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.areas or self.products or self.monitors or self.handles or self.charts or self.findings
            or self.open_runs
        )


# Building


def _handle_of(result: Any) -> str | None:
    value = result.get("handle") if isinstance(result, dict) else None
    return value if isinstance(value, str) and is_handle_id(value) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _handle_params(params: Mapping[str, Any]) -> tuple[str, ...]:
    """The handle ids among a step's parameters, in parameter order."""
    found: list[str] = []
    for value in params.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and is_handle_id(item) and item not in found:
                found.append(item)
    return tuple(found)


def _period(record: HandleRecord) -> str | None:
    period = record.descriptor.period
    if period is None:
        return None
    return period.start if period.start == period.end else f"{period.start} to {period.end}"


class _Builder:
    def __init__(self, context: RunContext, job_states: Mapping[str, JobState]) -> None:
        self.user_id = context.identity.user_id
        self.storage = context.storage
        self.job_states = job_states
        self.areas: dict[str, LedgerArea] = {}
        self.products: dict[tuple[str, str], list[str]] = {}
        self.monitors: dict[str, LedgerMonitors] = {}
        self.handles: dict[str, LedgerHandle] = {}
        self.charts: dict[str, LedgerChart] = {}
        self.findings: list[LedgerFindings] = []
        self.open_runs: list[LedgerOpenRun] = []
        self._records: dict[str, HandleRecord | None] = {}

    async def _record(self, handle_id: str) -> HandleRecord | None:
        if handle_id not in self._records:
            try:
                self._records[handle_id] = await self.storage.handles.get(self.user_id, handle_id)
            except HandleError:
                self._records[handle_id] = None
        return self._records[handle_id]

    def _area(self, handle_id: str, name: str | None, wording: str | None, run_id: str) -> None:
        if handle_id not in self.areas:
            self.areas[handle_id] = LedgerArea(handle=handle_id, name=name, wording=wording, run_id=run_id)

    async def _handle(self, handle_id: str, run_id: str) -> None:
        if handle_id in self.handles or handle_id in self.areas:
            return
        record = await self._record(handle_id)
        if record is None:
            self.handles[handle_id] = LedgerHandle(id=handle_id, run_id=run_id, known=False)
            return
        descriptor = record.descriptor
        region = descriptor.region if isinstance(descriptor.region, dict) else {}
        area_id = region.get("area")
        if isinstance(area_id, str) and is_handle_id(area_id):
            # A handle made over an area records it, as get_data does.
            self._area(area_id, _text(region.get("name")), _text(region.get("wording")), run_id)
        if record.kind == "polygons" and isinstance(region.get("bounds"), list) and "wording" in region:
            # An area handle itself, given to a run rather than resolved in it.
            self._area(record.id, _text(region.get("name")), _text(region.get("wording")), run_id)
            return
        self.handles[handle_id] = LedgerHandle(
            id=record.id,
            run_id=run_id,
            kind=record.kind,
            summary=descriptor.summary,
            quantity=descriptor.quantity,
            product=descriptor.product,
            unit=descriptor.unit,
            area=_text(region.get("name")),
            period=_period(record),
            evicted=record.payload.state == "evicted",
        )

    async def _entry(self, entry: TraceEntry, run_id: str) -> None:
        if entry.kind != "use" or not isinstance(entry.result, dict):
            return
        result = entry.result
        handle = _handle_of(result)
        if entry.capability == RESOLVE:
            if entry.status == "done" and handle is not None:
                summary = _mapping(result.get("summary"))
                self._area(handle, _text(summary.get("name")), _text(entry.params.get("place")), run_id)
            return
        values = _mapping(result.get("values"))
        product, rule = _text(result.get("product")), _text(values.get("rule"))
        if product is not None and rule is not None:
            runs = self.products.setdefault((product, rule), [])
            if run_id not in runs:
                runs.append(run_id)
        if entry.capability == FIND_MONITORS and entry.status == "done" and handle is not None:
            await self._monitors(entry, handle, values, run_id)
        if entry.capability and entry.capability.startswith(RENDER_PREFIX) and entry.status == "done":
            self._charts(entry, result, run_id)
        if handle is not None:
            await self._handle(handle, run_id)

    async def _monitors(self, entry: TraceEntry, handle: str, values: Mapping[str, Any], run_id: str) -> None:
        if handle in self.monitors:
            return
        params = entry.params
        area = params.get("area")
        area_id = area if isinstance(area, str) and is_handle_id(area) else None
        if area_id is not None:
            await self._handle(area_id, run_id)
        lat, lon = params.get("lat"), params.get("lon")
        point = (float(lat), float(lon)) if isinstance(lat, int | float) and isinstance(lon, int | float) else None
        start, end = _text(params.get("start")), _text(params.get("end"))
        period = None
        if start is not None:
            period = start if end is None or end == start else f"{start} to {end}"
        ids = values.get("monitor_ids")
        self.monitors[handle] = LedgerMonitors(
            handle=handle,
            monitor_ids=tuple(str(i) for i in ids) if isinstance(ids, list) else (),
            pollutant=_text(params.get("pollutant")),
            period=period,
            area=area_id,
            point=point,
            run_id=run_id,
        )

    def _charts(self, entry: TraceEntry, result: Mapping[str, Any], run_id: str) -> None:
        charts = result.get("charts")
        if not isinstance(charts, list) or entry.capability is None:
            return
        summary = _mapping(result.get("summary"))
        shown = _handle_params(entry.params)
        sources = summary.get("source_handles")
        if not shown and isinstance(sources, list):
            shown = tuple(h for h in sources if isinstance(h, str) and is_handle_id(h))
        for ref in charts:
            if isinstance(ref, str) and ref not in self.charts:
                self.charts[ref] = LedgerChart(
                    ref=ref, run_id=run_id, kind=entry.capability.removeprefix(RENDER_PREFIX),
                    title=_text(summary.get("title")), handles=shown,
                )

    async def _result(self, run: Run, result: Result | None) -> None:
        if result is not None:
            for handle in result.handles:
                await self._handle(handle, run.id)
            for ref in result.charts:
                self.charts.setdefault(ref, LedgerChart(ref=ref, run_id=run.id))
            if result.findings:
                self.findings.append(
                    LedgerFindings(run_id=run.id, mode_label=run.mode.label(), findings=result.findings)
                )
        if run.status not in NOT_DONE:
            return
        # The run's status is the trace's; the result is read only when it says the same.
        current = result if result is not None and result.status == run.status else None
        self.open_runs.append(LedgerOpenRun(
            run_id=run.id,
            status=run.status,
            mode_label=run.mode.label(),
            workflow=run.mode.workflow is not None,
            reason=current.reason if current is not None else None,
            question=current.question if current is not None else None,
            job_state=self.job_states.get(run.id, "unknown") if run.status == "waiting" else None,
            has_result=result is not None,
        ))

    async def build(self, conversation_id: str) -> Ledger:
        traces = self.storage.traces
        runs = await traces.list_runs(self.user_id, conversation_id)
        saved = await self.storage.results.for_runs(self.user_id, [run.id for run in runs])
        for run in runs:
            for entry in await traces.entries(self.user_id, run.id):
                await self._entry(entry, run.id)
            await self._result(run, saved.get(run.id))
        return Ledger(
            conversation_id=conversation_id,
            areas=tuple(self.areas.values()),
            products=tuple(
                LedgerProduct(product=product, rule=rule, run_ids=tuple(run_ids))
                for (product, rule), run_ids in self.products.items()
            ),
            monitors=tuple(self.monitors.values()),
            # Areas and monitors have sections of their own.
            handles=tuple(h for h in self.handles.values() if h.id not in self.areas and h.id not in self.monitors),
            charts=tuple(self.charts.values()),
            findings=tuple(self.findings),
            open_runs=tuple(self.open_runs),
        )


async def build_ledger(context: RunContext, *, job_states: Mapping[str, JobState] | None = None) -> Ledger:
    """The ledger of the context's conversation, read as the context's user.

    ``job_states`` maps a waiting run's id to the state of the retrieval jobs
    it waits on; a waiting run it does not name is ``unknown``.
    """
    return await _Builder(context, job_states or {}).build(context.identity.conversation_id)


# Rendering


def _line(text: str, limit: int | None = None) -> str:
    """``text`` on one line, cut to ``limit`` characters."""
    text = " ".join(text.split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _number(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return _line(str(value), _WORDING_CHARS)


def _area_line(area: LedgerArea) -> str:
    name = _line(area.name, _WORDING_CHARS) if area.name else "an area"
    asked = ""
    if area.wording and _line(area.wording).lower() != _line(area.name or "").lower():
        asked = f' (asked as "{_line(area.wording, _WORDING_CHARS)}")'
    return f"- {name}{asked}: {area.handle}, run {area.run_id}"


def _product_line(product: LedgerProduct) -> str:
    noun = "run" if len(product.run_ids) == 1 else "runs"
    return f"- {product.product}, chosen by rule {product.rule}: {noun} {', '.join(product.run_ids)}"


def _monitors_line(found: LedgerMonitors, areas: Mapping[str, LedgerArea]) -> str:
    if found.area is not None:
        area = areas.get(found.area)
        name = _line(area.name, _WORDING_CHARS) if area is not None and area.name else "the area"
        where = f"in {name} ({found.area})"
    elif found.point is not None:
        where = f"near {found.point[0]:g}, {found.point[1]:g}"
    else:
        where = "near a point"
    what = " ".join(part for part in (found.pollutant, "monitors") if part)
    when = f", {found.period}" if found.period else ""
    ids = ", ".join(found.monitor_ids) if found.monitor_ids else "none listed"
    return f"- {found.handle}: {len(found.monitor_ids)} {what} {where}{when}; monitor ids {ids} (run {found.run_id})"


def _chart_line(chart: LedgerChart) -> str:
    said = [f"- {chart.ref}:", chart.kind or "chart"]
    if chart.title:
        said.append(f'"{_line(chart.title, _WORDING_CHARS)}"')
    if chart.handles:
        said.append("of " + ", ".join(chart.handles))
    return f"{' '.join(said)} (run {chart.run_id})"


def _handle_line(handle: LedgerHandle, *, short: bool) -> str:
    if not handle.known:
        return f"- {handle.id}: no record"
    evicted = ", evicted" if handle.evicted else ""
    head = f"- {handle.id} {handle.kind}{evicted}"
    if short:
        return head
    summary = _line(handle.summary or "")
    parts = [summary] if summary else []
    for label, value in (
        ("product", handle.product), ("quantity", handle.quantity), ("area", handle.area),
        ("period", handle.period), ("unit", handle.unit),
    ):
        # Only what the summary does not already say.
        if value and _line(value) not in summary:
            parts.append(f"{label} {_line(value, _WORDING_CHARS)}")
    described = f": {'; '.join(parts)}" if parts else ""
    return f"{head}{described} (run {handle.run_id})"


def _finding_line(finding: Finding) -> str:
    unit = f" {_line(finding.unit, _WORDING_CHARS)}" if finding.unit else ""
    where = [_line(text, _WORDING_CHARS) for text in (finding.region, finding.period) if text]
    context = f", {', '.join(where)}" if where else ""
    return f"  - {_line(finding.name, _WORDING_CHARS)} = {_number(finding.value)}{unit}{context} (from {finding.source_handle})"


def _option(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _line(text, _OPTION_CHARS)


def _question_text(question: dict[str, Any], reason: str | None) -> list[str]:
    said: list[str] = []
    prompt = question.get("prompt")
    if isinstance(prompt, str) and prompt.strip() and _line(prompt) != _line(reason or ""):
        said.append(f"Question: {_line(prompt)}")
    options = question.get("options")
    if isinstance(options, list) and options:
        said.append("Options: " + " | ".join(_option(o) for o in options))
    items = question.get("items")
    if isinstance(items, list) and items:
        said.append("Items: " + " | ".join(_option(i) for i in items))
    inputs = question.get("inputs")
    if isinstance(inputs, list) and inputs:
        names = [_line(str(i.get("name")), _WORDING_CHARS) for i in inputs if isinstance(i, dict) and i.get("name")]
        if names:
            said.append("Inputs asked for: " + ", ".join(names))
    return said


def _waiting_advice(run: LedgerOpenRun) -> str:
    state = run.job_state or "unknown"
    if run.workflow:
        return {
            "ready": f"Its retrieval jobs are ready: continue it with resume_run({run.run_id}).",
            "running": f"Its retrieval jobs are still running; resume_run({run.run_id}) once they are ready.",
            "failed": (
                f"A retrieval job it waits on failed; resume_run({run.run_id}) submits it again, "
                "or ends the run once the step has been submitted too often."
            ),
            "unknown": (
                f"Its retrieval jobs have not been checked; resume_run({run.run_id}) collects them if they "
                "are ready, or the run waits again."
            ),
        }[state]
    jobs = {
        "ready": "Its retrieval jobs are ready.",
        "running": "Its retrieval jobs are still running.",
        "failed": "A retrieval job it waits on failed.",
        "unknown": "Its retrieval jobs have not been checked.",
    }[state]
    if state == "failed":
        return f"{jobs} An exploration cannot be resumed; asking again starts a new run."
    return f"{jobs} An exploration cannot be resumed; asking the same question again collects the saved job."


def _open_line(run: LedgerOpenRun) -> str:
    said = [f"- {run.run_id} {run.status}, {_line(run.mode_label)}:"]
    if run.reason:
        said.append(_line(run.reason))
    elif not run.has_result:
        said.append(NO_RESULT)
    if run.question:
        said.extend(_question_text(run.question, run.reason))
    if run.status == "waiting":
        said.append(_waiting_advice(run))
    return " ".join(said)


def _findings_count_line(group: LedgerFindings) -> str:
    noun = "finding" if len(group.findings) == 1 else "findings"
    return f"- {group.run_id}, {_line(group.mode_label)}: {len(group.findings)} {noun}"


def render_ledger(ledger: Ledger, *, max_handles: int = 30, max_finding_runs: int = 10) -> str:
    """The ledger as plain text lines under short headings, for a model.

    The newest ``max_handles`` handles are described in full; any older are
    shown one short line each, id and kind, so none is dropped. Likewise the
    newest ``max_finding_runs`` runs' findings are listed in full and older
    runs' as one line each with the run id and the number of findings. Areas,
    monitors and charts are always listed in full.
    """
    if max_handles < 0:
        raise ValueError("max_handles is at least 0")
    if max_finding_runs < 0:
        raise ValueError("max_finding_runs is at least 0")
    if ledger.is_empty():
        return EMPTY
    lines = [HEADING]
    if ledger.areas:
        lines.append("Areas:")
        lines.extend(_area_line(area) for area in ledger.areas)
    if ledger.products:
        lines.append("Products:")
        lines.extend(_product_line(product) for product in ledger.products)
    if ledger.monitors:
        areas = {area.handle: area for area in ledger.areas}
        lines.append("Monitors:")
        lines.extend(_monitors_line(found, areas) for found in ledger.monitors)
    if ledger.handles:
        older = max(0, len(ledger.handles) - max_handles)
        lines.append("Handles:" if not older else f"Handles ({older} oldest shown as id and kind only):")
        lines.extend(_handle_line(h, short=i < older) for i, h in enumerate(ledger.handles))
    if ledger.charts:
        lines.append("Charts:")
        lines.extend(_chart_line(chart) for chart in ledger.charts)
    if ledger.findings:
        older = max(0, len(ledger.findings) - max_finding_runs)
        lines.append("Findings:" if not older else f"Findings ({older} oldest runs shown as a count only):")
        for i, group in enumerate(ledger.findings):
            if i < older:
                lines.append(_findings_count_line(group))
                continue
            lines.append(f"- {group.run_id}, {_line(group.mode_label)}:")
            lines.extend(_finding_line(finding) for finding in group.findings)
    if ledger.open_runs:
        lines.append("Runs not done:")
        lines.extend(_open_line(run) for run in ledger.open_runs)
    return "\n".join(lines)
