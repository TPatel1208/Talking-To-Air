"""A result's export: its methods document, its provenance file, and one archive of both with its figures and data.

Everything is read as the requesting user through the harness's own stores:
the run and its trace, the saved result and the handle registry. Another
user's run is missing exactly as one that does not exist.

The methods document is assembled from that record and never written by a
model. It opens with the mode (a validated or draft workflow with its name
and version, or exploratory), then gives one sentence per step: the method
sentence the capability's template was filled with from the step's recorded
parameters, which carry the inputs the user confirmed. Then the products and
the rule that chose each, the masking recorded on each handle, and the
sources with their citations. A model-written interpretation is kept in a
section of its own, marked as interpretation.

The data service is asked only for the retrieved data's own lineage and for
dataset citations. Without it, both are left out and the documents say so.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tta_backend.domain.earthdata.provenance import merge_lineage
from tta_backend.harness.capabilities.sources.satellite import RULE_WORDING
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.lineage import dataset_handles, lineage
from tta_backend.harness.handles.model import HandleRecord, is_handle_id
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.results import RUN_ID_PATTERN, Result
from tta_backend.harness.results_store import ResultNotFound as SavedResultMissing
from tta_backend.harness.run_context import EarthdataSession
from tta_backend.harness.storage import Stores
from tta_backend.harness.trace import Run, RunNotFound, TraceEntry
from tta_backend.services.artifact_store import ArtifactStore, is_table_id
from tta_backend.services.chart_service import ChartLink, ChartService
from tta_backend.services.methods_export_service import _citation_text
from tta_backend.services.payload_figures import chart_rows, csv_bytes, figure_png

logger = logging.getLogger(__name__)

PROVENANCE_FORMAT = "tta-result-provenance"
PROVENANCE_VERSION = 1
RENDER_PREFIX = "render."
# Derived data files larger than this many rows are left out of the archive.
MAX_DATA_ROWS = 100_000

NO_RESULT = "No result was saved for this run."
NO_DATA_SERVICE = (
    "The data service could not be reached, so the retrieved data's own lineage and the dataset citations "
    "are not included."
)
INTERPRETATION_NOTE = (
    "This section was written by a language model. It interprets the findings above; it is not part of the method."
)


class ResultNotFound(Exception):
    """A run with no result for this user: missing, someone else's, or never saved."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"no result for run {run_id}")
        self.run_id = run_id


@dataclass(frozen=True)
class ProductChoice:
    """A data product behind a result and the rule that chose it, when the trace recorded one."""

    key: str
    rule: str | None
    rule_text: str | None

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "rule": self.rule, "rule_text": self.rule_text}


def rule_text(product: str, rule: str | None) -> str | None:
    """The rule that chose ``product`` as a plain sentence."""
    if rule is None:
        return None
    wording = RULE_WORDING.get(rule)
    if wording is None:
        return f"{product} was chosen by the rule {rule}."
    return f"{product} was chosen because {wording}."


@dataclass(frozen=True, kw_only=True)
class ResultRecord:
    """Everything a result's export reads, gathered once."""

    run: Run
    result: Result
    entries: tuple[TraceEntry, ...]
    # Every handle behind the result, by id: its own first, then their ancestors.
    records: dict[str, HandleRecord]
    missing: tuple[str, ...]
    products: tuple[ProductChoice, ...]
    # Chart and table ids, the result's own first.
    figures: tuple[str, ...]
    lineage: dict[str, Any]
    citations: tuple[dict[str, Any], ...]
    sources_note: str | None = None
    answers: tuple[dict[str, Any], ...] = field(default=())


# Reading the record


def _handle_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if is_handle_id(value) else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and is_handle_id(item)]
    return []


def _entry_handles(entry: TraceEntry) -> list[str]:
    """The handles a done step produced or drew."""
    if entry.kind != "use" or entry.status != "done" or not isinstance(entry.result, dict):
        return []
    found = _handle_ids(entry.result.get("handle"))
    if (entry.capability or "").startswith(RENDER_PREFIX):
        for value in entry.params.values():
            found.extend(_handle_ids(value))
        summary = entry.result.get("summary")
        if isinstance(summary, dict):
            found.extend(_handle_ids(summary.get("source_handles")))
    return found


def _entry_figures(entry: TraceEntry) -> list[str]:
    if not (entry.capability or "").startswith(RENDER_PREFIX) or not isinstance(entry.result, dict):
        return []
    charts = entry.result.get("charts")
    return [c for c in charts if isinstance(c, str)] if isinstance(charts, list) else []


async def walk_handles(stores: Stores, user_id: str, roots: Iterable[str]) -> tuple[dict[str, HandleRecord], list[str]]:
    """The user's records for ``roots`` and every ancestor, in that order, and the ids not found."""
    records: dict[str, HandleRecord] = {}
    missing: list[str] = []
    queue = list(dict.fromkeys(roots))
    seen = set(queue)
    while queue:
        handle = queue.pop(0)
        try:
            record = await stores.handles.get(user_id, handle)
        except HandleError:
            missing.append(handle)
            continue
        records[handle] = record
        for parent in record.parents:
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)
    return records, missing


def _rules(entries: Iterable[TraceEntry]) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """The rule each product was chosen by: per product, and per (product, handle made)."""
    by_product: dict[str, str] = {}
    by_handle: dict[tuple[str, str], str] = {}
    for entry in entries:
        result = entry.result if entry.kind == "use" and isinstance(entry.result, dict) else None
        if result is None:
            continue
        product, values = result.get("product"), result.get("values")
        rule = values.get("rule") if isinstance(values, dict) else None
        if not isinstance(product, str) or not isinstance(rule, str):
            continue
        by_product.setdefault(product, rule)
        handle = result.get("handle")
        if isinstance(handle, str):
            by_handle.setdefault((product, handle), rule)
    return by_product, by_handle


async def product_choices(
    stores: Stores, user_id: str, run: Run, records: dict[str, HandleRecord], entries: Sequence[TraceEntry],
) -> tuple[ProductChoice, ...]:
    """The products behind ``records``, each with the rule that chose it.

    The rule is the one recorded where that product's handle was made; a
    handle made in an earlier run of the conversation is looked up there.
    """
    products = list(dict.fromkeys(
        [str(e.result["product"]) for e in entries
         if e.kind == "use" and isinstance(e.result, dict) and isinstance(e.result.get("product"), str)]
        + [r.descriptor.product for r in records.values() if r.descriptor.product]
    ))
    by_product, by_handle = _rules(entries)

    def rule_of(product: str) -> str | None:
        return next((by_handle[(product, h)] for h in records if (product, h) in by_handle), None) or by_product.get(product)

    if any(rule_of(p) is None for p in products):
        for other in await stores.traces.list_runs(user_id, run.conversation_id):
            if other.id == run.id:
                continue
            _, made_there = _rules(await stores.traces.entries(user_id, other.id))
            for (product, handle), rule in made_there.items():
                if handle in records:
                    by_handle.setdefault((product, handle), rule)
    return tuple(ProductChoice(key=p, rule=rule_of(p), rule_text=rule_text(p, rule_of(p))) for p in products)


class _DataService:
    """The data service's lineage and citation lookups, for one export; anything it cannot answer is left out."""

    def __init__(self, earthdata: EarthdataSession | None) -> None:
        self._earthdata = earthdata
        self.failed = earthdata is None

    async def provenance(self, user_id: str, retrieval: list[str]) -> dict[str, Any]:
        if self._earthdata is None:
            return {"nodes": []}
        try:
            walks = [(h, await self._earthdata.call("get_provenance", {"handle": h})) for h in retrieval]
        except Exception as exc:
            logger.warning("export_provenance_unavailable", extra={"_error": type(exc).__name__})
            self.failed = True
            return {"nodes": []}
        return merge_lineage(walks)

    async def citations(self, datasets: list[str]) -> list[dict[str, Any]]:
        if self._earthdata is None:
            return []
        found = []
        for dataset in datasets:
            try:
                found.append(await self._earthdata.call("cite_dataset", {"dataset_handle": dataset}))
            except Exception as exc:
                logger.warning("export_citation_unavailable", extra={"_error": type(exc).__name__})
                self.failed = True
        return found


async def handle_lineage(
    stores: Stores, user_id: str, handles: Sequence[str], earthdata: EarthdataSession | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """The merged lineage of the user's ``handles``, their dataset citations, and whether the data service answered both."""
    service = _DataService(earthdata)
    merged = await lineage(stores.handles, user_id, list(handles), service.provenance) if handles else {"nodes": []}
    citations = await service.citations(dataset_handles(merged))
    return merged, citations, not service.failed


async def load_result(stores: Stores, user_id: str, run_id: str, earthdata: EarthdataSession | None) -> ResultRecord:
    """The record of the user's run ``run_id``; ResultNotFound when there is no result of theirs."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ResultNotFound(run_id)
    try:
        run = await stores.traces.get_run(user_id, run_id)
        result = await stores.results.get(user_id, run_id)
        entries = tuple(await stores.traces.entries(user_id, run_id))
    except (RunNotFound, SavedResultMissing):
        raise ResultNotFound(run_id) from None
    roots = [*result.handles, *(f.source_handle for f in result.findings)]
    for entry in entries:
        roots.extend(_entry_handles(entry))
    records, missing = await walk_handles(stores, user_id, roots)
    figures = list(result.charts)
    for entry in entries:
        figures.extend(_entry_figures(entry))
    products = await product_choices(stores, user_id, run, records, entries)
    merged, citations, answered = await handle_lineage(stores, user_id, list(records), earthdata)
    answers = tuple(
        {"step": e.step, "answer": e.result} for e in entries if e.kind == "ask" and e.status == "done" and e.result is not None
    )
    return ResultRecord(
        run=run, result=result, entries=entries, records=records, missing=tuple(missing), products=products,
        figures=tuple(dict.fromkeys(figures)), lineage=merged, citations=tuple(citations),
        sources_note=None if answered else NO_DATA_SERVICE, answers=answers,
    )


# The methods document


def _mode_line(record: ResultRecord) -> str:
    mode = record.run.mode
    if mode.workflow is None:
        return "**Mode:** Exploratory. These steps were chosen during the conversation; no validated workflow was run."
    if mode.kind == "draft":
        return (f"**Mode:** Draft workflow {mode.workflow.name}, version {mode.workflow.version}. "
                "The workflow has not been validated.")
    return f"**Mode:** Validated workflow {mode.workflow.name}, version {mode.workflow.version}."


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(", ", ": "), sort_keys=True, default=str)


def _step_lines(entries: Sequence[TraceEntry]) -> tuple[list[str], list[str]]:
    done: list[str] = []
    other: list[str] = []
    # A step that waited for its retrieval, or was retried, and then finished is listed once, as done.
    finished = {(e.step, e.capability) for e in entries if e.status == "done" and e.step is not None}
    for entry in entries:
        if entry.kind not in ("use", "agent", "stop"):
            continue
        if entry.status in ("failed", "waiting") and (entry.step, entry.capability) in finished:
            continue
        name = f"`{entry.step}`" if entry.step else f"`{entry.capability or entry.kind}`"
        method = entry.result.get("method") if isinstance(entry.result, dict) else None
        if entry.status == "done" and entry.kind == "use":
            done.append(f"{name}: {method}" if isinstance(method, str) and method else f"{name}: {entry.capability} ran.")
        elif entry.status == "skipped":
            other.append(f"- {name} was skipped.")
        elif entry.status in ("failed", "waiting"):
            category = f" ({entry.failure_category.value})" if entry.failure_category is not None else ""
            other.append(f"- {name} did not complete{category}: {entry.reason or 'no reason was recorded'}")
    return done, other


def methods_markdown(record: ResultRecord) -> str:
    """The methods document for ``record``, assembled from its lineage and trace."""
    result, run = record.result, record.run
    lines = [f"# Methods: {run.mode.label()}", "", _mode_line(record), ""]
    lines.append(f"Run `{run.id}`, harness release {result.harness_release}, started {run.started.isoformat()}"
                 + (f", ended {run.ended.isoformat()}." if run.ended else "."))
    if result.status != "done":
        lines.append(f"Status: {result.status}. {result.reason or ''}".rstrip())
    lines += ["", "## Steps", ""]
    done, other = _step_lines(record.entries)
    lines += [f"{i}. {line}" for i, line in enumerate(done, 1)] or ["No step completed."]
    if other:
        lines += ["", "Steps that did not run or did not complete:", "", *other]
    if record.answers:
        lines += ["", "## Answers confirmed by the user", ""]
        lines += [f"- `{a['step'] or 'question'}`: {_compact(a['answer'])}" for a in record.answers]
    lines += ["", "## Data products", ""]
    lines += [f"- {p.key}: {p.rule_text} (rule `{p.rule}`)" if p.rule else f"- {p.key}: no choosing rule was recorded."
              for p in record.products] or ["No data product was recorded."]
    lines += ["", "## Masking", ""]
    masked = [(h, r) for h, r in record.records.items() if r.descriptor.masking]
    lines += [f"- `{h}`: {_compact(r.descriptor.masking)}" for h, r in masked] or [
        "No masking was recorded on the handles behind this result."]
    lines += ["", "## Sources", ""]
    retrieved = [r for r in record.records.values() if r.origin == "retrieval"]
    for source in retrieved:
        kind = ", ".join(x for x in (source.kind, source.descriptor.product) if x)
        lines.append(f"- `{source.id}` ({kind}): {source.descriptor.summary}")
    if not retrieved:
        lines.append("No retrieved data is recorded behind this result.")
    if record.citations:
        lines += ["", "### Citations", ""]
        lines += [f"- {_citation_text(c)}" for c in record.citations]
    if record.sources_note:
        lines += ["", record.sources_note]
    if record.missing:
        lines += ["", "Handles no longer recorded: " + ", ".join(f"`{h}`" for h in record.missing) + "."]
    if result.findings:
        lines += ["", "## Findings", ""]
        for finding in result.findings:
            where = ", ".join(x for x in (finding.region, finding.period) if x)
            unit = f" {finding.unit}" if finding.unit else ""
            value = f"{finding.value:.6g}" if isinstance(finding.value, float) else str(finding.value)
            lines.append(f"- {finding.name}: {value}{unit}" + (f" ({where})" if where else "")
                         + f", from `{finding.source_handle}`")
    if result.disclosures:
        lines += ["", "## Disclosures", "", *(f"- {d}" for d in result.disclosures)]
    if result.gaps:
        lines += ["", "## Gaps", "", *(f"- `{g.step}`: {g.reason}" for g in result.gaps)]
    if result.interpretation:
        lines += ["", "## Interpretation (model-written)", "", f"*{INTERPRETATION_NOTE}*", "", result.interpretation]
    return "\n".join(lines) + "\n"


# The provenance file


def _handle_json(record: HandleRecord) -> dict[str, Any]:
    descriptor = record.descriptor
    return {
        "id": record.id, "kind": record.kind, "origin": record.origin, "summary": descriptor.summary,
        "product": descriptor.product, "quantity": descriptor.quantity, "unit": descriptor.unit,
        "region": descriptor.region, "period": None if descriptor.period is None else descriptor.period.model_dump(),
        "masking": descriptor.masking, "operation": record.operation, "parameters": record.parameters,
        "parents": list(record.parents), "run_id": record.run_id, "release": record.release,
        "created_at": record.created_at.isoformat(),
    }


def _operation_json(entry: TraceEntry) -> dict[str, Any]:
    result = entry.result if isinstance(entry.result, dict) else {}
    return {
        "seq": entry.seq, "parent_seq": entry.parent_seq, "kind": entry.kind, "step": entry.step,
        "capability": entry.capability, "status": entry.status, "parameters": entry.params,
        "method": result.get("method"), "output": result.get("handle"), "product": result.get("product"),
        "failure_category": None if entry.failure_category is None else entry.failure_category.value,
        "reason": entry.reason, "started": entry.started.isoformat(), "seconds": entry.seconds,
    }


def provenance_json(record: ResultRecord) -> dict[str, Any]:
    """The machine-readable record of how the result was made."""
    result, run = record.result, record.run
    mode = run.mode
    return {
        "format": PROVENANCE_FORMAT,
        "version": PROVENANCE_VERSION,
        "run_id": run.id,
        "status": result.status,
        "reason": result.reason,
        "harness_release": result.harness_release,
        "started": run.started.isoformat(),
        "ended": None if run.ended is None else run.ended.isoformat(),
        "mode": {**mode.model_dump(mode="json"), "label": mode.label()},
        "workflow": None if mode.workflow is None else mode.workflow.model_dump(),
        "products": [p.to_json() for p in record.products],
        "sources": [_handle_json(r) for r in record.records.values() if r.origin == "retrieval"],
        "citations": list(record.citations),
        "operations": [_operation_json(e) for e in record.entries],
        "answers": list(record.answers),
        "masking": [{"handle": h, "masking": r.descriptor.masking} for h, r in record.records.items() if r.descriptor.masking],
        "handles": [_handle_json(r) for r in record.records.values()],
        "missing_handles": list(record.missing),
        "lineage": record.lineage,
        "findings": [f.model_dump(mode="json") for f in result.findings],
        "disclosures": list(result.disclosures),
        "gaps": [g.model_dump() for g in result.gaps],
        "figures": list(record.figures),
        "interpretation": None if result.interpretation is None else {"text": result.interpretation, "written_by": "model"},
        "notes": [record.sources_note] if record.sources_note else [],
    }


# The archive


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()[:60] or "item"


@dataclass
class _Archive:
    files: dict[str, bytes] = field(default_factory=dict)
    left_out: list[str] = field(default_factory=list)

    def add(self, folder: str, name: str, suffix: str, content: bytes) -> None:
        base = f"{folder}/{len([f for f in self.files if f.startswith(folder + '/')]) + 1:02d}-{_slug(name)}"
        self.files[f"{base}.{suffix}"] = content

    def zip(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in self.files.items():
                archive.writestr(name, content)
            if self.left_out:
                archive.writestr("data/NOT_INCLUDED.txt", "\n".join(self.left_out) + "\n")
        return buffer.getvalue()


def _frame_rows(frame: Any) -> list[list[Any]] | None:
    if not hasattr(frame, "to_csv") or len(frame) > MAX_DATA_ROWS:
        return None
    return [list(frame.columns), *frame.astype(object).where(frame.notna(), None).values.tolist()]


class ResultExporter:
    """A result's methods document, provenance file and archive, for one user's runs."""

    def __init__(
        self,
        stores: Stores,
        *,
        chart_service: ChartService,
        artifacts: ArtifactStore,
        payload_root: str | None,
        earthdata: EarthdataSession | None = None,
    ) -> None:
        self._stores = stores
        self._charts = chart_service
        self._artifacts = artifacts
        self._payloads = PayloadStore(payload_root) if payload_root else None
        self._earthdata = earthdata

    async def record(self, user_id: str, run_id: str) -> ResultRecord:
        return await load_result(self._stores, user_id, run_id, self._earthdata)

    async def methods(self, user_id: str, run_id: str) -> str:
        return methods_markdown(await self.record(user_id, run_id))

    async def provenance(self, user_id: str, run_id: str) -> dict[str, Any]:
        return provenance_json(await self.record(user_id, run_id))

    async def archive(self, user_id: str, run_id: str) -> bytes:
        """The result's zip: methods.md, provenance.json, figures/ and data/."""
        record = await self.record(user_id, run_id)
        archive = _Archive()
        archive.files["methods.md"] = methods_markdown(record).encode("utf-8")
        archive.files["provenance.json"] = json.dumps(provenance_json(record), indent=2, default=str).encode("utf-8")
        charts = await self._charts.get_owned_charts(list(record.figures), user_id)
        for figure in record.figures:
            payload = charts.get(figure)
            if payload is not None:
                await self._chart_files(archive, figure, payload)
            elif is_table_id(figure):
                await self._table_file(archive, figure, user_id)
            else:
                archive.left_out.append(f"Figure {figure}: it is no longer stored.")
        for handle, handle_record in record.records.items():
            if handle in record.result.handles:
                await self._handle_file(archive, handle_record)
        return await asyncio.to_thread(archive.zip)

    async def _chart_files(self, archive: _Archive, chart_id: str, payload: dict[str, Any]) -> None:
        title = str(payload.get("title") or chart_id)
        try:
            png = await figure_png(payload)
        except Exception as exc:
            logger.warning("export_figure_failed", extra={"_chart_id": chart_id, "_error": type(exc).__name__})
            png = None
        if png is not None:
            archive.add("figures", title, "png", png)
        else:
            archive.left_out.append(f"Figure {chart_id} ({payload.get('type')}): this chart type is not drawn for export.")
        rows = await chart_rows(payload)
        if rows is not None:
            archive.add("data", title, "csv", csv_bytes(rows))

    async def _table_file(self, archive: _Archive, artifact_id: str, user_id: str) -> None:
        try:
            reference = await self._artifacts.reference(artifact_id, user_id)
            chunks = [chunk async for chunk in self._artifacts.iter_csv_chunks(artifact_id, user_id)]
        except KeyError:
            archive.left_out.append(f"Table {artifact_id}: it is no longer stored.")
            return
        archive.add("data", reference.title or artifact_id, "csv", b"".join(chunks))

    async def _handle_file(self, archive: _Archive, record: HandleRecord) -> None:
        if record.origin != "derived" or record.kind == "scalar":
            return
        if record.kind == "grid":
            archive.left_out.append(f"Handle {record.id} (grid): grids are not included; provenance.json records how it was made.")
            return
        if self._payloads is None:
            archive.left_out.append(f"Handle {record.id} ({record.kind}): the payload store is not configured here.")
            return
        try:
            data = await self._payloads.open(record)
        except HandleError as error:
            archive.left_out.append(f"Handle {record.id} ({record.kind}): its data is {error.reason.replace('_', ' ')}.")
            return
        if record.kind == "polygons":
            archive.add("data", f"{record.id}-{record.descriptor.summary}", "geojson", json.dumps(data).encode("utf-8"))
            return
        rows = _frame_rows(data)
        if rows is None:
            archive.left_out.append(f"Handle {record.id} ({record.kind}): more than {MAX_DATA_ROWS} rows.")
            return
        archive.add("data", f"{record.id}-{record.descriptor.summary}", "csv", csv_bytes(rows))


# A chart a harness run drew


async def _chart_records(stores: Stores, user_id: str, link: ChartLink) -> tuple[dict[str, HandleRecord], list[str]]:
    """The records behind every handle the chart renders, and those of its handles that are the user's."""
    records, _ = await walk_handles(stores, user_id, link.handles)
    return records, [h for h in link.handles if h in records]


async def chart_provenance(
    stores: Stores, user_id: str, link: ChartLink, earthdata: EarthdataSession | None,
) -> dict[str, Any]:
    """The provenance pane's view of a harness chart: the merged lineage of all its
    handles, the run and mode that drew it, and the products behind them, each
    with the rule that chose it. ``product`` is the first handle's, else the
    first behind the chart."""
    records, roots = await _chart_records(stores, user_id, link)
    merged, _, answered = await handle_lineage(stores, user_id, roots, earthdata)
    product = None
    products: list[dict[str, Any]] = []
    try:
        run = await stores.traces.get_run(user_id, link.run_id)
        entries = await stores.traces.entries(user_id, link.run_id)
        choices = await product_choices(stores, user_id, run, records, entries)
        behind = {r.descriptor.product for r in records.values() if r.descriptor.product}
        shown = [c for c in choices if c.key in behind] or list(choices)
        own = records[link.handle].descriptor.product if link.handle in records else None
        chosen = next((c for c in shown if c.key == own), None) or next(iter(shown), None)
        product = None if chosen is None else chosen.to_json()
        products = [c.to_json() for c in shown]
    except RunNotFound:
        pass
    view = {
        **merged, "run_id": link.run_id, "mode": link.mode.view(), "product": product, "products": products,
        "handle": link.handle, "handles": list(link.handles),
    }
    if not answered:
        view["note"] = NO_DATA_SERVICE
    return view


async def chart_citations(
    stores: Stores, user_id: str, link: ChartLink, earthdata: EarthdataSession | None,
) -> list[dict[str, Any]]:
    """The dataset citations behind all of a harness chart's handles."""
    _, roots = await _chart_records(stores, user_id, link)
    _, citations, _ = await handle_lineage(stores, user_id, roots, earthdata)
    return citations


__all__ = [
    "NO_RESULT",
    "ProductChoice",
    "ResultExporter",
    "ResultNotFound",
    "ResultRecord",
    "chart_citations",
    "chart_provenance",
    "load_result",
    "methods_markdown",
    "provenance_json",
    "rule_text",
]
