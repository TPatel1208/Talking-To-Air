"""Lineage, citations and recorded reliability facts for handles; small values only, no handle minted, nothing retrieved."""
from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import Field, JsonValue

from tta_backend.datasets.registry import COLLECTIONS_WITHOUT_QUALITY_FLAG, load_registry
from tta_backend.domain.earthdata.provenance import merge_lineage
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.registry import Group
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.handles.lineage import dataset_handles, lineage
from tta_backend.harness.handles.model import KINDS, HandleRecord

# Serialized bytes a list of steps or citations may take, well inside what a result may hold.
LIST_BUDGET = 2400
MAX_PARAMETERS = 6
MAX_TEXT = 80
MAX_NOTE = 400
MAX_CITATION_TEXT = 200
MAX_CITATIONS_PER_DATASET = 2
NO_EVIDENCE = (
    "No reliability evidence is recorded for this handle: no quality masking, valid-data rate or "
    "product maturity was recorded, so nothing can be said about it."
)
NO_DATASETS = "No dataset was found behind these handles, so there is nothing to cite."
PLAN_ONLY = (
    "Only the masking plan is recorded for this handle: no quality status, valid-data rate or product "
    "maturity was recorded, so nothing can be said about how reliable it is."
)
PLAN_NOTE = (
    "The masking plan was written before the data was opened and is applied whenever the grid is read; "
    "what masking it left (quality status and valid-data rates) is not recorded on this handle."
)
CONNECTOR = "earthdata"

GROUP = Group(
    name="explain",
    description="Where a result came from and how far to trust it: lineage, dataset citations, recorded reliability facts.",
    notes=(
        "All three take handles and return small facts; none retrieves data or makes a handle. "
        "lineage lists the steps that made the handles, earliest first, with the nearest steps kept when it is long. "
        "citations gives the dataset citations behind the handles, once per dataset. "
        "reliability reports only what was recorded for one handle: the quality rule applied or why none was, "
        "valid-data and quality pass rates, and the product's maturity. A retrieved grid records only its "
        "masking plan (masking_plan), which is not a measurement. It gives facts, never a verdict; "
        "when it says nothing is recorded, say so rather than judging the data."
    ),
)


class LineageParams(Params):
    handles: Annotated[list[str], handle_param(*KINDS)] = Field(
        min_length=1, max_length=20, description="One or more handle ids, of any kind."
    )


class CitationsParams(Params):
    handles: Annotated[list[str], handle_param(*KINDS)] = Field(
        min_length=1, max_length=20, description="One or more handle ids, of any kind."
    )


class ReliabilityParams(Params):
    handle: Annotated[str, handle_param(*KINDS)] = Field(description="The handle id to report recorded reliability facts for.")


def _brief(value: Any, limit: int = MAX_TEXT) -> JsonValue:
    """A scalar as is; text and nested values cut to ``limit`` characters."""
    if value is None or isinstance(value, bool | int | float):
        return value
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _key_parameters(parameters: Any) -> dict[str, JsonValue]:
    if not isinstance(parameters, dict):
        return {}
    return {str(key): _brief(value) for key, value in list(parameters.items())[:MAX_PARAMETERS]}


def _step(node: dict[str, Any]) -> dict[str, JsonValue]:
    """One lineage node as a compact step."""
    events = node.get("events") or []
    newest = events[-1] if events else {}
    detail = newest.get("detail") if isinstance(newest, dict) else None
    detail = detail if isinstance(detail, dict) else {}
    if newest and newest.get("event_type") == "operation":
        operation = detail.get("operation")
        parameters: Any = detail.get("parameters")
        parents = [str(parent) for parent in detail.get("parents") or []]
    else:
        operation = newest.get("event_type") if isinstance(newest, dict) else None
        parameters = detail
        parents = []
    step: dict[str, JsonValue] = {"handle": node["handle"], "kind": node.get("kind"), "operation": operation}
    key = _key_parameters(parameters)
    if key:
        step["parameters"] = key
    if parents:
        step["parents"] = list(parents)
    if node.get("note"):
        step["note"] = _brief(node["note"], MAX_NOTE)
    return step


def _size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":")).encode())


def _fit(items: list[Any], *, keep: str) -> tuple[list[Any], int]:
    """``items`` cut to the list budget, dropping from the front (``keep="last"``) or the end; and how many were left out."""
    kept = list(items)
    while len(kept) > 1 and _size(kept) > LIST_BUDGET:
        kept.pop(0 if keep == "last" else -1)
    return kept, len(items) - len(kept)


async def _lineage_of(call: Call[Any], handles: list[str]) -> dict[str, Any]:
    async def provenance(user_id: str, retrieval: list[str]) -> dict[str, Any]:
        walks = [(handle, await call.earthdata.call("get_provenance", {"handle": handle})) for handle in retrieval]
        return merge_lineage(walks)

    return await lineage(call.env.context.storage.handles, call.user_id, handles, provenance)


async def _lineage(call: Call[LineageParams]) -> CapabilityResult:
    merged = await _lineage_of(call, [record.id for record in call.handles("handles")])
    steps = [_step(node) for node in merged["nodes"]]
    kept, omitted = _fit(steps, keep="last")
    values: dict[str, JsonValue] = {"steps": list[JsonValue](kept), "total_steps": len(steps), "omitted_steps": omitted}
    disclosures = (f"The lineage is long: the {omitted} earliest step(s) are left out and the nearest are shown.",) if omitted else ()
    return call.done(summary={"steps": len(steps)}, values=values, disclosures=disclosures)


lineage_capability = Capability(
    name="explain.lineage",
    description="How handles were made: the merged steps behind them, earliest first, with each step's operation, key parameters and parents.",
    params=LineageParams,
    output_kind=None,
    label="Tracing how results were made",
    method="The lineage of {handles} was read, giving {summary.steps} step(s).",
    retrieves=False,
    retry_safe=True,
    # Needed only for handles the earthdata service made.
    connector=CONNECTOR,
    connector_required=False,
    run=_lineage,
)


def _citation(record: dict[str, Any]) -> dict[str, JsonValue]:
    compact: dict[str, JsonValue] = {
        key: _brief(record.get(key), MAX_CITATION_TEXT)
        for key in ("handle", "concept_id", "doi", "doi_authority", "reference_citation_count")
        if record.get(key) is not None
    }
    texts = record.get("collection_citations")
    if isinstance(texts, list) and texts:
        compact["collection_citations"] = [_brief(text, MAX_CITATION_TEXT) for text in texts[:MAX_CITATIONS_PER_DATASET]]
    return compact


async def _citations(call: Call[CitationsParams]) -> CapabilityResult:
    merged = await _lineage_of(call, [record.id for record in call.handles("handles")])
    datasets = dataset_handles(merged)
    found: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    for dataset in datasets:
        record = await call.earthdata.call("cite_dataset", {"dataset_handle": dataset})
        citation = _citation({"handle": dataset, **record})
        # The same dataset reached under two handles is cited once.
        identity = str(citation.get("doi") or citation.get("concept_id") or dataset)
        if identity not in seen:
            seen.add(identity)
            found.append(citation)
    kept, omitted = _fit(found, keep="first")
    values: dict[str, JsonValue] = {"citations": list[JsonValue](kept), "omitted_citations": omitted}
    disclosures: list[str] = []
    if not found:
        disclosures.append(NO_DATASETS)
    if omitted:
        disclosures.append(f"{omitted} further citation(s) are left out to keep the result small.")
    return call.done(summary={"datasets": len(found)}, values=values, disclosures=tuple(disclosures))


citations = Capability(
    name="explain.citations",
    description="The dataset citations behind handles, once per dataset.",
    params=CitationsParams,
    output_kind=None,
    label="Looking up dataset citations",
    method="The dataset citations behind {handles} were read, giving {summary.datasets} dataset(s).",
    retrieves=False,
    retry_safe=True,
    # Needed only for handles the earthdata service made.
    connector=CONNECTOR,
    connector_required=False,
    run=_citations,
)

_RATE_KEYS = ("valid_fraction", "valid_cells", "region_cells", "qa_pass_rate")
_MASKING_KEYS = ("variable", "quality_flag", "qa_status", "qa_source", "registry_entry")
# What a retrieval record's masking holds: the plan written before the data is opened.
_PLAN_KEYS = ("variable", "quality_flag", "geometry")


def _masking(record: HandleRecord) -> dict[str, Any]:
    return record.descriptor.masking if isinstance(record.descriptor.masking, dict) else {}


def _recorded_rates(record: HandleRecord) -> dict[str, JsonValue]:
    if record.origin == "retrieval":
        return {}
    masking = _masking(record)
    found: dict[str, JsonValue] = {}
    for key in _RATE_KEYS:
        for source in (masking, record.parameters):
            value = source.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                found[key] = value
                break
    return found


def _recorded_masking(record: HandleRecord) -> dict[str, JsonValue]:
    """The masking a derived handle records as applied; a retrieval record holds only a plan."""
    if record.origin == "retrieval":
        return {}
    masking = _masking(record)
    return {key: _brief(masking[key]) for key in _MASKING_KEYS if masking.get(key) is not None}


def _masking_plan(record: HandleRecord) -> dict[str, JsonValue]:
    if record.origin != "retrieval":
        return {}
    masking = _masking(record)
    return {key: _brief(masking[key]) for key in _PLAN_KEYS if masking.get(key) is not None}


def _registry_facts(product: str | None) -> dict[str, JsonValue]:
    cfg = load_registry().get(product) if product else None
    if cfg is None:
        return {}
    facts: dict[str, JsonValue] = {"maturity": cfg.maturity}
    if cfg.maturity_note:
        facts["maturity_note"] = _brief(" ".join(cfg.maturity_note.split()), MAX_NOTE)
    if cfg.quality_flag_var:
        facts["registry_quality_flag"] = cfg.quality_flag_var.rsplit("/", 1)[-1]
    elif product in COLLECTIONS_WITHOUT_QUALITY_FLAG:
        facts["registry_quality_flag"] = None
        facts["no_quality_flag_reason"] = _brief(COLLECTIONS_WITHOUT_QUALITY_FLAG[str(product)], MAX_NOTE)
    return facts


async def _reliability(call: Call[ReliabilityParams]) -> CapabilityResult:
    record = call.handle("handle")
    masking = _recorded_masking(record)
    plan = _masking_plan(record)
    rates = _recorded_rates(record)
    registry = _registry_facts(record.descriptor.product)
    # "unknown" is the registry's default and says nothing was checked, so it is not evidence.
    # A masking plan is not evidence either: it says what will be applied, not what it left.
    has_evidence = bool(masking or rates or registry.get("maturity") not in (None, "unknown") or "no_quality_flag_reason" in registry)
    summary: dict[str, JsonValue] = {"product": record.descriptor.product, "recorded": has_evidence}
    if masking:
        summary["masking"] = masking
    if plan:
        summary["masking_plan"] = plan
    summary.update(registry)
    values: dict[str, JsonValue] = {"subject": record.id, "has_evidence": has_evidence}
    disclosures: list[str] = []
    if plan:
        disclosures.append(PLAN_NOTE)
    if not has_evidence:
        statement = PLAN_ONLY if plan else NO_EVIDENCE
        values["statement"] = statement
        disclosures.append(statement)
    return call.done(
        quality=rates or None,
        summary=summary,
        values=values,
        disclosures=tuple(disclosures),
    )


reliability = Capability(
    name="explain.reliability",
    description="The reliability facts recorded for a handle: quality masking applied or why none, valid-data and pass rates, product maturity.",
    params=ReliabilityParams,
    output_kind=None,
    label="Reading recorded reliability facts",
    method="The recorded reliability facts for {handle} were read.",
    retrieves=False,
    retry_safe=True,
    run=_reliability,
)

CAPABILITIES = (lineage_capability, citations, reliability)
