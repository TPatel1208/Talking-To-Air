"""Gridded satellite data from the earthdata service: get data, point series, discovery.

``get_data`` is two-phase. ``begin`` chooses the product, finds the dataset,
checks coverage, gates the size and submits; it returns the job reference,
or settles the call without submitting (a size question for the user, or no
data). ``complete`` waits for the job within ``Limits``, records the
retrieval handle and measures its quality through the grid opener.

Job reference (what ``begin`` saves and ``complete`` reads)::

    {"product": "TEMPO_NO2", "rule": "approved_preference",
     "dataset": "dataset_...", "aoi": "aoi_...", "variable": "vertical_column_troposphere",
     "jobs": [{"job": "job_...", "day": "2024-06-01" | null}],
     "dropped": ["2024-06-03"], "skipped": [{"product": "...", "reason": "...", "detail": "..."}],
     "estimated_bytes": 123 | null}

``day`` is the date a per-day job retrieved, or null for one job over the
whole request. Still running after the timeout: ``complete`` raises
``JobStillRunning`` (a retryable ``source_unavailable`` failure holding the
job and the job ids still running). The runner keeps the job reference and
calls ``complete`` again later with it; ``complete`` never submits, so nothing
is retrieved twice.

The four no-data situations end as ``insufficient_evidence`` with
``values.situation`` one of ``outside_record``, ``not_published_yet``,
``gap`` or ``too_little_valid``, and fixed wording naming it. For a subject
date or range, a gap or too little valid data also lists in
``values.nearest_dates`` up to two nearby dates that have granules. Comparison
days without granules, or with nothing valid after masking, are dropped and
listed in ``values.dropped_dates`` (the second kind also in
``values.dropped_after_masking``).

A paused job is cancelled at the service before the step fails, and a job the
provider reports cancelled ends the step with that reason; neither is retried.
A job that failed at the provider, or that the service no longer holds, raises
``JobFailed`` (``source_unavailable``); a retry begins the step again with a
new ``attempt``, so the dead job is not answered again.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import Field, JsonValue, model_validator

from tta_backend.datasets.registry import CollectionConfig, load_registry
from tta_backend.domain.earthdata.opening import combine, quality_summary
from tta_backend.domain.earthdata.retrieval import (
    PAUSED_STATUS,
    POLL_TRANSIENT_FAILURE_LIMIT,
    TERMINAL_STATUSES,
    annotate_paused,
    aoi_handle_from_result,
    bbox_location,
    dataset_handle_from_search,
    estimated_bytes,
    day_time_range,
    granule_count,
    nearby_dates,
    pinned_companion_variables,
    request_variables,
    search_hits,
    MAX_UNSIZED_GRANULES,
    SizeDecision,
    size_gate,
    span_time_range,
)
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.failures import UNKNOWN_PRODUCT, JobFailed, JobStillRunning, failure
from tta_backend.harness.capabilities.grids import open_grid
from tta_backend.harness.capabilities.products import (
    BBox,
    DateRange,
    Resolution,
    _match_named,
    resolve_product,
)
from tta_backend.harness.capabilities.registry import Group
from tta_backend.harness.capabilities.result import CapabilityResult, JobRef, Question
from tta_backend.harness.capabilities.shapes import series_frame
from tta_backend.harness.capabilities.sources.areas import area_facts
from tta_backend.harness.handles.model import Descriptor, HandleRecord, Period
from tta_backend.harness.run_context import Limits
from tta_backend.harness.trace import FailureCategory, StepFailure

GROUP = Group(
    name="sources.satellite",
    description="Gridded satellite data: retrieve a quantity over an area and dates, a point series, or look up what exists.",
    notes=(
        "get_data takes a quantity (or the product or instrument the user named), an area handle "
        "from sources.areas.resolve, and dates with their role: subject (the day being studied, never "
        "substituted), comparison (days without data are dropped and listed) or range (gaps are "
        "allowed and counted). The product is chosen for you and named on every result with the rule "
        "that chose it; pass product to keep using it. Set confirmed only after the user agreed to a "
        "large retrieval. A result with status insufficient_evidence names why there is no data: "
        "outside_record, not_published_yet, gap or too_little_valid. Do not retry it with other dates "
        "or another product unless the user asks. When the user asks for values each day or each hour, "
        "set cadence, so a monthly-only product is refused instead of read as daily values. search, "
        "describe and check_availability are cheap lookups that retrieve nothing."
    ),
)

CONNECTOR = "earthdata"
MAX_DATES = 31
_NAME_CHARS = 120
_DETAIL_CHARS = 160
_LISTED = 10

Role = Literal["subject", "comparison", "range"]
Cadence = Literal["hourly", "daily", "monthly"]
Approved = list[str] | Literal["registry_default"]


class DateSpan(Params):
    """Every day from start to end, inclusive."""

    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("end is before start")
        return self


DateList = Annotated[list[date], Field(min_length=1, max_length=MAX_DATES)]


def _registry_quantities() -> tuple[str, ...]:
    """Every distinct registry quantity, sorted. Fails at import on an empty or malformed registry."""
    names = {cfg.quantity for cfg in load_registry().values() if cfg.quantity is not None}
    if not names:
        raise ValueError("the dataset registry names no quantity, so sources.satellite cannot offer one")
    for name in names:
        if any(ch.isspace() or ch.isupper() for ch in name):
            raise ValueError(f"registry quantity {name!r} must be lower_snake with no spaces or capitals")
    return tuple(sorted(names))


QUANTITIES = _registry_quantities()
_QUANTITY_DESCRIPTION = "Canonical quantity, one of: " + ", ".join(QUANTITIES) + "."
_PRODUCT_KEYS = tuple(sorted(load_registry()))
_PRODUCT_DESCRIPTION = "Registered product key, one of: " + ", ".join(_PRODUCT_KEYS) + "."


class _ProductChoice(Params):
    quantity: str | None = Field(default=None, max_length=100, description=_QUANTITY_DESCRIPTION)
    named: str | None = Field(default=None, max_length=100, description="A product or instrument the user named.")
    approved: Approved = Field(default="registry_default", description="The workflow's approved products in order, or registry_default.")
    product: str | None = Field(default=None, max_length=100, description="The product this run already uses for the quantity.")
    cadence: Cadence | None = Field(
        default=None,
        description=(
            "The finest cadence the analysis needs: set daily when the user asks for each day or daily "
            "values, hourly for hours. A product too coarse for it is refused, never used instead."
        ),
    )

    @model_validator(mode="after")
    def _something_to_choose_by(self) -> Self:
        if self.quantity is None and self.named is None and self.product is None:
            raise ValueError("give a quantity, a named product or instrument, or a pinned product")
        return self


class GetDataParams(_ProductChoice):
    area: Annotated[str, handle_param("polygons")]
    dates: date | DateSpan | DateList
    date_role: Role
    confirmed: bool = Field(default=False, description="Set only from the user's answer to a size question.")
    min_valid_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _dates_fit_role(self) -> Self:
        if self.date_role == "range" and not isinstance(self.dates, DateSpan):
            raise ValueError("a range role takes a start and end")
        if self.date_role == "subject" and isinstance(self.dates, list):
            raise ValueError("a subject role takes one date or a range")
        return self


class PointSeriesParams(_ProductChoice):
    area: Annotated[str, handle_param("polygons")]
    dates: DateSpan


class SearchParams(Params):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=_LISTED)


class DescribeParams(Params):
    product: str = Field(min_length=1, max_length=100, description=_PRODUCT_DESCRIPTION)


class AvailabilityParams(Params):
    product: str = Field(min_length=1, max_length=100, description=_PRODUCT_DESCRIPTION)
    area: Annotated[str, handle_param("polygons")]
    dates: date | DateSpan | DateList


# Fixed wording for each no-data situation; filled only from observed facts.
NO_DATA_REASONS: dict[str, str] = {
    "outside_record": "No data: {dates} is outside the record of every product that could serve {quantity} over {area}.",
    "not_published_yet": "No data yet: {quantity} for {dates} is not yet published by any product that could serve {area}.",
    "gap": "No data: {product} has no observations over {area} for {dates}; this is a gap in its record.",
    "too_little_valid": (
        "Too little valid data: after quality masking, {product} kept {percent} of {area} for {dates}."
    ),
}

RULE_WORDING: dict[str, str] = {
    "pinned": "this run already uses it for this quantity",
    "user_named": "it was named in the request",
    "only_candidate": "it is the only candidate that covers the area and dates",
    "approved_preference": "it is the first of the workflow's approved products that covers the area and dates",
    "highest_rank": "it is the registry's preferred product for this quantity that covers the area and dates",
    "highest_rank_tie_by_key": "it shares the registry's top rank for this quantity and comes first by name",
}


def _short(text: str, limit: int = _NAME_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── Dates ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Dates:
    kind: Literal["date", "range", "list"]
    start: date
    end: date
    listed: tuple[date, ...] = ()

    @classmethod
    def of(cls, value: date | DateSpan | list[date]) -> _Dates:
        if isinstance(value, DateSpan):
            return cls("range", value.start, value.end)
        if isinstance(value, list):
            days = tuple(sorted(set(value)))
            return cls("list", days[0], days[-1], days)
        return cls("date", value, value, (value,))

    def resolver_dates(self) -> date | DateRange | list[date]:
        if self.kind == "range":
            return DateRange(self.start, self.end)
        if self.kind == "list":
            return list(self.listed)
        return self.start

    def every_day(self) -> list[date]:
        if self.kind == "range":
            return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]
        return list(self.listed)

    def text(self) -> str:
        if self.kind == "range":
            return f"{self.start.isoformat()} to {self.end.isoformat()}"
        if self.kind == "list" and len(self.listed) > 1:
            return f"{len(self.listed)} days from {self.start.isoformat()} to {self.end.isoformat()}"
        return self.start.isoformat()


def _period(start: date, end: date) -> Period:
    return Period(start=start.isoformat(), end=end.isoformat())


# ── Product choice ─────────────────────────────────────────────────────────


def _quantity(choice: _ProductChoice, registry: Mapping[str, CollectionConfig]) -> str | None:
    if choice.quantity is not None:
        return choice.quantity
    if choice.product is not None and choice.product in registry:
        return registry[choice.product].quantity
    if choice.named is not None:
        keys, _ = _match_named(choice.named, registry)
        quantities = {registry[k].quantity for k in keys}
        if len(quantities) == 1:
            return quantities.pop()
    return None


def _skipped(resolution: Resolution) -> list[dict[str, JsonValue]]:
    out: list[dict[str, JsonValue]] = []
    for considered in resolution.skipped[:3]:
        first = considered.ruled_out[0]
        out.append({"product": considered.key, "reason": first.reason, "detail": _short(first.detail, _DETAIL_CHARS)})
    return out


def _choice_disclosures(job: Mapping[str, Any], cfg: CollectionConfig) -> list[str]:
    product = str(job["product"])
    label = f"{product} ({cfg.short_name})" if cfg.short_name else product
    lines = [f"Product: {label}, chosen because {RULE_WORDING.get(str(job['rule']), str(job['rule']))}."]
    for skipped in job.get("skipped") or []:
        lines.append(
            f"Moved from {skipped['product']} to {product}: {skipped['product']} was ruled out "
            f"({skipped['reason']}: {skipped['detail']})."
        )
    unsized = job.get("unsized_granules")
    if isinstance(unsized, int) and unsized > 0:
        lines.append(
            f"The data provider reports no size for {product}'s files, so this retrieval of {unsized} "
            f"granule{'s' if unsized != 1 else ''} was not checked against the size limits."
        )
    return lines


def _considered_values(resolution: Resolution) -> list[JsonValue]:
    return [
        {"product": c.key, "ruled_out": [r.reason for r in c.ruled_out]}
        for c in resolution.considered[:_LISTED]
    ]


def _unavailable(
    resolution: Resolution, *, quantity: str, area: str, dates: _Dates, name: str
) -> CapabilityResult:
    """The result, or the failure, when no product can serve the request."""
    reason = resolution.unavailable
    if reason == "named_not_registered":
        raise StepFailure(FailureCategory.BAD_INPUT, f"{name}: no registered product or instrument matches the name given.")
    if reason == "named_conflicts_with_pin":
        raise StepFailure(
            FailureCategory.BAD_INPUT,
            f"{name}: the product named differs from the product this run already uses for this quantity.",
        )
    if reason == "no_approved_products":
        raise StepFailure(FailureCategory.BAD_INPUT, f"{name}: the workflow approves no product for {quantity}.")
    facts = {"quantity": quantity, "area": area, "dates": dates.text()}
    values: dict[str, JsonValue] = {"considered": _considered_values(resolution), "unavailable": reason}
    if resolution.no_data is not None:
        values["situation"] = resolution.no_data
        text = NO_DATA_REASONS[resolution.no_data].format(**facts)
    elif reason == "no_registered_product":
        text = f"No registered product measures {quantity}."
    else:
        # The readable detail of each rule; the codes stay in values. A detail
        # may hold commas or semicolons, so products are set apart with " | ".
        ruled = " | ".join(
            f"{c.key}: {' and '.join(r.detail for r in c.ruled_out)}"
            for c in resolution.considered[:5]
            if not c.fits
        )
        text = _short(f"No product can serve {quantity} over {area} for {dates.text()} ({ruled}).", 900)
    return CapabilityResult(status="insufficient_evidence", reason=text, values=values)


def _resolve(
    call: Call[Any], choice: _ProductChoice, area_bounds: Sequence[float], dates: _Dates, area: str
) -> tuple[str, CollectionConfig, Resolution] | CapabilityResult:
    registry = load_registry()
    quantity = _quantity(choice, registry)
    if quantity is None:
        raise StepFailure(
            FailureCategory.BAD_INPUT,
            f"{call.capability.name} needs the quantity to retrieve, or a product or instrument that names one.",
        )
    west, south, east, north = (float(v) for v in area_bounds)
    resolution = resolve_product(
        quantity=quantity,
        area=BBox(west, south, east, north),
        dates=dates.resolver_dates(),
        registry=registry,
        today=call.today,
        approved=choice.approved,
        cadence_needed=choice.cadence,
        named=choice.named,
        pinned=choice.product,
    )
    if resolution.product is None:
        return _unavailable(resolution, quantity=quantity, area=area, dates=dates, name=call.capability.name)
    return resolution.product, registry[resolution.product], resolution


# ── Earthdata calls ────────────────────────────────────────────────────────


def _contract(call: Call[Any], what: str) -> StepFailure:
    return StepFailure(FailureCategory.INTERNAL_ERROR, f"{call.capability.name}: the data service returned no {what}.")


async def _dataset(call: Call[Any], key: str, cfg: CollectionConfig) -> str:
    """The dataset handle for exactly the chosen product's collection; never another collection.

    The search is filtered to the collection's concept id, and the reply is
    still checked for exactly that collection."""
    found = await call.earthdata.call(
        "search_datasets", {"query": cfg.collection_id, "filters": {"concept_id": cfg.collection_id}},
    )
    handle = dataset_handle_from_search(found, cfg.collection_id)
    if handle is None:
        raise StepFailure(
            FailureCategory.INTERNAL_ERROR,
            f"{call.capability.name}: the data service's search did not list the collection of {key} "
            f"({_short(cfg.collection_id, 60)}), so nothing was retrieved.",
        )
    return handle


async def _aoi(call: Call[Any], bounds: Sequence[float]) -> str:
    defined = await call.earthdata.call("define_area_of_interest", {"location": bbox_location(bounds)})
    handle = aoi_handle_from_result(defined)
    if handle is None:
        raise _contract(call, "area-of-interest handle")
    return handle


async def _granules(call: Call[Any], dataset: str, aoi: str, time_range: str) -> int:
    checked = await call.earthdata.call(
        "check_coverage", {"dataset_handle": dataset, "aoi_handle": aoi, "time_range": time_range}
    )
    count = granule_count(checked)
    if count is None:
        raise _contract(call, "granule count for the coverage check")
    return count


async def _nearest_with_data(
    call: Call[Any], dataset: str, aoi: str, dates: _Dates, cadence: str
) -> list[str]:
    """Up to two dates near ``dates``, nearest first, that have granules over the area."""
    found: list[str] = []
    for candidate in nearby_dates(dates.start, cadence, today=call.today, end=dates.end):
        if await _granules(call, dataset, aoi, day_time_range(candidate)):
            found.append(candidate.isoformat())
            if len(found) == 2:
                break
    return found


@dataclass(frozen=True)
class _Size:
    """A retrieval plan's estimate: the bytes of the ranges the provider sized,
    and the granules of those it could not."""

    sized_bytes: int
    sized_ranges: int
    unsized_granules: int
    unsized_ranges: int

    @property
    def estimated(self) -> int | None:
        """The sized part's bytes; None when no range was sized."""
        return self.sized_bytes if self.sized_ranges else None


async def _estimate(call: Call[Any], dataset: str, aoi: str, plan: Sequence[tuple[str, int]]) -> _Size:
    """The size of each ``(time_range, granules)`` in ``plan``, summed into its sized and unsized parts."""
    sized = sized_ranges = unsized = unsized_ranges = 0
    for time_range, granules in plan:
        estimate = await call.earthdata.call(
            "estimate_retrieval_size", {"dataset_handle": dataset, "aoi_handle": aoi, "time_range": time_range}
        )
        size = estimated_bytes(estimate)
        if size is None:
            unsized += granules
            unsized_ranges += 1
        else:
            sized += size
            sized_ranges += 1
    return _Size(sized_bytes=sized, sized_ranges=sized_ranges, unsized_granules=unsized, unsized_ranges=unsized_ranges)


def _gate(size: _Size, limits: Limits, confirmed: bool) -> SizeDecision:
    """The size gate on the sized part, then on the unsized part: the hard cap
    applies to the sized bytes whatever else is unsized, and the unsized-granule
    allowance only to the unsized part's granules."""
    decision = size_gate(
        size.sized_bytes, soft=limits.retrieval_soft_bytes, hard=limits.retrieval_hard_bytes, confirmed=confirmed,
    )
    if decision != "proceed" or not size.unsized_ranges:
        return decision
    return size_gate(
        None, soft=limits.retrieval_soft_bytes, hard=limits.retrieval_hard_bytes, confirmed=confirmed,
        granules=size.unsized_granules, max_unsized_granules=MAX_UNSIZED_GRANULES,
    )


def _megabytes(size: int) -> str:
    return f"{size / 1_000_000:,.0f} MB"


# ── get_data: begin ────────────────────────────────────────────────────────


async def _begin(call: Call[GetDataParams]) -> JobRef | CapabilityResult:
    p = call.params
    area = area_facts(call.handle("area"), call=call)
    area_name = _short(area.name)
    dates = _Dates.of(p.dates)
    chosen = _resolve(call, p, area.bounds, dates, area_name)
    if isinstance(chosen, CapabilityResult):
        return chosen
    key, cfg, resolution = chosen

    choice: dict[str, Any] = {"product": key, "rule": resolution.rule, "skipped": _skipped(resolution)}
    disclosed = tuple(_choice_disclosures(choice, cfg))
    dataset = await _dataset(call, key, cfg)
    aoi = await _aoi(call, area.bounds)
    facts = {"product": key, "area": area_name, "dates": dates.text(), "quantity": cfg.quantity or key}

    # Coverage, applied by the role of the dates.
    dropped: list[str] = []
    granules = 0
    if dates.kind == "range":
        granules = await _granules(call, dataset, aoi, span_time_range(dates.start, dates.end))
        if not granules:
            nearest: list[str] = []
            if p.date_role == "subject":
                nearest = await _nearest_with_data(call, dataset, aoi, dates, cfg.cadence)
            return _no_data("gap", facts, product=key, nearest_dates=nearest, disclosures=disclosed)
        plan: list[tuple[str, str | None, int]] = [(span_time_range(dates.start, dates.end), None, granules)]
    else:
        kept: list[tuple[date, int]] = []
        for day in dates.listed:
            count = await _granules(call, dataset, aoi, day_time_range(day))
            if count:
                kept.append((day, count))
            else:
                dropped.append(day.isoformat())
        if not kept:
            nearest = []
            if p.date_role == "subject":
                nearest = await _nearest_with_data(call, dataset, aoi, dates, cfg.cadence)
            listed = dropped if p.date_role == "comparison" else []
            return _no_data(
                "gap", facts, product=key, nearest_dates=nearest, dropped=listed, disclosures=disclosed,
            )
        plan = [(day_time_range(day), day.isoformat(), count) for day, count in kept]

    science, variables = request_variables(cfg)
    size = await _estimate(call, dataset, aoi, [(time_range, count) for time_range, _, count in plan])
    estimated = size.estimated
    limits = call.limits
    decision = _gate(size, limits, p.confirmed)
    # Said of the sized part alone when another part has no size.
    at_least = "at least " if size.unsized_ranges else ""
    if decision == "refuse":
        raise StepFailure(
            FailureCategory.REQUEST_TOO_LARGE,
            f"The retrieval of {key} over {area_name} for {dates.text()} is {at_least}about "
            f"{_megabytes(size.sized_bytes)}, above the {_megabytes(limits.retrieval_hard_bytes)} limit; "
            "narrow the area or the dates.",
        )
    if decision == "confirm":
        if estimated is None:
            described = "of unknown size"
        elif size.unsized_ranges:
            described = f"about {_megabytes(size.sized_bytes)} plus a part of unknown size"
        else:
            described = f"about {_megabytes(size.sized_bytes)}"
        over_soft = estimated is not None and size.sized_bytes > limits.retrieval_soft_bytes
        prompt = (
            f"The retrieval of {key} over {area_name} for {dates.text()} is {described}"
            + (f", above the {_megabytes(limits.retrieval_soft_bytes)} soft limit" if over_soft else "")
            + ". Confirm to go ahead, or narrow the area or the dates."
        )
        return CapabilityResult(
            status="needs_input",
            question=Question(kind="pick_one", prompt=_short(prompt, 500), options=("confirm", "narrow")),
            product=key,
            values={"estimated_bytes": estimated, "soft_limit_bytes": limits.retrieval_soft_bytes},
            disclosures=disclosed,
        )

    jobs: list[JsonValue] = []
    for time_range, when, _ in plan:
        request: dict[str, Any] = {
            "dataset_handle": dataset, "aoi_handle": aoi, "time_range": time_range,
            "variables": variables, "output_format": None,
        }
        submitted = await call.earthdata.call(
            "retrieve_subset", request, idempotency_key=call.idempotency_key("retrieve_subset", request),
        )
        job_handle = submitted.get("job_handle")
        if not isinstance(job_handle, str) or not job_handle:
            raise _contract(call, "job handle for the submitted retrieval")
        jobs.append({"job": job_handle, "day": when})
    call.progress(f"Submitted {len(jobs)} retrieval{'s' if len(jobs) > 1 else ''} of {key}")
    return {
        "product": key,
        "rule": resolution.rule,
        "dataset": dataset,
        "aoi": aoi,
        "variable": science,
        "jobs": jobs,
        "dropped": list(dropped),
        "skipped": list(choice["skipped"]),
        "estimated_bytes": estimated,
        # Set when the provider could not size part of the retrieval and that part went ahead unasked.
        "unsized_granules": size.unsized_granules if size.unsized_ranges and not p.confirmed else None,
    }


def _no_data(
    situation: str,
    facts: Mapping[str, str],
    *,
    product: str | None = None,
    nearest_dates: Sequence[str] = (),
    dropped: Sequence[str] = (),
    disclosures: tuple[str, ...] = (),
    **extra: JsonValue,
) -> CapabilityResult:
    values: dict[str, JsonValue] = {"situation": situation, **extra}
    if nearest_dates:
        values["nearest_dates"] = list(nearest_dates)
    if dropped:
        values["dropped_dates"] = list(dropped)
    return CapabilityResult(
        status="insufficient_evidence",
        reason=NO_DATA_REASONS[situation].format(**facts),
        product=product,
        values=values,
        disclosures=disclosures,
    )


# ── Waiting for jobs ───────────────────────────────────────────────────────


def _job_entries(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = job.get("jobs")
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


async def _cancel(call: Call[Any], job_id: str) -> bool:
    """Whether the service accepted cancelling ``job_id``."""
    try:
        await call.earthdata.call("cancel_retrieval", {"job_handle": job_id})
    except Exception as exc:
        if not isinstance(getattr(exc, "category", None), str):
            raise
        return False
    return True


async def _wait(call: Call[Any], job: Mapping[str, Any], product: str) -> dict[str, str]:
    """Each job id mapped to its result handle, once every job is ready.

    Raises JobStillRunning at the timeout, and a named failure for a paused,
    failed or vanished job.
    """
    limits = call.limits
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limits.retrieval_timeout_seconds
    interval = limits.poll_min_seconds
    pending = [str(entry["job"]) for entry in _job_entries(job)]
    ready: dict[str, str] = {}
    failures = 0
    while True:
        for job_id in list(pending):
            try:
                status = annotate_paused(await call.earthdata.call("get_retrieval_status", {"job_handle": job_id}))
            except Exception as exc:
                if getattr(exc, "category", None) != "provider_unavailable":
                    raise
                failures += 1
                if failures > POLL_TRANSIENT_FAILURE_LIMIT:
                    raise
                continue
            failures = 0
            state = status.get("status", "")
            if state == PAUSED_STATUS:
                cancelled = await _cancel(call, job_id)
                raise StepFailure(
                    FailureCategory.REQUEST_TOO_LARGE,
                    f"The data provider paused the retrieval of {product}, usually because the request is too "
                    "large; it will not finish on its own"
                    + (", so it was cancelled" if cancelled else ", and it could not be cancelled")
                    + ". Narrow the area or the dates.",
                    result={"job": dict(job), "paused": [job_id], "cancelled": cancelled},
                )
            if state == "ready":
                handle = status.get("obs_handle") or status.get("cube_handle")
                if not isinstance(handle, str) or not handle:
                    raise _contract(call, "result handle for the finished retrieval")
                ready[job_id] = handle
                pending.remove(job_id)
            elif state == "failed":
                raise JobFailed(
                    f"The retrieval of {product} failed at the data provider.", job=job, state="failed", job_ids=[job_id],
                )
            elif state == "cancelled":
                # Stopped at the provider: the step ends with that reason and is not tried again.
                raise StepFailure(
                    FailureCategory.BAD_INPUT,
                    f"The retrieval of {product} was cancelled at the data provider.",
                    result={"job": dict(job), "cancelled": [job_id]},
                )
            elif state in TERMINAL_STATUSES:
                raise JobFailed(
                    f"The retrieval job for {product} is no longer held by the data service; run the step again.",
                    job=job, state=state, job_ids=[job_id],
                )
        if not pending:
            return ready
        if loop.time() >= deadline:
            raise JobStillRunning(call.capability.name, job, pending)
        call.progress(f"Waiting for {len(pending)} retrieval{'s' if len(pending) > 1 else ''} of {product}")
        await asyncio.sleep(interval)
        interval = min(interval * 2, limits.poll_max_seconds)


# ── get_data: complete ─────────────────────────────────────────────────────


def _listed(days: Sequence[str]) -> str:
    return ", ".join(days[:_LISTED]) + ("." if len(days) <= _LISTED else ", and more.")


def _has_valid(grid: Any) -> bool:
    return bool(grid.data.notnull().any())


def _masking_plan(cfg: CollectionConfig, variable: str) -> dict[str, JsonValue]:
    flag = cfg.quality_flag_var.rsplit("/", 1)[-1] if cfg.quality_flag_var else None
    return {"variable": variable, "quality_flag": flag, "geometry": "area"}


async def _complete(call: Call[GetDataParams], job: JobRef) -> CapabilityResult:
    p = call.params
    area = area_facts(call.handle("area"), call=call)
    area_name = _short(area.name)
    dates = _Dates.of(p.dates)
    key = str(job["product"])
    cfg = load_registry()[key]
    ready = await _wait(call, job, key)

    region_value = area.region()
    entries = _job_entries(job)
    records: list[HandleRecord] = []
    for entry in entries:
        day = entry.get("day")
        start, end = (date.fromisoformat(day), date.fromisoformat(day)) if isinstance(day, str) else (dates.start, dates.end)
        when = start.isoformat() if start == end else f"{start.isoformat()} to {end.isoformat()}"
        descriptor = Descriptor(
            summary=_short(f"{key} {cfg.quantity or cfg.primary_var} over {area_name}, {when}", 500),
            quantity=cfg.quantity,
            unit=cfg.units,
            product=key,
            region=region_value,
            period=_period(start, end),
            masking=_masking_plan(cfg, str(job["variable"])),
        )
        records.append(await call.record_retrieval(ready[str(entry["job"])], "grid", descriptor))

    grids = [await open_grid(call, record) for record in records]
    emptied: list[str] = []
    if p.date_role == "comparison" and len(grids) > 1:
        # Comparison days with nothing valid after masking are dropped, unless every day is empty.
        valid = [await call.compute(_has_valid, grid) for grid in grids]
        if any(valid):
            emptied = [str(entry["day"]) for entry, ok in zip(entries, valid, strict=True) if not ok]
            kept = [i for i, ok in enumerate(valid) if ok]
            entries = [entries[i] for i in kept]
            records = [records[i] for i in kept]
            grids = [grids[i] for i in kept]
    if len(grids) == 1:
        grid, record = grids[0], records[0]
    else:
        days = [str(entry["day"]) for entry in entries]
        grid = await call.compute(combine, grids, days)
        masking: dict[str, JsonValue] = {**grid.masking, "region_cells": grid.region_cells}
        descriptor = Descriptor(
            summary=_short(f"{key} {cfg.quantity or cfg.primary_var} over {area_name}, {dates.text()}", 500),
            quantity=cfg.quantity,
            unit=cfg.units,
            product=key,
            region=region_value,
            period=_period(dates.start, dates.end),
            masking=masking,
        )
        dataset = await call.compute(lambda g: g.dataset().load(), grid)
        record = await call.mint(
            "grid", dataset, descriptor, parents=(area.id, *(r.id for r in records)),
        )

    dropped_raw = job.get("dropped")
    without_granules = [str(d) for d in dropped_raw] if isinstance(dropped_raw, list) else []
    dropped = sorted([*without_granules, *emptied])
    requested = [d.isoformat() for d in dates.every_day() if d.isoformat() not in dropped]
    quality = await call.compute(quality_summary, grid, requested_days=requested)
    disclosures = _choice_disclosures(job, cfg)
    if without_granules:
        disclosures.append(
            f"{len(without_granules)} of the requested days had no data and were left out: "
            f"{_listed(without_granules)}"
        )
    if emptied:
        disclosures.append(
            f"{len(emptied)} of the requested days had no valid data after quality masking and were left out: "
            f"{_listed(emptied)}"
        )
    missing = quality["missing_days"]
    if missing and dates.kind != "date":
        disclosures.append(
            f"{quality['days_with_data']} of {quality['days_requested']} days had valid data after quality "
            "masking; the others are gaps and are not filled."
        )

    values: dict[str, JsonValue] = {
        "area_name": area_name,
        "period_text": dates.text(),
        "rule": str(job["rule"]),
        "qa_status": str(grid.masking.get("qa_status") or "not recorded"),
        "date_role": p.date_role,
        "masking": {k: v for k, v in grid.masking.items() if k in ("variable", "quality_flag", "qa_status", "region")},
    }
    if dropped:
        values["dropped_dates"] = list[JsonValue](dropped)
    if emptied:
        values["dropped_after_masking"] = list[JsonValue](emptied)
    if dates.kind != "date":
        values["days_without_valid_data"] = missing

    fraction = float(quality["valid_fraction"])
    if fraction == 0.0 or fraction < p.min_valid_fraction:
        facts = {"product": key, "area": area_name, "dates": dates.text(), "percent": f"{fraction:.1%}"}
        if p.date_role == "subject":
            nearest = await _nearest_with_data(call, str(job["dataset"]), str(job["aoi"]), dates, cfg.cadence)
            if nearest:
                values["nearest_dates"] = list[JsonValue](nearest)
                disclosures.append(
                    "The nearest dates with data over the area are " + " and ".join(nearest)
                    + "; whether enough of it is valid there is known only once retrieved."
                )
        return CapabilityResult(
            status="insufficient_evidence",
            handle=record.id,
            kind="grid",
            unit=record.descriptor.unit,
            product=key,
            quality=quality,
            reason=NO_DATA_REASONS["too_little_valid"].format(**facts),
            values={**values, "situation": "too_little_valid"},
            disclosures=tuple(disclosures),
        )
    summary: dict[str, JsonValue] = {
        "product": key,
        "area": area_name,
        "period": {"start": dates.start.isoformat(), "end": dates.end.isoformat()},
        "parts": len(records),
    }
    return call.done(record, quality=quality, summary=summary, values=values, disclosures=tuple(disclosures))


get_data = Capability(
    name="sources.satellite.get_data",
    description="Retrieve a satellite quantity over an area for dates, quality-masked, as a grid handle.",
    params=GetDataParams,
    output_kind="grid",
    label="Retrieving satellite data",
    method=(
        "{product} was retrieved over {area_name} for {period_text}, chosen by rule {rule}, "
        "and masked to the area with the product's quality rules ({qa_status})."
    ),
    retrieves=True,
    retry_safe=True,
    connector=CONNECTOR,
    begin=_begin,
    complete=_complete,
)


# ── get_point_series ───────────────────────────────────────────────────────


async def _series_begin(call: Call[PointSeriesParams]) -> JobRef | CapabilityResult:
    p = call.params
    area = area_facts(call.handle("area"), call=call)
    area_name = _short(area.name)
    dates = _Dates.of(p.dates)
    span = (dates.end - dates.start).days + 1
    if span > call.limits.max_point_timeseries_days:
        raise StepFailure(
            FailureCategory.REQUEST_TOO_LARGE,
            f"A point series of {span} days is longer than the {call.limits.max_point_timeseries_days}-day "
            "limit; narrow the dates.",
        )
    chosen = _resolve(call, p, area.bounds, dates, area_name)
    if isinstance(chosen, CapabilityResult):
        return chosen
    key, cfg, resolution = chosen
    dataset = await _dataset(call, key, cfg)
    aoi = await _aoi(call, area.bounds)
    science = cfg.primary_var
    request: dict[str, Any] = {
        "dataset_handle": dataset, "aoi_handle": aoi,
        "time_range": span_time_range(dates.start, dates.end),
        "variables": [science, *pinned_companion_variables([science])],
        "point_sample": True,
    }
    submitted = await call.earthdata.call(
        "retrieve_timeseries", request, idempotency_key=call.idempotency_key("retrieve_timeseries", request),
    )
    job_handle = submitted.get("job_handle")
    if not isinstance(job_handle, str) or not job_handle:
        raise _contract(call, "job handle for the submitted point series")
    return {
        "product": key,
        "rule": resolution.rule,
        "dataset": dataset,
        "aoi": aoi,
        "variable": science.rsplit("/", 1)[-1],
        "jobs": [{"job": job_handle, "day": None}],
        "dropped": [],
        "skipped": list(_skipped(resolution)),
        "estimated_bytes": None,
    }


def _series_from_table(table: Any, variable: str, label: str) -> pd.DataFrame:
    """A point-sampled table as a series: its time column paired with the variable's column."""
    frame = table.to_pandas() if hasattr(table, "to_pandas") and not isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    columns = [str(c) for c in frame.columns]
    if not columns:
        return series_frame(pd.DataFrame({"time": [], "value": [], "series": []}))
    time_col = next((c for c in ("time", "date") if c in columns), columns[0])
    value_col = variable if variable in columns else next((c for c in columns if c != time_col), None)
    if value_col is None:
        return series_frame(pd.DataFrame({"time": [], "value": [], "series": []}))
    out = pd.DataFrame({
        "time": pd.to_datetime(frame[time_col], utc=True),
        "value": pd.to_numeric(frame[value_col], errors="coerce"),
    })
    out = out.dropna(subset=["value"])
    out["series"] = label
    return series_frame(out)


async def _series_complete(call: Call[PointSeriesParams], job: JobRef) -> CapabilityResult:
    p = call.params
    area = area_facts(call.handle("area"), call=call)
    area_name = _short(area.name)
    dates = _Dates.of(p.dates)
    key = str(job["product"])
    cfg = load_registry()[key]
    ready = await _wait(call, job, key)
    [entry] = _job_entries(job)
    region_value = area.region()
    raw = await call.record_retrieval(ready[str(entry["job"])], "table", Descriptor(
        summary=_short(f"{key} point samples at {area_name}, {dates.text()}", 500),
        quantity=cfg.quantity, unit=cfg.units, product=key, region=region_value,
        period=_period(dates.start, dates.end),
    ))
    table = await call.env.resolver.open(call.user_id, raw.id, "table")
    frame = await call.compute(_series_from_table, table, str(job["variable"]), key)
    disclosures = _choice_disclosures(job, cfg)
    disclosures.append("Point samples are reported as the data service sampled them; no quality flag was applied.")
    values: dict[str, JsonValue] = {"area_name": area_name, "period_text": dates.text(), "rule": str(job["rule"])}
    if frame.empty:
        facts = {"product": key, "area": area_name, "dates": dates.text(), "percent": f"{0:.1%}"}
        return CapabilityResult(
            status="insufficient_evidence", product=key,
            reason=NO_DATA_REASONS["too_little_valid"].format(**facts),
            values={**values, "situation": "too_little_valid"}, disclosures=tuple(disclosures),
        )
    descriptor = Descriptor(
        summary=_short(f"{key} point series at {area_name}, {dates.text()}", 500),
        quantity=cfg.quantity, unit=cfg.units, product=key, region=region_value,
        period=_period(dates.start, dates.end),
        masking={"variable": str(job["variable"]), "quality_flag": None, "point_sample": True},
    )
    record = await call.mint("series", frame, descriptor, parents=(area.id, raw.id))
    days = {t.date() for t in frame["time"]}
    summary: dict[str, JsonValue] = {
        "points": len(frame),
        "days_with_data": len(days),
        "first": frame["time"].min().isoformat(),
        "last": frame["time"].max().isoformat(),
    }
    return call.done(record, summary=summary, values=values, disclosures=tuple(disclosures))


get_point_series = Capability(
    name="sources.satellite.get_point_series",
    description="Retrieve a point-sampled satellite series at an area over a date range, as a series handle.",
    params=PointSeriesParams,
    output_kind="series",
    label="Retrieving a satellite point series",
    method="{product} was point-sampled at {area_name} for {period_text}, chosen by rule {rule}.",
    retrieves=True,
    retry_safe=True,
    connector=CONNECTOR,
    begin=_series_begin,
    complete=_series_complete,
)


# ── Discovery ──────────────────────────────────────────────────────────────


def _registry_key_for(entry: Mapping[str, Any], registry: Mapping[str, CollectionConfig]) -> str | None:
    concept = entry.get("concept_id")
    short = entry.get("short_name")
    for key, cfg in registry.items():
        if concept and cfg.collection_id == concept:
            return key
    for key, cfg in registry.items():
        if short and cfg.short_name and cfg.short_name.upper() == str(short).upper():
            return key
    return None


NO_REGISTERED_MATCH = "No registered product matched; only registered products can be retrieved."


async def _search(call: Call[SearchParams]) -> CapabilityResult:
    found = await call.earthdata.call("search_datasets", {"query": call.params.query, "filters": None})
    registry = load_registry()
    listed: list[JsonValue] = []
    any_registered = False
    for entry in search_hits(found):
        product = _registry_key_for(entry, registry)
        any_registered = any_registered or product is not None
        if len(listed) >= call.params.limit:
            continue
        handle = entry.get("handle") or entry.get("dataset_handle")
        listed.append({
            "dataset": handle if isinstance(handle, str) else None,
            "short_name": _short(str(entry.get("short_name") or ""), 60) or None,
            "title": _short(str(entry.get("entry_title") or entry.get("title") or ""), 100) or None,
            "product": product,
        })
    values: dict[str, JsonValue] = {"datasets": listed, "count": len(listed)}
    if not any_registered:
        values["note"] = NO_REGISTERED_MATCH
        values["registered_quantities"] = list(QUANTITIES)
    return call.done(values=values)


search = Capability(
    name="sources.satellite.search",
    description="Search the earthdata catalogue for datasets; returns a short list, retrieves nothing.",
    params=SearchParams,
    output_kind=None,
    label="Searching datasets",
    method="The catalogue was searched for '{query}' and returned {count} datasets.",
    retrieves=False,
    retry_safe=True,
    connector=CONNECTOR,
    run=_search,
)


def product_facts(key: str, cfg: CollectionConfig) -> dict[str, JsonValue]:
    """The registry facts a model may read about a product, beside its name and unit."""
    coverage = cfg.coverage
    return {
        "short_name": cfg.short_name or None,
        "description": _short(cfg.description, 200) or None,
        "instrument": cfg.instrument or None,
        "quantity": cfg.quantity,
        "cadence": cfg.cadence,
        "coverage": coverage if isinstance(coverage, str) or coverage is None else coverage.model_dump(),
        "record_start": cfg.record_start.isoformat() if cfg.record_start else None,
        "record_end": cfg.record_end.isoformat() if cfg.record_end else None,
        "stream": cfg.stream,
        "latency_days": cfg.latency_days,
        "maturity": cfg.maturity,
        "variable": cfg.primary_var,
        "quality_flag": cfg.quality_flag_var,
        "layered": bool(cfg.vertical_axis_vars),
    }


async def _describe(call: Call[DescribeParams]) -> CapabilityResult:
    registry = load_registry()
    key = call.params.product
    if key not in registry:
        raise failure(UNKNOWN_PRODUCT, name=call.capability.name, keys=", ".join(_PRODUCT_KEYS))
    cfg = registry[key]
    return call.done(product=key, unit=cfg.units, values=product_facts(key, cfg))


describe = Capability(
    name="sources.satellite.describe",
    description="Describe a registered product: quantity, coverage, record dates, cadence, latency, variable.",
    params=DescribeParams,
    output_kind=None,
    label="Describing a product",
    method="The registry facts for {product} were read.",
    retrieves=False,
    retry_safe=True,
    run=_describe,
)


async def _availability(call: Call[AvailabilityParams]) -> CapabilityResult:
    p = call.params
    registry = load_registry()
    if p.product not in registry:
        raise failure(UNKNOWN_PRODUCT, name=call.capability.name, keys=", ".join(_PRODUCT_KEYS))
    cfg = registry[p.product]
    area = area_facts(call.handle("area"), call=call)
    area_name = _short(area.name)
    dates = _Dates.of(p.dates)
    west, south, east, north = area.bounds
    resolution = resolve_product(
        quantity=cfg.quantity or "", area=BBox(west, south, east, north), dates=dates.resolver_dates(),
        registry=registry, today=call.today, approved=[p.product], pinned=p.product,
    )
    values: dict[str, JsonValue] = {"area_name": area_name, "period_text": dates.text()}
    [considered] = resolution.considered
    ruled = sorted(considered.reasons - {"wrong_quantity"})
    if resolution.no_data is not None or ruled:
        values.update({
            "available": False,
            "situation": resolution.no_data,
            "ruled_out": [r.reason for r in considered.ruled_out if r.reason in ruled],
        })
        return call.done(product=p.product, values=values)
    dataset = await _dataset(call, p.product, cfg)
    aoi = await _aoi(call, area.bounds)
    if dates.kind == "range":
        count = await _granules(call, dataset, aoi, span_time_range(dates.start, dates.end))
        values.update({"available": count > 0, "granules": count, "situation": None if count else "gap"})
        return call.done(product=p.product, values=values)
    with_data: list[JsonValue] = []
    without: list[JsonValue] = []
    total = 0
    for day in dates.listed:
        count = await _granules(call, dataset, aoi, day_time_range(day))
        total += count
        (with_data if count else without).append(day.isoformat())
    values.update({
        "available": bool(with_data), "granules": total, "situation": None if with_data else "gap",
        "days_with_data": with_data, "days_without_data": without,
    })
    return call.done(product=p.product, values=values)


check_availability = Capability(
    name="sources.satellite.check_availability",
    description="Check whether a product has data for an area and dates, without retrieving it.",
    params=AvailabilityParams,
    output_kind=None,
    label="Checking availability",
    method="The availability of {product} over {area_name} for {period_text} was checked.",
    retrieves=False,
    retry_safe=True,
    connector=CONNECTOR,
    run=_availability,
)

CAPABILITIES = (get_data, get_point_series, search, describe, check_availability)
