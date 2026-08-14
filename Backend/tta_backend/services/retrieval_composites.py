"""
services/retrieval_composites.py
==================================
Backend composites over the earthdata-retrieval MCP that absorb the two
ergonomic hazards of durable retrieval: an LLM burning turns polling job
status (await_retrieval), and an LLM free-running an expensive retrieval
(safe_retrieve).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import BaseTool

from tta_backend.config.settings import Settings, get_settings
from tta_backend.config.workflow_stages import STAGE_ESTIMATE, STAGE_PROGRESS, STAGE_SUBMIT
from tta_backend.datasets.registry import known_quality_flag_vars, load_registry
from tta_backend.earthdata_mcp.results import (
    CATEGORY_CONTRACT,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_TOO_LARGE,
    MCPToolError,
    parse_tool_result,
)
from tta_backend.services import retrieval_narration, scope_registry, variable_choice_registry
from tta_backend.utils.metrics import observe_harmony_fetch
from tta_backend.utils.streaming import emit_job_progress, emit_status

logger = logging.getLogger(__name__)

# "not_found" is a dead handle list_workspace still lists after the MCP has
# evicted the job (get_retrieval_status has no record of it) -- terminal
# because it never changes again and there's no live job a cancel could
# reach. On a long-lived workspace these can vastly outnumber real jobs
# (live repro: 134 not_found vs 46 real jobs); leaving it out sorted every
# dead row into the "active" group ahead of genuinely current tasks (see
# list_jobs' sort below and jobs_service._CACHEABLE_STATUSES, which already
# treated it as immutable).
TERMINAL_STATUSES = {"ready", "failed", "expired", "cancelled", "not_found"}

# await_retrieval keeps polling through this many CONSECUTIVE transient
# (provider_unavailable) status-poll failures before surfacing the error. A
# single network blip during a 15-minute Harmony job used to abandon the
# whole await — the agent reported failure while the job completed
# unobserved at the provider. One successful poll resets the count.
POLL_TRANSIENT_FAILURE_LIMIT = 3

# Harmony auto-pauses jobs it considers too large to run unattended (its
# previewing -> paused flow). The MCP's status mapper reports a paused job as
# status "running" (harmony-retrieval-mcp providers/harmony.py maps "paused"
# -> RUNNING), so the only paused signal that reaches this backend is the
# provider's own status text relayed in ``message`` ("The job is paused and
# may be resumed using the provided link", live-observed 2026-07-16 on
# job_142cbb2faa6aecc0). Without this detection a paused job polls as
# "running" until the await timeout — an ever-climbing chat timer over a job
# that will never finish on its own, and a jobs-panel row stuck at
# "Processing" forever.
PAUSED_STATUS = "paused"

_PAUSED_MESSAGE_RE = re.compile(r"\bpaused\b", re.IGNORECASE)

PAUSED_NOTE = (
    "The data provider paused this job — usually because the request was too "
    "large to process automatically. It will not finish on its own: cancel "
    "it, then narrow the time range or area and try again."
)


def annotate_paused(status_data: dict[str, Any]) -> dict[str, Any]:
    """Surface a provider-paused job honestly: a non-terminal status whose
    provider message says the job is paused becomes status ``"paused"`` (with
    a ``"paused at provider"`` phase and a plain-language ``note``) instead of
    masquerading as running forever. Terminal statuses and messages that never
    mention pausing pass through untouched."""
    status = status_data.get("status", "")
    message = status_data.get("message") or ""
    if status in TERMINAL_STATUSES or not _PAUSED_MESSAGE_RE.search(message):
        return status_data
    return {
        **status_data,
        "status": PAUSED_STATUS,
        "phase": "paused at provider",
        "note": PAUSED_NOTE,
    }


def _supports_variable_subsetting(variables: list[str]) -> bool:
    """Best-effort registry check: False only when every requested variable
    resolves to a collection explicitly marked ``supports_variable_subsetting:
    false`` (a Harmony capability flag captured by
    scripts/generate_collection.py). ``safe_retrieve`` only ever sees an
    opaque ``dataset_handle``, not a collection id, so this matches by
    variable short name instead -- a name that collides across registry
    entries always agrees on this flag in practice, and an unregistered
    name defaults to True (today's send-it-and-see behavior) rather than
    guessing.
    """
    index: dict[str, bool] = {}
    for cfg in load_registry().values():
        names = {cfg.primary_var}
        if cfg.quality_flag_var:
            names.add(cfg.quality_flag_var)
        names.update(v.rsplit("/", 1)[-1] for v in cfg.variables)
        for name in names:
            index[name] = cfg.supports_variable_subsetting or index.get(name, False)

    resolved = [index[v] for v in variables if v in index]
    return all(resolved) if resolved else True


def _leaf(name: str) -> str:
    return str(name or "").rsplit("/", 1)[-1]


def _pinned_companion_variables(variables: list[str]) -> list[str]:
    """The companions each requested science variable cannot be honestly
    interpreted without, in the spelling the registry itself uses.

    Two kinds, one doctrine. A subset that drops the **quality flag** cannot be
    masked at all: provenance degrades to "not applied — semantics unknown" and
    every not-normal pixel is plotted as if it were good (measured 2026-08-01:
    26.6% of a real TEMPO NO2 L3 scene). A subset that drops a layered
    product's **vertical axes** yields values with nothing to plot them against
    (measured 2026-08-08: the agent requested ``product/ozone_profile`` alone,
    Harmony obliged, and the profile rendered against a bare layer index --
    upside down, since layer 0 is the top of the atmosphere).

    Both are part of requesting the science variable, not separate decisions --
    and not ones left to the agent. The flag guard exists because the previous
    "a standard TEMPO retrieval always requests both" comment assumed without
    enforcing; the axes were registered in ``collections.yaml`` and made
    exactly the same assumption until the same thing happened again.

    Matched by bare leaf name, since the caller's spelling may be
    group-qualified (``product/vertical_column_troposphere``) or bare. The
    addition is emitted in whichever spelling the matching request used:
    handing the provider one variable list addressed two different ways is a
    request no collection's variable list ever looks like. A companion that
    lives in a *different* group from the science variable (the axes are in
    ``support_data``) keeps the registry's own qualified spelling, which is the
    only one that resolves.
    """
    requested = {_leaf(v) for v in variables}
    by_leaf = {_leaf(v): v for v in variables}
    additions: list[str] = []
    for cfg in load_registry().values():
        companions = [cfg.quality_flag_var, *cfg.vertical_axis_vars]
        companions = [c for c in companions if c]
        if not companions:
            continue
        collection_leaves = {_leaf(cfg.primary_var)} | {_leaf(v) for v in cfg.variables}
        matched = requested & collection_leaves
        if not matched:
            continue
        science_as_asked = by_leaf[sorted(matched)[0]]
        for companion in companions:
            companion_leaf = _leaf(companion)
            if companion_leaf in requested:
                continue                  # the caller already asked for it
            if "/" not in science_as_asked and "/" not in companion:
                spelling = companion_leaf
            else:
                # Prefer the registry's own qualified spelling; fall back to the
                # group the caller addressed the science variable through.
                spelling = next(
                    (v for v in cfg.variables if _leaf(v) == companion_leaf and "/" in v),
                    f"{science_as_asked.rsplit('/', 1)[0]}/{companion_leaf}"
                    if "/" in science_as_asked else companion_leaf,
                )
            if spelling not in additions:
                additions.append(spelling)
    return additions


def _normalize_time_range(time_range: str) -> str:
    """Widen a degenerate ``start/end`` interval where start == end — the
    shape a single-date request ("June 15, 2024" → '2024-06-15/2024-06-15')
    naturally produces — into a real span before any provider sees it.
    Harmony rejects start >= stop outright (live 2026-07-11: six jobs failed
    with "The temporal range's start must be earlier than its stop datetime",
    surviving only when the OPeNDAP fallback happened to work). A date-only
    pair means "that whole day"; a timestamped pair gets one second of width.
    Anything unparseable is returned unchanged for the MCP's own time_range
    validation to reject with its more specific message."""
    parts = time_range.split("/", 1)
    if len(parts) != 2 or parts[0] != parts[1]:
        return time_range
    start = parts[0].strip()
    try:
        parsed = datetime.fromisoformat(start)
    except ValueError:
        return time_range
    if "T" not in start:
        return f"{start}T00:00:00/{start}T23:59:59"
    return f"{start}/{(parsed + timedelta(seconds=1)).isoformat()}"


class RetrievalError(RuntimeError):
    """Base class for retrieval-composite failures."""


class RetrievalTimeoutError(TimeoutError, RetrievalError):
    """Raised when await_retrieval exceeds the configured polling timeout."""

    def __init__(self, message: str, *, job_handle: str, elapsed_seconds: float):
        super().__init__(message)
        self.job_handle = job_handle
        self.elapsed_seconds = elapsed_seconds


async def await_retrieval(
    job_handle: str,
    tools: dict[str, BaseTool],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Poll get_retrieval_status backend-side until the job reaches a terminal state.

    Spends one model turn instead of a polling loop: backs off from
    poll_min to poll_max seconds, emits a job_progress SSE event each poll,
    and returns the terminal status dict (including obs_handle on success).
    A failed/cancelled job is returned, not raised — the MCP's own stage/
    provider-prefixed error string is the caller's answer, verbatim.

    A provider-paused job (see :func:`annotate_paused`) is also returned
    immediately, with status ``"paused"`` and a plain-language ``note`` —
    polling it to the timeout would just narrate a job that cannot finish
    on its own.
    """
    settings = settings or get_settings()
    status_tool = tools["get_retrieval_status"]
    interval = settings.await_retrieval_poll_min_seconds
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + settings.await_retrieval_timeout_seconds

    consecutive_poll_failures = 0
    while True:
        try:
            raw = await status_tool.ainvoke({"job_handle": job_handle})
            data = annotate_paused(parse_tool_result(raw))
        except MCPToolError as exc:
            # Transient outages (the MCP's upstream call dying, a network
            # blip) are retried up to POLL_TRANSIENT_FAILURE_LIMIT
            # consecutive times — the job itself is still running at the
            # provider, so abandoning the await on the first blip throws
            # away real work. Anything non-transient (contract/user_input)
            # is a bug or a bad request and fails loud on the first poll.
            if exc.category != CATEGORY_PROVIDER_UNAVAILABLE:
                raise
            consecutive_poll_failures += 1
            if consecutive_poll_failures > POLL_TRANSIENT_FAILURE_LIMIT:
                raise
            logger.warning(
                "await_retrieval_poll_retry",
                extra={
                    "_event": "await_retrieval_poll_retry",
                    "_job_handle": job_handle,
                    "_consecutive_failures": consecutive_poll_failures,
                },
            )
            await asyncio.sleep(interval)
            interval = min(interval * 2, settings.await_retrieval_poll_max_seconds)
            continue
        consecutive_poll_failures = 0
        status = data.get("status", "")
        progress = data.get("progress")
        emit_job_progress(
            job_handle,
            status,
            progress,
            data.get("phase"),
            data.get("message"),
            note=data.get("note"),
        )
        if status == PAUSED_STATUS:
            emit_status(
                "Retrieval paused by the data provider — the request may be too large.",
                stage=STAGE_PROGRESS,
                detail=progress,
            )
            return data
        # `phase` is harmony-retrieval-mcp's qualitative label for the job and
        # reads better mid-flight than the durable `status` it qualifies
        # ("queued at provider" vs. "submitted") — the jobs panel already
        # prefers it for exactly that reason (Frontend/src/utils/jobCard.js),
        # so this line had been showing the worse of two strings it already
        # held. Terminal responses ignore it for the same reason that panel
        # does: a cancel carries no phase, or a stale one, and a "cancelled"
        # line reading "materializing" is a contradiction.
        label = status if status in TERMINAL_STATUSES else (data.get("phase") or status)
        # What the wait is *for*, recorded at submission. None for a job this
        # process didn't submit (open_handle's rematerialize path), which
        # degrades to the wording this line always had.
        subject = retrieval_narration.describe(job_handle) or "data"
        emit_status(f"Retrieving {subject} — {label}...", stage=STAGE_PROGRESS, detail=progress)
        if status in TERMINAL_STATUSES:
            if status == "ready":
                # T51: the one place that knows a retrieval's full span. Only
                # on ``ready``: the histogram describes a job that actually
                # downloaded something, and folding failures/cancellations in
                # would make its percentiles describe a different population
                # than they claim.
                observe_harmony_fetch(loop.time() - started)
                # T25: promote this job's recorded variable choice (if any)
                # onto the handle it resolved to, so a later plot/stat/
                # compare call inherits it instead of AggregationService.
                # to_dataarray refusing a multi-variable file all over again.
                handle = data.get("obs_handle") or data.get("cube_handle")
                variable_choice_registry.finalize(job_handle, handle)
                # T46: promote the requested scope (recorded at submission) onto
                # the resolved handle, so a later plot/stat can disclose any
                # silent substitution between requested and delivered scope.
                scope_registry.finalize(job_handle, handle)
            # A narration describes the wait, and the wait is over. No
            # finalize onto the result handle: unlike the two registries
            # above, nothing downstream reads this against a result.
            retrieval_narration.discard(job_handle)
            return data

        if loop.time() >= deadline:
            elapsed = loop.time() - started
            raise RetrievalTimeoutError(
                f"Retrieval job {job_handle} did not reach a terminal state within "
                f"{settings.await_retrieval_timeout_seconds}s",
                job_handle=job_handle,
                elapsed_seconds=elapsed,
            )

        await asyncio.sleep(interval)
        interval = min(interval * 2, settings.await_retrieval_poll_max_seconds)


async def safe_retrieve(
    dataset_handle: str,
    aoi_handle: str,
    time_range: str,
    variables: list[str],
    tools: dict[str, BaseTool],
    *,
    output_format: str | None = None,
    confirmed: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run estimate -> gate -> retrieve as one deterministic call.

    The soft/hard cap numbers live in config, not prompts, so neither the
    model nor a persuasive user can talk past the guardrail:

    - at or below the soft cap: proceeds automatically.
    - between the soft and hard cap: pauses for confirmation unless the
      caller already has one (``confirmed=True``, e.g. a retry after the
      supervisor relayed the user's approval).
    - above the hard cap: refused unconditionally, even if ``confirmed``.
    """
    settings = settings or get_settings()
    time_range = _normalize_time_range(time_range)
    emit_status("Estimating retrieval size...", stage=STAGE_ESTIMATE)
    estimate_raw = await tools["estimate_retrieval_size"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
    })
    estimate = parse_tool_result(estimate_raw)
    estimated_bytes = estimate.get("estimated_bytes")

    # An absent estimate must not sail past the caps as if it were zero
    # bytes — "couldn't price it" is not "free". Treat it like the
    # between-caps case: pause for confirmation, saying honestly that the
    # size is unknown (the bundle-open size gate still backstops at open
    # time, but only after the provider has already done the work).
    if estimated_bytes is None:
        if not confirmed:
            return {
                "status": "needs_confirmation",
                "estimated_bytes": None,
                "message": (
                    "The provider could not estimate this retrieval's size, so the "
                    "size guardrail cannot vet it. Reply to confirm proceeding "
                    "anyway, or narrow the area of interest, time range, or "
                    "variable list first."
                ),
            }
    elif estimated_bytes > settings.retrieval_hard_cap_bytes:
        return {
            "status": "refused",
            "estimated_bytes": estimated_bytes,
            "message": (
                f"Estimated retrieval size (~{estimated_bytes:,} bytes) exceeds the "
                f"{settings.retrieval_hard_cap_bytes:,}-byte hard cap. Narrow the "
                "area of interest, time range, or variable list and try again."
            ),
        }

    if estimated_bytes is not None and estimated_bytes > settings.retrieval_soft_cap_bytes and not confirmed:
        return {
            "status": "needs_confirmation",
            "estimated_bytes": estimated_bytes,
            "message": (
                f"Estimated retrieval size (~{estimated_bytes:,} bytes) is above the "
                f"{settings.retrieval_soft_cap_bytes:,}-byte soft cap. Reply to confirm "
                "or narrow the request."
            ),
        }

    emit_status("Submitting retrieval...", stage=STAGE_SUBMIT)
    # Skip the doomed subset attempt entirely for collections the registry
    # already knows don't support it (e.g. TROPOMI, MODIS AOD) -- avoids a
    # wasted failed-subset-then-full-retrieval round trip on every call.
    subset_variables = variables if _supports_variable_subsetting(variables) else []
    if subset_variables:
        subset_variables = [*subset_variables, *_pinned_companion_variables(subset_variables)]
    subset_raw = await tools["retrieve_subset"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
        "variables": subset_variables,
        "output_format": output_format,
    })
    subset = parse_tool_result(subset_raw)

    # T25: record the model's chosen science variable, keyed by the job this
    # retrieval submits as. A quality flag variable riding along is not a
    # science choice -- ``_pinned_companion_variables`` above adds one to
    # every registered subset request, so counting raw ``variables`` would see
    # 2 and record nothing, leaving the opened 2-var file to refuse downstream.
    # Exclude known flag vars (matched by bare leaf name, since ``variables``
    # is group-qualified) first: a single remaining science variable is still
    # an unambiguous choice worth recording; 0 or >1 leave the file's own
    # choice to be made downstream. Note this reads the caller's ``variables``,
    # not ``subset_variables`` -- the injected flag must not shift the count.
    flag_vars = known_quality_flag_vars()
    science_variables = [v for v in variables if v.rsplit("/", 1)[-1] not in flag_vars]
    if len(science_variables) == 1:
        variable_choice_registry.record_pending(subset.get("job_handle"), science_variables[0])

    # T46: safe_retrieve only sees an opaque aoi_handle (not a place name), so
    # it records just the requested time_range — enough to disclose a single-
    # day request served by a monthly mean. Region substitution on this path
    # is caught by the dispatch-layer AOI guard instead.
    scope_registry.record_pending(subset.get("job_handle"), {"time_range": time_range})

    # What the researcher will be waiting on, for the status line. Everything
    # here was already computed above and then dropped; the same aoi_handle
    # limitation the scope record just noted applies, so there is no place
    # name to offer — the size estimate stands in as the other bound on the
    # wait.
    retrieval_narration.record(
        subset.get("job_handle"),
        variable=science_variables[0] if len(science_variables) == 1 else None,
        time_range=time_range,
        estimated_bytes=estimated_bytes,
    )

    return {"status": "submitted", "estimated_bytes": estimated_bytes, **subset}


def _parse_time_span_days(time_range: str) -> int | None:
    """Days between an ISO 8601 'start/end' interval's two ends, or None if
    ``time_range`` isn't in that shape. An unparseable range is left for the
    MCP's own time_range validation to reject rather than folded into the
    too_large gate below."""
    try:
        start_str, end_str = time_range.split("/", 1)
        return (datetime.fromisoformat(end_str) - datetime.fromisoformat(start_str)).days
    except (ValueError, AttributeError):
        return None


async def point_timeseries(
    dataset_handle: str,
    location: str,
    time_range: str,
    variable: str,
    tools: dict[str, BaseTool],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Resolve AOI, gate the requested time span, submit a point-sampled
    timeseries retrieval, and await it to a terminal state (T20) — the
    point-timeseries analogue of safe_retrieve's gate-then-submit shape,
    chained straight into await_retrieval so the model spends one tool call
    instead of three.

    Point sampling is always on (this composite's identity is the MCP's
    AppEEARS-routed point path, never a gridded cube). Raises
    ``MCPToolError(category="too_large")`` before any MCP call when the
    requested span exceeds ``settings.retrieval_max_timeseries_days`` — a
    span gate rather than safe_retrieve's byte-size estimate, since a point
    series has no size to estimate.
    """
    settings = settings or get_settings()
    time_range = _normalize_time_range(time_range)

    span_days = _parse_time_span_days(time_range)
    if span_days is not None and span_days > settings.retrieval_max_timeseries_days:
        raise MCPToolError(
            CATEGORY_TOO_LARGE,
            f"Requested time span ({span_days} days) exceeds the "
            f"{settings.retrieval_max_timeseries_days}-day point-timeseries limit.",
            suggestion="Narrow the time range and try again.",
        )

    aoi_raw = await tools["define_area_of_interest"].ainvoke({"location": location})
    aoi = parse_tool_result(aoi_raw)
    aoi_handle = aoi.get("handle") or aoi.get("aoi_handle")
    # An AOI response with no handle used to flow on as None and submit a
    # retrieval against a null area; off-taxonomy that's a contract violation
    # the agent should be able to explain, not a silent wrong request.
    if not aoi_handle:
        raise MCPToolError(
            CATEGORY_CONTRACT,
            "define_area_of_interest returned no area-of-interest handle "
            "(expected a 'handle' or 'aoi_handle' field), so the point-timeseries "
            "retrieval can't be located.",
            suggestion="Try a different phrasing of the location, or a nearby named place.",
        )

    emit_status("Submitting point timeseries retrieval...", stage=STAGE_SUBMIT)
    submit_raw = await tools["retrieve_timeseries"].ainvoke({
        "dataset_handle": dataset_handle,
        "aoi_handle": aoi_handle,
        "time_range": time_range,
        "variables": [variable, *_pinned_companion_variables([variable])],
        "point_sample": True,
    })
    submit = parse_tool_result(submit_raw)
    # A submit response with no job_handle used to KeyError below off-taxonomy;
    # classify it so the failure reads as a contract violation naming the tool
    # and the absent field rather than an opaque traceback.
    job_handle = submit.get("job_handle")
    if not job_handle:
        raise MCPToolError(
            CATEGORY_CONTRACT,
            "retrieve_timeseries returned no 'job_handle', so the submitted "
            "point-timeseries retrieval can't be awaited.",
            suggestion="Retry the retrieval; if it persists, the product may not support point sampling.",
        )

    # T46: point_timeseries knows both the place name and the time range, so it
    # records the full requested scope for the job — promoted onto the cube
    # handle by await_retrieval's finalize.
    scope_registry.record_pending(job_handle, {"location": location, "time_range": time_range})

    # This composite awaits inline, so its own status line is the one the
    # researcher watches. Unlike safe_retrieve it holds the actual place name
    # (it resolved the AOI itself) but has no size estimate — a point series
    # has none to quote.
    retrieval_narration.record(
        job_handle, variable=variable, location=location, time_range=time_range
    )

    status = await await_retrieval(job_handle, tools, settings=settings)
    return {"aoi_handle": aoi_handle, **status}
