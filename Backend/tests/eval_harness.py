"""
eval_harness.py
=================
The scripted eval for the earthdata agent (PRD T04): 14 canned research
tasks run against the real agent wired to the fake-MCP seam, scored on
tool-call trace and terminal outcome, plus 3 end-to-end tasks (T16) that
enter through ChatStreamService.stream_chat_events with the real
intent_router and real sub-agents — one ground, one satellite, one
cross-source — exercising the router fast path and envelope handling under
the same gate. Lives beside the test suite behind the opt-in "eval" pytest
marker (tests/test_eval_harness.py) because it spends real model tokens —
this module is the harness itself, not a test file. The one "robustness"
task (T15) is the exception — it is scored at the finalization seam
instead of the live model loop, so it spends no tokens.

Every task also records wall-clock (elapsed_seconds) and fails its
category's latency budget (CATEGORY_BUDGETS, decision record 2026-07-06
§6); run_eval_suite fails the whole run if any provider retry/429 evidence
was logged during a single-user run (capture_rate_limit_evidence).

Pass threshold: >= 14/17, recorded here (test_eval_harness.py enforces it).
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, patch

EnvelopeCheck = Callable[[Any, list[str]], bool]
AqsGetHandler = Callable[[str, dict], Awaitable[dict]]


class RateLimitDetected(RuntimeError):
    """Raised when provider retry/429 evidence was logged during an eval run."""


# Groq's client logs retries on the "groq" logger family (e.g. "Retrying
# request to /v1/chat/completions in 5.2 seconds") rather than raising on
# the caller's side; httpx logs the raw response line, which carries the
# status code. Google's genai client (satellite/supervisor both default to
# google now) logs its own retry backoff via tenacity's before_sleep_log on
# the "google_genai._api_client" logger — the message embeds APIError's
# "{code} {status}. {details}" string, so the same "429" match already
# catches it once that logger is watched. Matching these loggers is cruder
# than instrumenting httpx/genai directly but has zero production footprint.
_RATE_LIMIT_LOG_PATTERN = re.compile(r"retrying request|\b429\b|too many requests|resource_exhausted", re.I)


class _RateLimitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.matches: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if _RATE_LIMIT_LOG_PATTERN.search(message):
            self.matches.append(message)


@contextmanager
def capture_rate_limit_evidence():
    """Watch the groq/httpx/google_genai loggers for retry/429 evidence for
    the duration of the ``with`` block. Yields the handler; ``handler.
    matches`` lists any matching log messages observed (single-user
    cleanliness is the bar — any evidence at all means rate-limit pressure
    returned)."""
    handler = _RateLimitLogHandler()
    watched_loggers = [
        logging.getLogger("groq"),
        logging.getLogger("httpx"),
        logging.getLogger("google_genai"),
    ]
    previous_levels = [logger.level for logger in watched_loggers]
    for logger in watched_loggers:
        # httpx logs its request-line summary (which carries the status
        # code) at INFO; without lowering the logger's own effective level
        # here, records below the ambient root level never reach handlers.
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        for logger, previous_level in zip(watched_loggers, previous_levels):
            logger.removeHandler(handler)
            logger.setLevel(previous_level)


def _always_valid(envelope, tool_calls: list[str]) -> bool:
    return envelope is not None


_STAGE_KEY_PATTERN = re.compile(r'"stage":\s*"([^"]+)"')

# The canonical satellite-workflow stage order (T19 Testing Decisions) —
# extra/duplicate stages (e.g. two "coverage" emissions, one per await_
# retrieval poll) are fine; contains_subsequence only requires these eight
# to appear in order, not contiguously.
_FULL_SATELLITE_WORKFLOW_STAGES = [
    "search", "aoi", "coverage", "estimate", "submit", "progress", "open", "render",
]


def _stage_sequence(joined: str) -> list[str]:
    """The ordered list of stage keys named in a joined SSE event stream's
    status events (T19) — e.g. 'event: status\\ndata: {..., "stage": "search"}'."""
    return _STAGE_KEY_PATTERN.findall(joined)


def _covers_full_satellite_workflow(stage_sequence: list[str]) -> bool:
    """True if ``stage_sequence`` contains the canonical satellite-workflow
    stage order as a subsequence — makes a workflow that goes silent after
    the pre-dispatch phase (2026-07-07 live test) a command-detectable
    regression rather than a UX issue discovered in a demo."""
    return contains_subsequence(stage_sequence, _FULL_SATELLITE_WORKFLOW_STAGES)


def contains_subsequence(trace: list[str], expected: list[str]) -> bool:
    """True if ``expected`` appears in ``trace``, in order, not necessarily
    contiguously — e.g. ["search_datasets", "safe_retrieve"] matches a trace
    that also called describe_dataset in between."""
    it = iter(trace)
    return all(name in it for name in expected)


@dataclass
class EvalTask:
    name: str
    category: str
    prompt: str
    handlers: dict[str, Callable[..., Awaitable[dict]]]
    expected_tool_calls: list[str]
    outcome_check: EnvelopeCheck = field(default=_always_valid)
    # Optional stub for epa_aqs_tools._aqs_get, applied only for the duration
    # of this task's agent run — validate_against_ground/exceedance_overlay
    # call the AQS HTTP boundary directly (not through the fake-MCP seam),
    # so tasks that exercise them must supply this to avoid a live EPA call.
    aqs_get: AqsGetHandler | None = None


@dataclass
class EvalTaskResult:
    task: EvalTask
    tool_calls: list[str]
    raw_text: str
    envelope: Any
    passed: bool
    elapsed_seconds: float = 0.0


DateCheck = Callable[[list[tuple[str, dict]]], bool]
StageCheck = Callable[[list[str]], bool]


@dataclass
class E2ETask:
    """A T16 end-to-end task: enters through ChatStreamService.stream_chat_
    events (real intent_router, real sub-agents, fake MCP) rather than
    run_eval_task's direct single-agent invocation. Scored on which
    agent(s) ran (``expects_ground``/``expects_satellite``) plus a
    non-error terminal answer; ``date_check`` is an optional extra
    assertion over EPA AQS calls captured during the run (see
    _dispatched_correct_relative_date); ``stage_check`` (T19) is an
    optional extra assertion over the ordered stage keys narrated during
    the run (see _stage_sequence/_covers_full_satellite_workflow)."""
    name: str
    category: str
    prompt: str
    handlers: dict[str, Callable[..., Awaitable[dict]]]
    expects_ground: bool = False
    expects_satellite: bool = False
    aqs_get: AqsGetHandler | None = None
    date_check: DateCheck | None = None
    stage_check: StageCheck | None = None


def _standard_handlers(
    *,
    dataset_handle: str = "dataset_1",
    aoi_handle: str = "aoi_1",
    obs_handle: str = "obs_1",
    granule_count: int = 5,
    estimated_bytes: int = 100,
):
    """The discovery -> AOI -> coverage -> gate -> retrieve handlers shared
    by most tasks. Individual tasks override entries to script a specific
    scenario (e.g. zero granules, an over-cap estimate)."""

    async def search_datasets(query, filters, workspace_id):
        return {"dataset_handle": dataset_handle, "short_name": query, "title": query}

    async def describe_dataset(dataset_handle, detail, workspace_id):
        return {
            "dataset_handle": dataset_handle,
            "variables": [{"name": "no2", "fill_value": -9999, "valid_min": 0, "valid_max": 1}],
        }

    async def define_area_of_interest(location, workspace_id):
        return {"aoi_handle": aoi_handle, "location": location}

    async def check_availability(dataset_handle, aoi_handle, time_range, workspace_id):
        return {"granule_count": granule_count}

    async def check_coverage(dataset_handle, aoi_handle, time_range, workspace_id):
        return {"granule_count": granule_count, "coverage_pct": 100 if granule_count else 0}

    async def estimate_retrieval_size(dataset_handle, aoi_handle, time_range, workspace_id):
        return {"estimated_bytes": estimated_bytes}

    async def retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
        return {"job_handle": f"job_{obs_handle}", "obs_handle": obs_handle}

    async def get_retrieval_status(job_handle, workspace_id):
        return {"job_handle": job_handle, "status": "ready", "obs_handle": obs_handle}

    return {
        "search_datasets": search_datasets,
        "describe_dataset": describe_dataset,
        "define_area_of_interest": define_area_of_interest,
        "check_availability": check_availability,
        "check_coverage": check_coverage,
        "estimate_retrieval_size": estimate_retrieval_size,
        "retrieve_subset": retrieve_subset,
        "get_retrieval_status": get_retrieval_status,
    }


def _handles_nonempty(envelope, tool_calls: list[str]) -> bool:
    return envelope is not None and len(envelope.handles) > 0


def _no_data_options_offered(envelope, tool_calls: list[str]) -> bool:
    if envelope is None or envelope.handles:
        return False
    summary = envelope.summary.lower()
    return "safe_retrieve" not in tool_calls and any(
        marker in summary for marker in ("broaden", "switch dataset", "different location", "cancel", "a)")
    )


def _refused_without_retrieving(envelope, tool_calls: list[str]) -> bool:
    return (
        envelope is not None
        and not envelope.handles
        and "await_retrieval" not in tool_calls
    )


def _point_timeseries_used_no_grid_retrieval(envelope, tool_calls: list[str]) -> bool:
    """T20: the point-timeseries composite is the whole retrieval — a grid
    pull (safe_retrieve) alongside it means the model reached for the
    heavier path this composite exists to replace."""
    return (
        envelope is not None
        and len(envelope.handles) > 0
        and "point_timeseries" in tool_calls
        and "safe_retrieve" not in tool_calls
    )


def build_eval_tasks(volume) -> list[EvalTask]:
    """Build the 13 canned tasks. ``volume`` is a fake_earthdata_mcp.HandleVolume
    used to back the plotting tasks with real (tiny) Zarr/Parquet fixtures so
    plot_singular/conduct_temporal_statistic/point_timeseries have something
    to open."""
    import xarray as xr

    def make_map_dataset():
        return xr.Dataset(
            {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
            coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
        )

    def make_timeseries_dataset():
        import pandas as pd

        times = pd.date_range("2024-01-01", periods=3, freq="D")
        return xr.Dataset(
            {"no2": (("time", "lat", "lon"), [[[1.0, 2.0]], [[2.0, 3.0]], [[3.0, 4.0]]], {"units": "mol/m^2"})},
            coords={"time": times, "lat": [10.0], "lon": [30.0, 40.0]},
        )

    def make_point_timeseries_table():
        import pyarrow as pa

        return pa.table({
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "no2": [1.0, 2.0, 3.0],
        }).replace_schema_metadata({b"units": b"mol/m^2"})

    volume.add_zarr("obs_map_1", make_map_dataset)
    volume.add_zarr("obs_ts_1", make_timeseries_dataset)
    volume.add_zarr("obs_cmp_a", make_map_dataset)
    volume.add_zarr("obs_cmp_b", make_map_dataset)
    volume.add_parquet("obs_pt_ts_1", make_point_timeseries_table)

    def volume_lifecycle_handlers(obs_handle: str) -> dict:
        return {
            "export_result": volume.export_result,
            "rematerialize": volume.rematerialize,
            "get_retrieval_status": volume.get_retrieval_status,
        }

    tasks: list[EvalTask] = []

    # ── Discovery (2) ────────────────────────────────────────────────────
    tasks.append(EvalTask(
        name="discovery_no2_dataset",
        category="discovery",
        prompt="What NASA datasets are available for NO2 column density over New Jersey?",
        handlers=_standard_handlers(),
        expected_tool_calls=["search_datasets"],
    ))
    tasks.append(EvalTask(
        name="discovery_soil_moisture",
        category="discovery",
        prompt="Find a soil moisture dataset I could use to map conditions over the Raritan basin.",
        handlers=_standard_handlers(dataset_handle="dataset_soil"),
        expected_tool_calls=["search_datasets"],
    ))

    # ── Retrieval (2) ────────────────────────────────────────────────────
    tasks.append(EvalTask(
        name="retrieval_tempo_no2",
        category="retrieval",
        prompt="Retrieve TEMPO NO2 over Houston for 2024-06-01.",
        handlers=_standard_handlers(obs_handle="obs_retrieval_1"),
        expected_tool_calls=["search_datasets", "define_area_of_interest", "safe_retrieve", "await_retrieval"],
        outcome_check=_handles_nonempty,
    ))
    tasks.append(EvalTask(
        name="retrieval_aod_month",
        category="retrieval",
        prompt="Retrieve MODIS aerosol optical depth over California for June 2024.",
        handlers=_standard_handlers(obs_handle="obs_retrieval_2"),
        expected_tool_calls=["search_datasets", "define_area_of_interest", "safe_retrieve", "await_retrieval"],
        outcome_check=_handles_nonempty,
    ))

    # ── Plotting (3) ─────────────────────────────────────────────────────
    tasks.append(EvalTask(
        name="plotting_single_map",
        category="plotting",
        prompt="Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
        handlers={**_standard_handlers(obs_handle="obs_map_1"), **volume_lifecycle_handlers("obs_map_1")},
        expected_tool_calls=["safe_retrieve", "await_retrieval", "plot_singular"],
        outcome_check=_handles_nonempty,
    ))
    tasks.append(EvalTask(
        name="plotting_timeseries",
        category="plotting",
        prompt="Show me how NO2 changed over Newark NJ during January 2024.",
        handlers={**_standard_handlers(obs_handle="obs_ts_1"), **volume_lifecycle_handlers("obs_ts_1")},
        expected_tool_calls=["safe_retrieve", "await_retrieval", "conduct_temporal_statistic"],
        outcome_check=_handles_nonempty,
    ))
    async def _pt_ts_retrieve_timeseries(dataset_handle, time_range, variables, aoi_handle, output_format, point_sample, workspace_id):
        return {"job_handle": "job_obs_pt_ts_1"}

    tasks.append(EvalTask(
        name="plotting_point_timeseries",
        category="plotting",
        prompt=(
            "What was the NO2 reading at the single point 40.735,-74.172 "
            "(Newark, NJ) each day during January 2024? I just need that "
            "one location's history, not an area average."
        ),
        handlers={
            "search_datasets": _standard_handlers()["search_datasets"],
            "define_area_of_interest": _standard_handlers()["define_area_of_interest"],
            "retrieve_timeseries": _pt_ts_retrieve_timeseries,
            **volume_lifecycle_handlers("obs_pt_ts_1"),
        },
        expected_tool_calls=["search_datasets", "point_timeseries"],
        outcome_check=_point_timeseries_used_no_grid_retrieval,
    ))

    # ── Comparison setup (2) ─────────────────────────────────────────────
    tasks.append(EvalTask(
        name="comparison_setup_two_cities",
        category="comparison_setup",
        prompt=(
            "Retrieve TROPOMI NO2 for both Newark NJ and Los Angeles CA on "
            "2024-01-15 so I can compare them side by side."
        ),
        handlers={**_standard_handlers(obs_handle="obs_cmp_a"), **volume_lifecycle_handlers("obs_cmp_a")},
        expected_tool_calls=["define_area_of_interest", "safe_retrieve", "await_retrieval"],
        outcome_check=_handles_nonempty,
    ))
    tasks.append(EvalTask(
        name="comparison_setup_two_periods",
        category="comparison_setup",
        prompt="Retrieve TEMPO NO2 over Houston for both January 2024 and July 2024.",
        handlers={**_standard_handlers(obs_handle="obs_cmp_b"), **volume_lifecycle_handlers("obs_cmp_b")},
        expected_tool_calls=["safe_retrieve", "await_retrieval"],
        outcome_check=_handles_nonempty,
    ))

    # ── Failure recovery (2) ─────────────────────────────────────────────
    tasks.append(EvalTask(
        name="failure_recovery_no_data",
        category="failure_recovery",
        prompt="Plot TEMPO NO2 over Death Valley for 1990-01-01.",
        handlers=_standard_handlers(granule_count=0),
        expected_tool_calls=["check_availability"],
        outcome_check=_no_data_options_offered,
    ))
    tasks.append(EvalTask(
        name="failure_recovery_retrieval_refused",
        category="failure_recovery",
        prompt=(
            "Retrieve TEMPO NO2 hourly data for all of 2024 over the entire "
            "continental United States."
        ),
        handlers=_standard_handlers(estimated_bytes=50 * 1024 ** 3),
        expected_tool_calls=["safe_retrieve"],
        outcome_check=_refused_without_retrieving,
    ))

    # ── Ground validation (1) — PRD T07 signature workflow ──────────────
    def make_ground_validation_dataset():
        import pandas as pd

        times = pd.date_range("2024-01-01", periods=3, freq="D")
        return xr.Dataset(
            {"no2": (
                ("time", "lat", "lon"),
                [[[1.0, 100.0], [100.0, 100.0]],
                 [[2.0, 100.0], [100.0, 100.0]],
                 [[3.0, 100.0], [100.0, 100.0]]],
                {"units": "mol/m^2"},
            )},
            coords={"time": times, "lat": [40.0, 41.0], "lon": [-74.0, -73.0]},
        )

    volume.add_zarr("obs_validate_1", make_ground_validation_dataset)

    async def _ground_validation_aqs_get(endpoint: str, params: dict) -> dict:
        if endpoint == "monitors/byBox":
            return {"Header": [{"status": "success"}], "Data": [{
                "latitude": "40.0", "longitude": "-74.0",
                "state_code": "34", "county_code": "017", "site_number": "0006",
                "local_site_name": "Newark Firehouse",
            }]}
        if endpoint == "dailyData/byBox":
            return {"Header": [{"status": "success"}], "Data": [
                {
                    "date_local": d, "arithmetic_mean": str(v), "state_code": "34",
                    "county_code": "017", "site_number": "0006", "units_of_measure": "ppb",
                    "pollutant_standard": "NO2 1-hour 2010", "local_site_name": "Newark Firehouse",
                }
                for d, v in [("2024-01-01", 2.0), ("2024-01-02", 4.0), ("2024-01-03", 6.0)]
            ]}
        return {"Header": [{"status": "success"}], "Data": []}

    tasks.append(EvalTask(
        name="ground_validation_tempo_vs_epa",
        category="ground_validation",
        prompt="Compare TEMPO NO2 with EPA ground monitors over Newark NJ for the first week of January 2024.",
        handlers={**_standard_handlers(obs_handle="obs_validate_1"), **volume_lifecycle_handlers("obs_validate_1")},
        expected_tool_calls=["safe_retrieve", "await_retrieval", "validate_against_ground"],
        outcome_check=_handles_nonempty,
        aqs_get=_ground_validation_aqs_get,
    ))

    # ── Region/period comparison (1) — PRD T08 signature workflow #2 ────
    def make_compare_period_a():
        return xr.Dataset(
            {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

    def make_compare_period_b():
        return xr.Dataset(
            {"no2": (("lat", "lon"), [[2.0, 4.0], [6.0, 8.0]], {"units": "mol/m^2"})},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

    def make_compare_aligned():
        return xr.Dataset(
            {"no2": (
                ("source", "lat", "lon"),
                [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]],
                {"units": "mol/m^2"},
            )},
            coords={"source": [0, 1], "lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

    volume.add_zarr("obs_compare_period_a", make_compare_period_a)
    volume.add_zarr("obs_compare_period_b", make_compare_period_b)
    volume.add_zarr("cube_compare_aligned", make_compare_aligned)

    def _compare_obs_handle_for(time_range: str) -> str:
        return "obs_compare_period_b" if "2026" in time_range else "obs_compare_period_a"

    async def _compare_retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
        obs_handle = _compare_obs_handle_for(time_range)
        return {"job_handle": f"job_{obs_handle}", "obs_handle": obs_handle}

    async def _compare_align(source_handles, method="outer", workspace_id="default"):
        return {"handle": "cube_compare_aligned", "status": "ok", "alignment_report": {"method": method}}

    tasks.append(EvalTask(
        name="comparison_period_tempo_no2",
        category="comparison",
        prompt="Compare TEMPO NO2 over New Jersey between June 2025 and June 2026 — did it change?",
        handlers={
            **_standard_handlers(dataset_handle="dataset_compare", aoi_handle="aoi_compare"),
            "retrieve_subset": _compare_retrieve_subset,
            "align": _compare_align,
            "export_result": volume.export_result,
            "rematerialize": volume.rematerialize,
            "get_retrieval_status": volume.get_retrieval_status,
        },
        expected_tool_calls=["safe_retrieve", "await_retrieval", "safe_retrieve", "await_retrieval", "compare"],
        outcome_check=_handles_nonempty,
    ))

    # ── Robustness (1) — PRD T15 malformed-envelope salvage ─────────────
    # A real model cannot be reliably scripted to emit malformed JSON, so
    # this task is scored at the finalization seam (run_robustness_task)
    # rather than through the live agent loop the other 12 tasks use — see
    # decision record §6.
    tasks.append(EvalTask(
        name="robustness_malformed_final_envelope",
        category="robustness",
        prompt="(scored at the finalization seam — no live model call; see run_robustness_task)",
        handlers={},
        expected_tool_calls=["_finalize_sub_agent_result"],
    ))

    return tasks


def _real_yesterday_yyyymmdd() -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y%m%d")


def _dispatched_correct_relative_date(aqs_calls: list[tuple[str, dict]]) -> bool:
    """True if any captured (endpoint, params) EPA AQS call used real UTC
    'yesterday' as its bdate/edate. The 2026-07-07 live test caught a
    "yesterday" query dispatched with *today's* date (T11 Further Notes);
    after T11 deleted the legacy date-conversion tools, relative-date
    correctness is enforced only here."""
    target = _real_yesterday_yyyymmdd()
    return any(
        params.get("bdate") == target or params.get("edate") == target
        for _, params in aqs_calls
    )


def _agents_consulted(text: str) -> set[str]:
    """Parse the "Agent consulted: ..." header both routing paths emit into
    the set of agent names named. The router fast path (T14) emits a
    single-source header ("Agent consulted: GROUND"); the supervisor path
    (cross-source, config/supervisor_prompt.py) emits one combined header
    ("Agent consulted: GROUND + SATELLITE") — this handles both shapes."""
    match = re.search(r"Agent consulted:\s*([A-Z ]+(?:\+ [A-Z]+)?)", text)
    if not match:
        return set()
    return {name.strip() for name in match.group(1).split("+") if name.strip()}


def build_e2e_tasks(volume) -> list[E2ETask]:
    """Build the 3 end-to-end tasks: one ground (fast path), one satellite
    (fast path), one cross-source (supervisor path, both agents)."""
    import xarray as xr

    def make_e2e_satellite_dataset():
        return xr.Dataset(
            {"no2": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

    volume.add_zarr("obs_e2e_satellite_1", make_e2e_satellite_dataset)

    def volume_lifecycle_handlers() -> dict:
        return {
            "export_result": volume.export_result,
            "rematerialize": volume.rematerialize,
            "get_retrieval_status": volume.get_retrieval_status,
        }

    async def _e2e_ground_aqs_get(endpoint: str, params: dict) -> dict:
        if endpoint == "monitors/byBox":
            return {"Header": [{"status": "success"}], "Data": [{
                "latitude": "40.0", "longitude": "-74.0",
                "state_code": "34", "county_code": "017", "site_number": "0006",
                "local_site_name": "Newark Firehouse",
            }]}
        if endpoint == "dailyData/byBox":
            return {"Header": [{"status": "success"}], "Data": [{
                "date_local": params.get("bdate", ""), "arithmetic_mean": "5.0", "state_code": "34",
                "county_code": "017", "site_number": "0006", "units_of_measure": "ppb",
                "pollutant_standard": "NO2 1-hour 2010", "local_site_name": "Newark Firehouse",
            }]}
        return {"Header": [{"status": "success"}], "Data": []}

    tasks: list[E2ETask] = [
        # ── Ground (router fast path) ────────────────────────────────────
        E2ETask(
            name="e2e_ground_relative_date",
            category="e2e_ground",
            prompt="What was the NO2 level in Newark, New Jersey yesterday?",
            handlers={},
            expects_ground=True,
            aqs_get=_e2e_ground_aqs_get,
            date_check=_dispatched_correct_relative_date,
        ),
        # ── Satellite (router fast path) ─────────────────────────────────
        E2ETask(
            name="e2e_satellite_plot",
            category="e2e_satellite",
            prompt="Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
            handlers={
                **_standard_handlers(obs_handle="obs_e2e_satellite_1"),
                **volume_lifecycle_handlers(),
            },
            expects_satellite=True,
            stage_check=_covers_full_satellite_workflow,
        ),
        # ── Cross-source (supervisor path, both agents) ──────────────────
        E2ETask(
            name="e2e_cross_source_ground_vs_satellite",
            category="e2e_cross_source",
            prompt=(
                "Compare the ground NO2 monitors with TROPOMI satellite NO2 "
                "over Newark NJ for January 2024."
            ),
            handlers={
                **_standard_handlers(obs_handle="obs_e2e_satellite_1"),
                **volume_lifecycle_handlers(),
            },
            expects_ground=True,
            expects_satellite=True,
            aqs_get=_e2e_ground_aqs_get,
        ),
    ]
    return tasks


def all_task_ids(volume) -> set[str]:
    """Every eval task id — direct-agent (build_eval_tasks) plus e2e
    (build_e2e_tasks) — the set T22's starter-prompt coverage check runs
    against."""
    return (
        {task.name for task in build_eval_tasks(volume)}
        | {task.name for task in build_e2e_tasks(volume)}
    )


def starter_prompts_missing_eval_tasks(volume) -> list[str]:
    """T22 story #4: the empty-chat starter list (config.starter_prompts)
    is a marketing surface — every entry's id must name a real eval task, or
    a starter prompt could rot into a broken promise with nothing to catch
    it. Returns the ids of any entries with no matching eval task (empty
    when the contract holds)."""
    from config.starter_prompts import STARTER_PROMPTS

    task_ids = all_task_ids(volume)
    return [entry["id"] for entry in STARTER_PROMPTS if entry["id"] not in task_ids]


async def run_eval_task(task: EvalTask, *, model: str | None = None) -> EvalTaskResult:
    from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
    from earthdata_mcp.client import load_raw_mcp_tools
    from config.settings import Settings
    from agents.earthdata_agent import build_earthdata_agent
    from models.agent_result import parse_sub_agent_envelope
    from utils.streaming import stream_response

    server = FakeEarthdataMCPServer(build_fake_mcp(task.handlers))
    server.start()
    aqs_patch = (
        patch("tools.ground_sensor_tools.epa_aqs_tools._aqs_get", AsyncMock(side_effect=task.aqs_get))
        if task.aqs_get is not None
        else nullcontext()
    )
    try:
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        mcp_tools = await load_raw_mcp_tools(settings)
        agent = build_earthdata_agent(model=model, mcp_tools=mcp_tools)

        tool_calls: list[str] = []
        text_parts: list[str] = []
        started = time.monotonic()
        with aqs_patch:
            async for event_type, data in stream_response(agent, task.prompt, thread_id=f"eval-{task.name}"):
                if event_type == "tool_call":
                    tool_calls.append(data["name"])
                elif event_type == "text":
                    text_parts.append(data if isinstance(data, str) else data.get("response", ""))
        elapsed_seconds = time.monotonic() - started

        raw_text = "".join(text_parts)
        envelope = parse_sub_agent_envelope(raw_text)
        within_budget = elapsed_seconds <= CATEGORY_BUDGETS[task.category]
        passed = (
            contains_subsequence(tool_calls, task.expected_tool_calls)
            and task.outcome_check(envelope, tool_calls)
            and within_budget
        )
        return EvalTaskResult(
            task=task, tool_calls=tool_calls, raw_text=raw_text, envelope=envelope,
            passed=passed, elapsed_seconds=elapsed_seconds,
        )
    finally:
        server.stop()


async def run_robustness_task() -> EvalTaskResult:
    """T15's malformed-envelope robustness task: exercises the salvage path
    directly at the finalization seam (services.subagent_dispatch.
    _finalize_sub_agent_result) — a successful tool workflow (already-
    collected chart + artifact) followed by a final message that is prose,
    not the {summary, artifact_ids, handles} envelope. Passes when the
    scored outcome is a non-error answer that still carries the artifact.
    """
    from models import AgentResult, ChartPayload
    from models.artifact import ArtifactReference
    from services.subagent_dispatch import _finalize_sub_agent_result

    task = EvalTask(
        name="robustness_malformed_final_envelope",
        category="robustness",
        prompt="(scored at the finalization seam — no live model call)",
        handlers={},
        expected_tool_calls=["_finalize_sub_agent_result"],
    )
    prose = "I plotted TROPOMI NO2 over New Jersey for 2024-01-15; the map is attached above."
    raw = AgentResult(
        text=prose,
        charts=[ChartPayload(type="heatmap", title="TROPOMI NO2 over NJ")],
        artifacts=[ArtifactReference(id="map_robustness_1", type="map", title="TROPOMI NO2 over NJ")],
    )

    started = time.monotonic()
    finalized = _finalize_sub_agent_result(raw, "earthdata")
    elapsed_seconds = time.monotonic() - started

    passed = (
        not finalized.metadata.get("error")
        and finalized.metadata.get("salvaged") is True
        and len(finalized.artifacts) == 1
        and finalized.artifacts[0].id == "map_robustness_1"
        and elapsed_seconds <= CATEGORY_BUDGETS[task.category]
    )
    return EvalTaskResult(
        task=task, tool_calls=[], raw_text=prose, envelope=None, passed=passed, elapsed_seconds=elapsed_seconds,
    )


class _UnusedAgent:
    """A fast-pathed e2e task never touches the other agent — stand-in that
    fails loudly (rather than silently no-oping) if it's invoked anyway."""

    def __getattr__(self, name):
        raise AssertionError(f"unexpected access to the untouched agent: {name}")


class _NoWritebackSupervisor:
    """Stand-in supervisor for a fast-pathed e2e task (ground- or satellite-
    only): the real supervisor's own model is never called on that path,
    only ChatStreamService._write_back_turn's aupdate_state — and a failed
    write-back is logged and swallowed by design, never fatal."""

    async def aupdate_state(self, *args, **kwargs):
        raise RuntimeError("no real supervisor built for this fast-pathed e2e task")


async def run_e2e_task(task: E2ETask, volume, *, model: str | None = None) -> EvalTaskResult:
    """Drive ``task.prompt`` through the real chat-streaming layer
    (ChatStreamService.stream_chat_events) with the real intent_router and
    real sub-agents, fake MCP — scored on which agent(s) ran plus a
    non-error terminal answer, per the Technical Implementation Guide."""
    from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer
    from earthdata_mcp.client import load_raw_mcp_tools
    from config.settings import Settings
    from agents.ground_sensor_agent import build_ground_agent
    from agents.earthdata_agent import build_earthdata_agent
    from agents.supervisor_agent import build_agent as build_supervisor_agent
    from services.chart_service import ChartService
    from services.chat_stream_service import ChatStreamService
    from utils.streaming import get_call_budget

    # A shared top-level context runs every task in the suite sequentially
    # (see run_eval_suite) — clear the T16 per-request budget holder so one
    # task's ground/satellite call count never bleeds into the next.
    get_call_budget().clear()

    aqs_calls: list[tuple[str, dict]] = []

    async def _capturing_aqs_get(endpoint: str, params: dict) -> dict:
        aqs_calls.append((endpoint, dict(params)))
        return await task.aqs_get(endpoint, params)

    aqs_patch = (
        patch("tools.ground_sensor_tools.epa_aqs_tools._aqs_get", AsyncMock(side_effect=_capturing_aqs_get))
        if task.aqs_get is not None
        else nullcontext()
    )

    server = FakeEarthdataMCPServer(build_fake_mcp(task.handlers)) if task.handlers else None
    if server is not None:
        server.start()

    try:
        ground_agent = build_ground_agent(model=model) if task.expects_ground else _UnusedAgent()

        if task.expects_satellite:
            settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
            mcp_tools = await load_raw_mcp_tools(settings)
            satellite_agent = build_earthdata_agent(model=model, mcp_tools=mcp_tools)
        else:
            satellite_agent = _UnusedAgent()

        if task.expects_ground and task.expects_satellite:
            # Cross-source: the genuinely ambiguous route goes through the
            # supervisor's own two model calls, so it needs the real,
            # checkpointed supervisor (real Postgres — this task must run
            # where the eval always runs, against a live database).
            supervisor_agent = await build_supervisor_agent(
                model=model, ground_agent=ground_agent, satellite_agent=satellite_agent,
            )
        else:
            supervisor_agent = _NoWritebackSupervisor()

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        events: list[str] = []
        started = time.monotonic()
        with aqs_patch:
            async for event in service.stream_chat_events(
                supervisor_agent, ground_agent, satellite_agent, task.prompt,
                f"eval-e2e-{task.name}", "eval-e2e-user", f"eval-e2e-{task.name}",
            ):
                events.append(event)
        elapsed_seconds = time.monotonic() - started
    finally:
        if server is not None:
            server.stop()

    joined = "".join(events)
    tool_calls = re.findall(r'event: tool_call\ndata: \{"name": "([^"]+)"', joined)
    non_error_done = "event: done" in joined and "event: error" not in joined

    consulted = _agents_consulted(joined)
    agents_ok = True
    if task.expects_ground:
        agents_ok = agents_ok and "GROUND" in consulted
    if task.expects_satellite:
        agents_ok = agents_ok and "SATELLITE" in consulted

    date_ok = task.date_check(aqs_calls) if task.date_check is not None else True
    stage_ok = task.stage_check(_stage_sequence(joined)) if task.stage_check is not None else True
    within_budget = elapsed_seconds <= CATEGORY_BUDGETS[task.category]
    passed = non_error_done and agents_ok and date_ok and stage_ok and within_budget

    # expected_tool_calls is intentionally empty — e2e tasks are scored on
    # which agent(s) ran (agents_ok) and outcome, not an MCP tool-call
    # sequence; contains_subsequence([], []) is vacuously true, so the
    # table's trace column reads "ok" rather than misrepresenting a miss.
    wrapping_task = EvalTask(
        name=task.name, category=task.category, prompt=task.prompt,
        handlers=task.handlers, expected_tool_calls=[],
    )
    return EvalTaskResult(
        task=wrapping_task, tool_calls=tool_calls, raw_text=joined, envelope=None,
        passed=passed, elapsed_seconds=elapsed_seconds,
    )


async def run_eval_suite(volume, *, model: str | None = None) -> list[EvalTaskResult]:
    tasks = build_eval_tasks(volume)
    e2e_tasks = build_e2e_tasks(volume)
    results = []
    with capture_rate_limit_evidence() as rate_limit_evidence:
        for task in tasks:
            if task.category == "robustness":
                results.append(await run_robustness_task())
            else:
                results.append(await run_eval_task(task, model=model))
        for e2e_task in e2e_tasks:
            results.append(await run_e2e_task(e2e_task, volume, model=model))
    if rate_limit_evidence.matches:
        raise RateLimitDetected(
            f"provider rate-limit evidence during a single-user eval run: {rate_limit_evidence.matches}"
        )
    return results


DIRECT_AGENT_TASK_COUNT = 14
PASS_THRESHOLD = 14
TOTAL_TASKS = DIRECT_AGENT_TASK_COUNT + 3

# Per-category wall-clock budgets (decision record 2026-07-06 §6). The
# satellite budget applies against the fake MCP — this measures the
# system's own overhead, not NASA's. A task that runs over its category's
# budget fails, even if its tool trace and outcome check both pass.
CATEGORY_BUDGETS: dict[str, float] = {
    "discovery": 15.0,
    "retrieval": 45.0,
    "plotting": 45.0,
    "comparison_setup": 45.0,
    "failure_recovery": 15.0,
    "ground_validation": 15.0,
    "comparison": 45.0,
    "robustness": 15.0,
    "e2e_ground": 15.0,
    "e2e_satellite": 45.0,
    "e2e_cross_source": 45.0,
}


def format_results_table(results: list[EvalTaskResult]) -> str:
    """Render the compact per-task table the eval prints after a run: name,
    category, pass/fail, trace verdict, seconds — so a regression's
    location is obvious from the output alone (user story 9)."""
    header = f"{'task':<42} {'category':<18} {'result':<6} {'trace':<6} {'seconds':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        trace_ok = contains_subsequence(r.tool_calls, r.task.expected_tool_calls)
        lines.append(
            f"{r.task.name:<42} {r.task.category:<18} {'PASS' if r.passed else 'FAIL':<6} "
            f"{'ok' if trace_ok else 'bad':<6} {r.elapsed_seconds:>8.2f}"
        )
    return "\n".join(lines)
