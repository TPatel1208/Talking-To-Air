"""
eval_harness.py
=================
The 10-task scripted eval for the earthdata agent (PRD T04): canned research
tasks run against the real agent wired to the fake-MCP seam, scored on
tool-call trace and terminal outcome. Lives beside the test suite behind the
opt-in "eval" pytest marker (tests/test_eval_harness.py) because it spends
real model tokens — this module is the harness itself, not a test file.

Pass threshold: >= 8/10, recorded here (test_eval_harness.py enforces it).
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, patch

EnvelopeCheck = Callable[[Any, list[str]], bool]
AqsGetHandler = Callable[[str, dict], Awaitable[dict]]


def _always_valid(envelope, tool_calls: list[str]) -> bool:
    return envelope is not None


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
        return {"job_handle": job_handle, "status": "materialized", "obs_handle": obs_handle}

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


def build_eval_tasks(volume) -> list[EvalTask]:
    """Build the 10 canned tasks. ``volume`` is a fake_earthdata_mcp.HandleVolume
    used to back the two plotting tasks with real (tiny) Zarr fixtures so
    plot_singular/conduct_temporal_statistic have something to open."""
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

    volume.add_zarr("obs_map_1", make_map_dataset)
    volume.add_zarr("obs_ts_1", make_timeseries_dataset)
    volume.add_zarr("obs_cmp_a", make_map_dataset)
    volume.add_zarr("obs_cmp_b", make_map_dataset)

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

    # ── Plotting (2) ─────────────────────────────────────────────────────
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

    return tasks


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
        with aqs_patch:
            async for event_type, data in stream_response(agent, task.prompt, thread_id=f"eval-{task.name}"):
                if event_type == "tool_call":
                    tool_calls.append(data["name"])
                elif event_type == "text":
                    text_parts.append(data if isinstance(data, str) else data.get("response", ""))

        raw_text = "".join(text_parts)
        envelope = parse_sub_agent_envelope(raw_text)
        passed = contains_subsequence(tool_calls, task.expected_tool_calls) and task.outcome_check(
            envelope, tool_calls
        )
        return EvalTaskResult(task=task, tool_calls=tool_calls, raw_text=raw_text, envelope=envelope, passed=passed)
    finally:
        server.stop()


async def run_eval_suite(volume, *, model: str | None = None) -> list[EvalTaskResult]:
    tasks = build_eval_tasks(volume)
    return [await run_eval_task(task, model=model) for task in tasks]


PASS_THRESHOLD = 9
TOTAL_TASKS = 11
