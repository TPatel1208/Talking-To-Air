"""The benchmark cases.

Each case is a question, the data the fake services return for it, and checks
on the outcome. Every case enters a system the way a user does, so none
presumes which agent or workflow answers. Cases name only the data boundary
(the earthdata tools and EPA AQS) and what the user sees, never a system's
own tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Awaitable, Callable

from benchmark.checks import (
    AnswerMatches,
    AnswerMentionsAny,
    AreaRequested,
    AqsDateRelativeToClock,
    BoundaryCalled,
    BoundaryNotCalled,
    ChartProduced,
    Check,
    Completed,
    GroundSourceNotUsed,
    GroundSourceUsed,
    HasResult,
    SatelliteSourceUsed,
    StageSubsequence,
    TimeRangeCovers,
)

Handler = Callable[..., Awaitable[dict]]
AqsGetHandler = Callable[[str, dict], Awaitable[dict]]
# A workflow's inputs, or a function of the run's date for relative dates.
WorkflowInputs = dict[str, Any] | Callable[[date], dict[str, Any]]


@dataclass(frozen=True)
class PatchSpec:
    """Replace ``target`` (a dotted path) with a mock returning ``return_value``."""

    target: str
    return_value: Any
    is_async: bool = True


@dataclass
class BenchmarkCase:
    id: str
    category: str
    prompt: str
    # Fake earthdata tool handlers; tools not named echo their arguments.
    handlers: dict[str, Handler]
    checks: tuple[Check, ...]
    # Stub for the EPA AQS request function; None answers every request with an empty success.
    aqs_get: AqsGetHandler | None = None
    patches: tuple[PatchSpec, ...] = field(default_factory=tuple)
    # The harness workflow that answers this case and the inputs it is run with;
    # None when no workflow does. Only the workflow adapter reads them.
    workflow: str | None = None
    workflow_inputs: WorkflowInputs | None = None

    def inputs_for(self, today: date) -> dict[str, Any]:
        """The workflow inputs, with a callable given the run's date."""
        inputs = self.workflow_inputs
        if inputs is None:
            return {}
        return dict(inputs(today) if callable(inputs) else inputs)


# Everything a satellite workflow narrates, in order; repeats and extras are fine.
SATELLITE_STAGES = ("search", "aoi", "coverage", "estimate", "submit", "progress", "open", "render")

# How an answer says a search came back empty. A pattern, not fixed phrases,
# because answers name the product in between: "No TEMPO NO2 data available".
NO_DATA_PATTERN = (
    r"\bno\b[^.\n]{0,40}\b(data|granules?|observations?|coverage|results?)\b"
    r"|not available|unavailable|could ?n[o']t find|does ?n[o']t exist|did ?n[o']t find"
)

# The science variables of the products the harness chooses for the covered cases.
TROPOMI_NO2_VARIABLE = "Tropospheric_NO2"
TEMPO_NO2_VARIABLE = "vertical_column_troposphere"
TEMPO_O3_VARIABLE = "column_amount_o3"
NO2_FILE_UNITS = "mol/m^2"

# What the geocoder returns for a region refusal case, so a buffer around
# "NYC" centres on a real point without a live Nominatim call.
GEOCODER_HIT = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "display_name": "New York, United States",
    "polygon": None,
    "bbox": [40.4774, 40.9176, -74.2591, -73.7002],
}


def _registry_entry(query: str, handle: str) -> dict[str, Any] | None:
    """The search hit for the registry collection ``query`` names, as the real service lists one.

    ``query`` is a collection's concept id or its registry key spelled as free
    text ("TEMPO NO2"); any other query names no collection.
    """
    from tta_backend.datasets.registry import load_registry

    wanted = query.strip().upper().replace(" ", "_")
    for key, cfg in load_registry().items():
        if query.strip() in (cfg.collection_id, cfg.short_name) or wanted == key.upper():
            return {
                "handle": handle, "dataset_handle": handle, "concept_id": cfg.collection_id,
                "short_name": cfg.short_name or key, "title": cfg.description or key,
            }
    return None


def _standard_handlers(
    *,
    dataset_handle: str = "dataset_1",
    aoi_handle: str = "aoi_1",
    obs_handle: str = "obs_1",
    granule_count: int = 5,
    estimated_bytes: int = 100,
    product: str | None = None,
    units: str | None = None,
) -> dict[str, Handler]:
    """The discovery, area, coverage, size, retrieve handlers most cases share.
    A case overrides entries to script a scenario (zero granules, an over-cap estimate).

    ``product`` is the registry key of the product the case's file holds; the
    dataset then describes that product's science variable (with its registry
    fill and valid range) in the file's ``units``, as the real service does.
    """

    async def search_datasets(query, filters, workspace_id):
        reply: dict[str, Any] = {"dataset_handle": dataset_handle, "short_name": query, "title": query}
        entry = _registry_entry(query, dataset_handle)
        if entry is not None:
            reply["datasets"] = [entry]
        return reply

    async def describe_dataset(dataset_handle, detail, workspace_id):
        if product is None:
            variable = {"name": "no2", "fill_value": -9999, "valid_min": 0, "valid_max": 1}
        else:
            from tta_backend.datasets.registry import load_registry

            cfg = load_registry()[product]
            variable = {
                "name": cfg.primary_var.rsplit("/", 1)[-1], "units": units or cfg.units,
                "fill_value": cfg.fill_value, "valid_min": cfg.valid_min, "valid_max": cfg.valid_max,
            }
        return {"dataset_handle": dataset_handle, "variables": [variable]}

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


def _volume_lifecycle_handlers(volume) -> dict[str, Handler]:
    return {
        "export_result": volume.export_result,
        "rematerialize": volume.rematerialize,
        "get_retrieval_status": volume.get_retrieval_status,
    }


def _aqs_ok(row: dict) -> dict:
    return {"Header": [{"status": "success"}], "Data": [row]}


_NEWARK_MONITOR = {
    "latitude": "40.0", "longitude": "-74.0",
    "state_code": "34", "county_code": "017", "site_number": "0006",
    "local_site_name": "Newark Firehouse",
}


def _daily_row(day: str, mean: float) -> dict:
    return {
        "date_local": day, "arithmetic_mean": str(mean), "state_code": "34",
        "county_code": "017", "site_number": "0006", "units_of_measure": "ppb",
        "pollutant_standard": "NO2 1-hour 2010", "local_site_name": "Newark Firehouse",
    }


async def _week_of_january_aqs_get(endpoint: str, params: dict) -> dict:
    if endpoint == "monitors/byBox":
        return _aqs_ok(_NEWARK_MONITOR)
    if endpoint == "dailyData/byBox":
        return {"Header": [{"status": "success"}], "Data": [
            _daily_row(day, mean) for day, mean in [("2024-01-01", 2.0), ("2024-01-02", 4.0), ("2024-01-03", 6.0)]
        ]}
    return {"Header": [{"status": "success"}], "Data": []}


async def _requested_day_aqs_get(endpoint: str, params: dict) -> dict:
    if endpoint == "monitors/byBox":
        return _aqs_ok(_NEWARK_MONITOR)
    if endpoint == "dailyData/byBox":
        return _aqs_ok(_daily_row(params.get("bdate", ""), 5.0))
    return {"Header": [{"status": "success"}], "Data": []}


def _dates(start: str, end: str | None = None) -> dict[str, str]:
    return {"start": start, "end": end or start}


def _days_before(today: date, days: int) -> str:
    return (today - timedelta(days=days)).isoformat()


def _covers(tool: str, *days: tuple[int, int, int]) -> tuple[Check, ...]:
    return tuple(TimeRangeCovers(tool, date(*day)) for day in days)


def build_cases(volume) -> list[BenchmarkCase]:
    """Build the 20 cases. ``volume`` is a fake_earthdata.HandleVolume; the
    plotting cases get tiny real Zarr/Parquet files in it to open."""
    import pandas as pd
    import pyarrow as pa
    import xarray as xr

    # Grids must lie over the area each prompt names: the system narrows data
    # to that area before plotting, and a grid elsewhere narrows to nothing.
    def grid(lat_start, lat_stop, lon_start, lon_stop, step):
        import numpy as np

        lats = np.round(np.arange(lat_start, lat_stop + step / 2, step), 4)
        lons = np.round(np.arange(lon_start, lon_stop + step / 2, step), 4)
        return lats, lons

    # Each file holds the science variable the harness's registry names for the
    # product the case is about, as the real service's files do.
    def map_dataset(variable="no2"):
        # New Jersey.
        import numpy as np

        lats, lons = grid(38.9, 41.4, -75.6, -73.9, 0.1)
        values = np.add.outer(lats - lats.min(), lons - lons.min()) + 1.0
        return xr.Dataset(
            {variable: (("lat", "lon"), values, {"units": "mol/m^2"})},
            coords={"lat": lats, "lon": lons},
        )

    def timeseries_dataset(variable="no2"):
        # Newark, NJ, every day of January 2024. Fine enough that the city's
        # own outline contains grid cells.
        import numpy as np

        times = pd.date_range("2024-01-01", "2024-01-31", freq="D")
        lats, lons = grid(40.60, 40.86, -74.30, -74.04, 0.02)
        daily = 1.0 + np.arange(len(times), dtype=float) / 10
        values = daily[:, None, None] + np.zeros((len(lats), len(lons)))
        return xr.Dataset(
            {variable: (("time", "lat", "lon"), values, {"units": "mol/m^2"})},
            coords={"time": times, "lat": lats, "lon": lons},
        )

    def point_timeseries_table():
        return pa.table({
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "no2": [1.0, 2.0, 3.0],
        }).replace_schema_metadata({b"units": b"mol/m^2"})

    def ground_validation_dataset():
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

    def newark_map_dataset(variable="no2"):
        return xr.Dataset(
            {variable: (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]], {"units": "mol/m^2"})},
            coords={"lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

    # A retrieved period carries a time axis, here a single step mid-period.
    def period_dataset(day, values):
        return xr.Dataset(
            {TEMPO_NO2_VARIABLE: (("time", "lat", "lon"), [values], {"units": "mol/m^2"})},
            coords={"time": pd.to_datetime([day]), "lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

    def period_a_dataset():
        return period_dataset("2025-06-15", [[1.0, 2.0], [3.0, 4.0]])

    def period_b_dataset():
        return period_dataset("2026-06-15", [[2.0, 4.0], [6.0, 8.0]])

    def ozone_dataset():
        # The Ozone Transport Region.
        import numpy as np

        lats, lons = grid(36.0, 46.0, -82.0, -66.0, 0.5)
        # Total column ozone, in the product's valid range of Dobson units.
        values = np.add.outer(lats - lats.min(), lons - lons.min()) + 280.0
        return xr.Dataset(
            {TEMPO_O3_VARIABLE: (("lat", "lon"), values, {"units": "DU"})},
            coords={"lat": lats, "lon": lons},
        )

    def aligned_dataset():
        return xr.Dataset(
            {"no2": (
                ("source", "lat", "lon"),
                [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]],
                {"units": "mol/m^2"},
            )},
            coords={"source": [0, 1], "lat": [40.0, 41.0], "lon": [-75.0, -74.0]},
        )

    volume.add_zarr("obs_map_1", lambda: map_dataset(TROPOMI_NO2_VARIABLE))
    volume.add_zarr("obs_ts_1", lambda: timeseries_dataset(TEMPO_NO2_VARIABLE))
    volume.add_zarr("obs_cmp_a", map_dataset)
    volume.add_zarr("obs_cmp_b", map_dataset)
    volume.add_parquet("obs_pt_ts_1", point_timeseries_table)
    volume.add_zarr("obs_validate_1", ground_validation_dataset)
    volume.add_zarr("obs_compare_period_a", period_a_dataset)
    volume.add_zarr("obs_compare_period_b", period_b_dataset)
    volume.add_zarr("cube_compare_aligned", aligned_dataset)
    volume.add_zarr("obs_e2e_satellite_1", newark_map_dataset)
    volume.add_zarr("obs_e2e_tropomi_1", lambda: newark_map_dataset(TROPOMI_NO2_VARIABLE))
    volume.add_zarr("obs_e2e_ozone_1", ozone_dataset)

    lifecycle = _volume_lifecycle_handlers(volume)

    async def point_timeseries_retrieve(dataset_handle, time_range, variables, aoi_handle, output_format, point_sample, workspace_id):
        return {"job_handle": "job_obs_pt_ts_1"}

    async def period_retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
        obs_handle = "obs_compare_period_b" if "2026" in time_range else "obs_compare_period_a"
        return {"job_handle": f"job_{obs_handle}", "obs_handle": obs_handle}

    async def aligned_cube(source_handles, method="outer", workspace_id="default"):
        return {"handle": "cube_compare_aligned", "status": "ok", "alignment_report": {"method": method}}

    standard = _standard_handlers()
    region_checks: tuple[Check, ...] = (
        Completed(),
        BoundaryNotCalled("define_area_of_interest"),
        BoundaryNotCalled("retrieve_subset"),
    )
    geocoder = (PatchSpec("tta_backend.utils.plotting.GeocodingService.ageocode", GEOCODER_HIT),)
    retrieval_base: tuple[Check, ...] = (Completed(), BoundaryCalled("retrieve_subset"), HasResult())

    return [
        # Discovery
        BenchmarkCase(
            id="discovery_no2_dataset",
            category="discovery",
            prompt="What NASA datasets are available for NO2 column density over New Jersey?",
            handlers=_standard_handlers(),
            checks=(Completed(), BoundaryCalled("search_datasets")),
        ),
        BenchmarkCase(
            id="discovery_soil_moisture",
            category="discovery",
            prompt="Find a soil moisture dataset I could use to map conditions over the Raritan basin.",
            handlers=_standard_handlers(dataset_handle="dataset_soil"),
            checks=(Completed(), BoundaryCalled("search_datasets")),
        ),
        # Retrieval
        BenchmarkCase(
            id="retrieval_tempo_no2",
            category="retrieval",
            prompt="Retrieve TEMPO NO2 over Houston for 2024-06-01.",
            handlers=_standard_handlers(obs_handle="obs_retrieval_1"),
            checks=(
                *retrieval_base,
                BoundaryCalled("search_datasets"),
                AreaRequested("Houston", (29.7604, -95.3698)),
                *_covers("retrieve_subset", (2024, 6, 1)),
            ),
        ),
        BenchmarkCase(
            id="retrieval_aod_month",
            category="retrieval",
            prompt="Retrieve MODIS aerosol optical depth over California for June 2024.",
            handlers=_standard_handlers(obs_handle="obs_retrieval_2"),
            checks=(
                *retrieval_base,
                BoundaryCalled("search_datasets"),
                AreaRequested("California", (36.7783, -119.4179)),
                *_covers("retrieve_subset", (2024, 6, 1), (2024, 6, 30)),
            ),
        ),
        # Plotting
        BenchmarkCase(
            id="plotting_single_map",
            category="plotting",
            prompt="Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
            handlers={**_standard_handlers(obs_handle="obs_map_1", product="TROPOMI_NO2", units=NO2_FILE_UNITS), **lifecycle},
            checks=(*retrieval_base, ChartProduced(), *_covers("retrieve_subset", (2024, 1, 15))),
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "New Jersey", "dates": _dates("2024-01-15"), "named": "TROPOMI"},
        ),
        BenchmarkCase(
            id="plotting_timeseries",
            category="plotting",
            prompt="Show me how NO2 changed over Newark NJ during January 2024.",
            handlers={**_standard_handlers(obs_handle="obs_ts_1", product="TEMPO_NO2", units=NO2_FILE_UNITS), **lifecycle},
            checks=(
                *retrieval_base,
                ChartProduced(),
                AreaRequested("Newark", (40.7357, -74.1724)),
                *_covers("retrieve_subset", (2024, 1, 1), (2024, 1, 31)),
            ),
            workflow="timeseries",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "Newark NJ", "dates": _dates("2024-01-01", "2024-01-31"), "mode": "region"},
        ),
        BenchmarkCase(
            id="plotting_point_timeseries",
            category="plotting",
            prompt=(
                "What was the TROPOMI NO2 reading at the single point 40.735,-74.172 "
                "(Newark, NJ) each day during January 2024? I just need that "
                "one location's history, not an area average."
            ),
            handlers={
                "search_datasets": standard["search_datasets"],
                "define_area_of_interest": standard["define_area_of_interest"],
                "retrieve_timeseries": point_timeseries_retrieve,
                **lifecycle,
            },
            checks=(
                Completed(),
                BoundaryCalled("retrieve_timeseries"),
                BoundaryNotCalled("retrieve_subset"),
                HasResult(),
                *_covers("retrieve_timeseries", (2024, 1, 1), (2024, 1, 31)),
            ),
            workflow="timeseries",
            workflow_inputs={
                "quantity": "no2_tropospheric_column", "area": "the single point 40.735,-74.172 (Newark, NJ)",
                "dates": _dates("2024-01-01", "2024-01-31"), "mode": "point", "named": "TROPOMI",
            },
        ),
        # Comparison setup
        BenchmarkCase(
            id="comparison_setup_two_cities",
            category="comparison_setup",
            prompt=(
                "Retrieve TROPOMI NO2 for both Newark NJ and Los Angeles CA on "
                "2024-01-15 so I can compare them side by side."
            ),
            handlers={**_standard_handlers(obs_handle="obs_cmp_a"), **lifecycle},
            checks=(
                *retrieval_base,
                AreaRequested("Newark", (40.7357, -74.1724)),
                AreaRequested("Los Angeles", (34.0522, -118.2437)),
            ),
        ),
        BenchmarkCase(
            id="comparison_setup_two_periods",
            category="comparison_setup",
            prompt="Retrieve TEMPO NO2 over Houston for both January 2024 and July 2024.",
            handlers={**_standard_handlers(obs_handle="obs_cmp_b"), **lifecycle},
            checks=(
                *retrieval_base,
                AreaRequested("Houston", (29.7604, -95.3698)),
                *_covers("retrieve_subset", (2024, 1, 15), (2024, 7, 15)),
            ),
        ),
        # Failure recovery
        BenchmarkCase(
            id="failure_recovery_no_data",
            category="failure_recovery",
            prompt="Plot TEMPO NO2 over Death Valley for 1990-01-01.",
            handlers=_standard_handlers(granule_count=0),
            checks=(Completed(), BoundaryNotCalled("retrieve_subset"), AnswerMatches(NO_DATA_PATTERN, "says no data was found")),
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "Death Valley", "dates": _dates("1990-01-01"), "named": "TEMPO"},
        ),
        BenchmarkCase(
            id="failure_recovery_retrieval_refused",
            category="failure_recovery",
            prompt=(
                "Retrieve TEMPO NO2 hourly data for all of 2024 over the entire "
                "continental United States."
            ),
            handlers=_standard_handlers(estimated_bytes=50 * 1024 ** 3),
            checks=(Completed(), BoundaryNotCalled("retrieve_subset")),
        ),
        # Ground validation
        BenchmarkCase(
            id="ground_validation_tempo_vs_epa",
            category="ground_validation",
            prompt="Compare TEMPO NO2 with EPA ground monitors over Newark NJ for the first week of January 2024.",
            handlers={**_standard_handlers(obs_handle="obs_validate_1"), **lifecycle},
            checks=(
                *retrieval_base,
                GroundSourceUsed(),
                AreaRequested("Newark", (40.7357, -74.1724)),
                *_covers("retrieve_subset", (2024, 1, 1), (2024, 1, 7)),
            ),
            aqs_get=_week_of_january_aqs_get,
        ),
        # Period comparison: the later period's range names 2026, which selects the second dataset.
        BenchmarkCase(
            id="comparison_period_tempo_no2",
            category="comparison",
            prompt="Compare TEMPO NO2 over New Jersey between June 2025 and June 2026 — did it change?",
            handlers={
                **_standard_handlers(
                    dataset_handle="dataset_compare", aoi_handle="aoi_compare", product="TEMPO_NO2", units=NO2_FILE_UNITS,
                ),
                "retrieve_subset": period_retrieve_subset,
                "align": aligned_cube,
                **lifecycle,
            },
            checks=(
                Completed(),
                HasResult(),
                *_covers("retrieve_subset", (2025, 6, 15), (2026, 6, 15)),
            ),
            workflow="period-comparison",
            workflow_inputs={
                "quantity": "no2_tropospheric_column", "area": "New Jersey", "first": _dates("2025-06-01", "2025-06-30"),
                "second": _dates("2026-06-01", "2026-06-30"), "change": "absolute", "named": "TEMPO",
            },
        ),
        # End to end: ground lookup
        BenchmarkCase(
            id="e2e_ground_relative_date",
            category="e2e_ground",
            prompt="What was the NO2 level in Newark, New Jersey yesterday?",
            handlers=_standard_handlers(),
            checks=(Completed(), GroundSourceUsed(), AqsDateRelativeToClock(days=-1)),
            aqs_get=_requested_day_aqs_get,
            workflow="monitor-readings",
            workflow_inputs=lambda today: {
                "pollutant": "no2", "area": "Newark, New Jersey",
                "dates": _dates(_days_before(today, 1)), "view": "series",
            },
        ),
        # End to end: satellite plot
        BenchmarkCase(
            id="e2e_satellite_plot",
            category="e2e_satellite",
            prompt="Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
            handlers={**_standard_handlers(obs_handle="obs_e2e_tropomi_1", product="TROPOMI_NO2", units=NO2_FILE_UNITS), **lifecycle},
            checks=(Completed(), SatelliteSourceUsed(), StageSubsequence(SATELLITE_STAGES)),
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "New Jersey", "dates": _dates("2024-01-15"), "named": "TROPOMI"},
        ),
        # End to end: both sources
        BenchmarkCase(
            id="e2e_cross_source_ground_vs_satellite",
            category="e2e_cross_source",
            prompt=(
                "Compare the ground NO2 monitors with TROPOMI satellite NO2 "
                "over Newark NJ for January 2024."
            ),
            handlers={**_standard_handlers(obs_handle="obs_e2e_tropomi_1", product="TROPOMI_NO2", units=NO2_FILE_UNITS), **lifecycle},
            checks=(Completed(), SatelliteSourceUsed(), GroundSourceUsed()),
            aqs_get=_requested_day_aqs_get,
        ),
        # End to end: routing. Prompts on which a lite model added a ground request nobody asked for.
        BenchmarkCase(
            id="e2e_routing_ozone_otc_satellite_only",
            category="e2e_routing",
            prompt="Show me what ozone looked like across the OTC on 2024-07-15.",
            handlers={**_standard_handlers(obs_handle="obs_e2e_ozone_1", product="TEMPO_O3TOT", units="DU"), **lifecycle},
            checks=(Completed(), SatelliteSourceUsed(), GroundSourceNotUsed()),
            workflow="map",
            workflow_inputs={"quantity": "o3_total_column", "area": "the OTC", "dates": _dates("2024-07-15")},
        ),
        BenchmarkCase(
            id="e2e_routing_non_aq_no_ground",
            category="e2e_routing",
            prompt="Show soil moisture over Iowa for June 2024 and check it against ground monitors.",
            handlers={**_standard_handlers(obs_handle="obs_e2e_satellite_1"), **lifecycle},
            checks=(Completed(), SatelliteSourceUsed(), GroundSourceNotUsed()),
        ),
        # Region refusal: a refused region must never become a retrieval of another one.
        BenchmarkCase(
            id="region_refusal_unresolvable_member",
            category="region_refusal",
            prompt="map NO2 over NY + Wakanda for June 1-7 2024",
            handlers=_standard_handlers(),
            checks=(*region_checks, AnswerMentionsAny(("wakanda",))),
            patches=geocoder,
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "NY + Wakanda", "dates": _dates("2024-06-01", "2024-06-07")},
        ),
        BenchmarkCase(
            id="region_refusal_buffer_too_large",
            category="region_refusal",
            prompt="map NO2 within 3000 miles of NYC for June 1-7 2024",
            handlers=_standard_handlers(),
            checks=region_checks,
            patches=geocoder,
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "within 3000 miles of NYC", "dates": _dates("2024-06-01", "2024-06-07")},
        ),
    ]
