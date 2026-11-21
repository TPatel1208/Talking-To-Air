"""The benchmark cases.

Each case is a question, the data the fake services return for it, and checks
on the outcome. Every case enters a system the way a user does, so none
presumes which agent or workflow answers. Cases name only the data boundary
(the earthdata tools and EPA AQS) and what the user sees, never a system's
own tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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
    StageSequenceAny,
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
    # Replaces ``checks`` when a harness adapter (workflow or explore) runs the case.
    harness_checks: tuple[Check, ...] | None = None
    # Stub for the EPA AQS request function; None answers every request with an empty success.
    aqs_get: AqsGetHandler | None = None
    patches: tuple[PatchSpec, ...] = field(default_factory=tuple)
    # The harness workflow that answers this case and the inputs it is run with;
    # None when no workflow does. Only the workflow adapter reads them.
    workflow: str | None = None
    workflow_inputs: WorkflowInputs | None = None
    # The specialist bundle that answers the case for the exploration adapter;
    # None when that adapter doesn't run it. Only the exploration adapter reads it.
    bundle: str | None = None
    # Each place the prompt names, copied exactly as written; the harness resolves them in code.
    explore_places: tuple[str, ...] = ()
    # True where stopping to ask the user a question is the right outcome, so a run that does counts as finished.
    expects_question: bool = False

    def inputs_for(self, today: date) -> dict[str, Any]:
        """The workflow inputs, with a callable given the run's date."""
        inputs = self.workflow_inputs
        if inputs is None:
            return {}
        return dict(inputs(today) if callable(inputs) else inputs)


# Everything a satellite workflow narrates, in order; repeats and extras are fine.
SATELLITE_STAGES = ("search", "aoi", "coverage", "estimate", "submit", "progress", "open", "render")
# The harness narrates capability names: the area resolved, satellite data, then a map.
HARNESS_SATELLITE_STAGES = ("sources.areas.resolve", "sources.satellite.get_data", "render.map")

# How an answer says a search came back empty. A pattern, not fixed phrases,
# because answers name the product in between: "No TEMPO NO2 data available".
NO_DATA_PATTERN = (
    r"\bno\b[^.\n]{0,40}\b(data|granules?|observations?|coverage|results?)\b"
    r"|not available|unavailable|could ?n[o']t find|does ?n[o']t exist|did ?n[o']t find"
    r"|outside (?:the |its )?[^.;,\n]{0,30}\brecord\b"
    r"|before (?:the |its )?[\w\- ,']{0,40}\brecords? (?:begins?|began|starts?)"
    r"|(?:instrument|satellite|sensor|tempo|tropomi|product)\b[^.;\n]{0,40}(?:did not|didn't) (?:exist|operate)"
    r"|(?:did not|didn't) (?:exist|operate) in \d{4}"
)

# How a harness answer says the product has no daily values: a refusal or negation tied to "daily".
DAILY_UNAVAILABLE_PATTERN = (
    r"\b(?:not|no|cannot|can't|could ?n[o']t|unable)\b[^.;\n]{0,60}\bdaily\b"
    r"|\bdaily\b[^.;\n]{0,60}(?:not available|unavailable|too coarse|only monthly|cadence_too_coarse)"
)

# The science variables of the products the harness chooses for the covered cases.
TROPOMI_NO2_VARIABLE = "Tropospheric_NO2"
TEMPO_NO2_VARIABLE = "vertical_column_troposphere"
TEMPO_O3_VARIABLE = "column_amount_o3"
NO2_FILE_UNITS = "mol/m^2"
MODIS_AOD_VARIABLE = "COMBINE_AOD_550_AVG"
MODIS_AOD_UNITS = "AOD (550nm)"
# South, north, west, east.
HOUSTON_BBOX = (29.4, 30.2, -96.0, -94.9)

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
            return {"handle": handle, "summary": {
                "concept_id": cfg.collection_id, "short_name": cfg.short_name or key,
                "version": cfg.version, "entry_title": cfg.description or key,
            }}
    return None


def _product_hit(key: str, handle: str) -> dict[str, Any]:
    """The search hit for registry product ``key``."""
    from tta_backend.datasets.registry import load_registry

    hit = _registry_entry(load_registry()[key].collection_id, handle)
    if hit is None:
        raise KeyError(key)
    return hit


def _no2_hits() -> list[dict[str, Any]]:
    return [_product_hit(key, f"dataset_{key.lower()}") for key in ("TEMPO_NO2", "TROPOMI_NO2", "OMI_NO2")]


def _soil_moisture_hits(handle: str) -> list[dict[str, Any]]:
    """SMAP soil moisture hits. Not registered, so the harness lists them with no product."""
    return [
        {"handle": handle, "summary": {
            "concept_id": "C0000000001-NSIDC_ECS", "short_name": "SPL3SMP_E",
            "entry_title": "SMAP L3 Radiometer Global Daily 9 km EASE-Grid Soil Moisture",
        }},
        {"handle": f"{handle}_l4", "summary": {
            "concept_id": "C0000000002-NSIDC_ECS", "short_name": "SPL4SMGP",
            "entry_title": "SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture",
        }},
    ]


def _standard_handlers(
    *,
    dataset_handle: str = "dataset_1",
    aoi_handle: str = "aoi_1",
    obs_handle: str = "obs_1",
    granule_count: int = 5,
    estimated_bytes: int = 100,
    product: str | None = None,
    units: str | None = None,
    search_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Handler]:
    """The discovery, area, coverage, size, retrieve handlers most cases share.
    A case overrides entries to script a scenario (zero granules, an over-cap estimate).

    ``product`` is the registry key of the product the case's file holds; the
    dataset then describes that product's science variable (with its registry
    fill and valid range) in the file's ``units``, as the real service does.
    Search lists that product's entry unless ``search_entries`` gives the hits;
    a ``concept_id`` filter keeps only the hits for that collection, as the
    real service's search does.
    """

    async def search_datasets(query, filters, workspace_id):
        reply: dict[str, Any] = {"dataset_handle": dataset_handle, "short_name": query, "title": query}
        hits = search_entries
        if hits is None and product is not None:
            hits = [_product_hit(product, dataset_handle)]
        if hits is None:
            entry = _registry_entry(query, dataset_handle)
            hits = [entry] if entry is not None else None
        concept_id = (filters or {}).get("concept_id")
        if hits and concept_id is not None:
            hits = [h for h in hits if (h.get("concept_id") or (h.get("summary") or {}).get("concept_id")) == concept_id]
        if hits:
            reply["datasets"] = hits
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
        # The service's own shape: megabytes over the sampled granules.
        return {"total_granules": granule_count, "total_size_mb": round(estimated_bytes / 1024 ** 2, 6)}

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


# Inside Newark's bounding box in the offline geocoder, a few km from its centre
# so a nearest-monitor distance is not zero.
_NEWARK_MONITOR_LAT, _NEWARK_MONITOR_LON = 40.7557, -74.1524
_NEWARK_MONITOR = {
    "latitude": str(_NEWARK_MONITOR_LAT), "longitude": str(_NEWARK_MONITOR_LON),
    "state_code": "34", "county_code": "017", "site_number": "0006",
    "local_site_name": "Newark Firehouse", "poc": "1", "parameter_name": "Nitrogen dioxide (NO2)",
}
_NO2_PARAM = "42602"


def _daily_row(day: str, mean: float) -> dict:
    return {
        # The daily 1-hour max sits above the daily mean, as it does for NO2.
        "date_local": day, "arithmetic_mean": str(mean), "first_max_value": str(round(mean * 1.5, 1)),
        "state_code": "34", "county_code": "017", "site_number": "0006", "poc": "1",
        "latitude": _NEWARK_MONITOR["latitude"], "longitude": _NEWARK_MONITOR["longitude"],
        "units_of_measure": "parts per billion", "pollutant_standard": "NO2 1-hour 2010",
        "local_site_name": "Newark Firehouse", "event_type": "No Events",
    }


def _newark_aqs_get(daily_rows: Callable[[dict], list[dict]]):
    """An AQS fake serving the Newark monitor and its NO2 daily rows on every
    endpoint the harness's monitor capabilities call."""

    def no2(params: dict) -> bool:
        return str(params.get("param", _NO2_PARAM)) == _NO2_PARAM

    def box_holds_monitor(params: dict) -> bool:
        try:
            return (
                float(params["minlat"]) <= _NEWARK_MONITOR_LAT <= float(params["maxlat"])
                and float(params["minlon"]) <= _NEWARK_MONITOR_LON <= float(params["maxlon"])
            )
        except (KeyError, TypeError, ValueError):
            return False

    async def get(endpoint: str, params: dict) -> dict:
        if endpoint == "monitors/byBox":
            if no2(params) and box_holds_monitor(params):
                return _aqs_ok(_NEWARK_MONITOR)
            return {"Header": [{"status": "success"}], "Data": []}
        if endpoint in ("dailyData/byBox", "dailyData/bySite"):
            at_site = (endpoint == "dailyData/byBox" and box_holds_monitor(params)) or (
                params.get("state") == "34"
                and params.get("county") == "017"
                and str(params.get("site")).zfill(4) == "0006"
            )
            if at_site and no2(params):
                return {"Header": [{"status": "success"}], "Data": daily_rows(params)}
        return {"Header": [{"status": "success"}], "Data": []}

    return get


def _days_requested(params: dict) -> list[date]:
    """Every day from the request's bdate to its edate, which AQS reads as YYYYMMDD."""
    first = datetime.strptime(str(params["bdate"]), "%Y%m%d").date()
    last = datetime.strptime(str(params["edate"]), "%Y%m%d").date()
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def _daily_means_aqs_get(means: dict[str, float]):
    """An AQS fake serving the Newark monitor's NO2 daily rows for the days of ``means`` the request spans."""

    def rows(params: dict) -> list[dict]:
        return [_daily_row(day.isoformat(), means[day.isoformat()]) for day in _days_requested(params) if day.isoformat() in means]

    return _newark_aqs_get(rows)


_week_of_january_aqs_get = _daily_means_aqs_get({f"2024-01-{day:02d}": 2.0 * day for day in range(1, 8)})
# A January of daily means that rises and falls with the satellite fake's steady climb, without matching it exactly.
_january_aqs_get = _daily_means_aqs_get(
    {f"2024-01-{day:02d}": round(2.0 + day * 0.5 + (day % 3) * 0.4, 1) for day in range(1, 32)}
)
_requested_day_aqs_get = _newark_aqs_get(lambda params: [_daily_row(str(params.get("bdate", "")), 5.0)])


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

    # A product's variable on a regular grid over one area, one step per day.
    def daily_dataset(variable, bbox, step, first_day, last_day, units=NO2_FILE_UNITS, level=1.0):
        import numpy as np

        south, north, west, east = bbox
        times = pd.date_range(first_day, last_day, freq="D")
        lats, lons = grid(south, north, west, east, step)
        daily = level + np.arange(len(times), dtype=float) / 10
        spatial = np.add.outer(lats - lats.min(), lons - lons.min()) * 0.1
        return xr.Dataset(
            {variable: (("time", "lat", "lon"), daily[:, None, None] + spatial, {"units": units})},
            coords={"time": times, "lat": lats, "lon": lons},
        )

    def point_timeseries_table(column="no2"):
        return pa.table({
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            column: [1.0, 2.0, 3.0],
        }).replace_schema_metadata({b"units": b"mol/m^2"})

    def ground_validation_dataset():
        # The cell holding the Newark monitor reads 1, 2, 3 ... over the first
        # week of January; every other cell reads 10.
        import numpy as np

        times = pd.date_range("2024-01-01", periods=7, freq="D")
        lats, lons = grid(40.0, 40.9, -74.3, -74.0, 0.02)
        values = np.full((len(times), len(lats), len(lons)), 10.0)
        lat_i = int(np.abs(lats - _NEWARK_MONITOR_LAT).argmin())
        lon_i = int(np.abs(lons - _NEWARK_MONITOR_LON).argmin())
        values[:, lat_i, lon_i] = np.arange(1, len(times) + 1, dtype=float)
        return xr.Dataset(
            {TEMPO_NO2_VARIABLE: (("time", "lat", "lon"), values, {"units": NO2_FILE_UNITS})},
            coords={"time": times, "lat": lats, "lon": lons},
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
    volume.add_zarr("obs_cmp_a", lambda: map_dataset(TROPOMI_NO2_VARIABLE))
    volume.add_zarr("obs_cmp_la", lambda: daily_dataset(
        TROPOMI_NO2_VARIABLE, (33.6, 34.4, -118.8, -118.0), 0.1, "2024-01-15", "2024-01-15",
    ))
    volume.add_zarr("obs_cmp_houston_jan", lambda: daily_dataset(
        TEMPO_NO2_VARIABLE, HOUSTON_BBOX, 0.05, "2024-01-01", "2024-01-31",
    ))
    volume.add_zarr("obs_cmp_houston_jul", lambda: daily_dataset(
        TEMPO_NO2_VARIABLE, HOUSTON_BBOX, 0.05, "2024-07-01", "2024-07-31", level=2.0,
    ))
    volume.add_zarr("obs_retrieval_1", lambda: daily_dataset(
        TEMPO_NO2_VARIABLE, HOUSTON_BBOX, 0.05, "2024-06-01", "2024-06-01",
    ))
    volume.add_zarr("obs_retrieval_2", lambda: daily_dataset(
        MODIS_AOD_VARIABLE, (32.0, 42.5, -124.5, -114.0), 0.5, "2024-06-01", "2024-06-30", units=MODIS_AOD_UNITS, level=0.2,
    ))
    volume.add_zarr("obs_e2e_cross_tropomi", lambda: timeseries_dataset(TROPOMI_NO2_VARIABLE))
    volume.add_zarr("obs_e2e_cross_tempo", lambda: timeseries_dataset(TEMPO_NO2_VARIABLE))
    volume.add_parquet("obs_e2e_cross_pt_tropomi", lambda: point_timeseries_table(TROPOMI_NO2_VARIABLE))
    volume.add_parquet("obs_e2e_cross_pt_tempo", lambda: point_timeseries_table(TEMPO_NO2_VARIABLE))
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

    # One file per area, so a retrieval of Los Angeles is not served New Jersey's data.
    async def city_aoi(location, workspace_id):
        west = "los angeles" in str(location).lower() or str(location).startswith("-118")
        return {"aoi_handle": "aoi_west" if west else "aoi_1", "location": location}

    async def city_retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
        obs_handle = "obs_cmp_la" if aoi_handle == "aoi_west" else "obs_cmp_a"
        return {"job_handle": f"job_{obs_handle}", "obs_handle": obs_handle}

    # The month the range falls in picks the file.
    async def month_retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
        obs_handle = "obs_cmp_houston_jul" if "-07-" in time_range else "obs_cmp_houston_jan"
        return {"job_handle": f"job_{obs_handle}", "obs_handle": obs_handle}

    # The variable asked for picks the product's file; no variable means the TROPOMI one.
    async def cross_source_retrieve_subset(dataset_handle, aoi_handle, time_range, variables, output_format, workspace_id):
        obs_handle = "obs_e2e_cross_tempo" if TEMPO_NO2_VARIABLE in variables else "obs_e2e_cross_tropomi"
        return {"job_handle": f"job_{obs_handle}", "obs_handle": obs_handle}

    async def cross_source_point_series(dataset_handle, time_range, variables, aoi_handle, output_format, point_sample, workspace_id):
        suffix = "tempo" if TEMPO_NO2_VARIABLE in variables else "tropomi"
        return {"job_handle": f"job_obs_e2e_cross_pt_{suffix}"}

    standard = _standard_handlers(search_entries=_no2_hits())
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
            bundle="find-data",
            explore_places=("New Jersey",),
            prompt="What NASA datasets are available for NO2 column density over New Jersey?",
            handlers=_standard_handlers(search_entries=_no2_hits()),
            checks=(Completed(), BoundaryCalled("search_datasets")),
        ),
        BenchmarkCase(
            id="discovery_soil_moisture",
            category="discovery",
            bundle="find-data",
            explore_places=("Raritan basin",),
            prompt="Find a soil moisture dataset I could use to map conditions over the Raritan basin.",
            handlers=_standard_handlers(dataset_handle="dataset_soil", search_entries=_soil_moisture_hits("dataset_soil")),
            checks=(Completed(), BoundaryCalled("search_datasets")),
        ),
        # Retrieval
        BenchmarkCase(
            id="retrieval_tempo_no2",
            category="retrieval",
            bundle="analyse",
            explore_places=("Houston",),
            prompt="Retrieve TEMPO NO2 over Houston for 2024-06-01.",
            handlers={**_standard_handlers(obs_handle="obs_retrieval_1", product="TEMPO_NO2", units=NO2_FILE_UNITS), **lifecycle},
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
            bundle="analyse",
            explore_places=("California",),
            prompt="Retrieve MODIS aerosol optical depth over California for June 2024.",
            handlers={**_standard_handlers(obs_handle="obs_retrieval_2", product="MODIS_AOD_TERRA"), **lifecycle},
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
            bundle="analyse",
            explore_places=("New Jersey",),
            prompt="Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
            handlers={**_standard_handlers(obs_handle="obs_map_1", product="TROPOMI_NO2", units=NO2_FILE_UNITS), **lifecycle},
            checks=(*retrieval_base, ChartProduced(), *_covers("retrieve_subset", (2024, 1, 15))),
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "New Jersey", "dates": _dates("2024-01-15"), "named": "TROPOMI"},
        ),
        BenchmarkCase(
            id="plotting_timeseries",
            category="plotting",
            bundle="analyse",
            explore_places=("Newark NJ",),
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
            bundle="analyse",
            explore_places=("Newark, NJ",),
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
            # The only TROPOMI NO2 product is monthly, so the harness correctly refuses a daily series.
            harness_checks=(
                Completed(),
                AnswerMatches(DAILY_UNAVAILABLE_PATTERN, "says daily values are not available"),
                BoundaryNotCalled("retrieve_timeseries"),
                BoundaryNotCalled("retrieve_subset"),
            ),
            workflow="timeseries",
            workflow_inputs={
                "quantity": "no2_tropospheric_column", "area": "Newark, NJ",
                "dates": _dates("2024-01-01", "2024-01-31"), "mode": "point", "named": "TROPOMI", "cadence": "daily",
            },
        ),
        # Comparison setup
        BenchmarkCase(
            id="comparison_setup_two_cities",
            category="comparison_setup",
            bundle="analyse",
            explore_places=("Newark NJ", "Los Angeles CA"),
            prompt=(
                "Retrieve TROPOMI NO2 for both Newark NJ and Los Angeles CA on "
                "2024-01-15 so I can compare them side by side."
            ),
            handlers={
                **_standard_handlers(obs_handle="obs_cmp_a", product="TROPOMI_NO2", units=NO2_FILE_UNITS),
                "define_area_of_interest": city_aoi,
                "retrieve_subset": city_retrieve_subset,
                **lifecycle,
            },
            checks=(
                *retrieval_base,
                AreaRequested("Newark", (40.7357, -74.1724)),
                AreaRequested("Los Angeles", (34.0522, -118.2437)),
            ),
        ),
        BenchmarkCase(
            id="comparison_setup_two_periods",
            category="comparison_setup",
            bundle="analyse",
            explore_places=("Houston",),
            prompt="Retrieve TEMPO NO2 over Houston for both January 2024 and July 2024.",
            handlers={
                **_standard_handlers(obs_handle="obs_cmp_houston_jan", product="TEMPO_NO2", units=NO2_FILE_UNITS),
                "retrieve_subset": month_retrieve_subset,
                **lifecycle,
            },
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
            bundle="analyse",
            explore_places=("Death Valley",),
            prompt="Plot TEMPO NO2 over Death Valley for 1990-01-01.",
            handlers=_standard_handlers(granule_count=0, search_entries=_no2_hits()),
            checks=(Completed(), BoundaryNotCalled("retrieve_subset"), AnswerMatches(NO_DATA_PATTERN, "says no data was found")),
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "Death Valley", "dates": _dates("1990-01-01"), "named": "TEMPO"},
        ),
        BenchmarkCase(
            id="failure_recovery_retrieval_refused",
            category="failure_recovery",
            bundle="analyse",
            explore_places=("the entire continental United States",),
            prompt=(
                "Retrieve TEMPO NO2 hourly data for all of 2024 over the entire "
                "continental United States."
            ),
            handlers=_standard_handlers(estimated_bytes=50 * 1024 ** 3, search_entries=_no2_hits()),
            checks=(Completed(), BoundaryNotCalled("retrieve_subset")),
        ),
        # Ground validation
        BenchmarkCase(
            id="ground_validation_tempo_vs_epa",
            category="ground_validation",
            bundle="analyse",
            explore_places=("Newark NJ",),
            prompt="Compare TEMPO NO2 with EPA ground monitors over Newark NJ for the first week of January 2024.",
            handlers={**_standard_handlers(obs_handle="obs_validate_1", product="TEMPO_NO2", units=NO2_FILE_UNITS), **lifecycle},
            checks=(
                *retrieval_base,
                GroundSourceUsed(),
                AreaRequested("Newark", (40.7357, -74.1724)),
                *_covers("retrieve_subset", (2024, 1, 1), (2024, 1, 7)),
            ),
            aqs_get=_week_of_january_aqs_get,
            workflow="satellite-vs-monitor",
            workflow_inputs={
                "quantity": "no2_tropospheric_column", "area": "Newark NJ", "dates": _dates("2024-01-01", "2024-01-07"),
                "named": "TEMPO",
            },
        ),
        # Period comparison: the later period's range names 2026, which selects the second dataset.
        BenchmarkCase(
            id="comparison_period_tempo_no2",
            category="comparison",
            bundle="analyse",
            explore_places=("New Jersey",),
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
            bundle="analyse",
            explore_places=("Newark, New Jersey",),
            prompt="What was the NO2 level in Newark, New Jersey yesterday?",
            handlers=_standard_handlers(search_entries=_no2_hits()),
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
            bundle="analyse",
            explore_places=("New Jersey",),
            prompt="Plot TROPOMI NO2 over New Jersey for 2024-01-15.",
            handlers={**_standard_handlers(obs_handle="obs_e2e_tropomi_1", product="TROPOMI_NO2", units=NO2_FILE_UNITS), **lifecycle},
            checks=(Completed(), SatelliteSourceUsed(), StageSequenceAny((SATELLITE_STAGES,), needing_result=(HARNESS_SATELLITE_STAGES,))),
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "New Jersey", "dates": _dates("2024-01-15"), "named": "TROPOMI"},
        ),
        # End to end: both sources
        BenchmarkCase(
            id="e2e_cross_source_ground_vs_satellite",
            category="e2e_cross_source",
            bundle="analyse",
            explore_places=("Newark NJ",),
            prompt=(
                "Compare the ground NO2 monitors with TROPOMI satellite NO2 "
                "over Newark NJ for January 2024."
            ),
            handlers={
                **_standard_handlers(
                    obs_handle="obs_e2e_cross_tropomi", product="TROPOMI_NO2", units=NO2_FILE_UNITS, search_entries=_no2_hits(),
                ),
                "retrieve_subset": cross_source_retrieve_subset,
                "retrieve_timeseries": cross_source_point_series,
                **lifecycle,
            },
            checks=(Completed(), SatelliteSourceUsed(), GroundSourceUsed()),
            # The only TROPOMI NO2 product is monthly and the workflow pairs by day, so the harness
            # correctly refuses instead of swapping in another instrument.
            harness_checks=(
                Completed(),
                AnswerMatches(DAILY_UNAVAILABLE_PATTERN, "says daily values are not available"),
                BoundaryNotCalled("retrieve_subset"),
                BoundaryNotCalled("retrieve_timeseries"),
            ),
            aqs_get=_january_aqs_get,
            workflow="satellite-vs-monitor",
            workflow_inputs={
                "quantity": "no2_tropospheric_column", "area": "Newark NJ", "dates": _dates("2024-01-01", "2024-01-31"),
                "named": "TROPOMI",
            },
        ),
        # End to end: routing. Prompts on which a lite model added a ground request nobody asked for.
        BenchmarkCase(
            id="e2e_routing_ozone_otc_satellite_only",
            category="e2e_routing",
            bundle="analyse",
            explore_places=("the OTC",),
            prompt="Show me what ozone looked like across the OTC on 2024-07-15.",
            handlers={**_standard_handlers(obs_handle="obs_e2e_ozone_1", product="TEMPO_O3TOT", units="DU"), **lifecycle},
            checks=(Completed(), SatelliteSourceUsed(), GroundSourceNotUsed()),
            workflow="map",
            workflow_inputs={"quantity": "o3_total_column", "area": "the OTC", "dates": _dates("2024-07-15")},
        ),
        BenchmarkCase(
            id="e2e_routing_non_aq_no_ground",
            category="e2e_routing",
            bundle="analyse",
            explore_places=("Iowa",),
            prompt="Show soil moisture over Iowa for June 2024 and check it against ground monitors.",
            handlers={
                **_standard_handlers(obs_handle="obs_e2e_satellite_1", search_entries=_soil_moisture_hits("dataset_1")),
                **lifecycle,
            },
            checks=(Completed(), SatelliteSourceUsed(), GroundSourceNotUsed()),
        ),
        # Region refusal: a refused region must never become a retrieval of another one.
        BenchmarkCase(
            id="region_refusal_unresolvable_member",
            category="region_refusal",
            bundle="analyse",
            explore_places=("NY + Wakanda",),
            prompt="map NO2 over NY + Wakanda for June 1-7 2024",
            handlers=_standard_handlers(search_entries=_no2_hits()),
            checks=(*region_checks, AnswerMentionsAny(("wakanda",))),
            patches=geocoder,
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "NY + Wakanda", "dates": _dates("2024-06-01", "2024-06-07")},
            expects_question=True,
        ),
        BenchmarkCase(
            id="region_refusal_buffer_too_large",
            category="region_refusal",
            bundle="analyse",
            explore_places=("within 3000 miles of NYC",),
            prompt="map NO2 within 3000 miles of NYC for June 1-7 2024",
            handlers=_standard_handlers(search_entries=_no2_hits()),
            checks=region_checks,
            patches=geocoder,
            workflow="map",
            workflow_inputs={"quantity": "no2_tropospheric_column", "area": "within 3000 miles of NYC", "dates": _dates("2024-06-01", "2024-06-07")},
            expects_question=True,
        ),
    ]
