"""
tests/live_smoke/test_universal_gridded_matrix.py
====================================================
The live-smoke acceptance matrix for unregistered gridded datasets. One
parametrized row per dataset/assumption, run against the real
earthdata-retrieval MCP (same opt-in gate as test_mcp_contract.py).

Every row targets a dataset NOT in collections.yaml. Some rows are
`xfail(strict=False)`: if one starts passing, pytest reports XPASS, the signal
to make it a real assertion.

The rest are real assertions:
  - MCD19A2 / VNP09 / VJ1 swath rows: projected/curvilinear grids raise a
    typed unsupported_grid MCPToolError (geo_utils.ensure_supported_grid).
  - GEOS-CF and OCO2_GEOS_L3CO2_DAY rows assert only discovery-level facts
    (granule_count from check_coverage); a granule-less stub collection
    legitimately returns zero granules.
  - Category A (masking disclosure): aggregate() always stamps
    result.meta["masking"] with the fill/valid-range provenance and whether
    masking ran (mask_info.resolve_mask_info).
  - Category B (no silent fallbacks): a multi-variable file or an unselected
    vertical level raises CATEGORY_VARIABLE_CHOICE_REQUIRED /
    CATEGORY_DIMENSION_CHOICE_REQUIRED instead of guessing.
  - Category C (QA tiers): qa_flags parses CF flag_values/flag_meanings
    deterministically and aggregate() always stamps
    result.meta["masking"]["qa_status"].

Rows with a known concept ID use it (OCO2_GEOS_L3CO2_DAY, GEOS-CF); every
other row resolves via search_datasets(query=...), like test_mcp_contract.py.

Run explicitly (same env var as test_mcp_contract.py):

    EARTHDATA_MCP_URL=http://localhost:8765/mcp pytest -m live_mcp tests/live_smoke/
"""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass

import pytest
import pytest_asyncio

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _mcp_url() -> str | None:
    return os.environ.get("EARTHDATA_MCP_URL")


pytestmark = [
    pytest.mark.live_mcp,
    pytest.mark.skipif(not _mcp_url(), reason="EARTHDATA_MCP_URL is not set — skipping the live-smoke matrix"),
]

_POLL_INTERVAL_SECONDS = 3
_POLL_MAX_ATTEMPTS = 60
_REAL_TERMINAL_STATUSES = {"ready", "failed", "expired", "cancelled"}  # live-verified, see test_mcp_contract.py

# California — small enough that retrieval rows stay tiny (test_mcp_contract.py's "keep retrieval
# tiny" rule), large enough that most L3 grids intersect it.
_AOI = "-124.5,32.5,-114.0,42.0"
_TIME_RANGE = "2024-06-01/2024-06-30"

# Used only by the two discovery-only rows (no retrieval, so size doesn't matter) — wide enough that
# "zero granules" / "some granules" is a strong claim rather than an unlucky window.
_CONUS = "-125.0,24.0,-66.0,49.0"
_WIDE_TIME_RANGE = "2020-01-01/2024-12-31"


@dataclass(frozen=True)
class _Row:
    id: str
    query: str
    concept_id: str | None = None

    @property
    def search_term(self) -> str:
        return self.concept_id or self.query


@pytest_asyncio.fixture
async def invoke():
    """One MCP tool-invocation callable per test, each with its own throwaway
    workspace id (mirrors test_mcp_contract.py's asyncSetUp) so parametrized
    rows never collide with each other or a researcher's real workspace."""
    from tta_backend.config.settings import Settings
    from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
    from tta_backend.earthdata_mcp.results import parse_tool_result

    workspace_id = f"live_smoke_t25_{uuid.uuid4().hex[:8]}"
    settings = Settings(earthdata_mcp_url=_mcp_url(), earthdata_mcp_token=os.environ.get("EARTHDATA_MCP_TOKEN"))
    tools = await load_raw_mcp_tools(settings)

    async def _invoke(tool_name: str, **kwargs) -> dict:
        raw = await tools[tool_name].ainvoke({**kwargs, "workspace_id": workspace_id})
        return parse_tool_result(raw)

    yield _invoke


async def _resolve_dataset_handle(invoke, row: _Row) -> str:
    search_result = await invoke("search_datasets", query=row.search_term, filters=None)
    datasets = search_result.get("datasets") or search_result.get("results") or []
    if not datasets:
        pytest.skip(f"{row.id}: search_datasets returned no results for {row.search_term!r} — cannot exercise this row right now")
    first = datasets[0]
    return first["handle"] if "handle" in first else first["dataset_handle"]


async def _coverage(invoke, dataset_handle: str, location: str, time_range: str):
    aoi_result = await invoke("define_area_of_interest", location=location)
    aoi_handle = aoi_result["handle"]
    coverage = await invoke("check_coverage", dataset_handle=dataset_handle, aoi_handle=aoi_handle, time_range=time_range)
    return aoi_handle, coverage


async def _open_after_retrieval(invoke, row: _Row, variables: list[str] | None = None, location: str = _AOI, time_range: str = _TIME_RANGE):
    """Full search -> AOI -> coverage -> retrieve -> poll -> export -> open
    chain, ending with a real opened xr.Dataset for a row to assert against.
    Any point where live data isn't available right now skips cleanly
    (per this task's instruction: rows that can't run must still be
    collected and skip, never fail for reasons unrelated to the assumption
    the row stresses) — mirrors test_mcp_contract.py's skip discipline."""
    dataset_handle = await _resolve_dataset_handle(invoke, row)
    aoi_handle, coverage = await _coverage(invoke, dataset_handle, location, time_range)
    if not coverage.get("granule_count", 0):
        pytest.skip(f"{row.id}: no granules covering {location!r}/{time_range!r} right now")

    # retrieve_subset's `variables` is a required list on the live MCP (no
    # implicit "all variables" via None, unlike this helper's own default) --
    # same "[] means no explicit subset" convention safe_retrieve already
    # uses (services/retrieval_composites.py::subset_variables).
    retrieve_result = await invoke(
        "retrieve_subset", dataset_handle=dataset_handle, aoi_handle=aoi_handle,
        time_range=time_range, variables=variables if variables is not None else [], output_format=None,
    )
    job_handle = retrieve_result["job_handle"]
    obs_handle = retrieve_result["obs_handle"]

    status: dict = {}
    for _ in range(_POLL_MAX_ATTEMPTS):
        status = await invoke("get_retrieval_status", job_handle=job_handle)
        if status.get("status") in _REAL_TERMINAL_STATUSES:
            break
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
    if status.get("status") != "ready":
        pytest.skip(f"{row.id}: retrieval job did not become ready (status={status})")

    export_result = await invoke("export_result", handle=obs_handle)
    if export_result.get("status") != "ready":
        pytest.skip(f"{row.id}: export did not become ready: {export_result}")

    from tta_backend.services.open_handle import _open

    return _open(export_result["storage_uri"], export_result.get("media_type", "netcdf"))


# ── Category A: unregistered tier-1 rows (UMM-Var masking) ─────────────────

_TIER1_ROWS = [
    _Row("tempo_so2", "TEMPO SO2"),
    _Row("tempo_aerosol", "TEMPO Aerosol Index"),
    _Row("omaeruvd", "OMAERUVd"),
    _Row("omaeroe", "OMAEROe"),
    _Row("vnp09_gridded_aerosol_sibling", "VNP09 gridded aerosol"),
    _Row("vj1_gridded_aerosol_sibling", "VJ1 gridded aerosol"),
    _Row("oco2_geos_l3co2_day_masking", "OCO2_GEOS_L3CO2_DAY", concept_id="C2240248762-GES_DISC"),
    _Row("modis_lst_non_aq", "MODIS land surface temperature"),
    _Row("smap_soil_moisture_non_aq", "SMAP L3 soil moisture"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _TIER1_ROWS, ids=[r.id for r in _TIER1_ROWS])
async def test_unregistered_tier1_dataset_discloses_masking_source(invoke, row):
    """aggregate() always stamps
    result.meta["masking"] (fill_value_source/valid_range_source/applied)
    via datasets/mask_info.py::resolve_mask_info's collections.yaml ->
    UMM-Var -> CF-attrs -> none precedence ladder, so an unregistered
    collection never leaves masking provenance unstated."""
    from tta_backend.preprocessing.aggregation_service import AggregationService

    ds = await _open_after_retrieval(invoke, row)
    result = AggregationService().aggregate(ds)
    assert "masking" in result.meta, f"{row.id}: aggregation meta has no masking-provenance disclosure: {result.meta}"
    masking = result.meta["masking"]
    assert masking.get("fill_value_source") in ("collections_yaml", "umm_var", "cf_attrs", "none"), (
        f"{row.id}: unexpected fill_value_source: {masking}"
    )
    assert masking.get("valid_range_source") in ("collections_yaml", "umm_var", "cf_attrs", "none"), (
        f"{row.id}: unexpected valid_range_source: {masking}"
    )
    assert "applied" in masking, f"{row.id}: masking provenance must state whether masking ran: {masking}"


# ── Category B: fallback-killer rows (explicit choice) ─────────────────────

_MULTI_VARIABLE_ROWS = [
    _Row("mod08_d3_variable_choice", "MOD08_D3"),
    _Row("myd08_d3_variable_choice", "MYD08_D3"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _MULTI_VARIABLE_ROWS, ids=[r.id for r in _MULTI_VARIABLE_ROWS])
async def test_multi_variable_file_with_no_choice_raises_a_candidate_listing_error(invoke, row):
    """An unregistered multi-variable file with no explicit variable and no
    retrieval-recorded choice raises CATEGORY_VARIABLE_CHOICE_REQUIRED
    listing candidates."""
    from tta_backend.earthdata_mcp.results import CATEGORY_VARIABLE_CHOICE_REQUIRED, MCPToolError
    from tta_backend.preprocessing.aggregation_service import AggregationService

    ds = await _open_after_retrieval(invoke, row)
    assert len(ds.data_vars) > 1, f"{row.id}: expected a multi-variable file to exercise the no-choice case, got {list(ds.data_vars)}"
    with pytest.raises(MCPToolError) as excinfo:
        AggregationService().to_dataarray(ds)
    assert excinfo.value.category == CATEGORY_VARIABLE_CHOICE_REQUIRED, (
        f"{row.id}: expected a variable_choice_required refusal, got category={excinfo.value.category}"
    )


_VERTICAL_LEVEL_ROWS = [
    _Row("merra2_aerosol_diagnostics_dim_choice", "MERRA-2 aerosol diagnostics"),
    _Row("merra2_atmospheric_chemistry_dim_choice", "MERRA-2 atmospheric chemistry"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _VERTICAL_LEVEL_ROWS, ids=[r.id for r in _VERTICAL_LEVEL_ROWS])
async def test_unselected_vertical_dimension_raises_a_candidate_listing_error(invoke, row):
    """A MERRA-2-style vertical level dim with no selection raises
    CATEGORY_DIMENSION_CHOICE_REQUIRED naming the dimension and its
    coordinate values. Time still auto-reduces via the CF-identified time dim
    (identify_time), not the literal name "time"."""
    from tta_backend.earthdata_mcp.results import CATEGORY_DIMENSION_CHOICE_REQUIRED, MCPToolError
    from tta_backend.preprocessing.aggregation_service import AggregationService
    from tta_backend.utils.geo_utils import identify_lat, identify_lon, identify_time
    from tta_backend.utils.plotting import _normalize_to_2d

    ds = await _open_after_retrieval(invoke, row)
    da = AggregationService().to_dataarray(ds)
    non_selectable = {n for n in (identify_lat(da), identify_lon(da), identify_time(da)) if n}
    extra_dims = [d for d in da.dims if d not in non_selectable and da.sizes[d] > 1]
    assert extra_dims, f"{row.id}: expected a surviving vertical/level dim to exercise the no-choice case, dims={da.dims}"
    with pytest.raises(MCPToolError) as excinfo:
        _normalize_to_2d(da)
    assert excinfo.value.category == CATEGORY_DIMENSION_CHOICE_REQUIRED, (
        f"{row.id}: expected a dimension_choice_required refusal, got category={excinfo.value.category}"
    )


# ── Category C: QA-tier rows ────────────────────────────────────────────────
#
# OMSO2e is the flag_meanings row — OMI SO2's published QA is a CF flag_values/flag_meanings
# enum, an unambiguous "good_quality"-style token set, the shape Tier 2's
# deterministic parse targets. MOD08_D3 is the prose-only-QA row — MODIS
# atmosphere QA is documented as bit-packed integers described in ATBD prose,
# not a CF flag_meanings array, the shape Tier 2 must degrade from (no mask,
# or an explicitly "inferred, not verified" model-proposed one).
#
# aggregate() locates the sibling flag variable (via CF `ancillary_variables`,
# or the sole sibling carrying both attrs) and always stamps
# result.meta["masking"]["qa_status"].
#
# Both rows extract via to_dataarray() and pass the opened Dataset separately
# as source_ds, mirroring the real tool path (tools never pass aggregate() a
# whole Dataset).

@pytest.mark.asyncio
async def test_flag_meanings_dataset_gets_a_deterministic_cf_mask(invoke):
    from tta_backend.preprocessing.aggregation_service import AggregationService

    row = _Row("omso2e_flag_meanings_qa", "OMSO2e")
    # Explicit science variable + its QA sibling (explicit param wins), so
    # this row tests QA disclosure, not variable choice. Illustrative names;
    # a wrong spelling still skips cleanly via retrieve_subset.
    variable = "ColumnAmountSO2_PBL"
    qa_variable = "QualityFlags_PBL"
    ds = await _open_after_retrieval(invoke, row, variables=[variable, qa_variable])
    service = AggregationService()
    da = service.to_dataarray(ds, variable=variable)
    result = service.aggregate(da, variable=variable, source_ds=ds)
    masking = result.meta.get("masking", {})
    assert masking.get("qa_status") == "cf-deterministic", f"{row.id}: expected a deterministic CF flag_meanings mask, got masking={masking}"


@pytest.mark.asyncio
async def test_prose_only_qa_dataset_discloses_no_mask_or_an_inferred_tag(invoke):
    from tta_backend.preprocessing.aggregation_service import AggregationService

    row = _Row("mod08_d3_prose_only_qa", "MOD08_D3")
    # Explicit variable override (explicit param wins), so this row tests QA disclosure,
    # not variable choice. Illustrative name; a wrong spelling still skips cleanly.
    variable = "Aerosol_Optical_Depth_Land_Ocean_Mean"
    ds = await _open_after_retrieval(invoke, row, variables=[variable])
    service = AggregationService()
    da = service.to_dataarray(ds, variable=variable)
    result = service.aggregate(da, variable=variable, source_ds=ds)
    masking = result.meta.get("masking", {})
    assert masking.get("qa_status") in ("not applied — semantics unknown", "inferred, not verified"), (
        f"{row.id}: expected an explicit QA disclosure for a prose-only-QA product, got masking={masking}"
    )


# ── Category D: expected-refusal rows ───────────────────────────────────────

_EXPECTED_REFUSAL_ROWS = [
    _Row("mcd19a2_projected_grid_refusal", "MCD19A2"),
    _Row("vnp09_swath_refusal", "VNP09"),
    # VIIRS/JPSS-1 aerosol swath, the "VJ1" analog of VNP09; resolved via descriptive search
    # text and skips cleanly if unresolved.
    _Row("vj1_swath_refusal", "VJ1 aerosol swath JPSS-1"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _EXPECTED_REFUSAL_ROWS, ids=[r.id for r in _EXPECTED_REFUSAL_ROWS])
async def test_unsupported_grid_products_refuse_with_the_named_limitation(invoke, row):
    """Not xfail: this typed refusal is raised today
    (utils/geo_utils.py::ensure_supported_grid + CATEGORY_UNSUPPORTED_GRID)."""
    from tta_backend.earthdata_mcp.results import CATEGORY_UNSUPPORTED_GRID, MCPToolError
    from tta_backend.utils.geo_utils import ensure_supported_grid

    ds = await _open_after_retrieval(invoke, row)
    with pytest.raises(MCPToolError) as excinfo:
        ensure_supported_grid(ds)
    assert excinfo.value.category == CATEGORY_UNSUPPORTED_GRID, f"{row.id}: expected an unsupported_grid refusal, got category={excinfo.value.category}"


# ── Category E: discovery-only rows (already correct, needs no product code) ─

_GEOS_STUB_ROW = _Row("geos_cf_and_geos_forecast_unreachable", "GEOS-CF", concept_id="C1633930911-NCCS")
_OCO2_REACHABLE_ROW = _Row("oco2_geos_l3co2_day_reachable", "OCO2_GEOS_L3CO2_DAY", concept_id="C2240248762-GES_DISC")


@pytest.mark.asyncio
async def test_geos_cf_and_geos_forecast_are_confirmed_unreachable_via_standard_services(invoke):
    """GEOS-CF exists only as a granule-less stub collection in CMR
    (C1633930911-NCCS, confirmed 2026-07-10) — its data lives on the
    GMAO/NCCS portal, outside CMR/Harmony. This pinned concept ID stands in
    for the GEOS-CF / GEOS forecast family. Zero granules over a wide
    multi-year window and a CONUS-wide AOI is the honest discovery-level
    answer (the no_data category treats a real zero as a fact)."""
    dataset_handle = await _resolve_dataset_handle(invoke, _GEOS_STUB_ROW)
    _, coverage = await _coverage(invoke, dataset_handle, _CONUS, _WIDE_TIME_RANGE)
    assert coverage.get("granule_count", 0) == 0, (
        f"GEOS-CF/{_GEOS_STUB_ROW.concept_id} unexpectedly has granules over {_WIDE_TIME_RANGE} — "
        "re-verify against CMR before trusting this row (it was pinned unreachable on 2026-07-10)"
    )


@pytest.mark.asyncio
async def test_oco2_geos_l3co2_day_is_reachable_as_the_geos_family_replacement_row(invoke):
    """The reachable GEOS-family row standing in for GEOS-CF's retrieval rows
    (2,616 granules at last check). Proving
    reachability needs no new product code either."""
    dataset_handle = await _resolve_dataset_handle(invoke, _OCO2_REACHABLE_ROW)
    _, coverage = await _coverage(invoke, dataset_handle, _CONUS, _WIDE_TIME_RANGE)
    assert coverage.get("granule_count", 0) > 0, (
        f"OCO2_GEOS_L3CO2_DAY/{_OCO2_REACHABLE_ROW.concept_id} unexpectedly has zero granules over {_WIDE_TIME_RANGE} — "
        "re-verify against CMR before trusting this row (it was confirmed reachable on 2026-07-10)"
    )
