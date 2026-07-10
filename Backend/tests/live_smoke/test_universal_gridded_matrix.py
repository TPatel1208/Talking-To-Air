"""
tests/live_smoke/test_universal_gridded_matrix.py
====================================================
PRD T25 Phase 0: the live-smoke acceptance matrix scaffold for "universal
gridded datasets". One parametrized row per dataset/assumption named in the
PRD's Acceptance Matrix section, run against the real earthdata-retrieval MCP
(same opt-in gate as test_mcp_contract.py — T17).

Today the satellite arm only trusts collections.yaml's ten registered
products (T25 Problem Statement); every row below targets a dataset NOT in
that registry, so today's silent-first-data-var / silent-mean-over-extra-
dims / no-masking-disclosure behavior is the *baseline* this suite pins.
Phases 1-4 turn rows green one at a time (see the PRD's Phasing section) —
most rows here are `xfail(strict=False)`: if a row unexpectedly starts
passing, pytest reports XPASS instead of failing the run, which is exactly
the "graduate this row" signal a later phase should notice and act on.

Several row shapes are real (non-xfail) assertions because today's behavior
is already correct, proved by work this PRD explicitly depends on:
  - MCD19A2 / VNP09 / VJ1 swath rows: T24 already raises a typed
    unsupported_grid MCPToolError for projected/curvilinear grids
    (utils/geo_utils.py::ensure_supported_grid). These are real
    "expected-refusal" rows, not placeholders.
  - The GEOS-CF/GEOS-forecast and OCO2_GEOS_L3CO2_DAY rows assert only
    discovery-level facts (granule_count from check_coverage) per this
    PRD's own instruction ("GEOS-CF rows assert the discovery-level honest
    answer, not retrieval") — a granule-less stub collection legitimately
    returns zero granules today, and OCO2's reachability needs no new
    product code either (T18's no_data category already treats a real zero
    as a fact, not a failure).
  - The Category A tier-1 masking-disclosure row graduated in Phase 1
    (metadata plumbing): AggregationService.aggregate() now always stamps
    result.meta["masking"] with the fill/valid-range provenance
    (collections_yaml/umm_var/cf_attrs/none) and whether masking actually
    ran (datasets/mask_info.py::resolve_mask_info), so this row no longer
    depends on a dataset actually being in collections.yaml.

Row-count note: the PRD's Acceptance Matrix paragraph is a summary of roles
("TEMPO SO2/Aerosol, OMAERUVd, OMAEROe, OMSO2e -> unregistered tier-1 rows",
etc.), not a literal enumerated table with concept IDs for every entry. This
file turns each named role into one or more rows (MOD08_D3 gets both a
choice-error row and, using an explicit variable override, a QA-disclosure
row) rather than inventing dataset names the PRD never mentioned. Where the
PRD pins a concept ID (OCO2_GEOS_L3CO2_DAY, GEOS-CF) this file uses it
directly; every other row resolves via search_datasets(query=...), the same
discovery path test_mcp_contract.py already relies on for TEMPO NO2.

Run explicitly (same env var as test_mcp_contract.py):

    EARTHDATA_MCP_URL=http://localhost:8765/mcp pytest -m live_mcp tests/live_smoke/
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass

import pytest
import pytest_asyncio

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


def _mcp_url() -> str | None:
    return os.environ.get("EARTHDATA_MCP_URL")


pytestmark = [
    pytest.mark.live_mcp,
    pytest.mark.skipif(not _mcp_url(), reason="EARTHDATA_MCP_URL is not set — skipping the T25 live-smoke matrix"),
]

_POLL_INTERVAL_SECONDS = 3
_POLL_MAX_ATTEMPTS = 60
_REAL_TERMINAL_STATUSES = {"ready", "failed", "expired", "cancelled"}  # T17 live-verified, see test_mcp_contract.py

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
    from config.settings import Settings
    from earthdata_mcp.client import load_raw_mcp_tools
    from earthdata_mcp.results import parse_tool_result

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

    retrieve_result = await invoke(
        "retrieve_subset", dataset_handle=dataset_handle, aoi_handle=aoi_handle,
        time_range=time_range, variables=variables, output_format=None,
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

    from services.open_handle import _open

    return _open(export_result["storage_uri"], export_result.get("media_type", "netcdf"))


# ── Category A: unregistered tier-1 rows (UMM-Var masking, Phase 1) ────────

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
    """T25 Phase 1 graduated this row: aggregate() always stamps
    result.meta["masking"] (fill_value_source/valid_range_source/applied)
    via datasets/mask_info.py::resolve_mask_info's collections.yaml ->
    UMM-Var -> CF-attrs -> none precedence ladder, so an unregistered
    collection never leaves masking provenance unstated."""
    from preprocessing.aggregation_service import AggregationService

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


# ── Category B: fallback-killer rows (explicit choice, Phase 2) ────────────

_MULTI_VARIABLE_ROWS = [
    _Row("mod08_d3_variable_choice", "MOD08_D3"),
    _Row("myd08_d3_variable_choice", "MYD08_D3"),
]


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "T25 Phase 2 (explicit variable choice): AggregationService.to_dataarray still falls back "
        "to next(iter(data.data_vars)) for an unregistered multi-variable file "
        "(aggregation_service.py:90) instead of raising a candidate-listing MCPToolError."
    ),
    strict=False,
)
@pytest.mark.parametrize("row", _MULTI_VARIABLE_ROWS, ids=[r.id for r in _MULTI_VARIABLE_ROWS])
async def test_multi_variable_file_with_no_choice_raises_a_candidate_listing_error(invoke, row):
    from earthdata_mcp.results import MCPToolError
    from preprocessing.aggregation_service import AggregationService

    ds = await _open_after_retrieval(invoke, row)
    assert len(ds.data_vars) > 1, f"{row.id}: expected a multi-variable file to exercise the no-choice case, got {list(ds.data_vars)}"
    with pytest.raises(MCPToolError):
        AggregationService().to_dataarray(ds)


_VERTICAL_LEVEL_ROWS = [
    _Row("merra2_aerosol_diagnostics_dim_choice", "MERRA-2 aerosol diagnostics"),
    _Row("merra2_atmospheric_chemistry_dim_choice", "MERRA-2 atmospheric chemistry"),
]


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "T25 Phase 2 (explicit dimension choice): utils/plotting.py::_normalize_to_2d still "
        "silently .mean()s over every surviving non-spatial dimension (e.g. MERRA-2's vertical "
        "level) instead of raising a candidate-listing error naming the coordinate values. Time "
        "still auto-reduces (unaffected by this row)."
    ),
    strict=False,
)
@pytest.mark.parametrize("row", _VERTICAL_LEVEL_ROWS, ids=[r.id for r in _VERTICAL_LEVEL_ROWS])
async def test_unselected_vertical_dimension_raises_a_candidate_listing_error(invoke, row):
    from earthdata_mcp.results import MCPToolError
    from preprocessing.aggregation_service import AggregationService
    from utils.plotting import _normalize_to_2d

    ds = await _open_after_retrieval(invoke, row)
    da = AggregationService().to_dataarray(ds)
    extra_dims = [d for d in da.dims if d not in ("lat", "lon", "latitude", "longitude", "time")]
    assert extra_dims, f"{row.id}: expected a surviving vertical/level dim to exercise the no-choice case, dims={da.dims}"
    with pytest.raises(MCPToolError):
        _normalize_to_2d(da)


# ── Category C: QA-tier rows (Phase 3) ──────────────────────────────────────
#
# Phase 0 picks (per this task's instruction, recorded here): OMSO2e is the
# flag_meanings row — OMI SO2's published QA is a CF flag_values/flag_meanings
# enum, an unambiguous "good_quality"-style token set, the shape Tier 2's
# deterministic parse targets. MOD08_D3 is the prose-only-QA row — MODIS
# atmosphere QA is documented as bit-packed integers described in ATBD prose,
# not a CF flag_meanings array, the shape Tier 2 must degrade from (no mask,
# or an explicitly "inferred, not verified" model-proposed one).

@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "T25 Phase 3 (QA tiers): CF flag_values/flag_meanings are not yet parsed into a "
        "deterministic mask, and no 'cf-deterministic' provenance tag is recorded anywhere."
    ),
    strict=False,
)
async def test_flag_meanings_dataset_gets_a_deterministic_cf_mask(invoke):
    from preprocessing.aggregation_service import AggregationService

    row = _Row("omso2e_flag_meanings_qa", "OMSO2e")
    ds = await _open_after_retrieval(invoke, row)
    result = AggregationService().aggregate(ds)
    masking = result.meta.get("masking", {})
    assert masking.get("qa_status") == "cf-deterministic", f"{row.id}: expected a deterministic CF flag_meanings mask, got masking={masking}"


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "T25 Phase 3 (QA tiers) / Phase 1 (masking disclosure): a prose-only-QA product should "
        "either skip masking with 'quality flags: not applied — semantics unknown' in meta, or "
        "carry an explicit 'inferred, not verified' tag — today aggregation meta says nothing "
        "about QA at all."
    ),
    strict=False,
)
async def test_prose_only_qa_dataset_discloses_no_mask_or_an_inferred_tag(invoke):
    from preprocessing.aggregation_service import AggregationService

    row = _Row("mod08_d3_prose_only_qa", "MOD08_D3")
    # Explicit variable override (PRD resolution order: explicit param wins) so this row exercises
    # QA disclosure without depending on Phase 2's variable-choice error landing first. Illustrative
    # name, not concept-ID-pinned by the PRD — a wrong spelling still skips cleanly via retrieve_subset.
    variable = "Aerosol_Optical_Depth_Land_Ocean_Mean"
    ds = await _open_after_retrieval(invoke, row, variables=[variable])
    result = AggregationService().aggregate(ds, variable=variable)
    masking = result.meta.get("masking", {})
    assert masking.get("qa_status") in ("not applied — semantics unknown", "inferred, not verified"), (
        f"{row.id}: expected an explicit QA disclosure for a prose-only-QA product, got masking={masking}"
    )


# ── Category D: expected-refusal rows (already correct, T24) ───────────────

_EXPECTED_REFUSAL_ROWS = [
    _Row("mcd19a2_projected_grid_refusal", "MCD19A2"),
    _Row("vnp09_swath_refusal", "VNP09"),
    # VIIRS/JPSS-1 aerosol swath, the "VJ1" analog of VNP09 named in the PRD — exact short name not
    # concept-ID-pinned there; resolved via descriptive search text and skips cleanly if unresolved.
    _Row("vj1_swath_refusal", "VJ1 aerosol swath JPSS-1"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _EXPECTED_REFUSAL_ROWS, ids=[r.id for r in _EXPECTED_REFUSAL_ROWS])
async def test_unsupported_grid_products_refuse_with_the_named_limitation(invoke, row):
    """Not xfail: T24 already raises this typed refusal today
    (utils/geo_utils.py::ensure_supported_grid + CATEGORY_UNSUPPORTED_GRID)."""
    from earthdata_mcp.results import CATEGORY_UNSUPPORTED_GRID, MCPToolError
    from utils.geo_utils import ensure_supported_grid

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
    GMAO/NCCS portal, outside CMR/Harmony. "GEOS-CF" and "GEOS forecast" name
    the same product family in the PRD; this single pinned concept ID stands
    in for both of the PRD's "confirmed unreachable" rows. Zero granules over
    a wide multi-year window and a CONUS-wide AOI is the discovery-level
    honest answer this row proves — no new product code needed (T18's
    no_data category already treats a real zero as a fact, not a failure)."""
    dataset_handle = await _resolve_dataset_handle(invoke, _GEOS_STUB_ROW)
    _, coverage = await _coverage(invoke, dataset_handle, _CONUS, _WIDE_TIME_RANGE)
    assert coverage.get("granule_count", 0) == 0, (
        f"GEOS-CF/{_GEOS_STUB_ROW.concept_id} unexpectedly has granules over {_WIDE_TIME_RANGE} — "
        "re-verify against CMR before trusting this row (it was pinned unreachable on 2026-07-10)"
    )


@pytest.mark.asyncio
async def test_oco2_geos_l3co2_day_is_reachable_as_the_geos_family_replacement_row(invoke):
    """The reachable GEOS-family row standing in for GEOS-CF's retrieval rows
    (PRD Acceptance Matrix) — 2,616 granules confirmed 2026-07-10. Proving
    reachability needs no new product code either."""
    dataset_handle = await _resolve_dataset_handle(invoke, _OCO2_REACHABLE_ROW)
    _, coverage = await _coverage(invoke, dataset_handle, _CONUS, _WIDE_TIME_RANGE)
    assert coverage.get("granule_count", 0) > 0, (
        f"OCO2_GEOS_L3CO2_DAY/{_OCO2_REACHABLE_ROW.concept_id} unexpectedly has zero granules over {_WIDE_TIME_RANGE} — "
        "re-verify against CMR before trusting this row (it was confirmed reachable on 2026-07-10)"
    )
