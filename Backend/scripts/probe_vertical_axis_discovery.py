"""PRD T58 Phase 1 — the CF-discovery GATE. Prints a scorecard, asserts nothing.

The question this answers is not "does discovery work on TEMPO_O3PROF" (T56
already proved that). It is: **on products nobody has looked at, does CF
metadata identify the vertical axis, and when it can't, does it say so instead
of returning a plausible wrong one?** D2 stakes the whole tier balance on the
answer, and D3 makes this a gate precisely because the counter-evidence is real
-- the CF tier has never once fired for QA flags on a real product.

Graded FOUR ways, deliberately not two (PRD Phase 1). Counting only successes
flatters registry-as-truth, which scores zero ``resolved-wrong`` by
construction:

    resolved-correct   found an axis, and it is the right one
    resolved-WRONG     found a plausible-but-incorrect axis  <- the target
    refused-correctly  ambiguous or hybrid, and it said so
    refused-wrongly    a real axis it should have found

MERRA-2 model-level's correct result is ``refused-correctly``: its ``lev`` is a
hybrid-eta level number, not a pressure. A scorecard marking that as a miss
would drive us to "fix" the one safety behaviour we want.

Two suites, run against two DIFFERENT metadata surfaces, which is the finding
the gate turns on:

  * ``--discovery``  UMM-Var via the live ``describe_dataset``. Cheap, covers
    every candidate. Carries units and standard_name but NO dimensions, so the
    dim-spanning filter in ``vertical_axes_for_dim`` cannot run here -- only
    the per-variable ``vertical_axis_kind`` primitive can.
  * ``--granule``    a real retrieved granule opened through the production
    path, where ``vertical_axes_for_dim`` runs for real, plus the four
    falsifiable checks (monotonic / plausible range / hydrostatic / round-trip)
    and the dominance distributions D6's refusal rule is derived from.

    python Backend/scripts/probe_vertical_axis_discovery.py --discovery
    python Backend/scripts/probe_vertical_axis_discovery.py --granule
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import uuid

import numpy as np

_BACKEND = pathlib.Path(__file__).resolve().parent.parent  # Backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from tta_backend.utils.geo_utils import vertical_axis_kind  # noqa: E402


# ── Candidates ────────────────────────────────────────────────────────────────
# `expect` is what a CORRECT discovery mechanism should conclude, written down
# BEFORE running so the scorecard cannot be graded to whatever came out.
# `truth` names the variable that genuinely is the vertical axis (None where the
# honest answer is a refusal).
CANDIDATES = [
    {
        "key": "TEMPO_O3PROF",
        "query": "TEMPO ozone profile",
        "short_name": "TEMPO_O3PROF_L3",
        "science_var": "product/ozone_profile",
        "expect": "resolved-correct",
        "truth": {"pressure": "support_data/ozone_profile_pressure",
                  "altitude": "support_data/ozone_profile_altitude"},
        "note": "per-pixel CF auxiliary coordinates; the T56 product",
    },
    {
        "key": "M2I3NPASM",
        "query": "M2I3NPASM",
        "short_name": "M2I3NPASM",
        "science_var": "O3",
        "expect": "resolved-correct",
        "truth": {"pressure": "lev"},
        "note": "PRESSURE-level product: lev genuinely is hPa (finding 6)",
    },
    {
        "key": "M2I3NVASM",
        "query": "M2I3NVASM",
        "short_name": "M2I3NVASM",
        "science_var": "O3",
        "expect": "refused-correctly",
        "truth": {},
        "note": "MODEL-level product: lev is a hybrid-eta level 1..72, NOT a pressure",
    },
    {
        "key": "GEOS-CF",
        "query": "GEOS CF Composition Forecast",
        "short_name": "GEOS-CF Products",
        "science_var": "O3",
        "expect": "resolved-correct",
        "truth": {"pressure": "lev"},
        "note": "model output; ensure_supported_grid refuses it for plotting regardless",
    },
    {
        "key": "S5P_O3_PR",
        "query": "Sentinel-5P TROPOMI ozone profile",
        "short_name": None,
        "science_var": "PRODUCT/ozone_profile",
        "expect": "resolved-correct",
        "truth": {"pressure": "pressure"},
        "note": "L2 swath: discovery-only, unplottable by ensure_supported_grid",
    },
    {
        "key": "ML2O3",
        "query": "MLS Aura ozone profile Level 2",
        "short_name": "ML2O3",
        "science_var": "HDFEOS/SWATHS/O3/Data Fields/O3",
        "expect": "resolved-correct",
        "truth": {"pressure": "Pressure"},
        "note": "L2 profile: discovery-only",
    },
    {
        "key": "AIRS3STD",
        "query": "AIRS3STD",
        "short_name": "AIRS3STD",
        "science_var": "Temperature_A",
        "expect": "resolved-correct",
        "truth": {"pressure": "StdPressureLev"},
        "note": "gridded L3 with a real pressure coordinate",
    },
]


def _rows(result: dict) -> list:
    return result.get("datasets") or result.get("results") or []


def _row_identity(row: dict) -> tuple:
    """``(short_name, concept_id, title)``. The live MCP nests all three under
    ``summary``; reading them off the row itself yields empty strings, which
    silently made every short-name match fail and picked whatever came first
    (the O3PROF *L2 swath* instead of the L3 grid) on the spike's first run."""
    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    short = summary.get("short_name") or row.get("short_name") or meta.get("short_name") or ""
    concept = summary.get("concept_id") or row.get("concept_id") or meta.get("concept_id") or ""
    title = summary.get("entry_title") or row.get("title") or meta.get("title") or ""
    return str(short).strip(), str(concept).strip(), str(title).strip()


def _classify_inventory(variables: list) -> dict:
    """Every variable UMM-Var metadata makes look like a vertical axis, by kind.

    This is the real ``vertical_axis_kind`` primitive, run over the metadata
    surface UMM-Var actually publishes. Nothing is filtered by dimension --
    UMM-Var carries none -- which is exactly the point: whatever shows up here
    is what a metadata-only tier would have to choose between.
    """
    found: dict[str, list] = {"pressure": [], "altitude": []}
    for v in variables:
        name = v.get("name") or ""
        attrs = {}
        if v.get("units"):
            attrs["units"] = v["units"]
        if v.get("standard_name"):
            attrs["standard_name"] = v["standard_name"]
        kind = vertical_axis_kind(_Attrs(attrs))
        if kind:
            found[kind].append((name, v.get("units"), v.get("standard_name"), v.get("long_name")))
    return found


class _Attrs:
    """Minimal stand-in exposing ``.attrs`` — ``vertical_axis_kind`` reads
    nothing else, so the real function is under test rather than a copy."""

    def __init__(self, attrs: dict):
        self.attrs = attrs


def _grade(cand: dict, found: dict) -> tuple[str, str]:
    """Four-way outcome for one candidate, plus the reason."""
    truth = cand["truth"]
    hits = {k: [n for n, *_ in v] for k, v in found.items() if v}
    if not hits:
        if not truth:
            return "refused-correctly", "no vertical axis proposed, and none should be"
        return "refused-wrongly", f"proposed nothing; the real axis is {sorted(truth.values())}"

    # A candidate set containing the true axis AND decoys is not a clean
    # resolution: a tier with no way to choose between them either picks wrong
    # or has to refuse. Both facts are reported.
    truth_names = {v.split("/")[-1] for v in truth.values()}
    proposed = {n.split("/")[-1] for names in hits.values() for n in names}
    matched = proposed & truth_names
    decoys = proposed - truth_names

    if not truth:
        return "resolved-WRONG", f"proposed {sorted(proposed)} on a product whose lev is not a physical level"
    if matched and not decoys:
        return "resolved-correct", f"proposed exactly {sorted(matched)}"
    if matched and decoys:
        return "ambiguous", f"proposed the real axis {sorted(matched)} alongside decoys {sorted(decoys)}"
    return "resolved-WRONG", f"proposed {sorted(decoys)}, none of which is the axis ({sorted(truth_names)})"


async def run_discovery(limit_keys: set | None) -> None:
    from tta_backend.config.settings import Settings
    from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
    from tta_backend.earthdata_mcp.results import parse_tool_result

    settings = Settings(
        earthdata_mcp_url=os.environ["EARTHDATA_MCP_URL"],
        earthdata_mcp_token=os.environ.get("EARTHDATA_MCP_TOKEN"),
    )
    tools = await load_raw_mcp_tools(settings)
    ws = f"t58_spike_{uuid.uuid4().hex[:8]}"

    async def inv(name, **kw):
        return parse_tool_result(await tools[name].ainvoke({**kw, "workspace_id": ws}))

    scorecard = []
    for cand in CANDIDATES:
        if limit_keys and cand["key"] not in limit_keys:
            continue
        print("=" * 78)
        print(f"{cand['key']}   ({cand['note']})")
        print(f"  expect: {cand['expect']}")
        try:
            search = await inv("search_datasets", query=cand["query"], filters=None)
        except Exception as e:  # noqa: BLE001 — a spike reports, it does not fail
            print(f"  SEARCH FAILED: {type(e).__name__}: {e}")
            scorecard.append((cand["key"], cand["expect"], "unreachable", str(e)[:60]))
            continue
        rows = _rows(search)
        print(f"  search hits: {len(rows)}")
        chosen = None
        for row in rows:
            short, concept, title = _row_identity(row)
            if cand["short_name"] and short.upper() == cand["short_name"].upper():
                chosen = row
                break
        if chosen is None and rows:
            chosen = rows[0]
        if chosen is None:
            print("  NO RESULTS")
            scorecard.append((cand["key"], cand["expect"], "unreachable", "no search results"))
            continue
        short, concept, title = _row_identity(chosen)
        print(f"  chose: short_name={short!r} concept_id={concept!r}")
        print(f"         title={title[:80]!r}")
        handle = chosen.get("handle") or chosen.get("dataset_handle")
        try:
            det = await inv("describe_dataset", dataset_handle=handle, detail=True)
        except Exception as e:  # noqa: BLE001
            print(f"  DESCRIBE FAILED: {type(e).__name__}: {e}")
            scorecard.append((cand["key"], cand["expect"], "unreachable", str(e)[:60]))
            continue
        variables = det.get("variables") or []
        print(f"  variables: {len(variables)}  (source={det.get('variable_source')})")
        # The decisive question for Phase 2's registry schema: does this
        # metadata surface carry DIMENSIONS? Production discovery
        # (vertical_axes_for_dim) filters candidates to those spanning the
        # science variable's vertical dim, and that filter is what separates
        # the real axis from every other pressure-like variable in the file.
        dim_keys = sorted({k for v in variables for k in v if "dim" in k.lower()})
        print(f"  dimension metadata present: {dim_keys or 'NONE'}")
        found = _classify_inventory(variables)
        for kind in ("pressure", "altitude"):
            for name, units, std, long_name in found[kind]:
                print(f"    [{kind:8s}] {name:44s} units={units!r} std={std!r} long={str(long_name)[:34]!r}")
        if not found["pressure"] and not found["altitude"]:
            print("    (nothing classified as a vertical axis)")
        outcome, reason = _grade(cand, found)
        print(f"  --> {outcome}: {reason}")
        scorecard.append((cand["key"], cand["expect"], outcome, reason))

    print()
    print("=" * 78)
    print("UMM-VAR (metadata-only) DISCOVERY SCORECARD")
    print("=" * 78)
    print(f"{'product':16s} {'expected':18s} {'actual':18s} reason")
    for key, expect, actual, reason in scorecard:
        flag = " " if actual == expect else "!"
        print(f"{flag}{key:15s} {expect:18s} {actual:18s} {reason[:70]}")
    agree = sum(1 for _, e, a, _ in scorecard if e == a)
    print(f"\n  {agree}/{len(scorecard)} matched the pre-registered expectation")
    wrong = [k for k, _, a, _ in scorecard if a == "resolved-WRONG"]
    print(f"  resolved-WRONG: {wrong or 'none'}")
    ambiguous = [k for k, _, a, _ in scorecard if a == "ambiguous"]
    print(f"  ambiguous (real axis + decoys): {ambiguous or 'none'}")


# ── Granule suite ─────────────────────────────────────────────────────────────

# The regional-mean vertical axes MEASURED by --granule on a real 11-timestep
# TEMPO_O3PROF_L3 retrieval over New Jersey, 2025-10-01/02 (spike 2026-08-08).
# Recorded so the hydrostatic check's calibration can be re-derived, and its
# negative controls re-run, without paying for another Harmony job.
MEASURED_O3PROF_PRESSURE_HPA = [
    0.1749, 0.416, 0.5884, 0.8321, 1.1767, 1.6641, 2.3534, 3.3283,
    4.7069, 6.6565, 9.4138, 13.3131, 18.8276, 26.6262, 37.6551, 53.2524,
    75.3103, 107.3553, 167.2165, 260.0476, 367.2513, 504.8014, 682.9116, 896.0633,
]
MEASURED_O3PROF_ALTITUDE_KM = [
    60.2146, 54.236, 51.6644, 49.0459, 46.4125, 43.7983, 41.2332, 38.7295,
    36.2888, 33.9031, 31.5568, 29.2459, 26.9749, 24.7501, 22.5718, 20.4352,
    18.3318, 16.185, 13.4236, 10.5367, 8.0879, 5.7026, 3.3105, 1.0823,
]

# Implied-scale-height band the hydrostatic check accepts, in km.
#
# The PRD proposed 7.0-8.5 (the textbook tropospheric range). MEASURED against
# the real O3PROF column it FAILS: median 6.92 km, only 29% of layers inside.
# That is the check being wrong, not the granule -- H = RT/g varies with
# temperature, so a column spanning 1-60 km runs ~6.4 km through the cold
# stratosphere and ~8.4 km near a 288 K surface, and no single band fits it
# tightly. Widened to a range that still fails every error this check exists to
# catch (verified by the negative controls in ``run_hydrostatic_controls``):
# a Pa/hPa mix-up moves ln(p) by 4.6 and an inversion turns H negative, both
# orders of magnitude outside this. It is an order-of-magnitude sanity check,
# never a precision test.
_HYDROSTATIC_SCALE_HEIGHTS = (5.5, 9.5)  # km

# Fraction of layers whose implied scale height must land inside that band.
#
# MEASURED, and the reason this check has two criteria instead of one. On the
# median alone an INVERTED axis pairing LEAKS: reversing pressure against
# altitude still yields a median of 6.78 km, comfortably inside the band, and
# the check passes an upside-down atmosphere -- the exact T56 failure class.
# The layer fraction separates them cleanly and with room to spare: the correct
# pairing scores 100%, both inversions score 12%.
_HYDROSTATIC_MIN_LAYER_FRACTION = 0.80


def check_monotonic(values: np.ndarray) -> tuple[bool, str]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return False, "fewer than 2 finite values"
    diffs = np.diff(finite)
    if np.all(diffs > 0):
        return True, "strictly increasing"
    if np.all(diffs < 0):
        return True, "strictly decreasing"
    return False, f"not monotonic ({int(np.sum(diffs > 0))} up / {int(np.sum(diffs < 0))} down)"


_PLAUSIBLE = {"pressure": (1e-4, 1100.0), "altitude": (-0.5, 120.0)}


def check_plausible_range(values: np.ndarray, kind: str) -> tuple[bool, str]:
    lo, hi = _PLAUSIBLE[kind]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False, "no finite values"
    vmin, vmax = float(finite.min()), float(finite.max())
    ok = lo <= vmin and vmax <= hi
    return ok, f"[{vmin:.4g}, {vmax:.4g}] vs plausible [{lo}, {hi}]"


def check_hydrostatic(pressure_hpa: np.ndarray, altitude_km: np.ndarray) -> tuple[bool, str]:
    """p ~= 1013 * exp(-z/H), H in 7..8.5 km. The only check that does not
    believe the metadata under test: it catches a Pa/hPa units error (a clean
    factor of 100), a wrong-variable pick, and an inversion."""
    ok_mask = np.isfinite(pressure_hpa) & np.isfinite(altitude_km) & (pressure_hpa > 0)
    p, z = pressure_hpa[ok_mask], altitude_km[ok_mask]
    if p.size < 2:
        return False, "fewer than 2 paired finite values"
    # Implied scale height at each layer, from p = 1013*exp(-z/H)  =>  H = -z/ln(p/1013)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = -z / np.log(p / 1013.25)
    implied = implied[np.isfinite(implied) & (implied != 0)]
    if implied.size == 0:
        return False, "no implied scale height computable"
    lo, hi = _HYDROSTATIC_SCALE_HEIGHTS
    median = float(np.median(implied))
    inside = float(np.mean((implied >= lo) & (implied <= hi)))
    ok = (lo <= median <= hi) and inside >= _HYDROSTATIC_MIN_LAYER_FRACTION
    return ok, (
        f"median implied scale height {median:.2f} km (expect {lo}-{hi}); "
        f"{inside * 100:.0f}% of layers inside (need >={_HYDROSTATIC_MIN_LAYER_FRACTION * 100:.0f}%)"
    )


def run_hydrostatic_controls(pressure_hpa: np.ndarray, altitude_km: np.ndarray) -> None:
    """Does the hydrostatic check actually REJECT the errors it claims to?

    A validation check that passes the real granule proves nothing on its own --
    a check that accepts everything would also pass. These are the three
    failures it is supposed to catch, injected into the same real axes, so its
    discriminating power is measured rather than asserted.
    """
    controls = [
        ("pressure mislabelled Pa (x100)", pressure_hpa * 100.0, altitude_km),
        ("pressure axis inverted", pressure_hpa[::-1], altitude_km),
        ("altitude mislabelled m (x1000)", pressure_hpa, altitude_km * 1000.0),
        ("altitude axis inverted", pressure_hpa, altitude_km[::-1]),
    ]
    print("       negative controls (each MUST fail):")
    for label, p, z in controls:
        ok, why = check_hydrostatic(p, z)
        verdict = "LEAKED (check accepted a known error)" if ok else "rejected"
        print(f"         {label:34s} -> {verdict}  [{why}]")


def check_round_trip(axis_values: np.ndarray, resolve) -> tuple[bool, str]:
    """For every layer k, requesting the value that SITS at layer k must return
    k. Tests the resolver, not just discovery."""
    misses = []
    for k, v in enumerate(axis_values):
        if not np.isfinite(v):
            continue
        got = resolve(float(v))
        if got != k:
            misses.append((k, float(v), got))
    if not misses:
        return True, f"all {int(np.sum(np.isfinite(axis_values)))} layers round-tripped"
    return False, f"{len(misses)} layers missed, e.g. {misses[:3]}"


def _nearest_index(axis: np.ndarray, requested: float) -> int:
    return int(np.nanargmin(np.abs(axis - requested)))


def dominance(per_pixel: np.ndarray, requested: float, vertical_axis_first: bool = True) -> dict:
    """Per-pixel agreement with the layer the REGIONAL MEAN axis picks (D5).

    ``per_pixel`` is (layer, ...) — the vertical axis with every spatial cell
    still present. Returns the disclosure D5 requires: which layer the regional
    mean resolves to, what fraction of analyzed pixels independently resolve to
    that same layer, the runner-up and its fraction, the margin between them,
    and the level error (how far the resolved layer's regional-mean level is
    from what was asked)."""
    n_layers = per_pixel.shape[0]
    flat = per_pixel.reshape(n_layers, -1)
    regional_mean = np.nanmean(flat, axis=1)
    resolved = _nearest_index(regional_mean, requested)

    usable = np.isfinite(flat).all(axis=0)
    cols = flat[:, usable]
    if cols.shape[1] == 0:
        return {"resolved": resolved, "n_pixels": 0}
    per_pixel_choice = np.argmin(np.abs(cols - requested), axis=0)
    counts = np.bincount(per_pixel_choice, minlength=n_layers)
    order = np.argsort(counts)[::-1]
    total = int(counts.sum())
    dominant, runner_up = int(order[0]), (int(order[1]) if counts[order[1]] else None)
    return {
        "resolved": resolved,
        "resolved_level": float(regional_mean[resolved]),
        "level_error": float(abs(regional_mean[resolved] - requested)),
        "n_pixels": total,
        "dominant": dominant,
        "dominant_fraction": counts[order[0]] / total,
        "runner_up": runner_up,
        "runner_up_fraction": (counts[order[1]] / total) if runner_up is not None else 0.0,
        "margin": (counts[order[0]] - counts[order[1]]) / total,
        "agrees_with_regional_mean": dominant == resolved,
    }


async def run_granule(location: str, time_range: str) -> None:
    """Retrieve a real TEMPO_O3PROF granule and run the four falsifiable checks
    plus the dominance distributions, on the NARROWED, PRE-AGGREGATION array —
    the only place the vertical axes still exist (finding 1)."""
    from tta_backend.config.settings import Settings
    from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
    from tta_backend.earthdata_mcp.results import parse_tool_result
    from tta_backend.datasets.registry import load_registry
    from tta_backend.utils.geo_utils import vertical_axes_for_dim
    from tta_backend.utils.plotting import mask_data_by_geometry
    from tta_backend.preprocessing.aggregation_service import AggregationService
    from tta_backend.tools.satellite_tools.plot_tools import _resolver

    settings = Settings(
        earthdata_mcp_url=os.environ["EARTHDATA_MCP_URL"],
        earthdata_mcp_token=os.environ.get("EARTHDATA_MCP_TOKEN"),
    )
    tools = await load_raw_mcp_tools(settings)
    ws = f"t58_spike_{uuid.uuid4().hex[:8]}"
    cfg = load_registry()["TEMPO_O3PROF"]

    async def inv(name, **kw):
        return parse_tool_result(await tools[name].ainvoke({**kw, "workspace_id": ws}))

    print(f"Retrieving {cfg.short_name} over {location!r} for {time_range} ...")
    search = await inv("search_datasets", query="TEMPO ozone profile", filters=None)
    rows = _rows(search)
    handle = None
    for row in rows:
        short, concept, _ = _row_identity(row)
        if concept == cfg.collection_id or short.upper() == (cfg.short_name or "").upper():
            handle = row.get("handle") or row.get("dataset_handle")
            break
    if handle is None and rows:
        handle = rows[0].get("handle") or rows[0].get("dataset_handle")
    aoi = await inv("define_area_of_interest", location=location)
    aoi_handle = aoi["handle"]
    variables = [cfg.primary_var and f"product/{cfg.primary_var}", *cfg.vertical_axis_vars]
    retrieve = await inv(
        "retrieve_subset", dataset_handle=handle, aoi_handle=aoi_handle,
        time_range=time_range, variables=[v for v in variables if v], output_format=None,
    )
    job_handle, obs_handle = retrieve["job_handle"], retrieve["obs_handle"]
    for _ in range(80):
        status = await inv("get_retrieval_status", job_handle=job_handle)
        if status.get("status") in ("ready", "failed", "expired", "cancelled"):
            break
        await asyncio.sleep(4)
    print("  job status:", status.get("status"))
    if status.get("status") != "ready":
        print("  cannot continue without a ready retrieval")
        return

    # Exported and opened directly rather than through ``open_handle``: that
    # composite binds its own workspace id, which is not the throwaway one this
    # spike retrieved under, so it cannot see the handle it just created.
    from tta_backend.services.open_handle import _open

    export = await inv("export_result", handle=obs_handle)
    print("  export:", export.get("status"), export.get("media_type"))
    ds = _open(export["storage_uri"], export.get("media_type", "netcdf"))
    da = AggregationService().to_dataarray(ds)
    print("  opened:", da.name, da.dims, da.shape)

    region = await _resolver.aresolve_location(location)
    narrowed = mask_data_by_geometry(da, region["geometry"])
    print("  narrowed coords:", list(narrowed.coords))

    vertical_dim = next((str(d) for d in narrowed.dims if d not in ("time", "latitude", "longitude", "lat", "lon")), None)
    print("  vertical dim:", vertical_dim)
    axes = vertical_axes_for_dim(narrowed, vertical_dim)
    print("  vertical_axes_for_dim ->", axes)

    from tta_backend.preprocessing.regional_reduction import reduce_keeping_axes

    means = {}
    for kind, name in axes.items():
        axis_da = narrowed[name]
        regional = reduce_keeping_axes(axis_da, keep=(vertical_dim,), stat="mean")
        values = np.asarray(regional.values, dtype="float64")
        means[kind] = values
        print(f"\n  --- {kind} ({name}) ---")
        print("   regional mean:", np.round(values, 4).tolist())
        ok, why = check_monotonic(values)
        print(f"   [1] monotonic       : {'PASS' if ok else 'FAIL'}  {why}")
        ok, why = check_plausible_range(values, kind)
        print(f"   [2] plausible range : {'PASS' if ok else 'FAIL'}  {why}")
        ok, why = check_round_trip(values, lambda v, a=values: _nearest_index(a, v))
        print(f"   [4] round-trip      : {'PASS' if ok else 'FAIL'}  {why}")

    if "pressure" in means and "altitude" in means:
        ok, why = check_hydrostatic(means["pressure"], means["altitude"])
        print(f"\n   [3] hydrostatic     : {'PASS' if ok else 'FAIL'}  {why}")
        run_hydrostatic_controls(means["pressure"], means["altitude"])
    else:
        print("\n   [3] hydrostatic     : SKIP (product publishes only one axis)")

    print("\n  --- dominance distributions (D6 evidence) ---")
    for kind, name in axes.items():
        axis_da = narrowed[name]
        collapsed = [d for d in axis_da.dims if d not in (vertical_dim,) and d == "time"]
        arr = axis_da.mean(dim=collapsed) if collapsed else axis_da
        arr = arr.transpose(vertical_dim, ...)
        per_pixel = np.asarray(arr.values, dtype="float64")
        requests = (
            [0.5, 10, 100, 300, 500, 850] if kind == "pressure" else [2, 5, 10, 20, 26, 40]
        )
        print(f"\n  {kind} ({name}), per-pixel grid {per_pixel.shape}")
        print(f"   {'requested':>10s} {'layer':>6s} {'dominant%':>10s} {'runner-up':>10s} {'margin':>8s} {'lvl err':>9s}")
        for req in requests:
            d = dominance(per_pixel, float(req))
            if not d.get("n_pixels"):
                print(f"   {req:>10g} (no fully-finite pixel columns)")
                continue
            ru = f"{d['runner_up']}@{d['runner_up_fraction'] * 100:.1f}%" if d["runner_up"] is not None else "-"
            print(
                f"   {req:>10g} {d['resolved']:>6d} {d['dominant_fraction'] * 100:>9.1f}% "
                f"{ru:>10s} {d['margin']:>7.2f} {d['level_error']:>9.3g}"
                + ("" if d["agrees_with_regional_mean"] else "   <-- per-pixel majority DISAGREES with regional mean")
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discovery", action="store_true", help="UMM-Var metadata suite over every candidate")
    ap.add_argument("--granule", action="store_true", help="real-granule suite: 4 checks + dominance")
    ap.add_argument(
        "--controls", action="store_true",
        help="re-run the hydrostatic check and its negative controls against the "
             "MEASURED O3PROF axes, without a retrieval",
    )
    ap.add_argument("--only", default="", help="comma-separated candidate keys for --discovery")
    ap.add_argument("--location", default="New Jersey")
    ap.add_argument("--time-range", default="2025-10-01/2025-10-02")
    args = ap.parse_args()
    if not (args.discovery or args.granule or args.controls):
        ap.error("pass --discovery, --granule and/or --controls")
    limit = {k.strip() for k in args.only.split(",") if k.strip()} or None
    if args.discovery:
        asyncio.run(run_discovery(limit))
    if args.granule:
        asyncio.run(run_granule(args.location, args.time_range))
    if args.controls:
        p = np.array(MEASURED_O3PROF_PRESSURE_HPA)
        z = np.array(MEASURED_O3PROF_ALTITUDE_KM)
        ok, why = check_hydrostatic(p, z)
        print(f"   [3] hydrostatic     : {'PASS' if ok else 'FAIL'}  {why}")
        run_hydrostatic_controls(p, z)


if __name__ == "__main__":
    main()
