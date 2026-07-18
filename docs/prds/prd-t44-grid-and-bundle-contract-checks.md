# PRD T44 — Grid & bundle contract checks: every documented assumption becomes an enforced one

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T24 (grid-kind classification), the bundle-open hardening series. Origin: reliability review 2026-07-16 (adaptability findings #3, #5, #9, #10 and the non-uniform-grid note).

## Recommended Model

**Claude Opus 4.8.** The irregular-grid tolerance has to be picked against real MERRA-2/TEMPO grids in-session, and several of these guards are the last line of defense against silent wrong-pixel masking — worth the stronger model for the judgment calls, even though most of the individual fixes are mechanical.

## Problem Statement

As a researcher, several open/retrieval paths rest on assumptions that hold for the 11 registered collections but are checked nowhere — and each fails *silently or uglily* on the off-registry datasets the universal pipeline exists to welcome:

1. **Non-uniform rectilinear grids mis-mask silently.** `geometry_mask`'s affine derives one resolution from the coordinate endpoints; `ensure_supported_grid` passes any 1-D grid, so non-uniform spacing gets a progressively misplaced mask — wrong pixels in the statistics, no error.
2. **Bundle time order is filename order.** `_open_netcdf_bundle` concatenates members sorted by name ("names sort chronologically"); a provider whose names don't sort yields a non-monotonic time axis, and later `sel(time=slice(...))` silently returns wrong subsets. Duplicate timestamps (overlapping orbits, reprocessed granules) break later selections with an opaque error.
3. **A grid with no recognizable coordinates dies off-taxonomy.** `_grid_kind` → `"none"` passes `ensure_supported_grid` as a no-op and fails later as a raw `ValueError("Could not find lat/lon coordinates")` instead of a T24-typed answer.
4. **`point_timeseries` trusts response shapes bare.** A missing AOI handle flows on as `None`; `submit["job_handle"]` KeyErrors off-taxonomy if the submit response is differently shaped.
5. **A registry typo prevents boot.** `PRESET_COLLECTIONS` is built at import with `reg[key]`; editing collections.yaml (the file whose header says "no code changes needed") can take the backend down with a bare KeyError.

## Solution

Each assumption becomes a check with an honest outcome: non-uniform spacing refuses with the T24 unsupported-grid shape (until a per-cell mask lands); bundles sort by time after concat and de-duplicate timestamps deterministically (keep-first, disclosed in a log event); a no-coordinates grid refuses with a typed error at the guard; `point_timeseries` validates the two response shapes and classifies failures; preset/registry loading validates at startup with a named-key error (fail loud at boot with a *useful* message — or degrade to the resolvable presets, decided in-session with the operator story in mind).

## User Stories

1. As a researcher, I want a non-uniformly-spaced product refused with "this grid isn't supported yet" rather than mis-masked, so that a wrong-region mean can't happen silently.
2. As a researcher, I want multi-granule aggregations correct regardless of the provider's file-naming scheme, so that time slicing never depends on alphabetics.
3. As a researcher, I want a product with unrecognizable coordinates to get the same typed unsupported-grid answer curvilinear products get, so that the agent can suggest an alternative instead of relaying a traceback.
4. As the earthdata agent, I want point-timeseries contract violations classified, so that I can tell the researcher what actually failed.
5. As an operator, I want a collections.yaml typo to produce an error naming the bad key (or a degraded preset list), so that dataset onboarding is as safe as its header promises.

## Implementation Decisions

- `_grid_kind` gains `"irregular"` for 1-D lat/lon whose spacing varies beyond a relative tolerance (~1e-3 of the median step, decided against real MERRA-2/TEMPO grids in-session); `ensure_supported_grid` refuses it and `"none"` with `CATEGORY_UNSUPPORTED_GRID` messages naming what was found.
- `_open_netcdf_bundle`: after concat, `sortby("time")` when a time coordinate exists; duplicate timestamps keep the first occurrence (stable, name-order deterministic) and log `bundle_duplicate_timestamps` with the count — never a crash, never double-counted granules in means.
- `point_timeseries`: missing AOI handle or `job_handle` raises `MCPToolError(CATEGORY_CONTRACT, ...)` naming the tool and the absent field.
- `preset_collections`: `get_preset_collections` validates keys against the registry and raises a startup error naming every missing key; module-level `PRESET_COLLECTIONS` moves behind a lazy accessor so import order can't turn a data error into an ImportError cascade. `registry.load_registry`'s own YAML failure already fails loud — keep, but verify its message names the file.

## Technical Implementation Guide

- `Backend/utils/geo_utils.py`: `_grid_kind`, `ensure_supported_grid`.
- `Backend/services/open_handle.py::_open_netcdf_bundle`.
- `Backend/services/retrieval_composites.py::point_timeseries`.
- `Backend/datasets/preset_collections.py` (+ its consumer in `config/earthdata_agent_prompt.py` and `/capabilities` if it reads the constant).

## Testing Decisions

- Grid: a 1-D lat axis with one stretched gap → typed refusal; a uniform grid with float jitter below tolerance → passes (regression for real grids).
- Bundle: two members whose names sort against their dates → time axis monotonic after open; two members with identical timestamps → one kept, event logged, `sel` works.
- point_timeseries: fake MCP returning `{}` from `define_area_of_interest`/`retrieve_timeseries` → contract-classified errors, no bare KeyError.
- Presets: a registry missing a preset key → error naming it; all-valid registry unchanged.
- Prior art: `test_grid_support.py`, `test_open_handle.py` bundle tests, `test_point_timeseries.py`, `test_preset_collections.py`.

## Out of Scope

- Actually supporting curvilinear/projected/irregular grids (refusal is the deliverable). Cross-granule spatial-grid mismatch detection (concat already fails loud there). The MCP's own bundle writer.

## Status update (2026-07-17)

Item 3 (no-coordinates grid → typed `unsupported_grid` refusal) **landed** with the QA blocker fixes: `geometry_mask`/`plot_map` now raise the typed error and both stat tools catch it (`test_grid_support.py`, `test_gpm_dimension_names.py`). Landed alongside (same session): per-variable `DimensionNames` recovery for scale-less GPM HDF5 (`_apply_declared_dimension_names` in `open_handle.py`) and the root-header-vars fix (root data_vars no longer hide nested science groups — GPM 3CMB opened as 3 header strings before).

**New item from the same live investigation — implicit GridHeader grids.** GPM L3 combined products (GPM_3CMB_DAY, live bundle `job_3d573642aeef04d8`) declare dims (`lnL/ltL` 72×28, `lnH/ltH` 1440×536) but ship **no lat/lon coordinate arrays at all** — the grid is implicit in the group's `GridHeader` attribute (origin/resolution strings). Today's honest answer is the typed refusal; actually supporting them means parsing `GridHeader` and synthesizing the 1-D lat/lon axes at open time (bounded, deterministic — same synthesis doctrine as the bundle time-coord fix). Also worth a look in the same session: dataset routing chose 3CMB for "GPM precipitation" when IMERG (`GPM_3IMERGDF`, which now works end-to-end) is the researcher-appropriate product — a preset/registry nudge would prevent the refusal from being the first answer.
