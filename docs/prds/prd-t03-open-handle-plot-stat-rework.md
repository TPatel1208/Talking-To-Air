# PRD T03 — Handle-based data access: `open_handle`, plot/stat rework, old loader deletion

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T02.

## Problem Statement

As the analysis layer, I still load data through the legacy path — a ~660-line hand-rolled Harmony downloader, a bespoke Zarr/PostGIS cache index, and plot/statistics tools that re-fetch by parameter dict — while retrievals now arrive as MCP handles. Two systems own data loading; one must go, and the one that stays must survive cache eviction without remembering how the data was originally requested.

## Solution

A single `open_handle` seam: handle in, xarray Dataset or Arrow table out. It exports the URI from the MCP, branches on URI scheme (never assumes local paths), opens by media type, and on eviction recovers via the MCP's `rematerialize` → await → re-export loop. Every plot and statistics tool converts to take handles; the legacy loader, cache index, and their dependencies are deleted; masking comes from `describe_dataset` metadata with a thin override table for wrong CMR records.

## User Stories

1. As a plot/statistics tool, I want `open_handle(handle)` to return an opened xarray Dataset (Zarr/NetCDF) or Arrow table (Parquet), so that all data access is one call against one seam.
2. As a consumer of a cache-managed URI, I want eviction handled inside `open_handle` (rematerialize → await → re-export → open), so that no tool above the seam ever sees a missing file.
3. As the future hosted deployment, I want `open_handle` to branch on URI scheme (`file://` opens locally, `https://` presigned opens remotely), so that moving off the shared volume changes zero analysis code.
4. As the earthdata agent, I want every plot tool (single, multiple) and statistics tool (statistic, temporal statistic, daily peak) to accept `obs_`/`cube_` handles, so that the retrieve → plot workflow threads one identifier through.
5. As a researcher, I want fill values and valid ranges applied from the dataset's own described metadata, so that masking works for any discovered dataset — not just the ten that used to be hand-pinned.
6. As a maintainer, I want the hand-rolled downloader, the cache index and its initialization SQL, and the Harmony/earthaccess/netCDF4 download dependencies deleted, so that the MCP is the single owner of retrieval and caching.
7. As a maintainer, I want the pinned-collections dict reduced to a small preset list used only as prompt guidance (suggestions, not a ceiling), so that discovery stays open-ended.
8. As a plotted artifact, I want the source handles recorded in my metadata, so that provenance (T10) can trace every figure to a re-materializable spec.

## Implementation Decisions

- `open_handle` lives in the analysis toolset layer, wraps `export_result` + the eviction-recovery loop, dispatches on `media_type` for the opener, and caches open datasets per-request only (no new persistent cache — the MCP owns caching).
- Eviction recovery is bounded: one rematerialize attempt per open; a second failure surfaces the MCP's structured error verbatim.
- Plot/stat tool signatures change from parameter dicts to `(handle, ...presentation args)`; the internal re-fetch pattern (`_fetch_params`/`_load_data`) is deleted with the loader.
- Masking: fill/valid-range read from `describe_dataset` per-variable enrichment at plot/stat time; a small override table keyed by short name corrects known-wrong UMM-Var records (populated from the live-matrix quirk ledger as entries arrive).
- Deletions land in the same session as the rewiring so the repo never has two live loading paths across a commit boundary.
- Artifact writes include `source_handles` from this PRD onward (the artifact model already stores metadata).

## Testing Decisions

- Seam: the fake-MCP server from T02, extended with a tmp-dir volume holding real small Zarr/Parquet fixtures.
- Tests: open a Zarr handle → Dataset with expected variables; open a Parquet handle → table; delete the fixture file then open → rematerialize path exercised → reopened; masking applied from described fill/valid-range on a synthetic dataset with sentinel fills; each converted plot/stat tool produces its artifact from a handle; the deleted modules stay deleted (import of the old loader fails).
- Prior art: existing service tests and artifact/chart repository tests.

## Out of Scope

- Prompt changes teaching the agent the new workflow (T04) — this session keeps prompts pointing at the new tools minimally functional.
- New artifact types (T06); new analysis workflows (T07/T08).

## Further Notes

Exit criterion for the Phase-1 swap overall (tracked here since this PRD completes it): today's demo conversations work end-to-end through the MCP, plus one previously-impossible discovery task ("find a soil-moisture dataset and map it over the Raritan basin").
