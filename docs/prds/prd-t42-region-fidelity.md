# PRD T42 — Region fidelity: the region you named is the region we computed

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T25 masking disclosure (the provenance surface these facts join). Origin: reliability review 2026-07-16 (scientific-accuracy findings #3, #6, and the geocoding notes).

## Recommended Model

**Claude Opus 4.8.** Three independent scientific-fidelity fixes (real polygons, empty-mask self-heal, sync/async parity) touch geometry, provenance plumbing, and frontend disclosure at once — the risk is a subtly-wrong region silently passing tests, so the extra reasoning care is worth it over Sonnet.

## Problem Statement

As a researcher, three things can quietly make "statistics over X" not mean X:

1. **Preset regions are crude bounding boxes.** `RegionResolver.global_regions` (`Backend/utils/plotting.py`) defines `'usa'` as `box(-125, 24, -66, 50)` — including a slice of Mexico, southern Canada, and open ocean. A mean labeled "United States" averages those pixels in. Nothing in the output discloses that the "region" was a rectangle.
2. **Small regions can false-negative to "no data".** `geometry_mask` rasterizes with center containment; a polygon smaller than a grid cell (or the 0.1° point-fallback box minted when Nominatim returns no polygon) can cover zero cell centers, and the tool answers "No valid data found — the region may be outside the data bbox" for data that is right there.
3. **Geocoding ambiguity is resolved silently.** Nominatim's first hit wins (`limit: 1`); "Georgia" or "Paris" quietly becomes whichever the geocoder ranks first. The display_name flows into titles (good) but nothing prompts a check when it's plausibly not what was meant. The sync path also skips the `"the "`-prefix strip the async path does — same input, different region.

## Solution

Regions carry a fidelity disclosure, empty masks self-heal honestly, and ambiguity is surfaced. Presets either gain real polygons (fetched once, cached in-repo) or are explicitly disclosed as bounding boxes in provenance; an all-False mask retries with `all_touched=True` and discloses it; the resolved display_name travels into the result payload (not just the title) so the agent can confirm when it diverges from the request; sync/async resolvers behave identically.

## User Stories

1. As a researcher, I want provenance to say `region_type: polygon | bounding_box | point_buffer | boundary_cells`, so that "mean over the US" is checkable against what was actually masked.
2. As a researcher, I want country/continent presets to use real boundaries where feasible, so that "United States" doesn't average in Sonora and the Atlantic.
3. As a researcher, I want a neighborhood-scale request on a coarse grid to return the boundary cells (disclosed) instead of "no data", so that small-area questions get their honest best answer.
4. As a researcher, I want the answer to name the place the geocoder resolved ("Paris, Texas, United States"), so that a wrong-place answer is catchable at a glance.
5. As the developer, I want one resolver behavior for sync and async callers, so that the same location string can't resolve differently by code path.

## Implementation Decisions

- `resolve_location`/`aresolve_location` return an added `region_type` and `display_name`; `_mask_col_info`-adjacent provenance plumbing (`_attach_reproducibility` in plot_tools, stats result dicts) carries both.
- Empty-mask retry: `geometry_mask` (or `mask_data_by_geometry`) detects an all-False mask, re-rasterizes with `all_touched=True`, and — only if that finds cells — proceeds with `region_type: boundary_cells`; still-empty keeps today's no-data error.
- Presets: ship simplified polygons for the multi-country presets (US/continents) as a small checked-in GeoJSON (source: Natural Earth 110m, simplified); pure-ocean/box concepts (`global`, `northeast us`) stay boxes with `region_type: bounding_box`. No runtime fetch.
- Sync resolver adopts the `"the "` strip; both share one normalization helper.
- Frontend: masking/metadata disclosure (T25's `MaskingDisclosure` surface) renders `region_type` when it isn't `polygon` — one line, no new tab.

## Technical Implementation Guide

- `Backend/utils/plotting.py`: `RegionResolver`, `geometry_mask`, `GeocodingService`.
- `Backend/tools/satellite_tools/plot_tools.py` `_attach_reproducibility`; `stat_tools.py` result dicts; `comparison_tools.py` panel metadata.
- New `Backend/data/preset_regions.geojson` (or `datasets/`), loaded lazily and cached.
- `Frontend/src/utils/resolveMasking.js` (or its current home) + the disclosure component.

## Testing Decisions

- Resolver: preset 'usa' returns polygon geometry + `region_type: polygon`; unknown place with no polygon → `point_buffer`; sync/async parity on "the netherlands".
- Mask: a polygon covering no cell centers on a coarse grid → boundary-cells result with disclosure; a genuinely off-grid polygon → no-data error unchanged.
- A stats result carries `region_type`/`display_name`; masking-execution tests extended rather than duplicated.
- Frontend disclosure unit test in the existing `.test.mjs` pattern.

## Out of Scope

- Weighted partial-cell (area-fraction) masking. Geocoder alternatives or multi-result disambiguation UI (the agent confirming from display_name is prompt territory). State/province-level preset polygons beyond what Natural Earth 110m gives.
