# PRD T23 — NASA-scientific interactive maps: MapLibre engine + server-rendered full-resolution data overlays

**Repo:** Talking-To-Air · **Session scope:** one feature branch (map-engine replacement — larger than the usual one-commit scope) · **Label:** ready-for-agent
**Depends on:** existing chart payload pipeline (`plot_tools._da_to_heatmap_payload` / `_save_chart` / `emit_chart`), the chart/artifact persistence path (`chart_service`, `chart_repository`, `artifact_registry`), and the export pipeline (`export_service`). Decision record 2026-07-08 (grilling session): Tier 3 full map-engine replacement; MapLibre for geo, Plotly for timeseries; server-rendered full-native-resolution overlay PNG; per-variable colormap registry; degrade-don't-die fallbacks.

## Problem Statement

As a researcher, the map I get back looks like a spreadsheet, not science. The inline chart renders my satellite field as a grid of colored **squares** on a flat land/ocean fill, using a red-yellow-green "traffic-light" scale. It reads as a business dashboard, not as a NASA data product. When I compare it to the maps NASA actually publishes — a smooth continuous field draped over shaded-relief terrain, in a perceptual colormap, under a labeled scientific colorbar — mine looks amateur, and that undermines my trust in the numbers behind it.

As the system owner, the current renderer has also hit its ceiling: it is built on Plotly's `scattergeo` trace, which physically cannot show a terrain basemap (no raster tiles) and cannot render a truly continuous field. I cannot style my way to the target look; the engine is the constraint.

## Solution

Replace the geo-chart engine. Heatmap and comparison maps render on **MapLibre GL JS**: a shaded-relief terrain basemap (computed hillshade), the data field as a **full-native-resolution, server-rendered, colormap-only PNG overlay** smoothed on the GPU, vector region borders on top, and a scientific colorbar. The map stays fully interactive — pan, zoom, per-cell hover, export. Timeseries charts stay on Plotly (they are not maps). The result looks like a NASA data product, on live user data, without sacrificing interactivity or scientific honesty (every colored pixel still traces to a real measurement; no-data stays transparent).

## User Stories

1. As a researcher, I want my heatmap drawn as a smooth continuous field, so that it reads as a scientific data product rather than a grid of squares.
2. As a researcher, I want a shaded-relief terrain basemap under my data, so that the geography is legible and the map matches published NASA imagery.
3. As a researcher, I want a perceptual colormap appropriate to the variable, so that color differences correspond honestly to value differences.
4. As a researcher viewing NO₂, I want the canonical NASA NO₂ palette, so that my map is directly comparable to NASA's own NO₂ products.
5. As a researcher viewing a difference/anomaly field, I want a diverging colormap centered on zero, so that positive and negative change are visually distinct and a sequential scale never misleads.
6. As a researcher, I want to pan and zoom the map, so that I can inspect a region closely.
7. As a researcher, I want to hover any location and read the underlying value, units, and coordinates, so that the map is an analysis tool, not just a picture.
8. As a researcher, I want zooming in to reveal the data's real native resolution, so that I see actual measured cells rather than an upsampled blur.
9. As a researcher, I want the smoothing to never invent structure across no-data gaps, so that I never read a measurement where none exists.
10. As a researcher, I want an optional "presentation smoothing" mode for a screenshot-ready hero look, so that I can produce a polished figure while the default view stays honest.
11. As a researcher, I want a horizontal scientific colorbar with title, units, and extend caps, so that the legend matches the conventions of the field.
12. As a researcher, I want the colorbar to depict exactly the colors on the map, so that the legend never lies about the pixels.
13. As a researcher comparing regions, I want the panels shown side by side so I can read them together, so that comparison is a glance rather than a click-through.
14. As a researcher, I want to click a comparison panel to open it in the full interactive map, so that I can drill into any one panel on demand.
15. As a researcher in dark mode, I want the basemap and chrome to follow my theme while the data colors stay fixed, so that the UI is consistent but my data stays comparable across screenshots.
16. As a researcher, I want to export the map as a PNG and the data as CSV, so that I can use it in reports — exactly as I can today.
17. As a researcher on a flaky network, I want the map to still show my data (on a plain basemap) if terrain tiles fail, so that a provider outage never blanks my result.
18. As a researcher, I want my data to still render (from the shipped grid) if the server overlay image is unavailable, so that the chart degrades instead of dying.
19. As the system owner, I want terrain/basemap tile URLs to be configuration, so that I can repoint to a keyed or self-hosted provider without a code change when traffic grows.
20. As the system owner, I want required attributions rendered on the map, so that we comply with the tile providers' terms.
21. As a developer, I want the colormap defined once and shared by the overlay renderer, the payload LUT, and the export renderer, so that the three can never drift apart.
22. As a developer, I want the overlay rendered eagerly at chart creation while the data handle is live, so that the live view is instant and never fails because a handle expired.
23. As the earthdata agent, I want the model-facing chart summary unchanged, so that this visual overhaul does not alter what the model reads or cites.

## Implementation Decisions

- **Engine split.** Geo render types (`heatmap`, `heatmap_multi`) move to a MapLibre GL JS component. `timeseries` stays on Plotly. The `ChartMessage` dispatch-on-`type` boundary is unchanged; only the geo panel internals are replaced. The `scattergeo` traces, the zoom-driven marker-resize logic, and the client-side `CMAP_MAP`/`SCALES` colorscale tables are removed.
- **Data overlay = server-rendered PNG.** A new overlay renderer produces a transparent, colormap-only raster (no basemap, borders, axes, or colorbar) at the field's full native resolution, in Web Mercator (EPSG:3857) to align with the mercator basemap. It reuses the aggregated `reduced` DataArray already computed in the plot tools (before downsampling). It is rendered **eagerly at chart-creation time** (handle guaranteed live), stored via the chart/artifact persistence path, and served by URL.
- **Payload additions (additive, back-compatible).** The heatmap payload gains: an `overlay` object (`url`, geographic `bounds`), a `colormap` object (resolved name + sampled RGBA `lut`), and the existing `vmin`/`vmax` reused as the color range. The existing `lats`/`lons`/`values`/`points` arrays and `provenance`/`query`/`export` blocks are kept unchanged.
- **Interaction rides on the shipped grid.** Hover is a client-side nearest-cell lookup against the (downsampled, ≤8k-cell) arrays already in the payload. Stats, histogram, and CSV export continue to compute from those arrays — no new endpoints, no behavior change. Visual fidelity (full-res PNG) and interaction resolution (arrays) are deliberately decoupled.
- **Smoothing.** Default is GPU bilinear resampling (`raster-resampling: linear`) of the honest native-resolution PNG — no server-side blur, no gap-crossing (no-data is transparent alpha and never interpolated into). An off-by-default "presentation smoothing" option applies a heavier Gaussian for a hero screenshot and is labeled as such.
- **Basemap composition.** Bottom-to-top: CARTO `light_nolabels` base → terrarium DEM `raster-dem` source + `hillshade` layer (blended) → data overlay image layer → vector region borders (reuse the region GeoJSON already fetched client-side, or a MapLibre line layer). Label-free by default. Tiles fetched directly by the browser. All tile/DEM/base URLs are backend/env configuration with free defaults; attribution strings are rendered on the map.
- **Colormap registry (single source of truth).** One backend module maps variable → colormap: NO₂ → cloned NASA/OMI-style LUT; other sequential variables → perceptual-uniform maps (plasma/inferno/turbo/viridis) per variable; difference/anomaly fields → a diverging map (`RdBu_r`) centered on zero. The same registry drives (a) the overlay PNG render, (b) the `colormap.lut` shipped in the payload, and (c) `export_service`'s PNG/legend — so map, legend, and export cannot diverge. Color range stays 2nd–98th percentile with explicit override.
- **Colorbar.** Rendered client-side as SVG (horizontal, extend caps, title, units), fixed as map furniture (does not pan/zoom), drawn from the payload `colormap.lut` + range. Theme-aware text; the gradient reflects the fixed LUT.
- **Comparisons.** `heatmap_multi` renders as static small-multiples using each panel's overlay PNG over a lightweight shared basemap thumbnail; clicking a panel promotes it into the single interactive MapLibre view. At most one live WebGL context at a time.
- **Theme.** Basemap + chrome follow the app light/dark theme (CARTO light/dark `nolabels`); the data colormap is frozen across themes. Dark mode lightens the terrain under the overlay so a colormap ending in near-black stays legible. Default appearance is light (matches the reference).
- **Degrade-don't-die.** Terrain/basemap tiles fail → flat land/ocean fill, overlay + borders still render. Overlay PNG missing/fails → client renders the field from the shipped arrays to a `<canvas>` (lower-res but alive). Consistent with the T17 resilience posture.
- **Projection scope.** Mercator + single-image overlay for v1. A single overlay is genuinely full-native-resolution up to the ~8192px/side WebGL texture limit; grids beyond that (very large global native fields) fall back to a capped-resolution render with a note. Globe projection and server-side XYZ tiling are explicitly deferred.

## Technical Implementation Guide

Paths current as of `talking-to-air-v2`.

- **Overlay renderer (primary backend seam).** Extend the payload builders in `Backend/tools/satellite_tools/plot_tools.py`: in `_da_to_heatmap_payload` (single) and the per-panel path in `make_plot_multiple`, render the full-res overlay from the aggregated `reduced` DataArray and attach `overlay`/`colormap` to the payload; do the render/store inside the existing `asyncio.to_thread` CPU-bound block, and wire persistence through `_save_chart` → `emit_chart` (mirror how `export` metadata is attached in `_attach_reproducibility`). The overlay PNG bytes persist alongside the chart via `services/artifact_registry.py` + `repositories/chart_repository.py`.
- **Colormap registry (new module).** New `Backend/utils/colormaps.py` (or `Backend/config/colormaps.py`): `resolve(variable, mode) -> (mpl_colormap, sampled_lut)`. Consumed by the overlay renderer, the payload builder (for `colormap.lut`), and `services/export_service.py` (replacing the hard-coded `"Spectral_r"` in `_plot_heatmap_axis*` and `build_chart_png`). Retire the frontend `CMAP_MAP`/`SCALES` tables in favor of the payload LUT.
- **Serving.** Serve the stored overlay bytes by URL from `api.py`, parallel to the existing `/chart/{id}/export.{png,csv}` routes (e.g. `GET /chart/{id}/overlay.png`), returning the eagerly-rendered bytes (not an on-demand re-render).
- **Frontend engine.** Add `maplibre-gl` to `Frontend/package.json`; new `MapLibreHeatmapPanel` replacing `HeatmapPanel`/`HeatmapMultiPanel` internals in `Frontend/src/components/ChartMessage.jsx` (keep the `ChartMessage` `switch (chart.type)` and `ChartToolbar`/`ProvenanceBlock` intact). Extract pure helpers for test seams: value→LUT-color, nearest-cell hover lookup, canvas-fallback frame builder, SVG-colorbar geometry. `TimeSeriesPanel` untouched.
- **Config + ops.** Tile/DEM/base URLs as settings in `Backend/config/settings.py` served to the frontend (or a small config endpoint); prod `nginx.conf` CSP `img-src`/`connect-src` extended for the CARTO + terrarium hosts; attribution strings surfaced in the map component.

## Testing Decisions

- Good tests assert **external behavior at the highest seam**, not rendering internals (a WebGL canvas is not unit-testable and should not be asserted pixel-wise).
- **Payload contract (primary, hermetic).** Extend `Backend/tests/test_satellite_plot_payload.py`: calling the payload builder on a known DataArray yields `overlay` (url + bounds) and `colormap` (name + non-empty LUT) alongside the unchanged `lats`/`lons`/`values`/`points`/`vmin`/`vmax`; a variable maps to the expected colormap family; a difference field maps to a diverging map; no-data cells remain absent from `points` (existing sparse-points test still passes). These tests gate on the existing optional-dependency skip (cartopy/xarray/…).
- **Colormap single-source.** A registry test proves the LUT shipped in the payload and the colormap used by `export_service` resolve from the same registry entry for a given variable (the anti-drift guarantee) — mirrors the spirit of the T22 starter/eval consistency test.
- **Streaming.** Extend `Backend/tests/test_streaming_chart_payload.py` to assert the new fields survive `emit_chart` → the `chart` SSE event.
- **Export unchanged.** `Backend/tests/test_export_service.py` continues to prove PNG/CSV export works, now sharing the registry colormap.
- **Frontend pure helpers.** `Frontend/tests/*.test.mjs` (node `--test`, prior art exists): value→LUT-color mapping, nearest-cell hover lookup returns the right value/coords, colorbar tick geometry, and canvas-fallback frame construction from arrays.

## Out of Scope

- Globe projection and server-side XYZ raster tiling (deferred; only needed for gigapixel/global-deep-zoom — v1 is Mercator single-overlay).
- A PNG-only payload that drops the grid arrays (rejected: it would force server round-trips for hover and precomputed stats; the grid stays).
- Client-side full-resolution rendering (rejected for the live view: full-res as JSON is untenable; canvas-from-arrays remains only as the degraded fallback).
- Per-panel simultaneous interactivity in comparisons (rejected: N live WebGL contexts; small-multiples + expand instead).
- 3D terrain extrusion, animation/time-scrubbing on the map, and drawing/annotation tools.
- Changing the model-facing chart summary, the envelope schema, or any agent prompt.
- A keyed/commercial tile provider integration (the config hook is in scope; choosing/paying for a provider is not).

## Further Notes

- This is a map-*engine* replacement, not a restyle — larger than the usual single-commit PRD. Landing the single-panel `heatmap` path first (engine + overlay + colorbar + hover + fallback), then `heatmap_multi` small-multiples, then theme polish, is a reasonable internal sequence within the branch.
- Two build-time unknowns to resolve early: the exact NASA/OMI NO₂ LUT stops, and behavior for native grids exceeding the WebGL texture cap (capped render + note is the assumed answer).
- The free tile providers (CARTO basemaps, AWS terrarium `elevation-tiles-prod`) carry no reliability SLA and terms that tighten under heavy/commercial use. The configurable-URL + attribution + fallback design is the compliance/reliability hedge; revisit before any high-volume or commercial deployment.
