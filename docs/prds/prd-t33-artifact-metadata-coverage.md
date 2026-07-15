# PRD T33 — Metadata coverage for table and ground-validation artifacts

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T32 (reuses its Overview/Details component and shared visual conventions — must be merged first). Independent of T34 (see Further Notes — corrects T32's original assumption).

## Problem Statement

T32 fixed the Metadata tab for chart-kind outputs. Every assistant message also carries a parallel `msg.artifacts[]` array (`artifact.type ∈ table, map, comparison, timeseries`), and T32 assumed all four artifact types uniformly "bypass tabs" and need the same treatment. Investigation shows that's only true for two of them:

- **`table` artifacts** are reachable and rendered (`TableArtifactMessage`, `Frontend/src/components/ArtifactMessage.jsx:171-380`) but have zero Metadata tab — just a paginated grid and a CSV export button. No dataset identity, no provenance, no "what am I looking at."
- **Ground-validation `timeseries` artifacts** — the satellite-vs-station composites produced by `validate_against_ground` and `exceedance_overlay` (`Backend/tools/satellite_tools/validation_tools.py`) — are worse than un-tabbed: they are **not reachable in the UI at all**. They call `build_artifact_reference` directly and never call `emit_chart`, so they never join `msg.charts`; and both places that decide which artifacts get a clickable card (`Frontend/src/components/Chat.jsx:505,517` and `Frontend/src/App.jsx:243`) filter to `artifact.type === 'table'` only. A researcher running a satellite/ground validation gets a tool response in chat but no card to click — the comparison itself is invisible.
- **`map`, `comparison`, and chart-backed `timeseries` artifacts** (render types `heatmap`, `heatmap_multi`, `timeseries` that *do* call `emit_chart`) are deliberately **not** surfaced as separate cards — per the existing code comment at `Chat.jsx:501-504`, they'd duplicate what the already-tabbed `chart` object shows. Their `ArtifactReference.metadata` is a thin stub (bbox/colorbar for map; panels[handle,title] for comparison) that carries far less than the parallel chart's provenance — building a second, thinner Overview/Details for these would drift out of sync with T32's richer chart view for no benefit. This PRD does not touch them.

## Solution

Extend T32's Overview/Details pattern to the two artifact types that actually need it — table and ground-validation timeseries — and wire up `MetadataTab`'s currently-dead `artifact` prop branch (`OutputPanel.jsx:161-169`, never invoked because the call site at line 406 only passes `chart`) to render it. Widen the card-reachability filters in `Chat.jsx` and `App.jsx` so ground-validation timeseries artifacts (no matching entry in `msg.charts`) become clickable at all, while leaving chart-backed artifact types exactly as un-carded as they are today. On the backend, attach the stats/coverage/exceedance-date facts that `validate_against_ground`/`exceedance_overlay` already compute but currently drop when building the artifact stub, so the new Overview/Details has real content instead of just `series[]`/`source_handles`.

## User Stories

1. As a general user, I want a table output to answer "what is this data" the same way a chart does, so transparency isn't chart-exclusive.
2. As a researcher running a satellite/ground validation, I want the resulting composite artifact to actually appear as a clickable card, so I can inspect it — today it silently doesn't show up.
3. As a researcher, I want that artifact's Overview to show the satellite variable, the ground station(s) being compared, and a correlation/coverage summary, so I can judge the comparison's validity at a glance.
4. As a researcher, I want the Details view for a ground-validation artifact to show exceedance dates (for exceedance-overlay results) or full correlation methodology (for direct validation), units for both series, and source handles, so I can audit the comparison.
5. As a developer, I want table and ground-validation Overview/Details built from the same shared primitives T32 introduced (`StatCard`, `MetaChip`, the accordion pattern), so visual consistency holds across every metadata-bearing surface.
6. As a developer, I want map/comparison/chart-backed-timeseries artifact stubs left completely untouched — no new cards, no new tabs — so we don't ship a second, thinner metadata view that duplicates and can drift from the chart card's already-richer one.

## Implementation Decisions

- **Reachability fix.** `Chat.jsx:505,517`'s artifact-card filter widens from `a.type === 'table'` to also include `a.type === 'timeseries'` artifacts that have no corresponding entry in `msg.charts` (checked by `chart_id`/`id` — ground-validation tools mint their own `chart_id` locally and never call `emit_chart`, so no chart will ever share that id). `App.jsx:243`'s auto-focus-on-reply effect gets the analogous widening. Chart-backed `timeseries` artifacts (which *do* have a matching chart) are excluded by this same check, so no duplicate card appears for them — preserving today's behavior exactly.
- **Wire up `MetadataTab`'s `artifact` branch.** Table and ground-validation-timeseries focus now passes `artifact` into `MetadataTab` for real. It renders a new artifact-shaped Overview/Details — a sibling to T32's chart version, sharing the accordion/`StatCard`/`MetaChip` primitives but with its own field-source adapter (artifact metadata shape, not chart provenance shape).
- **Table Overview:** title, row/column counts, source dataset if the producing tool sets one (verify per table-producing tool — e.g. EPA AQS — during implementation; where absent, render "Not available" per T32's convention).
- **Table Details:** full column list, CSV export (relocated/surfaced inside Details rather than only the table's own toolbar), raw metadata JSON toggle.
- **Ground-validation Overview:** satellite variable + ground station(s) from `metadata.series[]` (`label`/`source_kind`/`station_id`), a correlation/coverage summary line (new — see backend decision below), and a QA status line reusing T32's masking-summary approach if satellite-side masking info is available on the underlying handle.
- **Ground-validation Details:** full per-series breakdown (label, source_kind, station_id, units), exceedance-dates list for `exceedance_overlay` results or correlation-methodology stats for `validate_against_ground` results, source_handles, raw JSON toggle.
- **Backend enrichment.** `TimeseriesArtifactMetadata` (`Backend/models/artifact.py:48-63`) gains optional fields — `stats` (correlation/coverage, already computed in `validation_tools.py`'s `ts_payload["stats"]`/`["coverage"]` but currently dropped when building the artifact stub) and `exceedance_dates` (already computed for `exceedance_overlay`, same gap) — additive and optional, so existing consumers of the smaller shape are unaffected.
- **Explicitly untouched:** `MapArtifactMetadata`, `ComparisonArtifactMetadata`, `_RENDER_TYPE_TO_ARTIFACT_TYPE`, `MapArtifactCard`/`ComparisonArtifactCard`, and the existing `Chat.jsx:501-504` "chart-backed types duplicate `msg.charts`" behavior — no new cards, no new tabs for these.

## Testing Decisions

- Backend (`docker compose --profile test run --build --rm backend-test`): `validate_against_ground`/`exceedance_overlay` tests assert the enriched `TimeseriesArtifactMetadata` fields (`stats`/`coverage`/`exceedance_dates`) now appear on the built `ArtifactReference`.
- Frontend (`docker compose --profile test run --build --rm frontend-test`): a fixture message containing only a ground-validation timeseries artifact (no matching chart) now produces a clickable `OutputCard` — regression-tests the widened filter; a fixture with a chart-backed `heatmap_multi`/chart-timeseries artifact **still produces exactly one card** (the chart card, no duplicate) — guards against re-introducing the duplication the existing code deliberately avoids; `MetadataTab` given a table fixture and a ground-validation fixture each render the expected Overview/Details sections; a fixture with a missing optional field renders "Not available" per T32's established convention.
- Live verification: run an EPA/ground-station validation query in the browser, confirm the composite timeseries artifact is now clickable and its Overview shows real station/correlation info; open a table output and confirm Overview/Details now appear; run a plain heatmap and a region-comparison query and confirm each still shows exactly one card, not two.

## Out of Scope

- Overview/Details or new cards for `map`/`comparison`/chart-backed-`timeseries` artifact stubs — intentionally excluded; their content is already covered by T32's chart tabs (see Problem Statement).
- Compare-mode wiring — T34, and per Further Notes below, not a dependency of this PRD.
- Any change to `validate_against_ground`/`exceedance_overlay`'s actual statistical computation — this PRD only attaches facts the tools already compute onto the artifact stub; no new stats are invented.

## Further Notes

This PRD corrects an assumption made in T32's Further Notes/Out of Scope, which described a future T33 as extending Overview/Details "to map/table/comparison/timeseries-artifact outputs" uniformly. Investigation here found that framing wrong: `map`/`comparison`/chart-backed-`timeseries` artifacts are deliberately un-carded duplicates of the chart card by existing design, and only `table` plus the previously-*invisible* ground-validation `timeseries` artifacts needed real work. `docs/prds/prd-t32-metadata-tab-overview-details.md`'s references to T33 should be read in light of this narrower, corrected scope.

## Kickoff

**Recommended model:** Sonnet 5. The reachability-filter change needs careful regression testing (must not accidentally surface duplicate cards for chart-backed artifact types), but otherwise reuses T32's established component patterns directly.

**Starter prompt:**
> Implement PRD T33 (`docs/prds/prd-t33-artifact-metadata-coverage.md`) in Talking-to-Air. T32 must already be merged (chart Overview/Details, plus the `StatCard`/`MetaChip`/accordion primitives it introduces) — confirm that first. Widen the artifact-card reachability filters in `Frontend/src/components/Chat.jsx` (~line 505/517) and `Frontend/src/App.jsx` (~line 243) so ground-validation `timeseries` artifacts (type `timeseries`, no matching entry in `msg.charts`) become clickable, while chart-backed artifact types (which do have a matching chart) remain un-carded exactly as today — write a regression test proving no duplicate card appears for those. Wire `MetadataTab`'s dead `artifact` prop branch (`OutputPanel.jsx:161-169`, call site at line 406) to actually receive `artifact` for table and ground-validation-timeseries focus, and build the artifact-shaped Overview/Details per the PRD's Implementation Decisions, reusing T32's shared visual primitives. On the backend, add optional `stats`/`coverage`/`exceedance_dates` fields to `TimeseriesArtifactMetadata` (`Backend/models/artifact.py`) and populate them from what `Backend/tools/satellite_tools/validation_tools.py` already computes. Do not touch `MapArtifactMetadata`, `ComparisonArtifactMetadata`, or any chart-backed artifact rendering — those are explicitly out of scope. Write the tests in the PRD's Testing Decisions and run both suites via `docker compose --profile test run --build --rm backend-test` and `frontend-test` before considering this done.
