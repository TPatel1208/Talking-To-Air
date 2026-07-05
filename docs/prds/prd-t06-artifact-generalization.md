# PRD T06 — Artifact system generalization: map, comparison, timeseries

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T03 (handles + source_handles metadata), T05 (panel layout exists).

## Problem Statement

As a researcher, my outputs are limited to the two artifact types the chatbot era needed (table, chart). The research workflows this platform is being built for — mapped fields, side-by-side comparisons, satellite-vs-ground series — have no first-class representation: they'd be loose PNGs with no metadata, no gallery presence, no export, and no provenance hook.

## Solution

Extend the artifact model with three research-grade types — `map` (rendered field + bbox + colorbar/units), `comparison` (n-panel or difference view), `timeseries` (satellite series, optionally overlaid with ground series) — all carrying `source_handles`, all rendered in the artifact gallery (the Dashboard grown into the second workbench pane), all exportable through the existing export service.

## User Stories

1. As a researcher, I want a plotted field stored as a `map` artifact with its bbox, variable, units, and colorbar range, so that the image is interpretable and reproducible, not a bare PNG.
2. As a researcher, I want side-by-side or difference views stored as `comparison` artifacts recording both inputs, so that "A vs B" is one object I can revisit and export.
3. As a researcher, I want time-series plots stored as `timeseries` artifacts that can carry both satellite and ground series with their identities, so that the platform's signature satellite↔ground view is a first-class output.
4. As any artifact, I want `source_handles` (and for comparisons, per-panel handle attribution) in my metadata, so that the provenance pane (T10) can trace me to re-materializable specs.
5. As a researcher, I want the artifact gallery to render all five types with type-appropriate cards and detail views, so that my session's outputs are browsable in one place.
6. As a researcher, I want PNG export for visual artifacts and CSV export for tabular/series artifacts through the existing export flows, so that outputs leave the platform in usable forms.
7. As the earthdata agent, I want plot tools to register their outputs as typed artifacts and return artifact ids in the T04 envelope, so that chat and gallery stay consistent.

## Implementation Decisions

- The artifact model's type vocabulary grows by the three types; metadata schemas per type are defined once (map: bbox/variable/units/colorbar; comparison: panel list with handles + mode n-panel|difference; timeseries: series list with source kind satellite|ground, station ids where applicable) and validated at write time.
- Persistence and export extend the existing chart/artifact repository and export service patterns — same tables/flows, new types, no parallel system.
- The plot tools from T03 are updated to emit typed artifacts; rendering components extend the existing artifact-message/dashboard patterns.
- No new analysis logic in this PRD: the types and their plumbing land here; the workflows that populate comparison/timeseries richly arrive in T07/T08.

## Testing Decisions

- Service/repository seam (existing prior art: chart repository and export service tests): create/fetch/list each new type, metadata schema validation rejects malformed writes, exports produce the declared formats, `source_handles` round-trips.
- Endpoint seam: artifacts of new types serialize correctly in chat (envelope) and gallery listings.
- Frontend rendering demo-verified per the standing frontend-testing decision.

## Out of Scope

- The satellite↔ground pairing logic (T07) and alignment/differencing math (T08) — this PRD ships the containers, not the science.
- Provenance rendering (T10).

## Further Notes

Type metadata schemas are the contract T07/T08/T10 build against — treat changes after this session as breaking and version accordingly.
