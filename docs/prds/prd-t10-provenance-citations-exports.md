# PRD T10 — Provenance pane, citations, methods & data export

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T06 (source_handles on all artifacts), T07/T08 (the analyses worth tracing). Ships as one story with the lineage outputs per the pane-ordering decision.

## Problem Statement

As a researcher preparing results for a paper, I can make figures but I cannot defend them: no record of how a figure was produced, no dataset citations in the forms journals require, no methods text, and no way to hand a collaborator the underlying numbers. The platform's whole provenance substrate (spec-recording, ancestry, events) exists MCP-side but is invisible at the surface where research is judged.

## Solution

The fourth workbench pane plus the outputs it justifies: a provenance panel that walks any artifact's `source_handles` through the MCP's lineage tool and renders ancestry + lifecycle events ("how was this made"); dataset citations via the MCP's citation tool; a methods-export action generating the paragraph a paper's methods section needs (AOI, windows, datasets + DOIs, processing chain, retrieval dates); and data download endpoints (CSV/NetCDF) over the MCP's format conversion + export.

## User Stories

1. As a researcher, I want a "how was this made" view for any artifact — datasets, AOI, time window, transform chain, provider events (routed → submitted → materialized), granule counts — so that every figure traces to a re-materializable spec.
2. As a researcher, I want the lineage rendered as a readable chain/graph with timestamps, not raw JSON, so that provenance is legible to a non-developer.
3. As a researcher, I want formal citations (DOI + citation strings from CMR's own records) collected across all datasets behind an artifact or a whole session, so that my references list writes itself.
4. As a researcher, I want a methods-text export in Markdown (and Word via the existing export pathways) assembled from the actual provenance — not from the chat transcript — so that the text reflects what was done, not what was said.
5. As a researcher, I want to download an artifact's underlying data as CSV or NetCDF, so that collaborators and reviewers get numbers, not screenshots.
6. As a researcher, I want the methods export to state the retrieval dates and dataset versions, so that reproducibility claims carry the details that actually change results.
7. As a comparison artifact (T08), I want my aligned intermediates to appear in the rendered lineage, so that resampling steps are visible in the method, not hidden.

## Implementation Decisions

- Provenance panel: per-artifact action → backend walks `source_handles` → MCP lineage tool per handle → merged, deduplicated ancestry rendered newest-last; events shown with their provider details. Handles with no lineage (AOIs, datasets) render as leaf inputs with their descriptions.
- Citations: MCP citation tool per distinct dataset handle; deduplicated; rendered as a references block and included in the methods export.
- Methods generator: deterministic template over structured inputs (AOI geometry summary, time windows, dataset identities + versions + DOIs, transform/event chain, retrieval timestamps) — no LLM in the loop, so the same session always yields the same text; an optional LLM polish pass may rewrite prose but the structured facts are non-negotiable and re-validated after polish.
- Data download: backend endpoints that call the MCP's format conversion where needed, then stream from the exported URI; extends the existing export service and artifact CSV routes rather than adding a parallel download system.
- The MCP's format-conversion tool joins the backend's (not the agent's) callable set — downloads are UI-initiated, deterministic operations.

## Testing Decisions

- Backend seams against the fake MCP: lineage walk merges multi-handle ancestries correctly; citation collection deduplicates; download endpoints stream converted content with correct media types.
- Methods generator: golden tests — canned provenance/citation fixtures → exact expected Markdown; prior art: export service tests.
- Frontend pane demo-verified; exit script: run a T07 comparison → open provenance → export methods → verify the text names the real datasets, windows, and processing chain.

## Out of Scope

- Projects/multi-user workspaces (Phase 4). PDF typesetting. Automatic upload to external repositories (Zenodo etc.).
- Any provenance capture changes MCP-side — this PRD renders what already exists.

## Further Notes

Per the cut line (decision record §8.9) this pane is the second thing to slide if the schedule compresses — but it ships together with citations/methods so lineage UI and lineage outputs land as one coherent story when it does ship.
