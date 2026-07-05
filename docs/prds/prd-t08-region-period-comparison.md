# PRD T08 — Region/period comparison workflow

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T06. Second signature workflow.

## Problem Statement

As a researcher, my questions are frequently comparative — "how does NO2 over Newark compare to Philadelphia?", "did ozone change after the policy took effect?", "was this June anomalous against last June?" — and the platform can only plot one field at a time. Comparing two retrievals requires aligning grids and time bases, which is exactly the fiddly work that makes researchers give up and open a notebook.

## Solution

A comparison toolset: two retrievals (two AOIs, or one AOI × two periods), aligned onto a common grid via the MCP's `align` transform (exposed to the agent from this PRD on), differenced and summarized TTA-side, rendered as `comparison` artifacts — side-by-side panels or a difference map with symmetric color scaling — plus period-over-period anomaly summaries.

## User Stories

1. As a researcher, I want to ask for a two-region comparison of a variable over a period and get side-by-side maps with a shared color scale plus summary statistics per region, so that regional contrast is one request.
2. As a researcher, I want a two-period comparison over one region to produce a difference map (period B − period A) with a diverging, zero-centered scale, so that change is directly visible.
3. As the analysis layer, I want grid alignment done by the MCP's `align` transform (a provenance-recorded transform), so that the resampling that makes a difference map honest is itself traceable.
4. As a researcher, I want the anomaly summary (mean difference, percent change, area exceeding a threshold change) attached to the comparison artifact, so that the headline number accompanies the picture.
5. As a researcher, I want mismatched inputs (different variables, disjoint periods when comparing regions) rejected with a plain explanation, so that the tool refuses to produce a scientifically meaningless panel.
6. As a provenance consumer, I want the comparison artifact to record both input handles, the aligned intermediate handles, and the comparison mode, so that every panel traces to its spec.
7. As the earthdata agent, I want one comparison tool that accepts either two AOIs or two time ranges, so that the routing decision (region vs period mode) is explicit in one schema rather than spread over tools.

## Implementation Decisions

- One analysis tool: compare (dataset, variable, mode region|period, the two AOIs/periods → comparison artifact + stats). It composes `safe_retrieve` × 2, MCP `align` (added to the curated agent toolset now — the deferred-until-needed moment from T02 has arrived), `open_handle`, and TTA-side differencing.
- Differencing, masking of cells missing on either side, and summary statistics are TTA-side (MCP no-analysis rule); the shared/diverging color-scale logic lives with the map rendering from T06.
- Region mode never differences (different domains): side-by-side panels + per-region stats. Period mode differences on the aligned grid. The mode determines the artifact's panel structure.
- Validation front-loads: variable identity, non-empty overlap of each retrieval with its requested window, and grid compatibility post-align are checked before any rendering; failures return structured explanations the agent can relay.

## Testing Decisions

- Hermetic tests at the analysis-tool seam with synthetic aligned/misaligned cube pairs: difference math (including sign convention), missing-cell masking propagation, stats against hand-computed values, region-mode refusal to difference, rejection paths.
- Fake-MCP seam supplies retrievals and a canned `align` behavior (input handles → aligned fixture handles).
- One T04 eval task covers the conversational path ("compare June 2025 vs June 2026 NO2 over NJ").

## Out of Scope

- Statistical significance testing, trend fitting, seasonality decomposition — descriptive comparison only.
- Comparing different variables/datasets against each other (e.g. NO2 vs HCHO) — a later workflow with its own semantics.

## Further Notes

Signature workflow #2; never slides. The `align` exposure here is the pattern for all deferred plumbing tools: a tool joins the curated surface when a workflow needs it, with its own session.
