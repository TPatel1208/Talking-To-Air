# PRD T07 — Satellite↔ground validation workflow

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T06. First of the three signature workflows.

## Problem Statement

As an air-quality researcher, I want to know how satellite column measurements relate to what ground monitors actually breathe-level measured — the validation question at the heart of TEMPO-era research. Today the platform has satellite data and EPA AQS data in separate agents with no way to bring them together: no co-location, no pairing, no correlation, no shared plot.

## Solution

A satellite↔ground toolset in the analysis layer: one cube retrieval over the AOI, satellite values extracted at every monitor location at once (nearest-cell selection TTA-side), paired with the AQS series in time, rendered as overlay `timeseries` artifacts and summarized with correlation and exceedance-day statistics. Decided mechanics (2026-07-04): cube + local extraction, not per-monitor AppEEARS point jobs — works for every product including hourly TEMPO, one MCP job per comparison.

## User Stories

1. As a researcher, I want to ask "compare TEMPO NO2 with ground monitors over New Jersey for June" and get an overlaid satellite/ground timeseries per monitor, so that validation is one conversation, not a notebook.
2. As the analysis layer, I want monitor locations for the AOI from the existing AQS toolset (monitors-in-bbox), so that co-location uses the ground network we already have.
3. As the analysis layer, I want satellite values at all monitor coordinates extracted from one retrieved cube by nearest-cell selection, so that N monitors cost one retrieval, not N jobs.
4. As a researcher, I want satellite and ground series paired on a common time base (satellite sampled to the ground series' cadence, or both aggregated to daily), with the pairing rule stated on the artifact, so that the comparison is honest about temporal mismatch.
5. As a researcher, I want per-monitor and pooled correlation statistics (r, N, coverage fraction) reported with the artifact, so that agreement is quantified, not eyeballed.
6. As a researcher, I want exceedance-day views — days a monitor exceeded a standard, marked on the satellite series or mapped over the satellite field — so that regulatory-relevant events anchor the comparison.
7. As a researcher, I want units and quantity differences (column density vs surface concentration) labeled explicitly on every output, so that no plot implies the two measure the same quantity.
8. As a provenance consumer, I want the resulting artifacts to record the cube handle, monitor ids, and pairing parameters, so that the comparison is traceable end-to-end.

## Implementation Decisions

- New analysis tools (exposed to the earthdata agent): validate-against-ground (AOI, dataset, time range, pollutant → paired series + stats + artifacts) and exceedance-overlay; both compose `safe_retrieve`/`open_handle`, the AQS tools, and the T06 artifact types — no new retrieval paths.
- Extraction: nearest-cell selection at monitor coordinates from the opened cube; cells with fill/invalid values (per T03 masking) are excluded and the exclusion counted in coverage stats.
- Pairing: default both sides to daily aggregates (AQS daily summaries exist); hourly pairing available for TEMPO×hourly-monitor data behind an explicit parameter.
- Statistics stay TTA-side by architecture (the MCP has a hard no-analysis rule): plain correlation/summary math in the analysis toolset, no modeling.
- Satellite-vs-ground is a comparison of different physical quantities: outputs always carry both quantities' units; the prompt guidance (T04's regenerable section) gains wording that the agent must state this caveat.

## Testing Decisions

- Hermetic tests at the analysis-tool seam: synthetic cube fixtures with known values at known coordinates + canned AQS responses → assert nearest-cell selection picks the right cells, pairing aligns timestamps correctly, fills are excluded and counted, correlation matches hand-computed values, artifacts carry the full metadata.
- The fake-MCP seam supplies the cube; the AQS client is stubbed at its existing HTTP boundary (prior art: the AQS toolset's tests).
- One eval task in the T04 harness covers this workflow end-to-end conversationally.

## Out of Scope

- Region/period comparisons (T08). International ground networks (OpenAQ) — a future TTA toolset.
- Bias correction, regression modeling, or any inference beyond descriptive statistics.

## Further Notes

This is signature workflow #1 and the platform's differentiator — per the cut line (decision record §8.9), it never slides.
