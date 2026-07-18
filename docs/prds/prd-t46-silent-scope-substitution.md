# PRD T46 — Silent scope substitution: the answer discloses what was actually computed

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T18 (deterministic error surfacing — this extends its doctrine from hard errors to silent corrections), T42 (region fidelity — the resolver-side half of the same trust problem). Origin: exploratory QA 2026-07-17 (major finding #4, UX finding #1).

## Recommended Model

**Claude Opus 4.8.** The dispatch-layer guard and the "differ materially" comparison are exactly the kind of judgment call that's easy to get subtly too aggressive (false-positive nags) or too lax (misses a real substitution) — this is a trust-critical, deterministic-templating feature where the extra reasoning care pays for itself.

## Problem Statement

As a researcher, the system can quietly answer a different question than the one I asked, with zero disclosure in the prose I actually read:

1. **An impossible bounding box becomes a different continent.** Live repro (2026-07-17): "Plot NO2 for bounding box north=10, south=40, east=-70, west=-60" (south > north — invalid) returned **"Tropospheric_NO2 over North America"** as if that were the request. The MCP's `define_area_of_interest` correctly *rejects* an inverted bbox (`south latitude exceeds north latitude`) — the agent received that clear error and then improvised a substitute region instead of relaying it. No error, no clarifying question, and — worse — **no log trace at all**: the log auditor confirmed the substitution leaves nothing to grep for.
2. **A single-day request silently becomes a month.** A request for one day of a monthly-cadence product answers with the full-month mean; the substitution is disclosed only in the Metadata tab's fine print, never in the chat answer itself. A researcher trusting the prose gets scientifically wrong context with no red flag.
3. **A 226-year time range is silently clamped** (API-level repro, Subagent A) with only a vague "outside valid coverage" note — the answer doesn't say what range was actually used.

The common shape: a *user-fixable input problem* is "repaired" by the model instead of surfaced, and the repair is invisible. T18 made hard errors deterministic; this PRD does the same for silent corrections.

## Solution

Two layers, both deterministic:

1. **A scope echo in the artifact contract.** Retrieval-composite tools record the *requested* scope (the location/bbox string and time range the model passed) alongside the *delivered* scope (the AOI display name/bbox and time range that actually came back from the MCP) in chart provenance. When they differ materially, the chat-path answer carries a deterministic disclosure line (template, no model in the loop — T18 machinery): "Note: your request said X; the data shown covers Y."
2. **A no-substitution guard on AOI errors.** When `define_area_of_interest` fails with a `user_input`-category error in a turn, the earthdata agent's answer for that turn must be the T18 template for that error (relayed verbatim, with the suggestion) — the dispatch layer refuses to accept an envelope whose AOI differs from the requested one after such an error without an explicit clarification question. The failed validation is logged (`aoi_user_input_rejected`) so the event is greppable.

## User Stories

1. As a researcher who typed an impossible bbox, I want the answer to tell me the bbox was invalid (south > north) and ask what I meant, so that I never receive a confident map of a region I didn't ask about.
2. As a researcher who asked for one day of a monthly product, I want the chat answer itself to say "monthly product — showing the July 2024 mean", so that the substitution is impossible to miss.
3. As a researcher whose time range was clamped to the dataset's coverage, I want the answer to name the range actually retrieved, so that my trend conclusions use the right denominator.
4. As an operator, I want a rejected AOI input to leave a log event, so that silent-substitution regressions are discoverable from logs (2026-07-17: the live incident left no trace).
5. As the developer, I want the disclosure rendered from provenance facts by a template, so that the model can't paraphrase away an inconvenient correction.

## Implementation Decisions

- `safe_retrieve`/`point_timeseries`/compare composites stamp `requested_scope: {location, time_range}` and `delivered_scope: {region_name/bbox, start_date, end_date, cadence_note}` into the provenance dict they already build; chart payloads carry both.
- "Differ materially": time ranges compared after ISO normalization (any narrowing/widening counts); regions compared by resolved display-name mismatch against the requested string (exact-string fuzz is fine — the goal is catching *substitution*, not synonyms). Decided in-session against real cases.
- Disclosure rendering joins the T18/T37 template machinery (`config/error_templates.py` beside the taxonomy templates) — a `scope_note` appended to the envelope's text deterministically in `subagent_dispatch`/`_finalize_sub_agent_result`, not composed by the model.
- The AOI-error guard lives in the dispatch layer where T15 envelope enforcement already inspects tool outcomes: a `user_input` error from `define_area_of_interest` in the turn's trace + a completed retrieval for a *different* AOI ⇒ replace the answer with the error template (same mechanism as salvage; strictly deterministic).
- `bind_workspace`/tool wrapper logs `aoi_user_input_rejected` with the offending input at warning level.
- Frontend: none required (chat text carries the disclosure); the Metadata tab already shows the fine print.

## Technical Implementation Guide

- `Backend/services/retrieval_composites.py` (scope stamping), `Backend/services/subagent_dispatch.py` (guard + scope_note), `Backend/config/error_templates.py` (templates), `Backend/tools/satellite_tools/plot_tools.py`/`stat_tools.py` (provenance passthrough).
- Prior art: T15 salvage (`_finalize_sub_agent_result`), T37 template answers, `test_error_templates.py`, `test_subagent_dispatch.py`.

## Testing Decisions

- Composite test: a monthly-cadence retrieval for a one-day range → provenance carries both scopes; envelope text contains the deterministic cadence disclosure string.
- Dispatch test: a turn trace containing a `user_input` AOI rejection followed by a retrieval for a different region → the answer is the template relay, not the model's text; the log event fired.
- A matching request (no substitution) adds no note (regression: don't nag on exact matches).
- Live verification per CLAUDE.md: re-run the QA repro prompt ("north=10, south=40 ...") and confirm the answer names the invalid bbox instead of North America.

## Out of Scope

- Resolver-side fidelity (preset polygons, empty-mask retry, geocoder ambiguity) — that's T42. Clarification *dialogs* beyond the error-template relay (multi-turn disambiguation UX is its own design). MCP-side changes (its validation is already correct and clear).
