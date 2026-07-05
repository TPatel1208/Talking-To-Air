# PRD T04 — Earthdata agent: prompts, structured returns, per-agent models, scripted eval

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T03. Consumes quirk-ledger entries from harmony-retrieval-mcp PRD 019 as available.

## Problem Statement

As the supervisor, I receive prose from sub-agents and regex-scrape it for PNG paths — brittle, and blind to handles and artifacts. As the earthdata agent, my prompt still describes the old fetch workflow, and I run on a mid-size model that must now thread handles correctly through a multi-step tool chain. As the developer, I have no way to measure whether any of this actually works before a researcher hits it.

## Solution

Rewrite the satellite agent into the **earthdata agent** around the handle workflow; replace prose sub-agent returns with a JSON envelope `{summary, artifact_ids, handles}` and delete the regex scraping; make the model per-agent configurable with a frontier tool-use model on the earthdata agent; and build a 10-task scripted eval as the permanent regression gate for agent behavior.

## User Stories

1. As the earthdata agent, I want a prompt built around search → describe → AOI → coverage check → safe_retrieve → await → open/plot, so that my tool use follows the workflow the tools were designed for.
2. As the earthdata agent, I want explicit prompt rules — always check coverage and size before retrieving, keep AOIs tight and windows minimal (TEMPO is hourly), prefer described masking metadata — so that known failure modes are pre-empted, with collection-specific guidance sourced from the live-matrix quirk ledger.
3. As the supervisor, I want every sub-agent to return `{summary, artifact_ids, handles}` as a structured envelope, so that routing and API responses are parsed, never scraped.
4. As the API layer, I want to stop regex-scraping responses for image paths, so that artifacts flow through the artifact model alone.
5. As the system owner, I want the model selected per-agent in configuration (frontier tool-use model on earthdata; cheaper models where they suffice), so that capability is spent where the failure cost is highest.
6. As the developer, I want a scripted eval of 10 canned research tasks (discovery, retrieval, plotting, comparison setup, failure recovery) scored on tool-call correctness and terminal outcome, so that agent regressions are caught by a command, not a demo.
7. As the ground-sensor agent, I want the same envelope contract applied to my returns, so that the supervisor treats all sub-agents uniformly.

## Implementation Decisions

- Envelope: sub-agents return the JSON envelope as their final structured output; the supervisor validates it (missing/invalid envelope is a sub-agent failure with a structured error, not a silent prose fallback); the API layer serves artifacts by id from the artifact model.
- Prompt architecture: one earthdata-agent prompt centered on the handle workflow; a short preset list of curated collections included as suggestions with explicit "not a ceiling" wording; per-collection quirk guidance in a clearly delimited, regenerable section fed by the quirk ledger.
- Model config: per-agent model name + provider in settings; earthdata agent defaults to a frontier tool-use model (decision record §8.4); eval results, not opinions, justify any downgrade.
- Eval harness: canned tasks with expected tool-call traces and outcome assertions, run against the fake-MCP seam (deterministic canned data); scored pass/fail per task; lives beside the test suite behind an opt-in marker (it spends real model tokens).

## Testing Decisions

- Envelope handling tested hermetically at the existing endpoint/service seam: valid envelope routes artifacts and handles; malformed envelope produces the structured failure; regex path is gone (no test references it).
- The eval is the behavioral test for prompts and model choice: 10 tasks, run before/after any prompt or model change; the pass threshold (≥8/10 initially) is recorded with the harness.
- Prior art: intent-router and chat-endpoint tests for the parsing layer.

## Out of Scope

- Any UI (T05+). Ground-agent tool changes (only its return envelope changes here).
- Continuous eval automation/CI wiring — the harness is runnable by command; scheduling it is later hygiene.

## Further Notes

This is the highest-leverage session for perceived quality: T02/T03 make workflows possible; this one makes them reliable.
