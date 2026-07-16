# PRD T36 Phase 3 — Measurement explanation (on-demand, evidence-grounded)

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T35, T36 Phase 1, and T36 Phase 2 — all implemented and live-verified on `refactor/data_retrieval`. This is the **final** phase of T36: **P1 Discovery → P2 Evidence → P3 Explanation (this).** P2 computes the facts (`provenance.evidence`); P3 lets the agent explain them in words, only when asked, grounded strictly in them.

## Problem Statement

P2 computes a deterministic evidence summary (`provenance["evidence"]` — QA pass rate, uncertainty mean, context-band means, each with honest `coverage`) and the frontend renders it as a "Supporting information" block. But **the agent that talks to the user cannot see any of it.** `_chart_model_summary` (`plot_tools.py:313`) is the model-facing view of a chart, and by deliberate T13 design it carries only render_type/title/variable/units/dims/range/chart_id/source_handles — the docstring is explicit: *"the model only ever sees the compact summary"* (line 374). `provenance` (and thus `evidence`) is emitted out-of-band to the frontend and persisted (`agent_charts.payload->provenance->evidence`), never handed to the LLM.

So when a researcher asks *"why should I trust this NO₂ value?"* or *"how reliable is this?"*, the agent has **no facts to ground an answer** — it would either decline or improvise from general knowledge. Improvising is precisely the overconfident-inference failure the whole evidence/narrative split was built to prevent: the design principle is *deterministic evidence, computed once, explained on demand — never re-derived or invented by the LLM.* Today there is no bridge from the computed facts to a grounded explanation.

Two realities from P2's live verification shape this phase:
- **Evidence is usually empty.** `_safe_retrieve` pulls only the agent-chosen `variables`, which is typically just the plotted science variable — so a plain "plot O₃" yields **no** evidence. Rich evidence requires having retrieved the companions (QA flag, cloud fraction, …). P3 must handle empty evidence as the *common* case, honestly.
- **Evidence is factual, not a verdict.** P2 gives `cloud fraction = 0.098 @ coverage 0.974`, not "reliable." Turning facts into a calibrated confidence statement — without overclaiming — is the judgment P3 owns.

## Solution

Two pieces:

1. **A read-only accessor** the agent can call to retrieve a chart's already-computed evidence — `explain_measurement(chart_id)` (backend composite tool, like `safe_retrieve`/`point_timeseries`). It looks up the persisted chart payload and returns the compact `evidence` facts plus the `masking` qa_status — no recomputation (P2 owns that), no LLM. On-demand by construction: the agent calls it only when the user asks about reliability, so the compact chart summary stays lean for every other turn.

2. **A prompt rule** (`earthdata_agent_prompt.py`) — "Explaining measurement reliability" — that tells the agent: when the user asks to judge/interpret a measurement's confidence, call `explain_measurement` for the relevant chart_id, then explain **strictly** from what it returns: state facts as *evidence*, not conclusions; disclose caveats (low coverage, high uncertainty); **never assert a factor not in the returned evidence**; and when evidence is empty, say so plainly and offer to retrieve the companions that would supply it — closing the loop back to P1.

The empty-evidence case is the elegant hinge: "why trust this?" on a science-only chart → the agent honestly reports *no supporting evidence was retrieved* and offers to pull the QA flag / cloud fraction → the researcher re-plots with companions → P2 populates evidence → P3 explains it. Honest, grounded, and self-completing.

## User Stories

1. As a researcher, when I ask "how reliable is this measurement?", I want an answer built from the *actual computed facts* for this chart (QA pass rate, cloud fraction, uncertainty), not the model's general priors, so the explanation is trustworthy and specific.
2. As a researcher, I want the explanation to distinguish **evidence from inference** — "QA pass rate was 93% and cloud fraction was low (0.04), which generally supports confidence" — never a flat "this is reliable," so I'm told what's known vs. concluded.
3. As a researcher, I want caveats surfaced: if a context fact was computed over 38% coverage, or uncertainty is high, the explanation says so, so a thin fact isn't dressed up as solid.
4. As a researcher asking about a chart where only the science variable was retrieved, I want to be told honestly that there's no supporting evidence yet — and offered to retrieve the QA flag / cloud fraction — rather than given a confident-sounding answer with nothing behind it.
5. As a developer, I want the agent to *read* deterministic facts through a tool and explain them, never recompute or invent them, so the numbers stay P2's single source of truth and the LLM's role is strictly language.

## Implementation Decisions

- **`explain_measurement(chart_id)` backend composite tool.** Registered alongside the other composites the agent already uses. Reads the persisted chart payload (the chart-service / `agent_charts` persistence P2's evidence is already stored in) and returns a compact dict: the `evidence` list (name/role/stat/value/units/coverage), the `masking` qa_status/qa_source, the plotted variable/units, and an explicit `has_evidence: bool`. No recomputation, no MCP call, no LLM. If the chart_id isn't found or carries no provenance, it returns `has_evidence: false` with a reason — never an error the agent has to decode.
- **Prompt section "Explaining measurement reliability."** Added to `earthdata_agent_prompt.py`. Triggers on reliability/confidence/interpretation questions ("why should I trust", "how reliable", "how confident", "is this good data"). Directs: (a) identify the chart_id in question (from this turn's or a recent turn's envelope `artifact_ids`/chart references; if ambiguous, ask which chart); (b) call `explain_measurement`; (c) explain strictly from the returned facts with the evidence-vs-inference and caveat discipline; (d) on `has_evidence: false`, state plainly that no companion evidence was retrieved and offer to retrieve the QA flag / relevant context bands (a `suggested_followups` entry), never fabricating a confidence claim.
- **Grounding guardrails (the heart of the phase), stated in the prompt:**
  - Use only facts returned by `explain_measurement`. If a factor (PBL, geometry, aerosol) isn't in the returned evidence, do not mention it as if measured.
  - Present facts as evidence with directional framing ("low cloud fraction generally supports the retrieval"), never as a categorical verdict ("this is accurate").
  - Always surface `coverage` when it's low, and uncertainty magnitude when present.
  - Reuse, don't restate wrongly, the existing column-vs-surface and QA-disclosure rules already in the prompt.
- **Envelope handling.** A reliability explanation is a legitimate longer `summary` — relax the "one factual sentence or two" guidance *for this query class only* so the grounded explanation fits, while keeping the JSON-envelope output contract unchanged (still `summary`/`artifact_ids`/`handles`/`suggested_followups`).
- **No new computation, no frontend requirement.** The "Supporting information" block (P2) already shows the facts visually; P3 is the conversational counterpart. *Optional, secondary:* emit a "Why should I trust this measurement?" `suggested_followups` entry when a chart has non-empty evidence, so the on-demand path is discoverable (a natural tie to P1's chip surface) — include only if it fits cleanly.

## Testing Decisions

- Backend (`docker compose --profile test run --build --rm backend-test`):
  - `explain_measurement` returns the persisted `evidence` + qa_status for a chart_id whose payload carries evidence; returns `has_evidence: false` with a reason for a science-only chart and for an unknown chart_id (never raises).
  - It performs no recomputation — given a payload, the returned facts equal the stored `provenance.evidence` verbatim (guards against a second, drifting evidence path).
  - Prompt test (`test_earthdata_agent_prompt.py`): the "Explaining measurement reliability" section is present and contains the grounding guardrails (evidence-only, evidence-vs-inference, empty→offer-retrieval).
- Live (`test`/`1234`, per CLAUDE.md — rebuild backend+frontend):
  - Retrieve **TEMPO O₃ with `radiative_cloud_frac` + `uv_aerosol_index`**, plot, then ask "how reliable is this?" → agent calls `explain_measurement`, explains cloud/aerosol facts with coverage and evidence-not-verdict framing, invents no factor.
  - Retrieve **TEMPO NO₂ science-only**, plot, ask "why should I trust this?" → agent reports no supporting evidence retrieved and offers to pull the QA flag / cloud fraction (the P1 loop), with no confident-sounding confabulation.
  - Retrieve **NO₂ + `main_data_quality_flag`**, ask reliability → agent cites the real QA pass rate and qa_status `verified`.

## Out of Scope

- **Any change to what P2 computes** — P3 only reads and explains `provenance.evidence`. New evidence types (geometry suppression, added context stats) are P2 scope.
- **Actively retrieving companions on the user's behalf inside the explanation** — P3 *offers* to (a suggested followup / the P1 action path); it does not silently widen a retrieval mid-explanation.
- **A general "chat about any provenance field" capability** — `explain_measurement` returns the evidence/qa subset relevant to reliability, not the full provenance blob.
- **Frontend narrative rendering** — the explanation is the agent's conversational reply; the visual "Supporting information" block already exists from P2.

## Further Notes

This closes T36. The three phases realize one principle end to end: **P1** surfaces what companions exist and lets you act on them; **P2** computes deterministic facts from the companions you retrieved; **P3** explains those facts in words, on demand, grounded strictly in them. The LLM never touches a number it didn't get from `explain_measurement`, which never computes a number P2 didn't already store — so a confidence explanation is auditable back to a deterministic stat, which is the entire reason the evidence and narrative layers were split.

The empty-evidence-is-common reality (from P2's live findings) turned out to strengthen the design rather than weaken it: the honest answer to "trust this?" on a bare science plot is "I have nothing to judge it on yet — shall I retrieve the QA flag and cloud fraction?", which routes the researcher straight back into the retrieve→evidence→explain loop instead of manufacturing false confidence.

## Kickoff

**Recommended model:** Opus 4.8. The code is small (a read-only accessor + a prompt section), but the correctness lives entirely in judgment: grounding guardrails that don't leak into overconfident inference, and the honest empty-evidence handling. Getting the prompt discipline subtly wrong reintroduces exactly the overconfidence P2 was built to prevent.

**Starter prompt:**
> Implement PRD T36 Phase 3 (`docs/prds/prd-t36-phase3-measurement-explanation.md`) in Talking-to-Air. T35 + T36 P1/P2 are already implemented on `refactor/data_retrieval`. Add a read-only backend composite tool `explain_measurement(chart_id)` that looks up the persisted chart payload (the `agent_charts` persistence where P2's `provenance.evidence` is stored) and returns a compact dict — the `evidence` facts, `masking` qa_status, plotted variable/units, and an explicit `has_evidence` bool — with **no recomputation** (return the stored `provenance.evidence` verbatim) and no error path (unknown/empty → `has_evidence: false` + reason). Register it with the agent's tools like the other composites. Add an "Explaining measurement reliability" section to `Backend/config/earthdata_agent_prompt.py` that, on reliability/confidence questions, tells the agent to identify the chart_id, call `explain_measurement`, and explain **strictly** from the returned facts — evidence framed as evidence not verdicts, coverage/uncertainty caveats surfaced, no factor mentioned that isn't in the evidence, and on empty evidence say so plainly and offer via `suggested_followups` to retrieve the QA flag / cloud fraction (the P1 loop). Relax the summary-length guidance for this query class only; keep the JSON envelope contract unchanged. Add the backend accessor tests and the prompt test per Testing Decisions and run `docker compose --profile test run --build --rm backend-test` and `frontend-test`. Live-verify (rebuild backend+frontend, `test`/`1234`): O₃+companions → grounded explanation with caveats; NO₂ science-only → honest "no evidence retrieved, shall I pull QA/cloud?"; NO₂+flag → real QA pass rate cited. Do not change what P2 computes and do not silently widen any retrieval.
