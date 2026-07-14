# PRD T15 — Envelope enforcement and salvage: constrained output, graceful degradation, cheap retry

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T11 (truncation-order fix), T12 (model factory — constrained output is provider-aware). Decision record 2026-07-06 §5.

## Problem Statement

As a researcher, I sometimes watch the agent work for ninety seconds — tool calls succeeding, a chart artifact appearing — and then receive nothing but "the earthdata agent returned an invalid response envelope." The envelope contract is enforced only by prompt text, so a model under pressure occasionally emits prose instead of JSON, and the current policy throws away the completed work — including artifacts that were collected independently of the envelope — on a formatting technicality. Separately, when a sub-agent claims its tools are missing, the recovery re-runs the entire tool workflow from scratch, doubling the slowest path exactly when the provider is already throttling.

## Solution

Make envelope violations rare, then make the rare ones cheap. Rare: the sub-agents' final message is produced under provider-level constrained JSON output, so the schema is enforced by the API rather than requested by the prompt. Cheap: when a final message still fails to parse, the system salvages — the prose becomes the summary, the artifacts already collected from the tool stream are attached, and the contract violation is logged loudly with a metrics counter instead of being surfaced to the researcher. The refusal retry shrinks from a full workflow re-run to a single re-prompt of the final message.

## User Stories

1. As a researcher, I want a completed workflow to produce an answer even if the agent's last message was malformed, so that ninety seconds of correct work is never discarded over formatting.
2. As a researcher, I want charts and artifacts that were generated during the turn attached to the answer regardless of envelope validity, so that visible work is never orphaned.
3. As the earthdata agent, I want my final message generated under an API-enforced JSON schema, so that complying with the envelope contract does not depend on my discipline under context pressure.
4. As the ground-sensor agent, I want the same constrained final output, so that the supervisor continues to treat all sub-agents uniformly.
5. As the supervisor, I want a salvaged result marked as salvaged in its metadata, so that downstream consumers can distinguish a clean envelope from a rescued one.
6. As the system owner, I want every salvage logged with a named event and counted in metrics, so that contract rot stays visible in development even though users no longer see it.
7. As a researcher whose sub-agent wrongly claims its tools are missing, I want recovery via one cheap re-prompt of the final answer rather than a full workflow re-run, so that recovery does not double my wait under rate-limit pressure.
8. As the developer, I want a deliberately malformed final message covered in the scripted eval, so that the salvage path is exercised by a command, not discovered in production.
9. As an operator, I want zero user-facing "invalid response envelope" messages after this lands, so that structured-error philosophy is enforced against agents in logs, not against researchers in chat.

## Implementation Decisions

- Constrained output is requested through the model factory's provider-aware capabilities (both supported providers offer schema-constrained JSON); the envelope schema is defined once, shared by the prompt (as documentation) and the constraint (as enforcement).
- Constrained output applies to the final message only — tool-calling turns are unaffected.
- Salvage policy on parse failure: the raw prose (display-truncated per T11's ordering) becomes the summary; artifacts collected from the tool stream during the turn are attached; handles mentioned in tool results are attached when unambiguous; result metadata carries a salvage marker and a short raw preview; a warning-level named event and a metrics counter fire.
- The user-facing invalid-envelope error message is retired. The structured-error principle (a sub-agent's failure to report is not a legitimate answer) survives as observability — the decision record explicitly trades the hard-fail for salvage-plus-loud-logging.
- The refusal-marker detection stays, but its consequence becomes a single re-prompt asking the same agent to produce its final envelope from the work already in its context — never a second tool-workflow run. The deterministic-fallback plumbing that the old full retry fell through to is removed if it remains dead after this change.

## Technical Implementation Guide

Paths current as of `talking-to-air-v2` (c75f4da + working tree).

- **Schema source.** The envelope model and `parse_sub_agent_envelope` live in `Backend/models/agent_result.py`; that Pydantic model is the single schema fed to both enforcement and the prompt (`Backend/config/earthdata_agent_prompt.py`'s Output Format section stays as documentation).
- **Enforcement.** Constrained decoding must not break tool-calling turns, so do **not** set a blanket `response_format` on the agent's model. Two workable mechanisms, in preference order: (1) a `wrap_model_call` middleware (same pattern as the supervisor's `trim_middleware`) that inspects the request and applies the provider's JSON-schema response format only when tools are absent/final; (2) if that proves fragile across providers, enforce at the boundary instead — accept the loop's natural final message, and when it fails to parse, make **one** `llm.with_structured_output(SubAgentEnvelope)` call with the agent's message history to re-emit the envelope. Route provider specifics through the model factory's structured-output hook (T12). Groq: `response_format={"type": "json_schema", ...}`; Gemini: `with_structured_output`/`response_schema`.
- **Salvage.** `_finalize_sub_agent_result` in `Backend/agents/supervisor_agent.py` (moves to `subagent_dispatch.py` in T14 — coordinate) currently returns the "invalid response envelope" error result. Change: on parse failure, build `AgentResult(text=<display-truncated prose>, charts=<collected>, artifacts=<all collected refs>, metadata={"salvaged": True, "raw_preview": ...})`. Key insight: `result.artifacts` is populated from the tool stream (`_artifact_refs_from_content`) *before* envelope parsing, so salvage is a policy change — the artifacts are already in hand. Empty/whitespace text remains a structured failure. Emit `envelope_salvaged` WARNING + a counter via `Backend/utils/metrics.py`.
- **Retry demotion.** In `ask_earthdata_agent`, the `refusal_markers` branch currently calls `_run_satellite(retry_task)` — a full second tool-workflow run. (The `_try_direct_satellite_plot` fallback it used to chain into is deleted by T11.) Replace the full re-run with the single structured re-prompt from the enforcement mechanism above. Same demotion for the ground path's `_ground_retry_task` full re-run.
- **Eval robustness task.** Add a task whose fake-MCP handler workflow succeeds but whose scripted final message is prose (drive via a canned-model or by asserting the salvage function directly at the finalization seam if model scripting is impractical); the scored outcome is a non-error answer carrying the generated artifact.

## Testing Decisions

- Good tests feed final messages of varying shapes through the finalization seam and assert what the researcher receives: valid envelope → unchanged behavior; prose with collected artifacts → salvaged answer carrying those artifacts and the salvage marker; empty text → structured failure (salvage requires something to salvage).
- The constrained-output request is asserted at the model-factory seam (the request carries the schema), hermetically.
- The retry demotion is asserted at the supervisor-tool seam: a refusal-marked first result triggers exactly one further model interaction and zero additional tool-workflow runs.
- Prior art: agent-result envelope tests, supervisor agent artifact tests, eval harness robustness patterns.
- The scripted eval gains the malformed-envelope robustness task per decision record §6 and must hold its threshold.

## Out of Scope

- Envelope schema changes (fields stay `summary`, `artifact_ids`, `handles`). Supervisor synthesis behavior. Prompt rewrites beyond the enforcement note. Structured outputs for intermediate tool-calling turns.

## Further Notes

The artifacts-collected-independently observation is the crux: the stream already gathers artifact references before the envelope is parsed, so salvage is mostly a policy change, not new plumbing. This PRD converts the system's strictest accuracy failure mode into a logged degradation.
