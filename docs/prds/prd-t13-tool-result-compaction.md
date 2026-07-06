# PRD T13 — Tool-result compaction: chart payloads out of LLM history, subagent trim safety net

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T11. Decision record 2026-07-06 §4.

## Problem Statement

As a researcher, my longer satellite conversations degrade or die: every plot/statistics tool returns its full render payload — a grid of up to 8,000 cells, a parallel points list of up to 8,000 more values, plus provenance and export blocks — as the tool result the agent's model re-reads on every subsequent step. By the middle of a workflow the request body exceeds the provider's size cap, the request is rejected outright, and what survives is a mutilated history the model then reasons over — an accuracy failure disguised as an infrastructure hiccup. The supervisor already protects itself from these payloads by compacting them before its own model calls; the agent that generates them has no such protection.

## Solution

A tool result the model sees and the render payload the frontend sees become two different things. Plot/statistics tools return the model a compact summary — what was rendered, its dimensions and value range, the artifact id to cite, and the handles involved — while the full payload flows out-of-band through the existing chart/artifact pipeline exactly as today. A high-ceiling trim on the sub-agents acts as a last-resort safety net, sized so it never fires in a healthy workflow.

## User Stories

1. As a researcher, I want long multi-step satellite workflows to complete without the provider rejecting oversized requests, so that complex questions are answerable at all.
2. As a researcher, I want the charts in my chat to be exactly as detailed as they are today, so that compaction is invisible in the UI.
3. As the earthdata agent, I want each plot tool to tell me what it rendered, the artifact id, and the handle it used — and nothing bulkier — so that my message history stays small enough to reason over for the whole workflow.
4. As the earthdata agent, I want the compact result to include the value range and grid dimensions, so that I can describe the chart in my summary without re-reading raw data.
5. As the earthdata agent, I want the artifact id in the tool result, so that I can cite it in my envelope exactly as my prompt requires.
6. As the frontend, I want the full chart payload delivered through the existing chart/artifact events unchanged, so that no rendering, persistence, or export path changes.
7. As the supervisor, I want sub-agent tool results already compact when they reach me, so that my own input-compaction becomes a second line of defense rather than the only one.
8. As the system owner, I want a high-ceiling message trim on both sub-agents as a final safety net, so that even an unforeseen bloat source degrades gracefully instead of erroring — while never dropping messages in a healthy workflow, since dropped messages lose the handles the next step needs.
9. As an operator, I want a logged event whenever the safety-net trim actually fires, so that any remaining bloat source is discovered from logs, not user reports.
10. As the developer, I want the scripted eval to hold its threshold after compaction, so that removing raw data from the model's view demonstrably does not remove information it needed.

## Implementation Decisions

- The seam is the existing chart-persistence helper every plot/statistics tool already funnels through: it continues to build and emit the full payload (chart events, artifact references, durable persistence) but returns the model only the compact summary — render type, title, variable, units, dimensions, value range, artifact id, source handles.
- The compact summary is a structured (JSON) tool result so downstream parsing (artifact reference collection, envelope citation) keeps working; the artifact-reference convention embedded in tool results is preserved.
- The streaming layer's existing dual role (charts to the UI, text to the model) is unchanged; this PRD only changes what the tool returns to the model, not what it emits to the stream.
- The trim safety net wraps both sub-agents with a ceiling well above any healthy workflow (order of twenty thousand tokens), strategy "keep the most recent", and a logged event on activation. It exists to convert a hard provider rejection into a degraded-but-alive turn, nothing more.
- The supervisor's existing input compaction stays — defense in depth, and it still covers the ground agent's table payloads.

## Technical Implementation Guide

Paths current as of `talking-to-air-v2` (c75f4da + working tree).

- **Where the bloat is born.** `_save_chart` in `Backend/tools/satellite_tools/plot_tools.py` returns `json.dumps(payload)` — full grid (`values`, up to `_MAX_GRID_CELLS = 8000` cells), a parallel `points` dict (up to 8000 more), plus `provenance`/`query`/`export` blocks. That string is the tool's return value, which is simultaneously (a) the ToolMessage content re-sent to the model every subsequent turn and (b) the `tool_result` stream event the chart pipeline parses. The same pattern applies to `comparison_tools.py`, `validation_tools.py`, and `stat_tools.py` outputs that embed chart payloads.
- **The mechanism: split the two audiences with a chart emitter.** Mirror the existing `emit_job_progress` pattern in `Backend/utils/streaming.py`: add `emit_chart(payload: dict)` backed by a ContextVar emitter that `stream_response` wires into its queue as a new `("chart_payload", data)` event. Tools call `emit_chart(full_payload)` and **return** only the compact JSON summary: `{render_type, title, variable, units, grid_dims, vmin, vmax, chart_id, "_artifact_refs": [...], source_handles}`. Keep `_artifact_refs` in the return value — `_artifact_refs_from_content` in `supervisor_agent.py` and `_artifact_refs` in `chat_stream_service.py` both scrape it from tool-result content.
- **Consumers to rewire.** (1) `supervisor_agent.py::_run_satellite` currently harvests charts via `parse_chart_payload(content)` on `tool_result` events — harvest from the new `chart_payload` events instead. (2) `ChatStreamService._tool_result_events` parses charts out of tool-result content via `chart_service.parse_charts` — handle the new event type in `stream_chat_events` and persist via the existing `chart_service.persist_chart_payload`. (3) The ground/table path (`_artifact_refs` + `artifact_store.claim`) is already compact — leave it alone. The SSE events the *frontend* sees (`chart`, `artifact`) must not change shape (`Frontend/src/hooks/useChat.js` switches on them).
- **Supervisor input compaction stays.** `_compact_model_input_message`/`_compact_model_input_content` in `supervisor_agent.py` remain as second-line defense; after this PRD they should rarely match anything.
- **Trim safety net.** Reuse the supervisor's `trim_middleware` pattern (`wrap_model_call` + `trim_messages`) in `build_earthdata_agent` and `build_ground_agent` with `max_tokens` ≈ 20000, `strategy="last"`, and a `WARNING`-level named event (`subagent_trim_activated`) when the trimmed list is shorter than the input. It must never fire in the eval run — assert that in the eval output.

## Testing Decisions

- Good tests assert the two-audience contract at the tool seam: invoking a plot tool against the fake-MCP seam yields (a) a chart event carrying the full grid and (b) a tool result under a strict size bound that still contains the artifact id and handles. No test inspects how compaction is implemented.
- The chat-endpoint/stream-service seam asserts the frontend contract is byte-for-byte compatible: chart and artifact events unchanged in shape.
- The safety net is tested by constructing an oversized history and asserting the turn completes with the trim event logged.
- Prior art: satellite plot payload tests, artifact registry tests, chat-endpoint SSE tests.
- The scripted eval runs before/after; the plotting and comparison tasks are the ones this could plausibly regress, and they must hold.

## Out of Scope

- Reducing payload size sent to the frontend (already downsampled). Supervisor prompt or routing changes. Offloading CPU-bound grid work off the event loop (T16). Envelope schema changes (T15).

## Further Notes

Production logs from 2026-07-06 show an HTTP 413 from the model provider followed by truncation warnings mid-conversation — this PRD removes the cause rather than softening the symptom. It is also a prerequisite for T14's fast path being safe: once sub-agent results stream directly to users, they must already be compact.
