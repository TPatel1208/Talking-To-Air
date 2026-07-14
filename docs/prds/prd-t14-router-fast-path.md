# PRD T14 — Deterministic router fast path with checkpointed history write-back

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T12 (provider split), T13 (compact sub-agent results). Decision record 2026-07-06 §3.

## Problem Statement

As a researcher asking an unambiguous question — "plot NO2 over New Jersey yesterday" — I wait through two supervisor model calls that add no information: the intent router already classified my message deterministically, yet the supervisor is still invoked once to emit the tool call it was ordered to make and once more to restate the sub-agent's summary under a header. Those two calls are pure latency and rate-limit spend on the majority of queries, and they exist only because the supervisor is the sole owner of conversation memory.

## Solution

Finish the thought the intent router started: when it classifies a message as ground-only or satellite-only, the chat service dispatches directly to that sub-agent and streams its result as the answer — no supervisor model calls at all. The turn (user message and final answer) is then written into the supervisor's checkpointed thread, so the conversation the supervisor sees on its next genuinely ambiguous turn is complete and follow-ups keep their antecedents. Ambiguous and cross-source messages take the supervisor path exactly as today.

## User Stories

1. As a researcher asking an unambiguous single-source question, I want the answer to start streaming as soon as the sub-agent produces it, so that I stop paying two extra model calls of latency on the most common query shape.
2. As a researcher, I want a follow-up like "now compare that to last month" to work after a fast-pathed turn, so that taking the fast path never costs me conversation memory.
3. As a researcher asking an ambiguous or cross-source question, I want the supervisor to handle it with full history exactly as before, so that the fast path changes nothing it does not speed up.
4. As the supervisor, I want fast-pathed turns present in my checkpointed thread when I am next invoked, so that my context is the whole conversation, not the subset that happened to be ambiguous.
5. As the chat frontend, I want fast-pathed responses delivered through the same SSE event vocabulary (text, chart, artifact, tool events, done), so that the UI cannot tell which path produced an answer.
6. As a researcher, I want the fast-pathed answer to state which source answered (ground or satellite), so that the response format I am used to is preserved.
7. As the system owner, I want the routed-vs-supervised decision logged per request, so that the fast path's hit rate is measurable from logs.
8. As the ground-sensor path, I want the prior-monitor context that the supervisor used to inject into my tasks preserved on the fast path, so that follow-up monitor questions keep working.
9. As the developer, I want end-to-end tests proving each route (ground, satellite, both, ambiguous) invokes exactly the expected agent(s), so that routing regressions are caught by a command.
10. As a researcher whose fast-pathed sub-agent fails, I want a clear error answer and an intact conversation, so that one failed turn does not corrupt the thread for the next.

## Implementation Decisions

- The dispatch decision lives in the chat streaming service, at the point where the routing hint is computed today: ground-only and satellite-only classifications dispatch directly to the corresponding sub-agent; both/ambiguous classifications invoke the supervisor unchanged. The routing-hint prefix injection into supervisor messages is retired along with the prompt rules that consumed it.
- The sub-agent invocation on the fast path reuses the same wrappers the supervisor's tools use (context enrichment, envelope finalization, per-request budget, metrics), so behavior differs only in who initiated the call. Those wrappers move to a shared home both callers import.
- History write-back uses the checkpointer graph's state-update mechanism: after a fast-pathed turn completes, the user message and the final synthesized answer are appended to the supervisor's thread state for that thread id. If write-back fails, the turn still answers; the failure is logged loudly (memory degradation is acceptable, silent memory corruption is not).
- The response header convention ("Agent consulted: …") is produced deterministically on the fast path from the route taken.
- The ground path's cross-turn monitor context (previously held in supervisor process memory) moves to per-thread persisted metadata so both paths read and write the same context.

## Technical Implementation Guide

Paths current as of `talking-to-air-v2` (c75f4da + working tree).

- **Dispatch point.** `ChatStreamService.stream_chat_events` (`Backend/services/chat_stream_service.py`) currently calls `inject_routing_hint(message)` and always streams through the supervisor. Replace with: `intent = route_intent(message)` (`Backend/services/intent_router.py`); on `"GROUND"`/`"SATELLITE"` run the sub-agent dispatch below; on `"BOTH"`/`"LLM"` invoke the supervisor with the **raw** message (retire `inject_routing_hint` and the `[ROUTE:…]` rules in `Backend/config/supervisor_prompt.py` — they exist only to make the supervisor obey a decision already made).
- **Shared sub-agent runners.** The logic to reuse lives today as closures inside `build_agent` (`Backend/agents/supervisor_agent.py`): `_run_ground`, `_run_satellite`, `_inject_ground_context`, `_extract_ground_monitor_context`, `_finalize_sub_agent_result`, the call-budget counters, and `record_agent_request` metrics. Extract them into a new `Backend/services/subagent_dispatch.py` that takes the built sub-agents (constructed once at startup — note `api.py`'s lifespan currently builds sub-agents *inside* `build_agent`; hoist them so both the supervisor tools and the fast path share the same instances). The supervisor's `@tool` wrappers become thin shims over this module.
- **Streaming on the fast path.** `_run_satellite` already consumes `stream_response(satellite_agent, ...)` events — on the fast path, forward those events into the SSE stream (`tool_call`, `status`, `job_progress`, chart/artifact events per T13) instead of buffering, then emit the final `text`/`done` from the finalized envelope summary, prefixed deterministically with `Agent consulted: GROUND` / `Agent consulted: SATELLITE` (format defined in `supervisor_prompt.py`; `_strip_supervisor_preamble` in `chat_stream_service.py` expects it).
- **History write-back.** The supervisor from `create_agent(...)` is a compiled LangGraph graph with the Postgres checkpointer (`utils/db.py::get_checkpointer`). After a fast-pathed turn: `await agent.aupdate_state({"configurable": {"thread_id": thread_id}}, {"messages": [HumanMessage(user_message), AIMessage(final_answer)]})`. Verify the messages key/reducer matches the agent's state schema (it appends via `add_messages`). On failure: log `fast_path_writeback_failed` at WARNING and continue — the answer already streamed.
- **Monitor context.** `last_ground_monitor` is currently a closure dict in `build_agent` — process-wide, shared across *all* users/threads (a pre-existing bug this PRD fixes in passing). Move it to per-thread storage: `Backend/repositories/session_metadata_repository.py` already persists per-session metadata and is the natural home; key by `thread_id`.
- **History rendering.** `HistoryService` reads the checkpointed thread to rebuild the sidebar conversation — confirm written-back turns render correctly there (they are ordinary Human/AI messages, so they should).

## Testing Decisions

- Good tests observe which agents were invoked and what the SSE stream contained — never internal routing state. The chat-endpoint seam with stubbed agents (prior art: chat-endpoint and extracted-services tests) covers: ground-only → ground agent only; satellite-only → satellite agent only; both/ambiguous → supervisor; sub-agent failure → error event and intact thread.
- Write-back is tested at the same seam: fast-path a turn, then run a supervisor turn on the same thread and assert its input contains the fast-pathed exchange.
- The intent-router unit tests (existing prior art) are extended with the sentences the fast path must and must not catch.
- Three end-to-end eval tasks (one ground, one satellite, one cross-source) are added to the scripted eval per decision record §6, scored on which agent(s) ran and a non-error answer.

## Out of Scope

- Improving the intent router's classification quality (patterns stay as-is; misclassified-to-ambiguous is safe by construction). Supervisor prompt redesign. Streaming sub-agent tokens through the supervisor path.

## Further Notes

Latency accounting from the 2026-07-06 session: on the majority query shape this removes the two supervisor completions — the only remaining model spend is the sub-agent's own workflow, which T12 gave a dedicated rate budget and T13 made context-safe.
