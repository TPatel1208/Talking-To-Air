# PRD T16 — Eval hardening: latency budgets, end-to-end routing tasks, event-loop offload

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T12, T13, T14, T15 (this PRD measures and locks in their gains). Decision record 2026-07-06 §6.

## Problem Statement

As the developer, I can measure whether the earthdata agent calls the right tools, but not whether the system is fast — no number fails when a change makes answers slower, so the speed regressions that motivated this whole series could silently return. The eval also never touches the layers the series just rebuilt (routing fast path, envelope salvage), and two claims in the codebase are currently unverifiable or false: the scalability document says blocking work is offloaded to threads (it is not — grid crunching runs on the event loop and freezes every concurrent stream), and the per-request sub-agent budget may never fire depending on how context propagates into tool tasks.

## Solution

Make speed a scored dimension and the redesigned layers eval-covered: every eval task records wall-clock and fails its category budget (ground under fifteen seconds; satellite plotting under forty-five seconds against the fake MCP; zero provider rate-limit responses in a single-user run); three end-to-end tasks exercise the router fast path; one robustness task exercises envelope salvage. Alongside the measurement, the two falsifiable claims are made true: CPU-bound grid work moves off the event loop, and the sub-agent call budget is proven effective (or fixed) by a test.

## User Stories

1. As the developer, I want each eval task to record and report wall-clock time, so that speed regressions fail a command instead of surfacing in demos.
2. As the developer, I want per-category latency budgets asserted by the eval, so that the targets agreed in the decision record are enforced, not aspirational.
3. As the developer, I want the eval run to fail if any provider rate-limit response occurred in single-user conditions, so that rate-limit pressure — the original outage mode — can never silently reappear.
4. As the developer, I want end-to-end eval tasks that enter through the chat layer (one ground, one satellite, one cross-source), so that routing, fast path, and envelope handling are covered by the same gate as agent behavior.
5. As the developer, I want a robustness eval task with a deliberately malformed final message, so that the salvage path stays exercised.
6. As a researcher using the app while someone else's plot renders, I want grid processing off the event loop, so that my stream and the health endpoint stay responsive under concurrent load.
7. As an operator, I want the scalability documentation to describe what the code actually does, so that operational reasoning is based on true claims.
8. As the supervisor, I want the per-request sub-agent call budget proven to actually increment across calls within one request, so that the "call each agent once" guarantee is real and not an artifact of context-propagation luck.
9. As the developer, I want the eval to print a compact per-task table (pass/fail, tool trace verdict, seconds), so that a regression's location is obvious from the output alone.
10. As the system owner, I want the eval documented as the required before/after gate for any prompt, model, or routing change, so that the regression discipline survives this PRD series.

## Implementation Decisions

- Latency capture lives in the eval harness's task runner; budgets are per-category constants recorded beside the pass threshold. Budgets from decision record §6; the satellite budget applies against the fake MCP (the eval measures this system's overhead, not NASA's).
- Rate-limit detection is observed at the eval level (provider retry/429 evidence during the run fails the run) — single-user cleanliness is the bar.
- End-to-end tasks drive the chat streaming layer with the real router and real sub-agents against the fake-MCP seam, scored on which agent(s) ran and on a non-error terminal answer. These extend, not replace, the existing direct-agent tasks.
- CPU-bound work in the plot/statistics path (opening handles, masking, aggregation, payload serialization) is dispatched to worker threads at the tool boundary — the highest point that covers all of it without touching the numeric code itself. The scalability document is corrected to match.
- The sub-agent budget gets a hermetic test that invokes an agent tool twice within one request context and asserts the second call is refused; if context propagation defeats the current mechanism, the counter moves to a per-request holder that survives task boundaries. The test defines the contract; the mechanism serves it.

## Technical Implementation Guide

Paths current as of `talking-to-air-v2` (c75f4da + working tree).

- **Latency capture.** `run_eval_task` in `Backend/tests/eval_harness.py` wraps the agent run — record `time.monotonic()` around the `stream_response` loop and add `elapsed_seconds` to `EvalTaskResult`. Budgets: a `CATEGORY_BUDGETS` dict beside `PASS_THRESHOLD`/`TOTAL_TASKS` (ground validation 15s; plotting/retrieval/comparison 45s; discovery 15s). `test_eval_harness.py` asserts both the pass threshold and the budgets, and prints the per-task table (name, category, pass/fail, trace verdict, seconds).
- **429 detection.** Groq's client logs retries via the `groq._base_client` logger ("Retrying request to …"), visible in production logs. In the harness, attach a `logging.Handler` capturing records from `groq` and `httpx` loggers for the duration of the run; any record matching a retry/429 pattern fails the run. This is cruder than instrumenting httpx but has zero production footprint.
- **End-to-end tasks.** Three new tasks enter through `ChatStreamService.stream_chat_events` with the real `intent_router`, real sub-agents, and the fake MCP (`FakeEarthdataMCPServer`) — prior art for driving the service layer: `Backend/tests/test_chat_endpoint.py` and `test_extracted_services.py`. Ground task needs the `aqs_get` stub pattern already in the harness. Score: which agent(s) ran (observe `tool_call`/dispatch events) + non-error `done` event. These run under the same opt-in `eval` pytest marker.
- **Event-loop offload.** The synchronous hot path is inside the plot/stat tools (`Backend/tools/satellite_tools/plot_tools.py`, `stat_tools.py`, `comparison_tools.py`, `validation_tools.py`): `xr.open_zarr` (via `services/open_handle.py::_open`), `mask_data_by_geometry`, `AggregationService.aggregate`, and `_da_to_heatmap_payload` serialization. Highest single seam: make `_open` in `open_handle.py` run under `asyncio.to_thread`, and wrap each tool's mask→aggregate→payload block in one `asyncio.to_thread(lambda: ...)` call — the numeric code itself doesn't change. Then correct `Backend/SCALABILITY.md`, which currently claims this already happens.
- **Budget-counter proof.** `_ground_call_count`/`_satellite_call_count` in `supervisor_agent.py` are ContextVars `.set()` inside tool coroutines; if LangGraph executes each tool call in a fresh `asyncio.Task` (context copy), increments never propagate and the budget silently never fires. Test at the supervisor-tool seam: stub sub-agents, invoke the same tool twice in one request context, assert the second returns the budget-exceeded message. If it fails: replace the ContextVars with one mutable per-request holder (e.g. a dict set into a single ContextVar by `stream_response`, which already manages request-scoped context in `utils/streaming.py`) — mutating a dict through a copied context still works.
- **Documentation gate.** The eval-as-gate rule (run before/after any prompt/model/routing change) goes in `Backend/TESTING_AND_OBSERVABILITY.md`, which already documents the test suite.

## Testing Decisions

- Good tests here are the eval itself plus two hermetic guarantees. The eval (opt-in marker, spends real tokens — existing prior art) is the behavioral gate; its output table is the deliverable a human reads.
- Event-loop responsiveness is asserted hermetically: while a large synthetic grid renders, a concurrent trivial coroutine must complete within a tight bound — observable behavior, no thread-pool internals inspected.
- The budget test uses the existing supervisor-tool seam with stubbed sub-agents (prior art: supervisor agent artifact tests).
- Prior art throughout: the eval harness and its threshold test, chat-endpoint tests for the end-to-end entry.

## Out of Scope

- CI wiring/scheduling of the eval (stays run-by-command). Multi-user load testing (the load-test script exists separately). New product behavior — this PRD adds measurement and honesty, not features.

## Further Notes

This PRD intentionally lands last: it freezes the series' gains behind numbers. The decision record's acceptance bar becomes executable here — after this, "nothing works" claims are answerable with a command and a table.
