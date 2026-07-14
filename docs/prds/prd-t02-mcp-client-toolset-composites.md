# PRD T02 — Earthdata MCP client, curated toolset, and retrieval composites

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T01; harmony-retrieval-mcp PRD 017 + 018.

## Problem Statement

As the earthdata agent, I have no way to call the retrieval MCP: no client layer, no toolset, and no answer to the two ergonomic hazards of durable retrieval — an LLM burning turns polling job status, and an LLM free-running expensive retrievals (an hourly TEMPO cube over a loose AOI explodes into tens of GB and can evict everyone else's cache).

## Solution

An MCP client layer (langchain-mcp-adapters, streamable HTTP, bearer token) that loads a **curated ~12-tool** surface for the earthdata agent, a workspace-binding wrapper so the model never sees or invents `workspace_id`, and two backend composites: `await_retrieval` (the backend absorbs polling latency and streams progress) and `safe_retrieve` (estimate → gate → retrieve, with the gate numbers in config, not prompts).

## User Stories

1. As the backend, I want the MCP toolset loaded at startup over streamable HTTP with the bearer token, failing loud if the MCP is unreachable, so that a broken data layer is discovered at boot, not mid-conversation.
2. As the earthdata agent, I want exactly the curated tools — dataset search/describe/preview/summarize, AOI definition, availability/coverage checks, retrieval status, point timeseries, citation, provenance — plus the two composites, so that every expensive or error-prone path goes through deterministic code.
3. As the system owner, I want `workspace_id` bound to `user-{user_id}` by a wrapper on every call, so that the model can neither see nor forge workspace identity and a researcher's handles persist across their threads.
4. As the earthdata agent, I want one `await_retrieval(job_handle)` call to block backend-side (backoff 2→15 s, configurable timeout) and return the terminal status, so that I spend one model turn per retrieval instead of a polling loop.
5. As a researcher watching the chat, I want job-progress events streamed over the existing SSE channel while the backend polls, so that a minutes-long Harmony job reads as visible progress, not a hang.
6. As the earthdata agent, I want `safe_retrieve(dataset, aoi, time_range, variables)` to run size-estimate → gate → retrieve as one deterministic call, so that I cannot skip the estimate.
7. As a researcher, I want retrievals estimated between the soft and hard caps (2–10 GB) to pause and ask me in-chat before proceeding, so that I stay in control of big pulls without being blocked by them.
8. As the system owner, I want estimates above the hard cap (10 GB) refused with concrete tighten-your-request guidance, and the numbers in config rather than prompts, so that neither the model nor a persuasive user can talk past the guardrail.
9. As the earthdata agent, I want retrieval failures returned as the MCP's stage/provider-prefixed error string verbatim, so that I can explain to the researcher what actually failed.

## Implementation Decisions

- Client: langchain-mcp-adapters' multi-server client, configured from the T01 settings; tools loaded async at startup; the wrapper layer injects `workspace_id` and strips it from every tool schema the model sees.
- Curated surface (decision record §8.5): expose discovery (search/describe/preview/summarize), AOI, coverage (availability/coverage), `get_retrieval_status`, `retrieve_timeseries` (single-point questions on AppEEARS-covered products), `cite_dataset`, `get_provenance`, plus composites. Hidden: raw `retrieve_subset`/`retrieve_data`, transforms (`align` arrives with T08), format/inspection/cancel plumbing.
- `await_retrieval`: asyncio polling of `get_retrieval_status` with exponential backoff; emits `job_progress` SSE events (job handle, status, progress, phase, message) through the existing SSE pathway; returns terminal status including the `obs_handle`.
- `safe_retrieve`: calls `estimate_retrieval_size`; ≤ soft cap proceeds to `retrieve_subset`; between caps raises the agent-interrupt path so the supervisor surfaces a confirmation question to the user, resuming on approval; above hard cap returns a structured refusal. Caps and timeout in TTA config.
- The old satellite toolset (geocode, COLLECTIONS dict, availability checks, fetch) remains wired until T03 completes the swap — this PRD adds the new layer without ripping the old one out mid-session.

## Testing Decisions

- **New seam (the one new seam for the whole TTA series):** an in-process fake earthdata-MCP server — a FastMCP instance with canned tool responses and a tmp-dir "shared volume" — that the real client connects to over streamable HTTP in tests. Nothing mocks the adapter library's internals.
- Tests: toolset loads and matches the curated list exactly; `workspace_id` injected and invisible to tool schemas; `await_retrieval` returns terminal states (ready, failed, expired, cancelled) and emits SSE events in order; `safe_retrieve` takes all three gate branches; unreachable-MCP startup fails loud.
- Prior art: the existing service/endpoint pytest suites; async tests follow the repo's existing async test patterns.

## Out of Scope

- Rewiring plot/statistics tools to handles and deleting the old loader (T03).
- Prompt rewrites and model configuration (T04).
- The jobs panel UI consuming the SSE events (T05) — events are emitted now, rendered later.

## Further Notes

Gate numbers (2 GB ask / 10 GB refuse) are initial values in config — tune with lab experience, without touching prompts or code.
