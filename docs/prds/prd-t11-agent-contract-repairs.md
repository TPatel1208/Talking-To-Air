# PRD T11 — Contract repairs and legacy tool removal: MCP-first minimal toolset

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T10 committed (clean baseline — T10's provenance endpoints consume two tools this PRD demotes to internal). No other PRD depends on this landing first, but land it first anyway — it removes noise from every later measurement, and T12–T15 build on the reduced surface.

## Problem Statement

As a researcher, some of my queries fail or take twice as long for reasons that have nothing to do with data: the agent's prompt references a legacy date-conversion tool that was never registered, so any ambiguous date triggers a failed tool call and a full-workflow retry; the refusal-retry guidance lists tools from the deleted pre-MCP workflow and omits the sanctioned ones, steering recovery off the rails exactly when things are already failing; a valid final answer can be cut off mid-JSON before it is parsed, so I see "invalid response envelope" even though the agent complied; and each map request resolves the same place name through multiple independent geocoders, each behind its own one-request-per-second sleep against a free service whose usage policy the request header violates.

As the earthdata agent, my toolset is also bloated with tools my own workflow never uses and my prompt in places forbids: a legacy bounding-box geocoder that duplicates the MCP's area-of-interest tool, a raw timeseries retrieval that bypasses the size gate, a status-polling tool I am explicitly told never to call, and citation/provenance/summary tools only backend endpoints consume. Every unnecessary tool costs tokens in every model call and widens the space of wrong tool choices.

## Solution

Make the earthdata-retrieval MCP the primary and only data path, and make the remaining contracts true. Delete the legacy pre-MCP tools (date conversion, bbox geocoding, the dead deterministic query-parser fallback). Shrink the model-facing toolset to exactly what the prompt's handle workflow uses — discovery, AOI, coverage, preview, the safe-retrieve gate, the await composite, and the handle-based plot/statistics/comparison/validation tools — demoting everything else to internal-only. Regenerate the refusal-retry guidance from the reduced toolset so it can never drift again. Parse sub-agent final messages before truncating. One shared geocoding service (single cache, single rate limiter, policy-compliant User-Agent) backs the internal region resolution both plot tools and EPA tools use.

## User Stories

1. As the earthdata agent, I want my toolset to contain only the tools my workflow prompt actually uses, so that every model call spends fewer tokens on schemas and tool selection has fewer wrong options.
2. As the earthdata agent, I want the MCP's area-of-interest tool to be the only geocoding path I see, so that I never mint a raw bounding-box string no retrieval tool accepts.
3. As the earthdata agent, I want no prompt references to tools that are not registered, so that following my own instructions never produces a failed tool call.
4. As a researcher, I want every satellite retrieval to pass through the size-estimate gate, so that no model-facing tool can bypass the retrieval guardrail.
5. As the system owner, I want the status-polling tool hidden from the model (internal to the await composite only), so that the "never poll" rule is enforced by construction instead of prompt discipline.
6. As the backend provenance/citation endpoints, I want the citation and lineage tools to remain available internally after they leave the model surface, so that T10's features keep working unchanged.
7. As the supervisor, I want my refusal-retry guidance generated from the earthdata agent's actual registered toolset, so that a retry teaches the sanctioned MCP workflow and cannot orphan when tools change.
8. As the supervisor, I want a sub-agent's final message parsed as an envelope before any length limit is applied, so that a compliant answer is never destroyed by truncation.
9. As a researcher, I want long envelope summaries shortened for display rather than rejected as invalid, so that verbose-but-correct answers reach me.
10. As the ground-sensor agent, I want my toolset equally minimal (the legacy geocoder removed — my monitor tools geocode internally), so that both sub-agents follow the same MCP-first/minimal-surface principle.
11. As a researcher requesting a map, I want the place name geocoded once per request through one shared service, so that I do not pay duplicate network round-trips and rate-limit sleeps for caches that cannot see each other.
12. As the system owner, I want all Nominatim requests to carry a policy-compliant identifying User-Agent, so that the free geocoding service does not block the deployment's IP.
13. As the developer, I want dead code from the pre-MCP workflow (deterministic query-parser fallback) deleted rather than kept disabled, so that the codebase describes one workflow, not two.
14. As the developer, I want the scripted eval to hold its threshold before and after this change, so that the surface reduction demonstrably removes only what the agent never needed.

## Implementation Decisions

- **Model-facing earthdata toolset (the complete list):** dataset search, dataset description, dataset preview, area-of-interest definition, availability check, coverage check, the safe-retrieve composite, the await-retrieval composite, and the handle-based tools — single plot, multi plot, spatial statistic, temporal statistic, daily peak, ground validation, exceedance overlay, and region/period compare. Nothing else is visible to the model.
- **Demoted to internal-only** (still loaded from the MCP, still validated at boot, reachable by composites and endpoints, never model-visible): retrieval status (used by the await composite), citation and lineage (used by the provenance/citation endpoints), raw subset retrieval and size estimation (already internal). Raw timeseries retrieval and dataset summarization are removed from the required set entirely — nothing calls them; re-add to the internal list if and when a consumer appears.
- **Deleted outright:** the date-conversion tools and their prompt reference (the prompt already instructs the model to resolve dates itself, and the supervisor injects current UTC time into every satellite task); the bounding-box geocoder tool (from both agents' toolsets); the deterministic query-parser fallback and its plumbing (already a documented no-op).
- **Prompt updates:** the earthdata prompt drops the date-tool paragraph and the raw-timeseries warning (moot once the tool is gone — the safe-retrieve gate is the only retrieval path the model can see); the workflow steps and decision table are otherwise unchanged.
- **Retry guidance** is derived from the registered toolset at agent-build time (single source of truth) — it names exactly the reduced surface, never the raw/internal tools. The ground retry guidance is checked against the ground toolset the same way.
- **Truncation order:** envelope parsing happens on the untruncated final message; truncation applies afterwards, to the extracted summary only, and remains a logged event.
- **Geocoding stays as internal infrastructure** (plot tools and EPA tools resolve regions server-side); one module-level shared service instance backs every consumer — one cache, one rate limiter, one policy-compliant User-Agent identifying the project and a contact.

## Technical Implementation Guide

Paths current as of `talking-to-air-v2` (c75f4da + working tree); verify before relying on line numbers.

- **Curated/internal split.** `Backend/earthdata_mcp/client.py`: move `get_retrieval_status`, `cite_dataset`, `get_provenance` from `CURATED_TOOL_NAMES` to `INTERNAL_TOOL_NAMES`; delete `retrieve_timeseries` and `summarize_dataset` from `CURATED_TOOL_NAMES` (and from `REQUIRED_TOOL_NAMES` — nothing consumes them; grep first to confirm that is still true). `curated_model_tools` in `earthdata_mcp/toolset.py` needs no change — it reads the list. Update the mirror lists in `Backend/tests/fake_earthdata_mcp.py` (`CURATED_RAW_TOOL_NAMES`/`INTERNAL_RAW_TOOL_NAMES`) and the surface assertions in `test_earthdata_mcp_toolset.py` (its hidden-tools loop gains the demoted names) and `test_earthdata_mcp_client.py`.
- **Legacy deletions.** Remove `geocode_location` from `build_satellite_tools` (`Backend/tools/satellite_tools/factory.py`) and from `GROUND_TOOLS` (`Backend/tools/__init__.py`); delete `Backend/tools/satellite_tools/geocode_tools.py`. Delete `Backend/tools/satellite_tools/date_tools.py` and its re-exports in `tools/__init__.py`; drop the `convert_temporal_range_to_iso` paragraph from `Backend/config/earthdata_agent_prompt.py`. Delete `_try_direct_satellite_plot` and `_parse_simple_satellite_plot_task` from `Backend/agents/supervisor_agent.py` plus the `tools/satellite_tools/query_parser.py` import; delete `query_parser.py` itself unless something else imports it (`is_valid_location_candidate` is used by the geocode tool being deleted — check for other callers). Keep `utils/plotting.py`'s `GeocodingService`/`RegionResolver` — they are internal infrastructure for plot/EPA tools, not model tools.
- **Stale retry guidance.** In `Backend/agents/supervisor_agent.py`, `ask_earthdata_agent`'s `retry_task` string (~line 317) names `geocode_location`, `retrieve_timeseries`, `get_retrieval_status` — all leaving the surface. Derive the name list from the built toolset (e.g. `[t.name for t in tools]` captured at agent-build time, or a `sanctioned_tool_names()` helper exported by the factory) so guidance regenerates with the surface. Mirror-check `_GROUND_RETRY_TOOL_GUIDANCE` against `GROUND_TOOLS`.
- **Truncation order.** Satellite path: `_run_satellite` applies `truncate_text(text, 2000, ...)` (~line 288) *before* `_finalize_sub_agent_result` calls `parse_sub_agent_envelope`. Reorder: pass the untruncated join to finalization; truncate `envelope.summary` afterwards (keep the `response_truncated` log event from `Backend/utils/message_utils.py::truncate_text`). Ground path: `extract_last_text` truncates at 2000 with the same problem — parse first there too. Beware: `_run_satellite`'s comment about joining streamed deltas with `""` is load-bearing for JSON validity — don't change the join.
- **Geocoder unification.** Three independent `GeocodingService` instances exist: the `utils/plotting.py` singleton (`get_geocoding_service()`), `plot_tools.py`'s module-level `_resolver = RegionResolver()` (constructor builds its own), and `epa_aqs_tools.py`'s private `geocoding_service = GeocodingService()`. Make `RegionResolver.__init__` default to the shared singleton and point the EPA module at it. The offending User-Agent `'(Educational project)'` appears in both `geocode` and `ageocode` in `utils/plotting.py` — replace with one module constant identifying the project + contact per the Nominatim usage policy.

## Testing Decisions

- Good tests here assert external behavior only: the model-facing tool list matches the decision's complete list exactly (a snapshot-style assertion at the factory seam, so accidental surface growth fails a test); demoted tools remain invokable through the internal dict (provenance endpoint tests already cover cite/lineage); a retry prompt contains only registered tool names; an envelope longer than the display limit still parses and returns its artifacts; two resolutions of the same place name in one request hit the network once.
- Toolset surface is asserted at the tool-factory and MCP-toolset seams (prior art: existing satellite-tools factory tests and `test_earthdata_mcp_toolset.py`'s hidden-tools assertions — extend, don't replace).
- Truncation ordering is asserted at the supervisor helper seam (prior art: supervisor agent artifact tests and envelope tests).
- Geocoder unification is asserted at the geocoding-service seam with a stubbed HTTP layer (prior art: existing plotting/geocode tests; delete tests for the removed geocode tool).
- The scripted eval (fake-MCP seam, opt-in marker) runs before and after; the threshold must hold. Note the eval's expected tool traces use only surviving tools, so no eval task changes.

## Out of Scope

- Any model or provider change (T12). Any change to what tool results contain (T13). Routing changes (T14). Envelope schema or salvage behavior (T15). MCP-side tool changes — this PRD only changes which MCP tools this backend exposes to the model, not what the MCP serves.

## Further Notes

Every repair corresponds to a failure visible in production logs (failed tool calls, doubled retries, `response_truncated` firing before envelope parsing, duplicate geocode requests). The surface reduction converts three prompt rules ("never poll status", "never bulk-pull raw timeseries", date self-resolution) from discipline the model must remember into structure it cannot violate — the same containment principle `safe_retrieve` already applies to retrieval caps.
