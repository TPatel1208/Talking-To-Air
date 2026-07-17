# PRD T37 — Error-surface hardening: no endpoint answers off-taxonomy

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T18 (error taxonomy + `MCPToolError` handler). Origin: reliability review 2026-07-16 (three-review series; the fixed batch is committed separately).

## Problem Statement

As a researcher, most of the app answers failures in one structured voice (T18) — but four paths still leak raw internals or fail in shapes the frontend can't render honestly:

1. `GET /chart/{id}/export.csv` and `export.nc` read `app.state.earthdata_mcp_tools` directly (`Backend/api.py`), bypassing the T17 readiness gate every other MCP-backed endpoint goes through (`_earthdata_tools`). With the MCP not ready, the CSV path fails *inside* the `StreamingResponse` generator — after the 200 headers are sent — so the researcher downloads a silently truncated file that looks like success. Any mid-stream exception in `iter_chart_csv_chunks_async` produces the same truncated-200.
2. `GET /sessions`, `GET /session/{id}/history`, and `DELETE /session/{id}` catch `Exception` and return `str(e)` in the 500 detail — internal error text (driver messages, paths) shown to the client, in a shape that matches nothing else.
3. `run_ground`/`run_satellite` (`Backend/services/subagent_dispatch.py`) turn any unexpected exception into `text = str(exc)` — an arbitrary (possibly empty: `KeyError('x')` → `"'x'"`) exception string becomes the sub-agent's "answer", which the supervisor may dress up as an explanation.
4. `MissingUserContextError` (`Backend/earthdata_mcp/workspace.py`) is raised bare inside a bound tool call — correct to fail loud, but it surfaces as an unclassified traceback string instead of a T18 category.

## Solution

Every failure a researcher can observe traces back to the T18 taxonomy. Export endpoints go through the readiness gate and materialize their first chunk before committing to a 200; catch-alls return a generic detail and log the real exception; sub-agent dispatch renders unexpected exceptions through `render_error_answer` (classified category when it's an `MCPToolError`, contract otherwise); `MissingUserContextError` classifies as contract.

## User Stories

1. As a researcher, I want a CSV/NetCDF export with the data layer down to fail as a clear 503, so that I never open a half-written file that looked like a successful download.
2. As a researcher, I want an export that breaks mid-stream to be detectable, so that a truncated file is never mistaken for the full dataset.
3. As a security-conscious operator, I want 500 bodies to carry a generic detail while the real exception goes to logs, so that internals never reach a client.
4. As the supervisor agent, I want a failed sub-agent turn to hand me the taxonomy's honest error answer, so that I relay a real message instead of decorating a bare exception string.
5. As the developer, I want a bound tool call with no user context to classify as a contract error, so that the isolation guard's refusal is as legible as every other failure.

## Implementation Decisions

- `export_chart_csv`/`export_chart_netcdf` resolve tools via `_earthdata_tools(request)` (shared 503). Before returning the `StreamingResponse`, await the generator's first chunk (peek-and-chain) so the common failures (not-ready, missing handle, evicted export) become clean 4xx/5xx; a failure after streaming begins appends a clearly marked `# EXPORT INCOMPLETE — <category>` trailer line so the file self-identifies as truncated.
- Session endpoints: `raise HTTPException(500, detail="Internal server error")` + `logger.exception` with request context. No shape change on success paths.
- `subagent_dispatch`: in `_invoke`'s broad `except Exception`, render via `render_error_answer(CATEGORY_CONTRACT, ...)` unless the exception is an `MCPToolError` (use its category/message). Metrics outcomes unchanged.
- `workspace._call`: wrap the `MissingUserContextError` raise as `MCPToolError(CATEGORY_CONTRACT, ...)` returned via `to_tool_json()` like every other classified outcome (still logged loudly — it indicates a programming error).

## Technical Implementation Guide

- `Backend/api.py`: export endpoints (~lines 505–630), session endpoints (~lines 774–809).
- `Backend/services/export_service.py::iter_chart_csv_chunks_async` — first-chunk peek helper.
- `Backend/services/subagent_dispatch.py::run_ground/_invoke`, `run_satellite/_invoke`.
- `Backend/earthdata_mcp/workspace.py::_bind_one/_call`.

## Testing Decisions

- `test_chat_endpoint.py`-style FastAPI tests: export.csv with a not-ready manager → 503 with the taxonomy body; export.csv whose generator raises after the first chunk → trailer line present.
- Session endpoint 500s carry the generic detail and never the exception text (assert on a raised sentinel message).
- `test_subagent_dispatch.py`: a tool raising `RuntimeError("secret path leak")` yields the taxonomy's contract answer, not the raw string.
- Prior art: `test_error_templates.py`, `test_satellite_tools_mcp_errors.py`.

## Out of Scope

- Overall turn timeouts (T38). Retry logic. Frontend rendering changes for the trailer line (a follow-up chip if wanted).
