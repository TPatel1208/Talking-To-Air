# PRD T05 — Jobs panel: durable retrieval visibility

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T02 (SSE job events), T04 (envelopes). Requires harmony-retrieval-mcp PRD 019's `list_workspace` tool.

## Problem Statement

As a researcher, a Harmony or AppEEARS retrieval takes minutes, and today the only evidence it exists is a chat message scrolling away. If I reload the page or the backend restarts, running jobs vanish from view even though the MCP keeps them running durably. The durability I'm paying for is invisible — which makes the new architecture feel like a regression next to the old synchronous fetch.

## Solution

A jobs panel: the first of the four workbench panes. A backend jobs endpoint composes the MCP's `list_workspace` and `get_retrieval_status` into the researcher's job list; the panel renders it, updates live from the `job_progress` SSE events T02 already emits, survives reloads and restarts, and offers cancel.

## User Stories

1. As a researcher, I want a persistent panel listing my retrievals with status, progress/phase, dataset, and submission time, so that long jobs are visible outside chat scroll.
2. As a researcher, I want the panel populated on page load from the backend (not from chat history), so that reloading the page or restarting the backend never loses my jobs.
3. As a researcher, I want live updates while jobs run, so that progress is continuous rather than poll-on-refresh.
4. As a researcher, I want failed jobs to show the MCP's legible stage/provider-prefixed error, so that I know what to fix (AOI, window, credentials) without asking the agent.
5. As a researcher, I want to cancel a running job from the panel, so that a mistaken 10-GB request doesn't have to run to completion.
6. As a researcher, I want a completed job to show its result handle and a jump-off to the artifact/plot that used it (when one exists), so that jobs connect to outputs.
7. As a lab member sharing the box, I want the panel scoped to my own workspace, so that I see my jobs only.

## Implementation Decisions

- Backend: an authenticated jobs endpoint that calls `list_workspace` (job handles) and fans out `get_retrieval_status` per handle; response is the panel's list model. Cancel endpoint proxies the MCP's cancel tool (hidden from the agent, exposed to the UI).
- Frontend: a JobsPanel component alongside the chat (first pane of the workbench layout decided 2026-07-04: Jobs+Artifacts → Discovery → Provenance); subscribes to the existing SSE channel's `job_progress` events; reconciles push events with the pulled list by job handle.
- The SSE event vocabulary from T02 is frozen here as a contract (event name, fields); the panel and awaiter both conform to it.
- Workspace scoping rides the existing auth: user id → `workspace_id` mapping from T02's wrapper is reused by the endpoint.

## Testing Decisions

- Backend endpoint tested at the existing FastAPI endpoint seam against the fake MCP: list composition, status fan-out, error passthrough, cancel proxying, per-user scoping.
- Frontend: no automated tests (decision: demo-verified until a component-test rig earns its cost); the exit-criteria script is — start a retrieval, reload the page mid-run, watch it complete in the panel, cancel a second one.

## Out of Scope

- Artifact gallery generalization (T06) — the jump-off link renders only if the artifact already exists.
- Discovery and provenance panes (T09/T10).

## Further Notes

This panel is the payoff of the durable-jobs architecture — the risk register calls it "not polish." It ships immediately after Phase 1 for exactly that reason.
