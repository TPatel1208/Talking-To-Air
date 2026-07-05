# PRD T01 — Consolidate on talking-to-air-v2 + joint compose topology

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** harmony-retrieval-mcp PRD 018 (the MCP stack must expose HTTP + the shared volume to attach to).

## Problem Statement

As the developer, I have two divergent working copies of this project — an older `main` (simple agents, regex-scraped plot paths, hand-rolled Harmony loader) and a richer `talking-to-air-v2` branch (auth, artifact model, exports, intent router, real test suite) — and no deployment topology connecting either to the earthdata-retrieval MCP stack. Every subsequent PRD needs one canonical codebase and a working cross-stack connection.

## Solution

Make `talking-to-air-v2` the single canonical line: fold in anything still wanted from `main` (post-May prompt wording, supervisor fixes), make it the default branch, archive the stale copy. Then join the two compose stacks: an external Docker network both stacks attach to, and the MCP's materialization volume mounted read-only into the TTA backend at the same absolute path, so `file://` URIs resolve identically in both containers.

## User Stories

1. As the developer, I want one canonical branch containing the best of both working copies, so that every later PRD lands in one place.
2. As the TTA backend, I want the earthdata MCP reachable at a stable service URL on a shared external network, so that the MCP client (T02) is pure configuration.
3. As the TTA backend, I want the MCP's data volume mounted read-only at the identical absolute path, so that exported `file://` URIs open without translation.
4. As an operator, I want TTA's obsolete zarr-cache volume and its initialization SQL gone, so that there is exactly one owner of data caching (the MCP).
5. As an operator, I want the MCP URL and bearer token in TTA's environment configuration with sane example values, so that bringing up the joined stacks is documented and reproducible.
6. As the developer, I want the stale working copy archived (not deleted), so that nothing from `main` is lost if the fold-in missed something.

## Implementation Decisions

- Branch consolidation: merge `main`'s post-May commits into `talking-to-air-v2`; resolve in favor of v2's architecture everywhere they conflict; make v2 the repository default branch; tag the old tip before archiving.
- Compose: both stacks join a pre-created external network; the TTA backend adds a read-only mount of the MCP's named data volume; TTA's own cache volume, its cache-index initialization SQL, and their references are deleted.
- Config: MCP URL + bearer token enter TTA's settings module as first-class env-driven settings, alongside the existing settings pattern.
- No application-code changes beyond configuration — the client layer is T02's job. This PRD is repo surgery + topology only.

## Testing Decisions

- The existing v2 pytest suite must pass after the merge — that is the regression gate for the fold-in.
- Topology is verified by a smoke check documented in the README: bring up both stacks, exec into the TTA backend, list the shared mount, and curl the MCP endpoint with the token.
- No new automated tests — nothing behavioral changed TTA-side yet.

## Out of Scope

- Any MCP client code, tool changes, or prompt changes (T02+).
- Deleting the old Harmony loader code paths (T03 — they still serve the app until the swap lands).

## Further Notes

Decided in the 2026-07-04 grilling session: v2 is the base (it already has auth, artifacts, exports — the three primitives the research platform needs).
