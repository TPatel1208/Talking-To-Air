# PRD T27 — Jobs panel: metadata cards + cancel UX

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T26 (the panel lists real, correctly-scoped jobs and knows the true status vocabulary) and harmony-retrieval-mcp PRD 021 (enriched `get_retrieval_status` + provider-side cancel with an `upstream` flag). Phase 3 of 3 (T26 → PRD 021 → T27).

## Problem Statement

As a researcher, once the panel actually lists my correctly-scoped jobs (T26), each one is still a bare handle with a spinner. I can't tell what a job pertains to, a finished job offers no way to see its result, and cancelling is a single unguarded click that (before PRD 021) never reached NASA. The durable-jobs architecture is visible but illegible: I can see *that* a job exists, not *what it is*, *what it produced*, or *whether stopping it actually stopped it*.

## Solution

Turn each job row into a metadata card that reads out what the job pertains to (from PRD 021's enriched status), links a finished job to its result, and makes cancel a guarded, honest action that reflects PRD 021's local-first best-effort upstream cancel. Fit it into the 308px workbench pane with a compact card plus expandable detail.

## User Stories

1. As a researcher, I want each job card to show its dataset, and on expand its variables, area, time window, provider, output format, and granule count, so that I know what the job is without asking the agent.
2. As a researcher with a finished (`ready`) job, I want a "View result" action that opens its result in the workbench, so that jobs connect to outputs (T05 story 6, finally cashed).
3. As a researcher, I want an `expired` job to read as expired with a hint to re-run, and a `failed` job to show its provider-prefixed error prominently, so that I know what to do next.
4. As a researcher, I want cancel to take a deliberate second click and then reflect what happened — cancelling locally, and whether the provider was actually asked to stop — so that I don't misclick away a job and I'm not misled about NASA compute.
5. As a researcher scanning a narrow pane, I want active jobs at the top and finished ones below, compact by default, so that in-flight work stays visible without the list becoming a wall.

## Implementation Decisions

- **Compact card + expandable details (`JobsPanel.jsx`).** Always visible: dataset `short_name` (title), phase-driven status badge, progress bar while non-terminal, submitted time, and the single primary action. A disclosure toggle reveals the enriched metadata from PRD 021: `variables`, `aoi_bbox`, `time_range`, `provider`, `output_format`, `granule_count`. Ordering is set by T26's `list_jobs` (active first, then newest-first); the card only renders it.
- **Per-terminal-state treatment.** `ready` → teal "Ready" badge + **"View result"** button that calls `sendMessage` to have the agent open the `obs_handle` (mirrors `App.jsx::handleRetrieve` — no new backend surface, result renders where every artifact does). `expired` → amber "Expired" badge + a subtle "re-run to regenerate" hint (no result link — it's gone). `failed` → prominent `error`/`message`. `cancelled` → muted, terminal. Badge label from `phase`, color/terminality from `status` (vocabulary already corrected in T26).
- **Cancel UX.** Inline two-step confirm — the Cancel button flips to "Confirm cancel?" in place (no modal), reverting on blur/timeout — then an optimistic "cancelling…" state until the response settles to `cancelled`. Read PRD 021's `upstream` flag from the cancel response and show a **subtle** secondary line only when meaningful ("stop requested at provider" / "provider had already finished" / "provider stop failed"). Honest that "cancelled here" and "stopped at NASA" are distinct.
- **`useJobs` (`hooks/useJobs.js`).** `applyJobProgress` already reconciles by `job_handle`; thread the new metadata and the cancel response's `upstream` field through without changing the reconciliation model. Static metadata arrives on the pulled list; live status keeps arriving via the existing `job_progress` SSE channel.

## Testing Decisions

- Frontend follows T05's precedent (demo-verified, no component-test rig) unless the vitest rig has since earned its cost — if a card renderer is pure enough to unit-test (badge/label/action selection by status; upstream-flag line selection), test that seam.
- Extend the faithful fake (T26) so `get_retrieval_status` carries the enriched fields and `cancel_retrieval` returns an `upstream` flag, keeping any endpoint/hook tests contract-true.
- Exit-criteria demo: submit a retrieval as a logged-in user → it appears under `user-<id>` with dataset + expandable metadata → complete it and use "View result" to render the output → cancel a second one, confirm the two-step guard, the optimistic transition, and the upstream-outcome line.

## Out of Scope

- Artifact gallery generalization (still T06) — "View result" reuses the agent's `open_handle` path, it does not build a new artifact surface.
- Any MCP-side change — the enriched status and provider cancel are PRD 021; this phase only consumes them.
- Pause/resume controls.
- Bulk actions (cancel-all, clear-finished) — a later polish item if asked.

## Further Notes

This is the payoff phase: T26 made the panel true and scoped, PRD 021 made it informative and its cancel real, and T27 makes it legible. The one load-bearing UX decision is honesty about the two services — the upstream-outcome line exists precisely because "cancelled in my workspace" and "stopped on NASA's servers" are different facts, and the researcher paying for compute deserves to know which one happened.
