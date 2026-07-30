// Pure selectors over a job row from useJobs (T27). Kept side-effect-free so
// JobsPanel's badge/action/upstream-line choices are unit-testable without a
// component-test rig, per T05's precedent.
//
// A job merges list_workspace's handle summary with get_retrieval_status's
// response (services/jobs_service.list_jobs): `status` is the durable state
// (pending/submitted/running/materializing/ready/failed/expired/cancelled)
// and `phase` is harmony-retrieval-mcp's qualitative label for it (PRD 007 —
// e.g. "queued at provider", "materializing"). Badge color always comes from
// `status`; the label prefers `phase` since it reads better mid-flight
// ("queued at provider" vs. "submitted") and coincides with `status` once a
// job is terminal.

// "error" is synthesized by services/jobs_service.py's fault-isolated
// status fan-out (a single handle's get_retrieval_status call failed) --
// terminal because the backend has nothing further to report and there's
// no live job underneath a cancel could reach.
// "not_found" is a dead handle list_workspace still lists after the job was
// evicted (get_retrieval_status no longer has a record of it) -- terminal
// for the same reason as "error": nothing further to report, no live job a
// cancel could reach. On a long-lived workspace these can outnumber real
// jobs many times over (live repro: 134 not_found vs 46 real jobs), so
// leaving this out of TERMINAL_STATUSES rendered a floor of dead rows as if
// they were actively running (progress bar, live Cancel button) and sorted
// them ahead of genuinely current tasks.
export const TERMINAL_STATUSES = new Set(['ready', 'failed', 'expired', 'cancelled', 'error', 'not_found'])

const STATUS_COLORS = {
  ready: 'var(--teal-text)',
  failed: 'var(--error)',
  expired: 'var(--warning)',
  cancelled: 'var(--text-muted)',
  error: 'var(--error)',
  // "paused" is derived by the backend (services/retrieval_composites.
  // annotate_paused) from a Harmony auto-pause the MCP would otherwise
  // report as "running" forever. Non-terminal on purpose: the row keeps its
  // Cancel button, and the paused guidance arrives as `note`.
  paused: 'var(--warning)',
}

// PRD 021's `upstream` outcomes on a cancel response. "unsupported" (OPeNDAP,
// or a job never submitted upstream) has nothing meaningful to report, so it
// intentionally has no line — showing one would imply a provider call that
// never happened.
const UPSTREAM_LINES = {
  requested: 'Stop requested at provider',
  already_terminal: 'Provider had already finished',
  error: 'Provider stop failed',
}

// Statuses that don't advance on their own, so live-polling them just re-runs
// the backend's per-job get_retrieval_status fan-out (services/jobs_service.
// list_jobs) forever. Terminal states never change; "paused" (a Harmony
// auto-pause the MCP would otherwise report as "running" indefinitely) needs
// user action to resume or cancel — polling never moves it; "not_found" is a
// dead handle list_workspace still lists after the job was evicted, and on a
// long-lived workspace these dominate (the live repro had 134 of them). The
// live poller (hooks/useJobs.js) must stop once every job is one of these, or
// a workspace with no running job keeps hundreds of MCP calls/min flowing
// while the panel sits idle.
export const NON_PROGRESSING_STATUSES = new Set([...TERMINAL_STATUSES, 'paused', 'not_found'])

// True while at least one job is still genuinely advancing toward a terminal
// state (running/pending/submitted/materializing/...). Drives whether the
// panel keeps its 15s live poll open.
export function hasProgressingJob(jobs) {
  return (jobs || []).some(job => !NON_PROGRESSING_STATUSES.has(job.status))
}

export function titleCase(value) {
  if (!value) return ''
  const spaced = value.replace(/_/g, ' ')
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

export function statusBadge(job) {
  if (!job) return { label: '', color: 'var(--text-secondary)' }
  // Once terminal, label from `status`, not `phase`: a cancel response carries
  // `status: "cancelled"` but no `phase`, so a merged just-cancelled job still
  // holds its stale running phase ("processing"). Rendering that under a
  // cancelled-grey chip reads as a contradiction, so terminal states ignore
  // the (possibly stale) phase entirely.
  const label = TERMINAL_STATUSES.has(job.status)
    ? job.status
    : (job.phase || job.status || '')
  return {
    label: titleCase(label),
    color: STATUS_COLORS[job.status] || 'var(--text-secondary)',
  }
}

// 'view-result' (ready with a result handle), 'cancel' (still in flight), or
// null (nothing to do — failed/expired/cancelled are read-only terminal
// states, and a ready job with no obs_handle has no result to open).
export function primaryAction(job) {
  if (!job) return null
  if (job.status === 'ready') return job.obs_handle ? 'view-result' : null
  if (TERMINAL_STATUSES.has(job.status)) return null
  return 'cancel'
}

export function upstreamLine(upstream) {
  return UPSTREAM_LINES[upstream] || null
}

export function formatVariables(variables) {
  if (!Array.isArray(variables) || !variables.length) return ''
  return variables.map(v => String(v).split('/').pop()).join(', ')
}

export function formatBbox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return ''
  return bbox.map(n => Number(n).toFixed(2)).join(', ')
}

export function formatOutputFormat(mediaType) {
  if (!mediaType) return ''
  const sub = mediaType.split('/').pop() || mediaType
  return sub.replace(/^x-/, '').toUpperCase()
}

function formatDate(iso) {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString()
}

export function formatTimeRange(timeRange) {
  if (!timeRange) return ''
  const [start, end] = timeRange.split('/')
  if (!start) return timeRange
  return end ? `${formatDate(start)} – ${formatDate(end)}` : formatDate(start)
}

// Mirrors services/jobs_service.py's list_jobs ordering: active jobs sort
// before terminal ones, newest-first within each group. Re-applied after
// every SSE merge (not just the initial fetch) so a job crossing into a
// terminal status moves out of the active group immediately.
export function sortJobs(jobs) {
  const byRecency = [...jobs].sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
  return byRecency.sort((a, b) => Number(TERMINAL_STATUSES.has(a.status)) - Number(TERMINAL_STATUSES.has(b.status)))
}
