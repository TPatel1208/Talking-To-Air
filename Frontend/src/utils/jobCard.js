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

export const TERMINAL_STATUSES = new Set(['ready', 'failed', 'expired', 'cancelled'])

const STATUS_COLORS = {
  ready: 'var(--teal-text)',
  failed: 'var(--error)',
  expired: 'var(--warning)',
  cancelled: 'var(--text-muted)',
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

export function titleCase(value) {
  if (!value) return ''
  return value.charAt(0).toUpperCase() + value.slice(1)
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
