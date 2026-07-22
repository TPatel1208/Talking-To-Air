import test from 'node:test'
import assert from 'node:assert/strict'

import {
  statusBadge, primaryAction, upstreamLine,
  formatVariables, formatBbox, formatOutputFormat, formatTimeRange,
  sortJobs, TERMINAL_STATUSES, hasProgressingJob,
} from '../src/utils/jobCard.js'

test('statusBadge prefers phase for the label but status for the color', () => {
  const job = { status: 'running', phase: 'queued at provider' }
  const badge = statusBadge(job)
  assert.equal(badge.label, 'Queued at provider')
  assert.equal(badge.color, 'var(--text-secondary)')
})

test('statusBadge falls back to status when phase is absent', () => {
  assert.equal(statusBadge({ status: 'ready' }).label, 'Ready')
})

test('statusBadge ignores a stale phase once the job is terminal', () => {
  // A cancel response carries status but no phase, so a just-cancelled job
  // still holds its stale running phase — the badge must read the status.
  const badge = statusBadge({ status: 'cancelled', phase: 'processing' })
  assert.equal(badge.label, 'Cancelled')
  assert.equal(badge.color, 'var(--text-muted)')
})

test('statusBadge colors terminal states from status', () => {
  assert.equal(statusBadge({ status: 'ready' }).color, 'var(--teal-text)')
  assert.equal(statusBadge({ status: 'failed' }).color, 'var(--error)')
  assert.equal(statusBadge({ status: 'expired' }).color, 'var(--warning)')
  assert.equal(statusBadge({ status: 'cancelled' }).color, 'var(--text-muted)')
})

test('primaryAction is view-result only when ready with a result handle', () => {
  assert.equal(primaryAction({ status: 'ready', obs_handle: 'obs_1' }), 'view-result')
})

test('primaryAction is null for a ready job with no result handle', () => {
  // Nothing to open — offering "View result" would send the agent "(undefined)".
  assert.equal(primaryAction({ status: 'ready' }), null)
})

test('primaryAction is cancel for any non-terminal status', () => {
  for (const status of ['pending', 'submitted', 'running', 'materializing']) {
    assert.equal(primaryAction({ status }), 'cancel')
  }
})

test('primaryAction is null for read-only terminal states', () => {
  for (const status of ['failed', 'expired', 'cancelled']) {
    assert.equal(primaryAction({ status }), null)
  }
})

test('a job whose status read itself failed (synthesized "error") is terminal, not a live running job', () => {
  // services/jobs_service.py's fault-isolated status fan-out synthesizes
  // status: "error" when a single handle's get_retrieval_status call fails.
  // Without "error" in TERMINAL_STATUSES that row rendered as running, with
  // a live Cancel button, for a job the backend can no longer even ask
  // about.
  assert.equal(TERMINAL_STATUSES.has('error'), true)
  assert.equal(primaryAction({ status: 'error' }), null)
  assert.equal(statusBadge({ status: 'error' }).color, 'var(--error)')
})

test('a not_found (dead/evicted) job is terminal, not a live running job', () => {
  // list_workspace still lists handles for jobs the MCP has evicted
  // (get_retrieval_status returns status: "not_found"). On a long-lived
  // workspace these dominate (live repro: 134 not_found vs 46 real jobs).
  // Without "not_found" in TERMINAL_STATUSES every one of those dead rows
  // rendered as running (progress bar + live Cancel button) and sorted into
  // the active group ahead of/mixed with genuinely current tasks.
  assert.equal(TERMINAL_STATUSES.has('not_found'), true)
  assert.equal(primaryAction({ status: 'not_found' }), null)
})

test('statusBadge reads a multi-word status label with spaces, not a literal underscore', () => {
  assert.equal(statusBadge({ status: 'not_found' }).label, 'Not found')
})

test('a provider-paused job renders as a warning badge with Cancel still available', () => {
  // The backend derives status "paused" (annotate_paused) from a Harmony
  // auto-pause the MCP reports as "running" forever. It must not be
  // terminal (the researcher's only way out is Cancel) and must not render
  // as an eternally processing row.
  const job = { status: 'paused', phase: 'paused at provider' }
  assert.equal(TERMINAL_STATUSES.has('paused'), false)
  assert.equal(primaryAction(job), 'cancel')
  const badge = statusBadge(job)
  assert.equal(badge.label, 'Paused at provider')
  assert.equal(badge.color, 'var(--warning)')
})

test('upstreamLine maps PRD 021 outcomes to the honest subtle-line copy', () => {
  assert.equal(upstreamLine('requested'), 'Stop requested at provider')
  assert.equal(upstreamLine('already_terminal'), 'Provider had already finished')
  assert.equal(upstreamLine('error'), 'Provider stop failed')
})

test('upstreamLine is null for unsupported and unknown outcomes', () => {
  assert.equal(upstreamLine('unsupported'), null)
  assert.equal(upstreamLine(undefined), null)
})

test('formatVariables strips group prefixes and joins', () => {
  assert.equal(formatVariables(['geolocation/latitude', 'product/no2']), 'latitude, no2')
  assert.equal(formatVariables([]), '')
  assert.equal(formatVariables(undefined), '')
})

test('formatBbox rounds to 2 decimals', () => {
  assert.equal(formatBbox([-74.5123, 40.1, -73.9, 40.99]), '-74.51, 40.10, -73.90, 40.99')
  assert.equal(formatBbox(undefined), '')
  assert.equal(formatBbox([1, 2, 3]), '')
})

test('formatOutputFormat strips the mime prefix and x- marker', () => {
  assert.equal(formatOutputFormat('application/netcdf4'), 'NETCDF4')
  assert.equal(formatOutputFormat('application/x-parquet'), 'PARQUET')
  assert.equal(formatOutputFormat(''), '')
})

test('formatTimeRange renders a start-end pair as localized dates', () => {
  const result = formatTimeRange('2026-07-01T00:00:00/2026-07-02T00:00:00')
  assert.match(result, /2026/)
  assert.match(result, /–/)
})

test('formatTimeRange passes through an unparseable range unchanged', () => {
  assert.equal(formatTimeRange(''), '')
  assert.equal(formatTimeRange('not-a-range'), 'not-a-range')
})

test('sortJobs puts active jobs before terminal ones, newest first within each group', () => {
  const jobs = [
    { job_handle: 'old-terminal', status: 'ready', created_at: '2026-01-01T00:00:00Z' },
    { job_handle: 'new-active', status: 'running', created_at: '2026-01-03T00:00:00Z' },
    { job_handle: 'new-terminal', status: 'failed', created_at: '2026-01-02T00:00:00Z' },
    { job_handle: 'old-active', status: 'pending', created_at: '2026-01-01T12:00:00Z' },
  ]

  const sorted = sortJobs(jobs).map(job => job.job_handle)

  assert.deepEqual(sorted, ['new-active', 'old-active', 'new-terminal', 'old-terminal'])
})

test('sortJobs pushes not_found (dead) jobs into the terminal group, never ahead of real active jobs', () => {
  const jobs = [
    { job_handle: 'dead-1', status: 'not_found', created_at: '2026-01-03T00:00:00Z' },
    { job_handle: 'dead-2', status: 'not_found', created_at: '2026-01-02T12:00:00Z' },
    { job_handle: 'real-active', status: 'running', created_at: '2026-01-01T00:00:00Z' },
    { job_handle: 'real-terminal', status: 'ready', created_at: '2026-01-02T00:00:00Z' },
  ]

  const sorted = sortJobs(jobs).map(job => job.job_handle)

  assert.deepEqual(sorted, ['real-active', 'dead-1', 'dead-2', 'real-terminal'])
})

test('sortJobs moves a job that just went terminal out of the active group without a manual refresh', () => {
  // Mirrors the useJobs SSE merge path: an in-flight job's status flips to
  // terminal via applyJobProgress, and the list must reflect the new
  // active/terminal grouping immediately.
  const jobs = [
    { job_handle: 'a', status: 'running', created_at: '2026-01-02T00:00:00Z' },
    { job_handle: 'b', status: 'running', created_at: '2026-01-01T00:00:00Z' },
  ]
  const justFinished = jobs.map(job => (job.job_handle === 'b' ? { ...job, status: 'ready' } : job))

  const sorted = sortJobs(justFinished).map(job => job.job_handle)

  assert.deepEqual(sorted, ['a', 'b'])
})

test('hasProgressingJob keeps the poll alive only for genuinely-advancing jobs', () => {
  // The live poller (hooks/useJobs.js) runs while this is true. A running or
  // pending job legitimately keeps it alive — it will reach a terminal state
  // on its own, at which point polling stops.
  assert.equal(hasProgressingJob([{ status: 'running' }]), true)
  assert.equal(hasProgressingJob([{ status: 'pending' }, { status: 'ready' }]), true)
})

test('hasProgressingJob does not keep polling for paused or terminal jobs', () => {
  // The runaway: a Harmony-auto-paused job reports non-terminal ("paused")
  // forever, so treating it as active kept the 15s poll — and its per-job
  // get_retrieval_status fan-out — running indefinitely. Paused needs user
  // action (resume/cancel), never advances on its own, so it must NOT keep
  // the poller alive. Terminal jobs never keep it alive either.
  assert.equal(hasProgressingJob([{ status: 'paused' }]), false)
  assert.equal(hasProgressingJob([{ status: 'ready' }, { status: 'failed' }]), false)
  assert.equal(hasProgressingJob([{ status: 'paused' }, { status: 'cancelled' }]), false)
  assert.equal(hasProgressingJob([]), false)
})

test('hasProgressingJob does not poll a workspace of only dead (not_found) handles', () => {
  // The live repro: 134 not_found handles (evicted jobs list_workspace still
  // lists) and no running job. Treated as progressing, they kept the poller —
  // and its ~180-call per-poll fan-out — alive forever on an idle panel.
  assert.equal(hasProgressingJob([{ status: 'not_found' }, { status: 'ready' }]), false)
  assert.equal(hasProgressingJob([{ status: 'not_found' }, { status: 'running' }]), true)
})
