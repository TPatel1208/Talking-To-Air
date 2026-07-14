import test from 'node:test'
import assert from 'node:assert/strict'

import {
  statusBadge, primaryAction, upstreamLine,
  formatVariables, formatBbox, formatOutputFormat, formatTimeRange,
  sortJobs, TERMINAL_STATUSES,
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
