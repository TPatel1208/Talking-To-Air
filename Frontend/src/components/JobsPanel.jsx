import { useEffect, useRef, useState } from 'react'
import {
  TERMINAL_STATUSES, statusBadge, primaryAction, upstreamLine,
  formatVariables, formatBbox, formatOutputFormat, formatTimeRange,
} from '../utils/jobCard'

const CONFIRM_TIMEOUT_MS = 4000

const buttonBase = {
  fontSize: '11px', padding: '4px 10px', borderRadius: '6px', cursor: 'pointer', fontWeight: 600,
}
const ghostButtonStyle = { ...buttonBase, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-secondary)' }
const primaryButtonStyle = { ...buttonBase, border: 'none', background: 'var(--teal)', color: 'white' }
const dangerButtonStyle = { ...buttonBase, border: 'none', background: 'var(--error)', color: 'white' }

function formatSubmittedAt(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function MetadataRow({ label, value }) {
  if (!value) return null
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', fontSize: '11.5px' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-secondary)', textAlign: 'right' }}>{value}</span>
    </div>
  )
}

// PRD 021's enriched get_retrieval_status fields — undefined for whichever
// don't apply to this job's provider/shape, so each row hides itself rather
// than showing a blank value.
function JobMetadata({ job }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: '5px',
      padding: '9px 10px', background: 'var(--bg-secondary)', borderRadius: '8px',
    }}>
      <MetadataRow label="Variables" value={formatVariables(job.variables)} />
      <MetadataRow label="Area" value={formatBbox(job.aoi_bbox)} />
      <MetadataRow label="Time range" value={formatTimeRange(job.time_range)} />
      <MetadataRow label="Provider" value={job.provider} />
      <MetadataRow label="Format" value={formatOutputFormat(job.output_format)} />
      <MetadataRow label="Granules" value={job.granule_count != null ? String(job.granule_count) : ''} />
    </div>
  )
}

function ProgressBar({ progress }) {
  const pct = typeof progress === 'number' ? Math.max(0, Math.min(100, progress)) : null
  return (
    <div style={{ height: '4px', borderRadius: '2px', background: 'var(--bg-secondary)', overflow: 'hidden' }}>
      <div style={{
        height: '100%',
        width: pct != null ? `${pct}%` : '35%',
        borderRadius: '2px',
        background: 'var(--teal)',
        opacity: pct != null ? 1 : 0.5,
      }} />
    </div>
  )
}

function JobRow({ job, onCancel, onViewResult }) {
  const [expanded, setExpanded] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const confirmTimeoutRef = useRef(null)

  useEffect(() => () => clearTimeout(confirmTimeoutRef.current), [])

  const isTerminal = TERMINAL_STATUSES.has(job.status)
  const badge = statusBadge(job)
  const action = primaryAction(job)
  const subtleLine = upstreamLine(job.upstream) || job.note || null
  const hasMetadata = Boolean(
    (Array.isArray(job.variables) && job.variables.length) ||
    (Array.isArray(job.aoi_bbox) && job.aoi_bbox.length) ||
    job.time_range || job.provider || job.output_format || job.granule_count != null
  )
  const title = job.short_name || job.dataset_handle || job.job_handle

  function handleCancelClick() {
    if (!confirming) {
      setConfirming(true)
      confirmTimeoutRef.current = setTimeout(() => setConfirming(false), CONFIRM_TIMEOUT_MS)
      return
    }
    clearTimeout(confirmTimeoutRef.current)
    setConfirming(false)
    setCancelling(true)
    Promise.resolve(onCancel(job.job_handle)).finally(() => setCancelling(false))
  }

  function handleCancelBlur() {
    clearTimeout(confirmTimeoutRef.current)
    setConfirming(false)
  }

  return (
    <div style={{
      padding:       '10px 12px',
      borderRadius:  '10px',
      background:    'var(--bg-card)',
      boxShadow:     'var(--shadow-sm)',
      display:       'flex',
      flexDirection: 'column',
      gap:           '6px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <span style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {title}
        </span>
        <span style={{ fontSize: '11px', fontWeight: 500, color: cancelling ? 'var(--text-muted)' : badge.color, textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>
          {cancelling ? 'Cancelling…' : badge.label}
        </span>
      </div>

      {!isTerminal && !cancelling && (
        <>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {badge.label}{typeof job.progress === 'number' ? ` — ${job.progress}%` : ''}
          </div>
          <ProgressBar progress={job.progress} />
        </>
      )}

      {job.status === 'failed' && job.message && (
        <div style={{ fontSize: '12px', color: 'var(--error)', fontWeight: 600 }}>{job.message}</div>
      )}

      {job.status === 'expired' && (
        <div style={{ fontSize: '11.5px', color: 'var(--text-hint)', fontStyle: 'italic' }}>
          Expired — re-run the retrieval to regenerate this result.
        </div>
      )}

      {subtleLine && (
        <div style={{ fontSize: '11px', color: 'var(--text-hint)' }}>{subtleLine}</div>
      )}

      {expanded && hasMetadata && <JobMetadata job={job} />}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <span style={{ fontSize: '11px', color: 'var(--text-hint)' }}>
          {formatSubmittedAt(job.submitted_at || job.created_at)}
        </span>
        <div style={{ display: 'flex', gap: '6px' }}>
          {hasMetadata && (
            <button onClick={() => setExpanded(e => !e)} style={ghostButtonStyle}>
              {expanded ? 'Less' : 'Details'}
            </button>
          )}
          {action === 'view-result' && (
            <button onClick={() => onViewResult(job)} style={primaryButtonStyle}>
              View result
            </button>
          )}
          {action === 'cancel' && !cancelling && (
            <button
              onClick={handleCancelClick}
              onBlur={handleCancelBlur}
              style={confirming ? dangerButtonStyle : ghostButtonStyle}
            >
              {confirming ? 'Confirm cancel?' : 'Cancel'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default function JobsPanel({ jobs, error, onCancel, onRefresh, onViewResult }) {
  return (
    <div style={{
      flex:          1,
      minHeight:     0,
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <span style={{ fontSize: '15px', fontWeight: 800, color: 'var(--text-primary)' }}>Jobs</span>
        <button
          onClick={onRefresh}
          title="Refresh"
          style={{
            background: 'transparent', border: 'none', color: 'var(--teal-text)',
            cursor: 'pointer', fontSize: '11.5px', fontWeight: 700,
          }}
        >
          Refresh
        </button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {error && (
          <div style={{ color: 'var(--error)', fontSize: '12px', padding: '4px 2px' }}>{error}</div>
        )}
        {jobs.length === 0 && !error && (
          <div style={{ padding: '12px 6px', color: 'var(--text-hint)', fontSize: '12px', fontStyle: 'italic' }}>
            No retrievals yet
          </div>
        )}
        {jobs.map(job => (
          <JobRow key={job.job_handle} job={job} onCancel={onCancel} onViewResult={onViewResult} />
        ))}
      </div>
    </div>
  )
}
