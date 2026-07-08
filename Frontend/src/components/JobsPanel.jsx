const TERMINAL_STATUSES = new Set(['materialized', 'failed', 'cancelled'])

const STATUS_COLORS = {
  materialized: 'var(--teal-text)',
  failed: 'var(--error)',
  cancelled: 'var(--text-muted)',
}

function formatSubmittedAt(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function JobRow({ job, onCancel }) {
  const isTerminal = TERMINAL_STATUSES.has(job.status)
  const statusColor = STATUS_COLORS[job.status] || 'var(--text-secondary)'

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
          {job.dataset || job.job_handle}
        </span>
        <span style={{ fontSize: '11px', fontWeight: 500, color: statusColor, textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>
          {job.status}
        </span>
      </div>

      {!isTerminal && (
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          {job.phase || 'in progress'}{typeof job.progress === 'number' ? ` — ${job.progress}%` : ''}
        </div>
      )}

      {job.status === 'failed' && job.message && (
        <div style={{ fontSize: '12px', color: 'var(--error)' }}>{job.message}</div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <span style={{ fontSize: '11px', color: 'var(--text-hint)' }}>
          {formatSubmittedAt(job.submitted_at)}
        </span>
        {!isTerminal && (
          <button
            onClick={() => onCancel(job.job_handle)}
            style={{
              fontSize:     '11px',
              padding:      '4px 10px',
              borderRadius: '6px',
              border:       '1px solid var(--border)',
              background:   'transparent',
              color:        'var(--text-secondary)',
              cursor:       'pointer',
            }}
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  )
}

export default function JobsPanel({ jobs, error, onCancel, onRefresh }) {
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
          <JobRow key={job.job_handle} job={job} onCancel={onCancel} />
        ))}
      </div>
    </div>
  )
}
