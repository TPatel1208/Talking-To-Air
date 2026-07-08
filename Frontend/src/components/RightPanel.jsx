import { useState } from 'react'
import DiscoveryPane from './DiscoveryPane'
import JobsPanel from './JobsPanel'

const TERMINAL_STATUSES = new Set(['materialized', 'failed', 'cancelled'])

export default function RightPanel({ discovery, jobs, jobsError, onCancelJob, onRefreshJobs, onRetrieve }) {
  const [tab, setTab] = useState('discover')
  const hasRunningJobs = jobs.some(job => !TERMINAL_STATUSES.has(job.status))

  return (
    <div style={{
      width:         '308px',
      flexShrink:    0,
      borderLeft:    '1px solid var(--border)',
      background:    'var(--bg-card)',
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
      padding:       '18px',
    }}>
      <div style={{ display: 'flex', gap: '4px', background: 'var(--bg-secondary)', borderRadius: '9px', padding: '3px', marginBottom: '18px', flexShrink: 0 }}>
        <div
          onClick={() => setTab('discover')}
          style={{
            flex: 1, textAlign: 'center', fontSize: '12.5px', fontWeight: 700, padding: '8px',
            borderRadius: '7px', cursor: 'pointer',
            background: tab === 'discover' ? 'var(--bg-card)' : 'transparent',
            color: tab === 'discover' ? 'var(--text-primary)' : 'var(--text-muted)',
            boxShadow: tab === 'discover' ? 'var(--shadow-sm)' : 'none',
          }}
        >
          Discover
        </div>
        <div
          onClick={() => setTab('jobs')}
          style={{
            flex: 1, textAlign: 'center', fontSize: '12.5px', fontWeight: 700, padding: '8px',
            borderRadius: '7px', cursor: 'pointer',
            background: tab === 'jobs' ? 'var(--bg-card)' : 'transparent',
            color: tab === 'jobs' ? 'var(--text-primary)' : 'var(--text-muted)',
            boxShadow: tab === 'jobs' ? 'var(--shadow-sm)' : 'none',
          }}
        >
          Jobs
          {hasRunningJobs && (
            <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--warning)', marginLeft: '5px' }} />
          )}
        </div>
      </div>

      {tab === 'discover' && (
        <DiscoveryPane
          query={discovery.query}
          setQuery={discovery.setQuery}
          location={discovery.location}
          setLocation={discovery.setLocation}
          timeRange={discovery.timeRange}
          setTimeRange={discovery.setTimeRange}
          results={discovery.results}
          loading={discovery.loading}
          error={discovery.error}
          previews={discovery.previews}
          coverages={discovery.coverages}
          granules={discovery.granules}
          onSearch={discovery.search}
          onPreview={discovery.preview}
          onCoverage={discovery.checkCoverage}
          onGranules={discovery.inspectGranules}
          onRetrieve={onRetrieve}
        />
      )}

      {tab === 'jobs' && (
        <JobsPanel jobs={jobs} error={jobsError} onCancel={onCancelJob} onRefresh={onRefreshJobs} />
      )}
    </div>
  )
}
