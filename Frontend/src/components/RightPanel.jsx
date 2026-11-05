import DiscoveryPane from './DiscoveryPane'
import JobsPanel from './JobsPanel'
import DiscoverJobsControl from './DiscoverJobsControl'
import { PANEL_HEADER_HEIGHT } from '../utils/panelLayout'

const TERMINAL_STATUSES = new Set(['ready', 'failed', 'expired', 'cancelled'])

export default function RightPanel({
  tab = null,
  onSelectTab,
  discovery,
  jobs,
  jobsError,
  onCancelJob,
  onRefreshJobs,
  onRetrieve,
  onViewResult,
}) {
  const hasRunningJobs = jobs.some(job => !TERMINAL_STATUSES.has(job.status))
  const isActive = tab !== null

  return (
    <div style={{
      width:         '308px',
      flexShrink:    0,
      borderLeft:    '1px solid var(--border)',
      background:    isActive ? 'var(--bg-card)' : 'var(--bg-primary)',
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
    }}>
      <div style={{
        height: `${PANEL_HEADER_HEIGHT}px`,
        boxSizing: 'border-box',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
        padding: '0 18px',
        flexShrink: 0,
      }}>
        <DiscoverJobsControl
          tab={tab}
          onSelectTab={onSelectTab}
          hasRunningJobs={hasRunningJobs}
        />
      </div>

      {tab !== null && (
        <div style={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          padding: '16px 18px 18px',
        }}>
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
              inventories={discovery.inventories}
              onSearch={discovery.search}
              onPreview={discovery.preview}
              onCoverage={discovery.checkCoverage}
              onGranules={discovery.inspectGranules}
              onInspect={discovery.inspectVariables}
              onRetrieve={onRetrieve}
            />
          )}

          {tab === 'jobs' && (
            <JobsPanel
              jobs={jobs}
              error={jobsError}
              onCancel={onCancelJob}
              onRefresh={onRefreshJobs}
              onViewResult={onViewResult}
            />
          )}
        </div>
      )}
    </div>
  )
}
