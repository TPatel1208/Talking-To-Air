export default function DiscoverJobsControl({ tab, onSelectTab, hasRunningJobs, style: customStyle = {} }) {
  const isMinimized = tab === null

  return (
    <div style={{
      width: '100%',
      display: 'flex',
      alignItems: 'center',
      gap: '4px',
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border)',
      borderRadius: '9px',
      padding: '3px',
      boxSizing: 'border-box',
      ...customStyle,
    }}>
      {isMinimized ? (
        <>
          <button
            type="button"
            onClick={() => onSelectTab('discover')}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'transparent',
              border: 'none',
              fontSize: '12.5px',
              fontWeight: 700,
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '6px 8px',
              borderRadius: '6px',
            }}
          >
            Discover
          </button>
          <span style={{ color: 'var(--text-hint)', fontSize: '12px', userSelect: 'none', flexShrink: 0, margin: '0 1px' }}>|</span>
          <button
            type="button"
            onClick={() => onSelectTab('jobs')}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'transparent',
              border: 'none',
              fontSize: '12.5px',
              fontWeight: 700,
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '6px 8px',
              borderRadius: '6px',
              gap: '5px',
            }}
          >
            Jobs
            {hasRunningJobs && (
              <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--warning)' }} />
            )}
          </button>
          <button
            type="button"
            onClick={() => onSelectTab('discover')}
            title="Expand jobs and discover"
            aria-label="Expand jobs and discover"
            style={{
              background: 'transparent',
              border: 'none',
              padding: '4px 6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-muted)',
              flexShrink: 0,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
        </>
      ) : (
        <>
          <button
            type="button"
            onClick={() => onSelectTab(tab === 'discover' ? null : 'discover')}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: 'none',
              fontSize: '12.5px',
              fontWeight: 700,
              padding: '6px 12px',
              borderRadius: '7px',
              cursor: 'pointer',
              background: tab === 'discover' ? 'var(--bg-card)' : 'transparent',
              color: tab === 'discover' ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: tab === 'discover' ? 'var(--shadow-sm)' : 'none',
            }}
          >
            Discover
          </button>
          <button
            type="button"
            onClick={() => onSelectTab(tab === 'jobs' ? null : 'jobs')}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: 'none',
              fontSize: '12.5px',
              fontWeight: 700,
              padding: '6px 12px',
              borderRadius: '7px',
              cursor: 'pointer',
              background: tab === 'jobs' ? 'var(--bg-card)' : 'transparent',
              color: tab === 'jobs' ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: tab === 'jobs' ? 'var(--shadow-sm)' : 'none',
              gap: '5px',
            }}
          >
            Jobs
            {hasRunningJobs && (
              <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--warning)' }} />
            )}
          </button>
          <button
            type="button"
            onClick={() => onSelectTab(null)}
            title="Collapse jobs and discover"
            aria-label="Collapse jobs and discover"
            style={{
              background: 'transparent',
              border: 'none',
              padding: '4px 6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-muted)',
              flexShrink: 0,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="18 15 12 9 6 15" />
            </svg>
          </button>
        </>
      )}
    </div>
  )
}
