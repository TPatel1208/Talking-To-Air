export default function SessionSidebar({ sessions, threadId, onSwitch, onNew, onDelete, onLogout }) {
  const getSessionId = (session) => typeof session === 'string' ? session : session?.id
  const getSessionTitle = (session, index) => {
    if (session && typeof session === 'object' && session.title) return session.title
    return `Session ${index + 1}`
  }

  const handleDelete = (event, id) => {
    event.stopPropagation()
    if (!window.confirm('Delete this session?\n\nThis action cannot be undone.')) return
    onDelete(id)
  }

  return (
    <div style={{
      width:         '220px',
      flexShrink:    0,
      background:    'var(--bg-secondary)',
      borderRight:   '1px solid var(--border)',
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding:        '20px 14px 12px',
        borderBottom:   '1px solid var(--border)',
      }}>
        <span style={{
          fontSize:      '10px',
          fontWeight:    '500',
          color:         'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.09em',
        }}>
          Sessions
        </span>
      </div>

      {/* Session list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
        {sessions.length === 0 && (
          <div style={{
            padding:    '12px 6px',
            color:      'var(--text-hint)',
            fontSize:   '12px',
            fontStyle:  'italic',
          }}>
            No sessions yet
          </div>
        )}

        {sessions.map((session, i) => {
          const id = getSessionId(session)
          const isActive = id === threadId
          const title = getSessionTitle(session, i)
          return (
            <div
              key={id}
              onClick={() => onSwitch(id)}
              style={{
                display:        'flex',
                alignItems:     'center',
                justifyContent: 'space-between',
                gap:            '8px',
                padding:        '8px 10px',
                borderRadius:   '10px',
                cursor:         'pointer',
                background:     isActive ? 'var(--bg-card)' : 'transparent',
                boxShadow:      isActive ? 'var(--shadow-sm)' : 'none',
                transition:     'background 0.15s, box-shadow 0.15s',
              }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = 'rgba(0,0,0,0.04)' }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
            >
              {/* Left: icon + label */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                {/* Message circle icon (SVG) */}
                <svg
                  width="14" height="14" viewBox="0 0 24 24"
                  fill="none" stroke={isActive ? 'var(--teal)' : 'var(--text-muted)'}
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                  style={{ flexShrink: 0, transition: 'stroke 0.15s' }}
                >
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span style={{
                  fontSize:     '13px',
                  fontWeight:   isActive ? '500' : '400',
                  color:        isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                  overflow:     'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace:   'nowrap',
                  transition:   'color 0.15s',
                }}>
                  {title}
                </span>
              </div>

              {/* Delete button */}
              <button
                onClick={e => handleDelete(e, id)}
                style={{
                  background: 'transparent',
                  border:     'none',
                  color:      'var(--text-hint)',
                  cursor:     'pointer',
                  fontSize:   '13px',
                  padding:    '2px 4px',
                  borderRadius: '4px',
                  flexShrink: 0,
                  lineHeight: 1,
                  opacity:    0,
                  transition: 'opacity 0.15s, color 0.15s',
                }}
                className="session-delete-btn"
                title="Delete session"
              >
                ✕
              </button>
            </div>
          )
        })}
      </div>

      {/* Actions */}
      <div style={{ padding: '12px 10px 16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <button
          onClick={onNew}
          style={{
            display:        'flex',
            alignItems:     'center',
            gap:            '8px',
            width:          '100%',
            padding:        '10px 14px',
            borderRadius:   '10px',
            background:     'var(--accent)',
            color:          'var(--accent-text)',
            border:         'none',
            fontSize:       '13px',
            fontWeight:     '500',
            fontFamily:     'var(--font)',
            cursor:         'pointer',
            transition:     'opacity 0.15s',
          }}
          onMouseEnter={e => e.currentTarget.style.opacity = '0.88'}
          onMouseLeave={e => e.currentTarget.style.opacity = '1'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          New chat
        </button>
        <button
          onClick={onLogout}
          style={{
            display:        'flex',
            alignItems:     'center',
            gap:            '8px',
            width:          '100%',
            padding:        '9px 14px',
            borderRadius:   '10px',
            background:     'transparent',
            color:          'var(--text-secondary)',
            border:         '1px solid var(--border)',
            fontSize:       '13px',
            fontWeight:     '500',
            fontFamily:     'var(--font)',
            cursor:         'pointer',
            transition:     'background 0.15s, color 0.15s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'var(--bg-card)'
            e.currentTarget.style.color = 'var(--text-primary)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'transparent'
            e.currentTarget.style.color = 'var(--text-secondary)'
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
            <path d="M16 17l5-5-5-5"/>
            <path d="M21 12H9"/>
          </svg>
          Sign out
        </button>
      </div>

      <style>{`
        div:hover > div > .session-delete-btn,
        .session-delete-btn:focus { opacity: 1 !important; }
      `}</style>
    </div>
  )
}
