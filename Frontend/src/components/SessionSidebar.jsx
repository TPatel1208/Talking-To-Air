export default function SessionSidebar({ sessions, threadId, onSwitch, onNew, onDelete }) {
  return (
    <div style={{
      width:         '200px',
      flexShrink:    0,
      background:    'var(--bg-secondary)',
      borderRight:   '1px solid var(--border)',
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
    }}>
      <div style={{
        padding:      '12px',
        borderBottom: '1px solid var(--border)',
        display:      'flex',
        justifyContent: 'space-between',
        alignItems:   'center',
      }}>
        <span style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Sessions
        </span>
        <button onClick={onNew} style={{
          background:   'var(--accent)',
          border:       'none',
          borderRadius: '4px',
          color:        '#fff',
          fontSize:     '16px',
          cursor:       'pointer',
          width:        '22px',
          height:       '22px',
          display:      'flex',
          alignItems:   'center',
          justifyContent: 'center',
          lineHeight:   1,
        }}>+</button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {sessions.length === 0 && (
          <div style={{ padding: '12px', color: 'var(--text-secondary)', fontSize: '12px' }}>
            No sessions yet
          </div>
        )}
        {sessions.map((id, i) => (
          <div key={id} onClick={() => onSwitch(id)} style={{
            display:        'flex',
            alignItems:     'center',
            justifyContent: 'space-between',
            padding:        '8px 12px',
            cursor:         'pointer',
            background:     id === threadId ? 'var(--bg-tertiary)' : 'transparent',
            borderLeft:     id === threadId ? '2px solid var(--accent)' : '2px solid transparent',
          }}>
            <span style={{
              fontSize:     '12px',
              color:        id === threadId ? 'var(--text-primary)' : 'var(--text-secondary)',
              overflow:     'hidden',
              textOverflow: 'ellipsis',
              whiteSpace:   'nowrap',
            }}>
              Session {i + 1}
            </span>
            <button
              onClick={e => { e.stopPropagation(); onDelete(id); }}
              style={{
                background: 'transparent',
                border:     'none',
                color:      'var(--text-secondary)',
                cursor:     'pointer',
                fontSize:   '14px',
                padding:    '0 2px',
                flexShrink: 0,
              }}
            >×</button>
          </div>
        ))}
      </div>
    </div>
  )
}