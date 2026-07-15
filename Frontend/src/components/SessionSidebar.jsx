import { useState } from 'react'
import ConnectorsPanel from './ConnectorsPanel'

const API_BASE = '/api'

function authHeaders(accessToken) {
  return accessToken ? { Authorization: `Bearer ${accessToken}` } : {}
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function filenameFromDisposition(disposition, fallback) {
  const match = /filename="?([^";]+)"?/i.exec(disposition || '')
  return match?.[1]?.trim() || fallback
}

async function downloadArtifact(artifact, accessToken) {
  const url = artifact.type === 'table'
    ? `${API_BASE}/artifacts/${artifact.id}/csv`
    : `${API_BASE}/chart/${artifact.id}/export.csv`
  const res = await fetch(url, { headers: authHeaders(accessToken) })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const blob = await res.blob()
  downloadBlob(filenameFromDisposition(res.headers.get('content-disposition'), `${artifact.title || artifact.id}.csv`), blob)
}

async function downloadImage(url, accessToken) {
  const res = await fetch(url, { headers: authHeaders(accessToken) })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const blob = await res.blob()
  downloadBlob(url.split('/').pop() || 'image.png', blob)
}

const TYPE_TAG = { table: 'CSV', map: 'MAP', comparison: 'CMP', timeseries: 'TS' }

function FileRow({ tag, title, subtitle, onDownload }) {
  const [state, setState] = useState('')

  const handleDownload = async () => {
    setState('downloading')
    try {
      await onDownload()
      setState('')
    } catch (err) {
      setState(err.message || 'Failed')
      window.setTimeout(() => setState(''), 2000)
    }
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '9px',
      padding: '9px 10px', borderRadius: '8px',
      background: 'var(--bg-card)', border: '1px solid var(--border)',
    }}>
      <div style={{
        width: '26px', height: '26px', borderRadius: '6px', flexShrink: 0,
        background: 'var(--teal-light)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '8.5px', fontWeight: 800, letterSpacing: '0.02em', color: 'var(--teal-text)',
      }}>
        {tag}
      </div>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontSize: '12px', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {title}
        </div>
        <div style={{ fontSize: '10.5px', color: state && state !== 'downloading' ? 'var(--error)' : 'var(--text-muted)', marginTop: '1px' }}>
          {state === 'downloading' ? 'Downloading…' : (state || subtitle)}
        </div>
      </div>
      <svg
        onClick={handleDownload}
        width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
        style={{ color: 'var(--text-muted)', cursor: 'pointer', flexShrink: 0 }}
      >
        <path d="M12 4v12m0 0l-4.5-4.5M12 16l4.5-4.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M5 19h14" strokeLinecap="round" />
      </svg>
    </div>
  )
}

export default function SessionSidebar({ sessions, threadId, onSwitch, onNew, onDelete, onLogout, images = [], artifacts = [], accessToken, onCollapse }) {
  const [nav, setNav] = useState('chats')

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
      width:         '232px',
      flexShrink:    0,
      background:    'var(--bg-card)',
      borderRight:   '1px solid var(--border)',
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
      padding:       '16px 14px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <div
          onClick={onNew}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
            padding: '9px 12px', borderRadius: '9px', background: 'var(--teal)', color: 'white',
            fontWeight: 700, fontSize: '13px', cursor: 'pointer', flex: 1,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
            <span style={{ fontSize: '15px', lineHeight: 1 }}>+</span> New chat
          </div>
        </div>
        {onCollapse && (
          <button
            type="button"
            onClick={onCollapse}
            title="Collapse sessions"
            aria-label="Collapse sessions"
            style={{
              background: 'transparent', border: 'none', padding: '2px', cursor: 'pointer',
              display: 'flex', flexShrink: 0, color: 'var(--text-muted)',
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 6 9 12 15 18" />
            </svg>
          </button>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', marginBottom: '16px' }}>
        <div
          onClick={() => setNav('chats')}
          style={{
            display: 'flex', alignItems: 'center', gap: '9px', padding: '8px 10px', borderRadius: '7px',
            fontSize: '13px', fontWeight: nav === 'chats' ? 700 : 600,
            color: nav === 'chats' ? 'var(--text-primary)' : 'var(--text-secondary)',
            background: nav === 'chats' ? 'var(--bg-secondary)' : 'transparent', cursor: 'pointer',
          }}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" style={{ flexShrink: 0 }}>
            <path d="M4 5.5h16v10.5H9.5L5 20v-4H4z" strokeLinejoin="round" />
          </svg>
          Chats
        </div>
        <div
          onClick={() => setNav('files')}
          style={{
            display: 'flex', alignItems: 'center', gap: '9px', padding: '8px 10px', borderRadius: '7px',
            fontSize: '13px', fontWeight: nav === 'files' ? 700 : 600,
            color: nav === 'files' ? 'var(--text-primary)' : 'var(--text-secondary)',
            background: nav === 'files' ? 'var(--bg-secondary)' : 'transparent', cursor: 'pointer',
          }}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" style={{ flexShrink: 0 }}>
            <path d="M4 6.5h6l1.6 2H20V18H4z" strokeLinejoin="round" />
          </svg>
          Files
        </div>
        <div
          onClick={() => setNav('connectors')}
          style={{
            display: 'flex', alignItems: 'center', gap: '9px', padding: '8px 10px', borderRadius: '7px',
            fontSize: '13px', fontWeight: nav === 'connectors' ? 700 : 600,
            color: nav === 'connectors' ? 'var(--text-primary)' : 'var(--text-secondary)',
            background: nav === 'connectors' ? 'var(--bg-secondary)' : 'transparent', cursor: 'pointer',
          }}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" style={{ flexShrink: 0 }}>
            <path d="M9 12a3 3 0 1 0 6 0 3 3 0 0 0-6 0z" />
            <path d="M6 12H3m18 0h-3M12 6V3m0 18v-3" strokeLinecap="round" />
          </svg>
          Connectors
        </div>
      </div>

      {nav === 'chats' && (
        <>
          <div style={{ fontSize: '10.5px', fontWeight: 700, letterSpacing: '0.06em', color: 'var(--text-muted)', textTransform: 'uppercase', padding: '0 10px 8px' }}>
            Recent analyses
          </div>
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2px' }}>
            {sessions.length === 0 && (
              <div style={{ padding: '12px 10px', color: 'var(--text-hint, var(--text-muted))', fontSize: '12px', fontStyle: 'italic' }}>
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
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
                    padding: '9px 10px', borderRadius: '8px', cursor: 'pointer',
                    background: isActive ? 'var(--teal-light)' : 'transparent',
                    borderLeft: `2px solid ${isActive ? 'var(--teal)' : 'transparent'}`,
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = 'var(--bg-secondary)' }}
                  onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                    <span style={{
                      width: '6px', height: '6px', borderRadius: '50%', flexShrink: 0,
                      background: isActive ? 'var(--teal)' : 'var(--border-hover)',
                    }} />
                    <span style={{
                      fontSize: '12.5px', fontWeight: isActive ? 700 : 500,
                      color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {title}
                    </span>
                  </div>
                  <button
                    onClick={e => handleDelete(e, id)}
                    className="session-delete-btn"
                    title="Delete session"
                    style={{
                      background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer',
                      fontSize: '13px', padding: '2px 4px', borderRadius: '4px', flexShrink: 0, lineHeight: 1,
                      opacity: 0, transition: 'opacity 0.15s, color 0.15s',
                    }}
                  >
                    ✕
                  </button>
                </div>
              )
            })}
          </div>
        </>
      )}

      {nav === 'files' && (
        <>
          <div style={{ fontSize: '10.5px', fontWeight: 700, letterSpacing: '0.06em', color: 'var(--text-muted)', textTransform: 'uppercase', padding: '0 10px 8px' }}>
            Exported files
          </div>
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {artifacts.length === 0 && images.length === 0 && (
              <div style={{ padding: '12px 10px', color: 'var(--text-muted)', fontSize: '12px', fontStyle: 'italic' }}>
                No outputs yet
              </div>
            )}
            {artifacts.map((artifact, i) => (
              <FileRow
                key={artifact.id || i}
                tag={TYPE_TAG[artifact.type] || 'OUT'}
                title={artifact.title || 'Untitled output'}
                subtitle={artifact.type}
                onDownload={() => downloadArtifact(artifact, accessToken)}
              />
            ))}
            {images.map((url, i) => (
              <FileRow
                key={`img-${i}`}
                tag="IMG"
                title={url.split('/').pop() || `image-${i + 1}`}
                subtitle="Image"
                onDownload={() => downloadImage(url, accessToken)}
              />
            ))}
          </div>
        </>
      )}

      {nav === 'connectors' && (
        <>
          <div style={{ fontSize: '10.5px', fontWeight: 700, letterSpacing: '0.06em', color: 'var(--text-muted)', textTransform: 'uppercase', padding: '0 10px 8px' }}>
            Connected services
          </div>
          <ConnectorsPanel accessToken={accessToken} />
        </>
      )}

      <div style={{ borderTop: '1px solid var(--border)', paddingTop: '14px', marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div style={{
          width: '32px', height: '32px', borderRadius: '50%', background: 'var(--teal)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700,
          fontSize: '12px', flexShrink: 0, color: 'white',
        }}>
          T
        </div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: '12.5px', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-primary)' }}>
            Talking to Air
          </div>
        </div>
        <button
          onClick={onLogout}
          title="Sign out"
          aria-label="Sign out"
          style={{ background: 'transparent', border: 'none', padding: '4px', cursor: 'pointer', display: 'flex', flexShrink: 0 }}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" style={{ color: 'var(--text-muted)' }}>
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <path d="M16 17l5-5-5-5" />
            <path d="M21 12H9" />
          </svg>
        </button>
      </div>

      <style>{`
        div:hover > .session-delete-btn,
        .session-delete-btn:focus { opacity: 1 !important; }
      `}</style>
    </div>
  )
}
