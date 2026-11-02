import { useEffect, useRef, useState } from 'react'
import { accountSummary } from '../utils/accountDisplay.js'

// The sidebar footer: who is signed in, and a popup with their account
// details, a shortcut to connectors, and sign-out.

const avatarStyle = (size) => ({
  width: size, height: size, borderRadius: '50%', background: 'var(--teal)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700,
  fontSize: size > 32 ? '14px' : '12px', flexShrink: 0, color: 'white',
})

const ellipsis = { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }

const menuButtonStyle = {
  display: 'flex', alignItems: 'center', gap: '8px', width: '100%', padding: '8px 10px',
  background: 'transparent', border: 'none', borderRadius: '6px', cursor: 'pointer',
  fontSize: '12.5px', color: 'var(--text-primary)', textAlign: 'left', fontFamily: 'var(--font)',
}

function DetailRow({ label, value }) {
  if (!value) return null
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', fontSize: '11.5px' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-secondary)', ...ellipsis }}>{value}</span>
    </div>
  )
}

function SignOutIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <path d="M16 17l5-5-5-5" />
      <path d="M21 12H9" />
    </svg>
  )
}

export default function AccountMenu({ user, onLogout, onManageConnectors }) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)
  const account = accountSummary(user)

  useEffect(() => {
    if (!open) return undefined
    const closeOnOutside = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false)
    }
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', closeOnOutside)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('mousedown', closeOnOutside)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [open])

  return (
    <div ref={rootRef} style={{ position: 'relative', borderTop: '1px solid var(--border)', paddingTop: '14px', marginTop: '10px' }}>
      {open && (
        <div
          role="dialog"
          aria-label="Account"
          style={{
            position: 'absolute', bottom: 'calc(100% + 6px)', left: 0, right: 0, zIndex: 20,
            background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: '10px',
            boxShadow: 'var(--shadow-md)', padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={avatarStyle(40)}>{account.initials}</div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', ...ellipsis }}>{account.name}</div>
              {account.email && (
                <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', ...ellipsis }} title={account.email}>{account.email}</div>
              )}
            </div>
          </div>
          {(account.memberSince || account.lastSignIn) && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <DetailRow label="Member since" value={account.memberSince} />
              <DetailRow label="Last sign-in" value={account.lastSignIn} />
            </div>
          )}
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: '6px', display: 'flex', flexDirection: 'column' }}>
            <button
              className="account-menu-item"
              style={menuButtonStyle}
              onClick={() => { setOpen(false); onManageConnectors() }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
              Manage connectors
            </button>
            <button className="account-menu-item" style={menuButtonStyle} onClick={onLogout}>
              <SignOutIcon />
              Sign out
            </button>
          </div>
        </div>
      )}

      <button
        onClick={() => setOpen(value => !value)}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Account"
        className="account-menu-item"
        style={{ ...menuButtonStyle, padding: '4px', gap: '10px' }}
      >
        <div style={avatarStyle(32)}>{account.initials}</div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)', ...ellipsis }}>{account.name}</div>
          {account.email && (
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', ...ellipsis }}>{account.email}</div>
          )}
        </div>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
          <path d="M6 15l6-6 6 6" />
        </svg>
      </button>

      <style>{`
        .account-menu-item:hover { background: var(--bg-primary) !important; }
      `}</style>
    </div>
  )
}
