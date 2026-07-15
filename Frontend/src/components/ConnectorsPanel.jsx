import { useState } from 'react'
import { useConnectors } from '../hooks/useConnectors'
import { connectorBadge, isConnectorLinked } from '../utils/connectorCard'

function ConnectorCard({ connector, onSave, onDisconnect }) {
  const [expanded, setExpanded] = useState(false)
  const [tokenInput, setTokenInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [cardError, setCardError] = useState(null)

  const badge = connectorBadge(connector)
  const linked = isConnectorLinked(connector)
  const connectedStyle = connector?.status === 'connected'
  const shortLabel = connectedStyle ? 'Connected'
    : connector?.status === 'expired' ? 'Expired'
    : connector?.status === 'error' ? 'Error'
    : 'Connect'

  const handleSave = async (event) => {
    event.preventDefault()
    const token = tokenInput.trim()
    if (!token) return
    setBusy(true)
    setCardError(null)
    try {
      await onSave(connector.connector_type, token)
      setTokenInput('')
      setExpanded(false)
    } catch (err) {
      setCardError(err.message || 'Failed to save token')
    } finally {
      setBusy(false)
    }
  }

  const handleDisconnect = async () => {
    if (!window.confirm(`Disconnect ${connector.display_name}?`)) return
    setBusy(true)
    setCardError(null)
    try {
      await onDisconnect(connector.connector_type)
    } catch (err) {
      setCardError(err.message || 'Failed to disconnect')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{
      borderRadius: '8px', background: 'var(--bg-card)',
      border: '1px solid var(--border)', overflow: 'hidden',
    }}>
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        style={{
          width: '100%', display: 'flex', flexDirection: 'column', gap: '4px',
          padding: '12px', background: 'transparent', border: 0, cursor: 'pointer', textAlign: 'left',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            flexShrink: 0, width: '24px', height: '24px', borderRadius: '6px',
            background: 'var(--bg-primary)', border: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px',
          }}>
            🛰️
          </div>
          <div style={{
            flex: 1, minWidth: 0, fontSize: '13px', fontWeight: 700,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {connector.display_name}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
          <div style={{ flex: 1, minWidth: 0, fontSize: '11px', fontWeight: 600, color: badge.color }}>
            {badge.label}
          </div>
          <span
            style={{
              flexShrink: 0, fontSize: '10.5px', fontWeight: 700, padding: '4px 10px', borderRadius: '999px',
              whiteSpace: 'nowrap', transition: 'background 0.15s, color 0.15s, border-color 0.15s',
              background: connectedStyle ? 'var(--teal)' : 'transparent',
              color: connectedStyle ? 'white' : badge.color,
              border: connectedStyle ? '1px solid var(--teal)' : '1px solid var(--border)',
            }}
          >
            {shortLabel}
          </span>
        </div>
      </button>

      {expanded && (
        <div style={{
          padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: '8px',
          borderTop: '1px solid var(--border)', paddingTop: '10px',
        }}>
          <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
            {connector.description}
          </div>

          <form onSubmit={handleSave} style={{ display: 'flex', gap: '6px' }}>
            <input
              type="password"
              value={tokenInput}
              onChange={e => setTokenInput(e.target.value)}
              placeholder={linked ? 'Paste a new token to replace it' : 'Paste your token'}
              autoComplete="off"
              autoFocus
              style={{
                flex: 1, minWidth: 0, height: '30px', border: '1px solid var(--border)', borderRadius: '6px',
                background: 'var(--bg-primary)', color: 'var(--text-primary)', padding: '0 8px', fontSize: '12px',
              }}
            />
            <button
              type="submit"
              disabled={busy || !tokenInput.trim()}
              style={{
                flexShrink: 0, height: '30px', padding: '0 10px', border: 0, borderRadius: '6px',
                background: 'var(--teal)', color: 'white', fontSize: '12px', fontWeight: 600,
                cursor: busy ? 'not-allowed' : 'pointer', opacity: busy ? 0.75 : 1,
              }}
            >
              {busy ? 'Saving…' : 'Save'}
            </button>
          </form>

          {cardError && <div style={{ fontSize: '11px', color: 'var(--error)' }}>{cardError}</div>}

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
            <a
              href={connector.token_docs_url}
              target="_blank"
              rel="noreferrer"
              style={{ fontSize: '11px', color: 'var(--teal-text)' }}
            >
              Generate a token at Earthdata Login
            </a>
            {linked && (
              <button
                type="button"
                onClick={handleDisconnect}
                disabled={busy}
                style={{
                  background: 'transparent', border: 'none', color: 'var(--text-muted)',
                  cursor: busy ? 'not-allowed' : 'pointer', fontSize: '11px', padding: 0,
                }}
              >
                Disconnect
              </button>
            )}
          </div>

          <div style={{ fontSize: '10.5px', color: 'var(--text-hint, var(--text-muted))', fontStyle: 'italic' }}>
            Stored encrypted. Never shown again, to anyone, after save.
          </div>
        </div>
      )}
    </div>
  )
}

export default function ConnectorsPanel({ accessToken }) {
  const { connectors, loading, error, notConfigured, setToken, disconnect } = useConnectors(accessToken)

  if (notConfigured) {
    return (
      <div style={{ padding: '12px 10px', fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        Connectors aren't configured on this deployment. That's a server-side choice, not a bug.
      </div>
    )
  }

  return (
    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {loading && connectors.length === 0 && (
        <div style={{ padding: '12px 10px', fontSize: '12px', color: 'var(--text-muted)' }}>Loading…</div>
      )}
      {error && <div style={{ padding: '0 10px', fontSize: '12px', color: 'var(--error)' }}>{error}</div>}
      {!loading && connectors.length === 0 && !error && (
        <div style={{ padding: '12px 10px', fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          No connectors available
        </div>
      )}
      {connectors.map(connector => (
        <ConnectorCard
          key={connector.connector_type}
          connector={connector}
          onSave={setToken}
          onDisconnect={disconnect}
        />
      ))}
    </div>
  )
}
