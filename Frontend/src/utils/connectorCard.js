// T30: pure status->badge mapping for the Connectors tab card, kept out of
// the component so it's testable the same way jobCard.js's statusBadge is.

export function formatExpiry(isoString) {
  if (!isoString) return ''
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

export function connectorBadge(connector) {
  const status = connector?.status
  if (status === 'connected') {
    const expiry = formatExpiry(connector.expires_at)
    return { label: expiry ? `Connected until ${expiry}` : 'Connected', color: 'var(--teal-text)' }
  }
  if (status === 'expired') return { label: 'Expired', color: 'var(--warning)' }
  if (status === 'error') return { label: 'Error', color: 'var(--error)' }
  return { label: 'Not connected', color: 'var(--text-muted)' }
}

// A card offers Disconnect whenever a row exists server-side, regardless of
// whether it's still live -- an expired or errored token is still something
// to sever.
export function isConnectorLinked(connector) {
  return connector?.status === 'connected' || connector?.status === 'expired' || connector?.status === 'error'
}
