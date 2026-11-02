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

// The inputs a card's form shows, by the registry's auth_method: "token"
// pastes one token, "email_key" (EPA AQS) takes the signup email and key.
export function credentialFields(connector) {
  if (connector?.auth_method === 'email_key') {
    return [
      { name: 'email', type: 'email', placeholder: 'Email you signed up with', autoComplete: 'email' },
      { name: 'key', type: 'password', placeholder: 'API key EPA emailed you', autoComplete: 'off' },
    ]
  }
  return [{ name: 'token', type: 'password', placeholder: 'Paste your token', autoComplete: 'off' }]
}

// The PUT a filled-in form sends (path relative to API_BASE), or null while
// any field is still blank.
export function buildCredentialRequest(connector, values) {
  const body = {}
  for (const field of credentialFields(connector)) {
    const value = (values?.[field.name] ?? '').trim()
    if (!value) return null
    body[field.name] = value
  }
  const endpoint = connector?.auth_method === 'email_key' ? 'credentials' : 'token'
  return { path: `/connectors/${connector.connector_type}/${endpoint}`, body }
}

const FIELD_ERROR_TEXT = { email: 'Check the email address.', key: 'Check the key.', token: 'Check the token.' }

// A save failure's message. FastAPI's own 422 for a malformed field carries a
// list of validation objects, not a sentence.
export function errorDetail(data) {
  const detail = data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail.length > 0) {
    const field = detail[0]?.loc?.[detail[0].loc.length - 1]
    return FIELD_ERROR_TEXT[field] || 'Check every field and try again.'
  }
  return ''
}
