import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildCredentialRequest,
  connectorBadge,
  credentialFields,
  errorDetail,
  formatExpiry,
  isConnectorLinked,
} from '../src/utils/connectorCard.js'

test('connectorBadge renders the not_connected state with a clear one-line explanation cue', () => {
  const badge = connectorBadge({ status: 'not_connected' })
  assert.equal(badge.label, 'Not connected')
  assert.equal(badge.color, 'var(--text-muted)')
})

test('connectorBadge shows the expiry date for a connected token', () => {
  const badge = connectorBadge({ status: 'connected', expires_at: '2026-09-01T00:00:00Z' })
  assert.match(badge.label, /^Connected until /)
  assert.equal(badge.color, 'var(--teal-text)')
})

test('connectorBadge falls back to plain "Connected" when expires_at is missing', () => {
  assert.equal(connectorBadge({ status: 'connected' }).label, 'Connected')
})

test('connectorBadge flips to expired once expires_at has passed, independent of stored status', () => {
  const badge = connectorBadge({ status: 'expired', expires_at: '2020-01-01T00:00:00Z' })
  assert.equal(badge.label, 'Expired')
  assert.equal(badge.color, 'var(--warning)')
})

test('connectorBadge surfaces an error state distinctly from expired', () => {
  const badge = connectorBadge({ status: 'error' })
  assert.equal(badge.label, 'Error')
  assert.equal(badge.color, 'var(--error)')
})

test('connectorBadge treats an unrecognized/undefined status as not connected', () => {
  assert.equal(connectorBadge(undefined).label, 'Not connected')
  assert.equal(connectorBadge({}).label, 'Not connected')
})

test('formatExpiry renders a parseable date and passes through junk as empty', () => {
  assert.match(formatExpiry('2026-09-01T00:00:00Z'), /2026/)
  assert.equal(formatExpiry(''), '')
  assert.equal(formatExpiry(null), '')
  assert.equal(formatExpiry('not-a-date'), '')
})

test('isConnectorLinked offers Disconnect for connected, expired, and error rows', () => {
  assert.equal(isConnectorLinked({ status: 'connected' }), true)
  assert.equal(isConnectorLinked({ status: 'expired' }), true)
  assert.equal(isConnectorLinked({ status: 'error' }), true)
})

test('isConnectorLinked withholds Disconnect when nothing is stored server-side', () => {
  assert.equal(isConnectorLinked({ status: 'not_connected' }), false)
  assert.equal(isConnectorLinked(undefined), false)
})

test('credentialFields asks a token connector for one secret field', () => {
  const fields = credentialFields({ auth_method: 'token' })
  assert.deepEqual(fields.map(f => [f.name, f.type]), [['token', 'password']])
})

test('credentialFields asks an email_key connector for an email and a masked key', () => {
  const fields = credentialFields({ auth_method: 'email_key' })
  assert.deepEqual(fields.map(f => [f.name, f.type]), [['email', 'email'], ['key', 'password']])
})

test('buildCredentialRequest sends a token to the token endpoint, trimmed', () => {
  const request = buildCredentialRequest({ connector_type: 'earthdata', auth_method: 'token' }, { token: '  abc ' })
  assert.deepEqual(request, { path: '/connectors/earthdata/token', body: { token: 'abc' } })
})

test('buildCredentialRequest sends an email + key pair to the credentials endpoint, trimmed', () => {
  const request = buildCredentialRequest(
    { connector_type: 'epa_aqs', auth_method: 'email_key' },
    { email: ' me@lab.edu ', key: ' k1 ' },
  )
  assert.deepEqual(request, { path: '/connectors/epa_aqs/credentials', body: { email: 'me@lab.edu', key: 'k1' } })
})

test('buildCredentialRequest returns null until every field is filled', () => {
  const aqs = { connector_type: 'epa_aqs', auth_method: 'email_key' }
  assert.equal(buildCredentialRequest(aqs, { email: 'me@lab.edu', key: '  ' }), null)
  assert.equal(buildCredentialRequest(aqs, {}), null)
  assert.equal(buildCredentialRequest({ connector_type: 'earthdata', auth_method: 'token' }, { token: '' }), null)
})

test('errorDetail passes a server message through and turns field-validation lists into plain text', () => {
  assert.equal(errorDetail({ detail: "EPA didn't accept that email and key." }), "EPA didn't accept that email and key.")
  assert.equal(
    errorDetail({ detail: [{ loc: ['body', 'email'], msg: 'String should match pattern' }] }),
    'Check the email address.',
  )
  assert.equal(errorDetail({ detail: [{ loc: ['body', 'key'], msg: 'too short' }] }), 'Check the key.')
  assert.equal(errorDetail(null), '')
})
