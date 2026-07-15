import test from 'node:test'
import assert from 'node:assert/strict'

import { connectorBadge, formatExpiry, isConnectorLinked } from '../src/utils/connectorCard.js'

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
