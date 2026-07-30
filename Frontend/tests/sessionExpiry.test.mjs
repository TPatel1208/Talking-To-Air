import test from 'node:test'
import assert from 'node:assert/strict'

import {
  shouldPromptReauth,
  sessionExpiryReducer,
  authTransition,
} from '../src/utils/sessionExpiry.js'

test('shouldPromptReauth fires only on 401 — the "sign in again" signal', () => {
  assert.equal(shouldPromptReauth(401), true)
})

test('shouldPromptReauth ignores 404/5xx/network — those are not auth failures', () => {
  assert.equal(shouldPromptReauth(404), false)
  assert.equal(shouldPromptReauth(500), false)
  assert.equal(shouldPromptReauth(503), false)
  assert.equal(shouldPromptReauth(undefined), false)
})

test('a 401 raises the expired-session modal instead of logging the user out', () => {
  const next = sessionExpiryReducer({ sessionExpired: false }, { type: 'unauthorized' })
  assert.equal(next.sessionExpired, true)
  // The whole point of T47: a 401 must NOT read as a logout. The reducer has
  // no "logged out" outcome to reach from an auth failure — it can only ask
  // the user to sign back in, preserving the view underneath.
  assert.equal(next.loggedOut, undefined)
})

test('a successful re-login clears the expired state so the modal closes', () => {
  const next = sessionExpiryReducer({ sessionExpired: true }, { type: 'reauthenticated' })
  assert.equal(next.sessionExpired, false)
})

test('unknown actions leave the state untouched', () => {
  const prev = { sessionExpired: true }
  assert.equal(sessionExpiryReducer(prev, { type: 'whatever' }), prev)
})

test('re-auth resumes in place — it does not clear the active thread', () => {
  assert.equal(authTransition('reauth').clearActiveThread, false)
})

test('a fresh login starts clean — it clears any stale active thread', () => {
  assert.equal(authTransition('login').clearActiveThread, true)
})
