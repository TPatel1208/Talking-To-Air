import test from 'node:test'
import assert from 'node:assert/strict'

import { accountSummary } from '../src/utils/accountDisplay.js'

const user = {
  id: '6f1c2c1e-0000-4000-8000-000000000000',
  email: 'pat.lee@example.edu',
  created_at: '2026-08-27T14:03:00Z',
  last_sign_in_at: '2026-10-17T09:30:00Z',
  user_metadata: {},
}

test('the name comes from profile metadata when the account has one', () => {
  for (const key of ['full_name', 'name', 'display_name']) {
    const summary = accountSummary({ ...user, user_metadata: { [key]: 'Pat Lee' } })
    assert.equal(summary.name, 'Pat Lee', key)
    assert.equal(summary.initials, 'PL', key)
  }
})

test('with no profile name, the email local part stands in, tidied', () => {
  const summary = accountSummary(user)
  assert.equal(summary.name, 'Pat Lee')
  assert.equal(summary.initials, 'PL')
  assert.equal(summary.email, 'pat.lee@example.edu')
})

test('a single-word local part gives one initial', () => {
  const summary = accountSummary({ ...user, email: 'researcher42@example.edu' })
  assert.equal(summary.name, 'Researcher42')
  assert.equal(summary.initials, 'R')
})

test('dates read as calendar dates, absent ones are omitted', () => {
  const summary = accountSummary(user)
  assert.equal(summary.memberSince, 'Aug 27, 2026')
  assert.equal(summary.lastSignIn, 'Oct 17, 2026')
  const bare = accountSummary({ email: 'a@b.c' })
  assert.equal(bare.memberSince, null)
  assert.equal(bare.lastSignIn, null)
})

test('no user still renders something sensible', () => {
  for (const missing of [null, undefined, {}]) {
    const summary = accountSummary(missing)
    assert.equal(summary.name, 'Signed in')
    assert.equal(summary.initials, '?')
    assert.equal(summary.email, null)
  }
})
