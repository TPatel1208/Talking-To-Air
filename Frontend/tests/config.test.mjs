import test from 'node:test'
import assert from 'node:assert/strict'

import { API_BASE, resolveApiBase } from '../src/config.js'

test('unset or blank VITE_API_URL falls back to same-origin /api', () => {
  assert.equal(resolveApiBase(undefined), '/api')
  assert.equal(resolveApiBase(''), '/api')
  assert.equal(resolveApiBase('   '), '/api')
})

test('a configured base is used as given', () => {
  assert.equal(resolveApiBase('/api'), '/api')
  assert.equal(resolveApiBase('https://api.example.com'), 'https://api.example.com')
  assert.equal(resolveApiBase('https://example.com/backend'), 'https://example.com/backend')
})

test('trailing slashes are stripped so joined paths have no double slash', () => {
  assert.equal(resolveApiBase('https://api.example.com/'), 'https://api.example.com')
  assert.equal(resolveApiBase('/api//'), '/api')
  assert.equal(resolveApiBase('/'), '/api')
})

test('outside Vite (no import.meta.env) the module still loads with the default', () => {
  assert.equal(API_BASE, '/api')
})
