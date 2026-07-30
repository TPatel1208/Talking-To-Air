import test from 'node:test'
import assert from 'node:assert/strict'

import { classifyHistoryFetchFailure, historyStateReducer, HISTORY_ERROR_MESSAGE } from '../src/utils/historyLoad.js'

test('classifyHistoryFetchFailure treats a 404 as the session genuinely gone', () => {
  assert.equal(classifyHistoryFetchFailure(404), 'not-found')
})

test('classifyHistoryFetchFailure treats other statuses (and network exceptions, no status) as transient', () => {
  assert.equal(classifyHistoryFetchFailure(500), 'error')
  assert.equal(classifyHistoryFetchFailure(503), 'error')
  assert.equal(classifyHistoryFetchFailure(undefined), 'error')
})

test('historyStateReducer: a successful load replaces messages and clears any error', () => {
  const prev = { messages: [], historyError: 'stale error' }
  const next = historyStateReducer(prev, { type: 'loaded', messages: [{ role: 'user', content: 'hi' }] })
  assert.deepEqual(next, { messages: [{ role: 'user', content: 'hi' }], historyError: null })
})

test('historyStateReducer: a transient failure keeps the current messages and sets an error', () => {
  const prev = { messages: [{ role: 'assistant', content: 'partial' }], historyError: null }
  const next = historyStateReducer(prev, { type: 'failed' })
  assert.deepEqual(next.messages, prev.messages)
  assert.equal(next.historyError, HISTORY_ERROR_MESSAGE)
})

test('historyStateReducer: not-found clears messages and carries no error (session is genuinely gone)', () => {
  const prev = { messages: [{ role: 'assistant', content: 'stale' }], historyError: null }
  const next = historyStateReducer(prev, { type: 'not-found' })
  assert.deepEqual(next, { messages: [], historyError: null })
})

test('historyStateReducer: retry clears the error without touching messages', () => {
  const prev = { messages: [{ role: 'assistant', content: 'partial' }], historyError: HISTORY_ERROR_MESSAGE }
  const next = historyStateReducer(prev, { type: 'retry' })
  assert.deepEqual(next, { messages: prev.messages, historyError: null })
})
