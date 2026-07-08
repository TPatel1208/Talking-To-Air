import assert from 'node:assert/strict'
import test from 'node:test'
import { extractSuggestedFollowups } from '../src/utils/followups.js'

test('extracts suggestions from a done event that carries them', () => {
  const data = { thread_id: 'a', response: 'ok', suggested_followups: ['What about last month?', 'Any exceedances?'] }

  assert.deepEqual(extractSuggestedFollowups(data), ['What about last month?', 'Any exceedances?'])
})

test('returns an empty array when the done event omits the field', () => {
  const data = { thread_id: 'a', response: 'ok' }

  assert.deepEqual(extractSuggestedFollowups(data), [])
})

test('returns an empty array for a null/undefined done payload', () => {
  assert.deepEqual(extractSuggestedFollowups(null), [])
  assert.deepEqual(extractSuggestedFollowups(undefined), [])
})

test('drops non-string entries rather than rendering them', () => {
  const data = { suggested_followups: ['A real question?', 42, null] }

  assert.deepEqual(extractSuggestedFollowups(data), ['A real question?'])
})
