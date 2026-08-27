import test from 'node:test'
import assert from 'node:assert/strict'

import { turnCompletionFocus } from '../src/utils/turnFocus.js'

const assistantWithCharts = {
  role: 'assistant',
  charts: [{ chart_id: 'c1' }, { chart_id: 'c2' }],
  artifacts: [{ id: 'a1', type: 'table' }],
}

// Only the true -> false edge counts. Every other combination has to leave
// focus alone, because focus between turns belongs to whatever the user
// clicked in the chat history.
test('mid-turn renders never move focus', () => {
  assert.equal(turnCompletionFocus(true, true, [assistantWithCharts]), null)
})

test('a turn starting (false -> true) never moves focus', () => {
  assert.equal(turnCompletionFocus(false, true, [assistantWithCharts]), null)
})

test('a steady idle render (false -> false) never moves focus', () => {
  assert.equal(turnCompletionFocus(false, false, [assistantWithCharts]), null)
})

test('a settled turn focuses the NEWEST chart of the last assistant message', () => {
  const focus = turnCompletionFocus(true, false, [{ role: 'user' }, assistantWithCharts])
  assert.deepEqual(focus, { kind: 'chart', data: { chart_id: 'c2' } })
})

test('charts win over artifacts when the turn produced both', () => {
  const focus = turnCompletionFocus(true, false, [assistantWithCharts])
  assert.equal(focus.kind, 'chart')
})

test('with no charts, the first REACHABLE artifact takes focus', () => {
  // `map` artifacts get no card of their own (artifactReachability), so the
  // table behind it is the first thing actually reachable.
  const msg = {
    role: 'assistant',
    charts: [],
    artifacts: [{ id: 'a0', type: 'map' }, { id: 'a1', type: 'table' }],
  }
  assert.deepEqual(
    turnCompletionFocus(true, false, [msg]),
    { kind: 'artifact', data: { id: 'a1', type: 'table' } },
  )
})

test('a turn whose only artifacts are unreachable leaves focus alone', () => {
  const msg = { role: 'assistant', charts: [], artifacts: [{ id: 'a0', type: 'map' }] }
  assert.equal(turnCompletionFocus(true, false, [msg]), null)
})

test('a text-only reply leaves focus alone rather than blanking the panel', () => {
  assert.equal(turnCompletionFocus(true, false, [{ role: 'assistant', content: 'hi' }]), null)
})

test('a turn that ends on a non-assistant message leaves focus alone', () => {
  assert.equal(turnCompletionFocus(true, false, [assistantWithCharts, { role: 'user' }]), null)
})

test('an empty or missing message list is not a crash', () => {
  assert.equal(turnCompletionFocus(true, false, []), null)
  assert.equal(turnCompletionFocus(true, false, undefined), null)
})
