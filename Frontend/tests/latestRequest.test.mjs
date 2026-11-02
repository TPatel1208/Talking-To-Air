import test from 'node:test'
import assert from 'node:assert/strict'

import { createLatest, debounce } from '../src/utils/latestRequest.js'

test('only the most recently started request counts as current', () => {
  const latest = createLatest()
  const first = latest.next()
  const second = latest.next()
  assert.equal(latest.isLatest(first), false)
  assert.equal(latest.isLatest(second), true)
})

test('a debounced call runs once, with the last arguments, after the quiet period', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const calls = []
  const run = debounce((value) => calls.push(value), 400)
  run('0002-06-05')
  run('0020-06-05')
  run('2026-06-05')
  t.mock.timers.tick(399)
  assert.deepEqual(calls, [])
  t.mock.timers.tick(1)
  assert.deepEqual(calls, ['2026-06-05'])
})

test('a cancelled debounce never runs', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const calls = []
  const run = debounce((value) => calls.push(value), 400)
  run('x')
  run.cancel()
  t.mock.timers.tick(1000)
  assert.deepEqual(calls, [])
})
