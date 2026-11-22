import assert from 'node:assert/strict'
import test from 'node:test'
import {
  CONTINUE_MESSAGE, continueRequest, normalizeRuns, runActions, runBadge, runQuestion, runStatusLabel, withoutRun,
} from '../src/utils/pausedRuns.js'

const A = 'run_' + 'a'.repeat(32)
const B = 'run_' + 'b'.repeat(32)
const MODE = { kind: 'draft', label: 'Draft workflow: map v1', workflow: { name: 'map', version: 1 } }

function asked(over) {
  return {
    run_id: A, mode: MODE, status: 'needs_input', reason: 'Waiting for your answer.',
    question: { run_id: A, step: 'data', kind: 'pick_one', question: 'Go?', options: ['confirm', 'narrow'] },
    ...over,
  }
}

function waiting(over) {
  return { run_id: B, mode: MODE, status: 'waiting', reason: 'Waiting for the retrieval.', ready: false, ...over }
}

test('only paused rows are listed', () => {
  const rows = normalizeRuns({ runs: [asked(), waiting(), { run_id: 'x', status: 'done' }, null, { status: 'waiting' }] })
  assert.deepEqual(rows.map(r => r.run_id), [A, B])
  assert.deepEqual(normalizeRuns({}), [])
  assert.deepEqual(normalizeRuns(null), [])
})

test('a run on a question offers Answer and Cancel', () => {
  assert.deepEqual(runActions(asked()), { continue: false, answer: true, cancel: true })
})

test('a waiting run offers Continue only once its jobs are ready', () => {
  assert.deepEqual(runActions(waiting()), { continue: false, answer: false, cancel: true })
  assert.deepEqual(runActions(waiting({ ready: true })), { continue: true, answer: false, cancel: true })
  assert.deepEqual(runActions(waiting({ ready: undefined })), { continue: false, answer: false, cancel: true })
})

test('a question the app cannot answer offers no Answer but can still be cancelled', () => {
  for (const question of [null, undefined, { kind: 'inputs', step: 'x', run_id: A }]) {
    assert.deepEqual(runActions(asked({ question })), { continue: false, answer: false, cancel: true })
  }
  assert.equal(runQuestion(waiting({ question: asked().question })), null)
})

test('the status reads as a fixed label', () => {
  assert.equal(runStatusLabel(asked()), 'Needs your answer')
  assert.equal(runStatusLabel(waiting()), 'Waiting on a retrieval')
  assert.equal(runStatusLabel(waiting({ ready: true })), 'Ready to continue')
})

test('the badge is the mode badge', () => {
  assert.deepEqual(runBadge(asked()), { kind: 'draft', text: 'Draft workflow: map v1' })
  assert.deepEqual(runBadge(asked({ mode: { kind: 'exploratory', label: 'Exploratory', workflow: null } })),
    { kind: 'exploratory', text: 'Exploratory' })
  assert.equal(runBadge(asked({ mode: undefined })), null)
})

test('Continue sends fixed text and the run to resume', () => {
  assert.deepEqual(continueRequest(waiting({ ready: true })), { message: CONTINUE_MESSAGE, resume: { run_id: B } })
})

test('a cancelled row leaves the list', () => {
  assert.deepEqual(withoutRun([asked(), waiting()], A).map(r => r.run_id), [B])
})
