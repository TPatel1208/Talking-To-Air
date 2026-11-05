import assert from 'node:assert/strict'
import test from 'node:test'
import { applyWorkflowEvent, INITIAL_WORKFLOW_STATE } from '../src/utils/workflowStage.js'

test('a stage status event sets stage/label/detail and marks the strip active', () => {
  const next = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Checking coverage...', stage: 'coverage', detail: 14,
  })

  assert.equal(next.stage, 'coverage')
  assert.equal(next.label, 'Checking coverage...')
  assert.equal(next.detail, 14)
  assert.equal(next.active, true)
  assert.equal(next.failedStage, null)
})

test('a bare status event with no stage key keeps the label current without a stage', () => {
  const next = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', { message: 'Working on it...' })

  assert.equal(next.stage, null)
  assert.equal(next.label, 'Working on it...')
  assert.equal(next.active, true)
})

test('a later stage status overwrites the label/detail of an earlier one', () => {
  const afterSearch = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Searching datasets...', stage: 'search',
  })
  const afterAoi = applyWorkflowEvent(afterSearch, 'status', {
    message: 'Resolving area of interest...', stage: 'aoi',
  })

  assert.equal(afterAoi.stage, 'aoi')
  assert.equal(afterAoi.label, 'Resolving area of interest...')
})

test('a job_progress event with a numeric progress forwards it as detail', () => {
  const withStage = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Retrieving data...', stage: 'progress',
  })
  const next = applyWorkflowEvent(withStage, 'job_progress', { progress: 42 })

  assert.equal(next.detail, 42)
  assert.equal(next.stage, 'progress')
})

test('a job_progress event with no numeric progress leaves state unchanged', () => {
  const withStage = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Retrieving data...', stage: 'progress', detail: 10,
  })
  const next = applyWorkflowEvent(withStage, 'job_progress', { status: 'queued', progress: null })

  assert.equal(next.detail, 10)
})

test('the first text token collapses the strip (active becomes false)', () => {
  const active = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Rendering...', stage: 'render',
  })

  const next = applyWorkflowEvent(active, 'text', 'Here is your plot.')

  assert.equal(next.active, false)
  // The last stage is preserved for context even though the strip collapses.
  assert.equal(next.stage, 'render')
})

test('an error event freezes the current stage as the failed stage and collapses the strip', () => {
  const active = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Checking coverage...', stage: 'coverage',
  })

  const next = applyWorkflowEvent(active, 'error', { detail: 'boom' })

  assert.equal(next.failedStage, 'coverage')
  assert.equal(next.active, false)
})

test('a done event resets the strip back to its initial state', () => {
  const active = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Rendering...', stage: 'render',
  })

  const next = applyWorkflowEvent(active, 'done', {})

  assert.deepEqual(next, INITIAL_WORKFLOW_STATE)
})

test('cleanStatusMessage strips elapsed seconds phrases', () => {
  import('../src/utils/workflowStage.js').then(({ cleanStatusMessage }) => {
    assert.equal(cleanStatusMessage('Still working — 10s elapsed'), 'Still working')
    assert.equal(cleanStatusMessage('Still working - 10 seconds elapsed'), 'Still working')
    assert.equal(cleanStatusMessage('Still working (10s elapsed)'), 'Still working')
    assert.equal(cleanStatusMessage('- 10 seconds elapsed'), '')
    assert.equal(cleanStatusMessage('Resolving area of interest...'), 'Resolving area of interest...')
    assert.equal(cleanStatusMessage('Checking coverage — 14 granules...'), 'Checking coverage — 14 granules...')
  })
})

test('a stage change does not retain detail from an earlier stage', () => {
  const withDetail = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Checking coverage...', stage: 'coverage', detail: 14,
  })
  assert.equal(withDetail.detail, 14)

  const nextStage = applyWorkflowEvent(withDetail, 'status', {
    message: 'Resolving area of interest...', stage: 'aoi',
  })
  assert.equal(nextStage.stage, 'aoi')
  assert.equal(nextStage.detail, null)
})

test('stage working suppresses heartbeat elapsed detail from becoming strip detail', () => {
  const working = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Still working — 10s elapsed', stage: 'working', detail: 10,
  })
  assert.equal(working.stage, 'working')
  assert.equal(working.label, 'Still working')
  assert.equal(working.detail, null)
})

// T64 Phase 4: narration lanes. Sub-agents the supervisor runs in parallel
// narrate into one stream; an event carrying a lane updates that lane's own
// row instead of overwriting the single strip.

const NJ = { lane: '1.0', lane_label: 'Satellite · NO2 · the OTC' }
const GROUND = { lane: '1.1', lane_label: 'Ground · O3 · New Jersey' }

test('a laned status fills its own row and keeps the strip active', () => {
  const next = applyWorkflowEvent(INITIAL_WORKFLOW_STATE, 'status', {
    message: 'Searching datasets...', stage: 'search', ...NJ,
  })

  assert.deepEqual(next.lanes, [{
    lane: '1.0', label: 'Satellite · NO2 · the OTC', stage: 'search',
    message: 'Searching datasets...', detail: null, done: false,
  }])
  assert.equal(next.active, true)
})

test('lane rows keep first-appearance order, update in place, and finish as done', () => {
  let state = INITIAL_WORKFLOW_STATE
  state = applyWorkflowEvent(state, 'status', { message: 'Running', stage: 'running', ...NJ })
  state = applyWorkflowEvent(state, 'status', { message: 'Running', stage: 'running', ...GROUND })
  state = applyWorkflowEvent(state, 'status', { message: 'Checking coverage...', stage: 'coverage', ...NJ })
  state = applyWorkflowEvent(state, 'status', { message: 'Done', stage: 'finished', ...GROUND })

  assert.deepEqual(
    state.lanes.map(row => [row.lane, row.stage, row.message, row.done]),
    [['1.0', 'coverage', 'Checking coverage...', false], ['1.1', 'finished', 'Done', true]],
  )
})

test('laned job progress sets that lane\'s detail only, and the turn\'s done clears every lane', () => {
  let state = INITIAL_WORKFLOW_STATE
  state = applyWorkflowEvent(state, 'status', { message: 'Retrieving data...', stage: 'progress', ...NJ })
  state = applyWorkflowEvent(state, 'status', { message: 'Running', stage: 'running', ...GROUND })
  state = applyWorkflowEvent(state, 'job_progress', { job_handle: 'job_1', progress: 40, ...NJ })

  assert.deepEqual(state.lanes.map(row => row.detail), [40, null])
  assert.equal(state.detail, null)

  assert.deepEqual(applyWorkflowEvent(state, 'done', {}).lanes, [])
})

test('an error during a laned turn names the lane that was still running and its stage', () => {
  let state = INITIAL_WORKFLOW_STATE
  state = applyWorkflowEvent(state, 'status', { message: 'Done', stage: 'finished', ...GROUND })
  state = applyWorkflowEvent(state, 'status', { message: 'Opening retrieved data...', stage: 'open', ...NJ })
  // The outer turn's own heartbeat is unlaned.
  state = applyWorkflowEvent(state, 'status', { message: 'Still working', stage: 'working' })

  const failed = applyWorkflowEvent(state, 'error', { detail: 'boom' })

  assert.equal(failed.failedStage, 'open')
  assert.equal(failed.failedLane, 'Satellite · NO2 · the OTC')
  assert.equal(failed.active, false)
})
