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
