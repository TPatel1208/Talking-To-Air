import assert from 'node:assert/strict'
import test from 'node:test'
import {
  applyWorkflowEvent, INITIAL_WORKFLOW_STATE, runRowView, stageLabel,
} from '../src/utils/workflowStage.js'

const WORKFLOW_FIRST = {
  message: 'Starting no2_regional_trend',
  stage: 'workflow.no2_regional_trend',
  run_id: 'run-wf-1',
  detail: {
    workflow: 'no2_regional_trend', version: 2, mode: 'validated',
    steps: ['area', 'fetch', 'reduce', 'draw'],
  },
}
const EXPLORE_FIRST = {
  message: 'Exploring', stage: 'explore', run_id: 'run-ex-1',
  detail: { objective: 'Compare ozone and NO2 near the coast', bundle: 'satellite' },
}

function fold(frames, state = INITIAL_WORKFLOW_STATE) {
  return frames.reduce((acc, frame) => applyWorkflowEvent(acc, 'status', frame), state)
}

test('a workflow run opens one row with its name, version and steps', () => {
  const state = fold([WORKFLOW_FIRST])

  assert.equal(state.runs.length, 1)
  assert.equal(state.runs[0].runId, 'run-wf-1')
  assert.equal(state.runs[0].kind, 'workflow')
  assert.equal(state.runs[0].name, 'no2_regional_trend')
  assert.equal(state.runs[0].version, 2)
  assert.deepEqual(state.runs[0].steps, ['area', 'fetch', 'reduce', 'draw'])
  assert.equal(state.active, true)
  // The single strip is untouched by a run's frames.
  assert.equal(state.stage, null)
})

test('later frames of a run move its position through the steps', () => {
  const state = fold([
    WORKFLOW_FIRST,
    { message: 'Retrieving satellite data', stage: 'sources.satellite.get_data', run_id: 'run-wf-1', detail: { step: 'fetch' } },
  ])

  assert.equal(state.runs.length, 1)
  assert.equal(state.runs[0].step, 'fetch')
  const view = runRowView(state.runs[0])
  assert.equal(view.title, 'No2 regional trend')
  assert.equal(view.position, 'Step 2 of 4')
  assert.equal(view.activity, 'Retrieving satellite data')
})

test('a workflow row with no step yet shows how many steps it has', () => {
  const view = runRowView(fold([WORKFLOW_FIRST]).runs[0])
  assert.equal(view.position, '4 steps')
})

test('an exploration row shows Exploring and its objective', () => {
  const state = fold([
    EXPLORE_FIRST,
    { message: 'Searching', stage: 'sources.satellite.search', run_id: 'run-ex-1' },
  ])

  assert.equal(state.runs.length, 1)
  const view = runRowView(state.runs[0])
  assert.equal(view.title, 'Exploring')
  assert.equal(view.activity, 'Compare ozone and NO2 near the coast')
  assert.equal(view.current, 'Searching datasets')
})

test('runs keep first-appearance order and are keyed by run_id', () => {
  const state = fold([
    WORKFLOW_FIRST,
    EXPLORE_FIRST,
    { message: 'Drawing', stage: 'render.map', run_id: 'run-wf-1', detail: { step: 'draw' } },
  ])

  assert.deepEqual(state.runs.map(row => row.runId), ['run-wf-1', 'run-ex-1'])
  assert.equal(state.runs[0].step, 'draw')
})

test('a run id nested in detail still keys the row', () => {
  const state = fold([{ message: 'Resolving', stage: 'sources.areas.resolve', detail: { run_id: 'run-nested' } }])
  assert.equal(state.runs[0].runId, 'run-nested')
})

test('frames without a run id keep driving the single strip', () => {
  const state = fold([{ message: 'Still working — 10s elapsed', stage: 'working', detail: 10 }])

  assert.deepEqual(state.runs, [])
  assert.equal(state.stage, 'working')
})

test('a finished run is marked done and the turn done clears every row', () => {
  const finished = fold([WORKFLOW_FIRST, { message: 'Done', stage: 'finished', run_id: 'run-wf-1' }])
  assert.equal(finished.runs[0].done, true)

  assert.deepEqual(applyWorkflowEvent(finished, 'done', {}).runs, [])
})

test('an error names the run that was still going', () => {
  const state = fold([
    WORKFLOW_FIRST,
    { message: 'Retrieving satellite data', stage: 'sources.satellite.get_data', run_id: 'run-wf-1', detail: { step: 'fetch' } },
  ])
  const failed = applyWorkflowEvent(state, 'error', { detail: 'boom' })

  assert.equal(failed.failedStage, 'sources.satellite.get_data')
  assert.equal(failed.failedLane, 'No2 regional trend')
  assert.equal(failed.active, false)
})

test('capability and workflow stages have short labels; unknown ones fall back to the message', () => {
  assert.equal(stageLabel('sources.satellite.get_data'), 'Retrieving satellite data')
  assert.equal(stageLabel('render.map'), 'Drawing a map')
  assert.equal(stageLabel('render.timeseries'), 'Drawing a time series')
  assert.equal(stageLabel('workflow.no2_regional_trend'), 'Workflow: No2 regional trend')
  assert.equal(stageLabel('sources.monitors.daily_summary'), 'Fetching daily monitor summaries')
  assert.equal(stageLabel('something.new'), null)

  const state = fold([{ message: 'Doing a new thing...', stage: 'something.new', run_id: 'run-x' }])
  assert.equal(runRowView(state.runs[0]).title, 'Doing a new thing...')
})

test('the finished frame for a run closes only that row and keeps the status it ended in', () => {
  const state = fold([
    WORKFLOW_FIRST,
    EXPLORE_FIRST,
    { message: 'Failed', stage: 'finished', run_id: 'run-ex-1', detail: { status: 'failed' } },
  ])

  const [workflow, explore] = state.runs
  assert.equal(workflow.done, false)
  assert.equal(explore.done, true)
  assert.equal(explore.status, 'failed')
  const done = fold([{ message: 'Done', stage: 'finished', run_id: 'run-wf-1', detail: { status: 'done' } }], state)
  assert.equal(done.runs[0].done, true)
  assert.equal(done.runs[0].status, 'done')
})

test('an error points at the run the results say failed', () => {
  const state = fold([
    WORKFLOW_FIRST,
    EXPLORE_FIRST,
    { message: 'Searching', stage: 'sources.satellite.search', run_id: 'run-ex-1' },
  ])
  const failed = applyWorkflowEvent(state, 'error', { results: [{ run_id: 'run-wf-1', status: 'failed' }] })

  assert.equal(failed.failedLane, 'No2 regional trend')
  assert.equal(failed.failedStage, 'workflow.no2_regional_trend')
})

test('an error points at the run whose own closing frame said it failed', () => {
  const state = fold([
    WORKFLOW_FIRST,
    { message: 'Retrieving satellite data', stage: 'sources.satellite.get_data', run_id: 'run-wf-1', detail: { step: 'fetch' } },
    EXPLORE_FIRST,
    { message: 'Failed', stage: 'finished', run_id: 'run-wf-1', detail: { status: 'failed' } },
  ])
  const failed = applyWorkflowEvent(state, 'error', {})

  assert.equal(failed.failedLane, 'No2 regional trend')
  assert.equal(failed.failedStage, 'sources.satellite.get_data')
})

test('without a failed run named, an error points at the most recently started run still going', () => {
  const state = fold([
    { ...EXPLORE_FIRST, run_id: 'run-ex-queued', detail: { objective: 'Later task', bundle: 'analyse', queued: true } },
    WORKFLOW_FIRST,
    EXPLORE_FIRST,
    { message: 'Searching', stage: 'sources.satellite.search', run_id: 'run-ex-1' },
  ])
  const failed = applyWorkflowEvent(state, 'error', {})

  assert.equal(failed.failedLane, 'Exploring')
  assert.equal(failed.failedStage, 'sources.satellite.search')
})

test('a queued exploration task waits under its objective until its start frame', () => {
  const queued = fold([
    { message: 'Waiting to explore: Map smoke', stage: 'explore', run_id: 'run-q',
      detail: { objective: 'Map smoke', bundle: 'analyse', queued: true } },
  ])
  assert.equal(queued.runs[0].queued, true)
  assert.equal(runRowView(queued.runs[0]).title, 'Waiting to explore: Map smoke')

  const started = fold([{ message: 'Exploring', stage: 'explore', run_id: 'run-q',
    detail: { objective: 'Map smoke', bundle: 'analyse' } }], queued)
  assert.equal(started.runs[0].queued, false)
  const view = runRowView(started.runs[0])
  assert.equal(view.title, 'Exploring')
  assert.equal(view.activity, 'Map smoke')
})
