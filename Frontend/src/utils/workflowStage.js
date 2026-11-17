// Pure state machine driving the chat's workflow strip. Takes the same
// SSE vocabulary useChat.js already switches on (status/job_progress/text/
// error/done) — no new event names — and folds each event into the small
// bit of state the strip renders: current stage, its human label, an
// optional detail (granule count / percent), whether the strip is active,
// and the stage a failure happened at (frozen once set, until the next
// turn's 'done'/new message resets it).

export const INITIAL_WORKFLOW_STATE = {
  stage: null,
  label: '',
  detail: null,
  failedStage: null,
  // The lane label that was running when a laned turn failed.
  failedLane: null,
  active: false,
  // One row per sub-agent the supervisor runs in parallel,
  // in order of first appearance. Events carrying a `lane` land here.
  lanes: [],
  // One row per workflow run or exploration task, keyed by run_id, in order
  // of first appearance.
  runs: [],
}

const STAGE_LABELS = {
  search: 'Searching',
  aoi: 'Resolving area',
  coverage: 'Checking coverage',
  estimate: 'Estimating size',
  submit: 'Submitting retrieval',
  progress: 'Retrieving data',
  open: 'Opening data',
  render: 'Rendering',
  working: 'Still working',
  queued: 'Waiting for a free slot',
  running: 'Running',
  finished: 'Done',
  explore: 'Exploring',
  'sources.areas.resolve': 'Resolving the area',
  'sources.monitors.find': 'Finding monitors',
  'sources.monitors.readings': 'Fetching monitor readings',
  'sources.monitors.exceedance_days': 'Finding exceedance days',
  'sources.satellite.search': 'Searching datasets',
  'sources.satellite.describe': 'Describing a product',
  'sources.satellite.check_availability': 'Checking availability',
  'sources.satellite.get_data': 'Retrieving satellite data',
  'sources.satellite.get_point_series': 'Retrieving a point series',
  'sources.smoke.get_extent': 'Retrieving smoke extent',
  'operations.aggregate_time': 'Reducing over time',
  'operations.daily_peak': 'Finding daily peaks',
  'operations.difference': 'Differencing two grids',
  'operations.regional_series': 'Computing a regional series',
  'operations.statistic': 'Computing a statistic',
  'operations.threshold': 'Checking against a threshold',
  'render.map': 'Drawing a map',
  'render.monitor_map': 'Drawing a monitor map',
  'render.polygon_map': 'Drawing a polygon map',
  'render.profile': 'Drawing a vertical profile',
  'render.table': 'Preparing a table',
  'render.timeseries': 'Drawing a time series',
  'explain.citations': 'Looking up citations',
  'explain.lineage': 'Tracing how results were made',
  'explain.reliability': 'Reading reliability facts',
}

const WORKFLOW_PREFIX = 'workflow.'

function humanize(name) {
  const text = String(name || '').replace(/[_-]+/g, ' ').trim()
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : ''
}

// A short readable label for a stage name, or null when it is not one we know.
export function stageLabel(stage) {
  if (!stage || typeof stage !== 'string') return null
  if (STAGE_LABELS[stage]) return STAGE_LABELS[stage]
  if (stage.startsWith(WORKFLOW_PREFIX)) return `Workflow: ${humanize(stage.slice(WORKFLOW_PREFIX.length))}`
  const summary = /^sources\.monitors\.(\w+)_summary$/.exec(stage)
  if (summary) return `Fetching ${summary[1]} monitor summaries`
  return null
}

// The run a status frame belongs to; null on the turn's own frames.
export function runIdOf(data) {
  if (typeof data?.run_id === 'string' && data.run_id) return data.run_id
  const nested = data?.detail && typeof data.detail === 'object' ? data.detail.run_id : null
  return typeof nested === 'string' && nested ? nested : null
}

function updateRun(runs, data) {
  const runId = runIdOf(data)
  const existing = runs.find(row => row.runId === runId)
  const row = existing || {
    runId, kind: 'task', name: null, version: null, mode: null, objective: null,
    steps: [], step: null, stage: null, message: '', done: false,
  }
  const detail = data.detail && typeof data.detail === 'object' ? data.detail : {}
  const next = { ...row }
  if (typeof data.stage === 'string' && data.stage.startsWith(WORKFLOW_PREFIX)) {
    next.kind = 'workflow'
    next.name = detail.workflow ?? data.stage.slice(WORKFLOW_PREFIX.length)
    next.version = detail.version ?? row.version
    next.mode = detail.mode ?? row.mode
    next.steps = Array.isArray(detail.steps) ? detail.steps : row.steps
  } else if (data.stage === 'explore') {
    next.kind = 'explore'
    next.objective = detail.objective ?? row.objective
  }
  if (typeof detail.step === 'string') next.step = detail.step
  next.stage = data.stage ?? row.stage
  next.message = data.message ?? row.message
  next.done = data.stage === 'finished' || row.done
  return existing ? runs.map(r => (r === existing ? next : r)) : [...runs, next]
}

// What a run row shows: a title, where it is, and what it is doing now.
export function runRowView(row) {
  const current = stageLabel(row.stage)
  const activity = row.done ? 'Done' : (current || cleanStatusMessage(row.message) || '')
  if (row.kind === 'workflow') {
    const total = row.steps.length
    const index = row.step == null ? -1 : row.steps.indexOf(row.step)
    let position = ''
    if (index >= 0) position = `Step ${index + 1} of ${total}`
    else if (total) position = `${total} steps`
    return { title: humanize(row.name), position, activity: row.stage?.startsWith(WORKFLOW_PREFIX) ? '' : activity }
  }
  if (row.kind === 'explore') {
    return { title: 'Exploring', position: '', activity: row.objective || activity, current: row.stage === 'explore' ? '' : activity }
  }
  return { title: activity || stageLabel(row.stage) || row.stage || '', position: '', activity: '' }
}

// The row for `data.lane`, updated by `change`, appended if it is new.
function updateLane(lanes, data, change) {
  const existing = lanes.find(row => row.lane === data.lane)
  const row = existing || {
    lane: data.lane, label: data.lane_label || '', stage: null, message: '', detail: null, done: false,
  }
  const updated = { ...row, ...change(row) }
  return existing ? lanes.map(r => (r === existing ? updated : r)) : [...lanes, updated]
}

export function cleanStatusMessage(msg) {
  if (!msg || typeof msg !== 'string') return ''
  return msg
    .replace(/\s*[([]?\s*[-—–]?\s*\d+\s*(?:seconds?|secs?|s)\s*elapsed\s*[)\]]?(?:\.\.\.)?\s*/gi, '')
    .trim()
}

export function applyWorkflowEvent(state, event, data) {
  switch (event) {
    case 'status': {
      if (runIdOf(data) && !data?.lane) {
        return { ...state, runs: updateRun(state.runs || [], data), active: true }
      }
      if (data?.lane) {
        const lanes = updateLane(state.lanes || [], data, row => ({
          stage: data.stage ?? row.stage,
          message: data.message ?? row.message,
          detail: 'detail' in data && data.detail != null ? data.detail : row.detail,
          // The executor closes every lane with stage 'finished' (message
          // "Done" or "Failed"); the row stays, so the other lanes still read.
          done: data.stage === 'finished' || row.done,
        }))
        return { ...state, lanes, active: true }
      }
      const message = cleanStatusMessage(data?.message ?? state.label)
      const stage = data?.stage ?? state.stage
      const isNewStage = data?.stage != null && data.stage !== state.stage
      const rawDetail = data && 'detail' in data ? data.detail : null
      const detail = stage === 'working'
        ? null
        : (rawDetail != null ? rawDetail : (isNewStage ? null : state.detail))
      return { ...state, stage, label: message, detail, active: true }
    }
    case 'job_progress': {
      if (data?.lane && typeof data.progress === 'number') {
        return { ...state, lanes: updateLane(state.lanes || [], data, () => ({ detail: data.progress })), active: true }
      }
      if (data && typeof data.progress === 'number') {
        return { ...state, detail: data.progress, active: true }
      }
      return state
    }
    case 'text':
      // Collapses cleanly the moment the first answer token arrives — the
      // last stage/label are preserved (not cleared) so a strip that
      // re-expands later (e.g. a second tool call) has context, but
      // `active: false` is what the strip actually reads to hide itself.
      return { ...state, active: false }
    case 'error': {
      // On a laned turn the single-strip stage is only the outer heartbeat;
      // what failed is the lane still running.
      const running = (state.lanes || []).find(row => !row.done)
      if (running) {
        return { ...state, failedStage: running.stage, failedLane: running.label, active: false }
      }
      const runningRun = (state.runs || []).find(row => !row.done)
      if (runningRun) {
        const { title } = runRowView(runningRun)
        return { ...state, failedStage: runningRun.stage, failedLane: title, active: false }
      }
      return { ...state, failedStage: state.stage, active: false }
    }
    case 'done':
      return { ...INITIAL_WORKFLOW_STATE }
    default:
      return state
  }
}
