// T19: pure state machine driving the chat's workflow strip. Takes the same
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
  // T64 Phase 4: one row per sub-agent the supervisor runs in parallel,
  // in order of first appearance. Events carrying a `lane` land here.
  lanes: [],
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
      const message = cleanStatusMessage(data?.message ?? state.label)
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
      const message = data?.message ?? state.label
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
      return { ...state, failedStage: state.stage, active: false }
    }
    case 'done':
      return { ...INITIAL_WORKFLOW_STATE }
    default:
      return state
  }
}
