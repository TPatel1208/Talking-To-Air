// The runs a conversation has paused, as the runs panel lists them: rows of
// `GET /runs` ({run_id, mode, status, started, paused, reason, question |
// ready}). Pure selectors, so which buttons a row shows is unit-tested.
import { modeBadge } from './resultMode.js'
import { normalizeQuestion } from './pausedQuestion.js'

// The text Continue sends as the user's message.
export const CONTINUE_MESSAGE = 'Continue the paused run'

const STATUSES = new Set(['needs_input', 'waiting'])

export function normalizeRuns(payload) {
  const rows = Array.isArray(payload?.runs) ? payload.runs : []
  return rows.filter(run => run && typeof run.run_id === 'string' && STATUSES.has(run.status))
}

export function runStatusLabel(run) {
  if (run.status === 'needs_input') return 'Needs your answer'
  return run.ready ? 'Ready to continue' : 'Waiting on a retrieval'
}

export function runBadge(run) {
  return modeBadge(run.mode)
}

// The question Answer opens, or null when the app cannot answer this run.
export function runQuestion(run) {
  return run.status === 'needs_input' ? normalizeQuestion(run.question) : null
}

// Which buttons a row shows. Every paused run can be cancelled; Continue is
// for a waiting run whose retrieval jobs are finished; Answer for a run on a
// question the app can answer.
export function runActions(run) {
  return {
    continue: run.status === 'waiting' && run.ready === true,
    answer: runQuestion(run) !== null,
    cancel: true,
  }
}

// The body fields POST /chat takes to continue the run.
export function continueRequest(run) {
  return { message: CONTINUE_MESSAGE, resume: { run_id: run.run_id } }
}

export function withoutRun(runs, runId) {
  return runs.filter(run => run.run_id !== runId)
}

// The request fields a send carries for a paused run: one answer or one resume.
export function pausedRunFields(request) {
  if (request?.answer) return { answer: request.answer }
  if (request?.resume) return { resume: request.resume }
  return {}
}
