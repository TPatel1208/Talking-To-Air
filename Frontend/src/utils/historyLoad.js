// Pure state transitions for `/session/{id}/history` load outcomes (T41).
// Kept side-effect-free so useChat's "keep messages on a transient failure,
// only clear on a genuine 404" logic is unit-testable without a fetch mock.

export const HISTORY_ERROR_MESSAGE = 'Could not load this conversation. Please retry.'

// A 404 means the session itself is gone -- clearing the view is correct.
// Anything else (5xx, or a network exception with no status at all) is
// transient: the backend/connection is having a bad moment, not "this
// conversation never existed."
export function classifyHistoryFetchFailure(status) {
  return status === 404 ? 'not-found' : 'error'
}

// prevState: { messages, historyError }
// action:
//   { type: 'loaded', messages }              -- fetch succeeded
//   { type: 'not-found' }                      -- 404: session is really gone
//   { type: 'failed', message? }                -- transient failure: keep messages
//   { type: 'retry' }                           -- user clicked retry: clear the error
export function historyStateReducer(prevState, action) {
  switch (action.type) {
    case 'loaded':
      return { messages: action.messages, historyError: null }
    case 'not-found':
      return { messages: [], historyError: null }
    case 'failed':
      return { messages: prevState.messages, historyError: action.message || HISTORY_ERROR_MESSAGE }
    case 'retry':
      return { ...prevState, historyError: null }
    default:
      return prevState
  }
}
