// Pure state transitions for the "session expired mid-analysis" flow (T47).
// Kept side-effect-free so App/useChat's 401 handling is unit-testable without
// a fetch mock, and so the strict react-hooks lint is satisfied by driving all
// of this from event handlers rather than effects.
//
// The doctrine T47 fixes: a mid-session auth failure used to hard-log-out the
// researcher (token wiped, dropped to a blank login screen, 40 minutes of
// context gone). Instead, a 401 raises an in-place "sign in to continue" modal
// over the preserved view; on re-login we resume the same thread. An auth
// failure is recoverable, not a dead end.

// A 401 is the only status that means "your login lapsed, sign in again." A
// 404 is an honest empty state, a 5xx / network blip is transient (T41) — none
// of those should prompt re-auth.
export function shouldPromptReauth(status) {
  return status === 401
}

// prevState: { sessionExpired: boolean }
// action:
//   { type: 'unauthorized' }     -- a 401 arrived: raise the re-auth modal
//   { type: 'reauthenticated' }  -- re-login succeeded: close the modal, resume
export function sessionExpiryReducer(prevState, action) {
  switch (action.type) {
    case 'unauthorized':
      return { sessionExpired: true }
    case 'reauthenticated':
      return { sessionExpired: false }
    default:
      return prevState
  }
}

// The one storage difference between a fresh login and an in-place re-auth: a
// fresh login starts clean (drop any stale active thread), but a re-auth must
// preserve the active thread so the researcher lands back on the exact
// conversation they were in, its history reloaded via T41's loadHistory.
export function authTransition(kind) {
  return { clearActiveThread: kind !== 'reauth' }
}
