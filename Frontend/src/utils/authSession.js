import { sessionExpiryReducer } from './sessionExpiry.js'

// The auth session lifecycle as a pure state machine.
//
// It lives outside App.jsx because this repo has no jsdom: logic left inside a
// component is verifiable only by reading its source. Everything here is
// decidable without a DOM, so it is testable as behavior.

/**
 * Translate the GET /config/auth body into the config the client is built
 * from. Decision 11: served at runtime rather than baked into the bundle, so
 * one image runs against either the dev or the prod Supabase project.
 *
 * A rename of the wire names, nothing more -- an absent field is caught and
 * named at the client seam by ensureSupabaseClient, which is the last point
 * before the value would reach supabase-js.
 */
export function readAuthConfig(body) {
  return {
    supabaseUrl: body?.supabase_url,
    publishableKey: body?.supabase_publishable_key,
  }
}

export const initialAuthState = {
  config: null,
  configError: null,
  // supabase-js restores any stored session asynchronously and reports the
  // outcome through INITIAL_SESSION. Until that lands we know nothing, which
  // is a different state from "signed out".
  bootstrapped: false,
  session: null,
  // Whether an authenticated tree is, or has been, mounted. Distinct from
  // holding a session: it is what keeps the view on 'app' when a session is
  // lost involuntarily.
  hadSession: false,
  // Owned by T47's sessionExpiryReducer, which this machine composes rather
  // than reimplements -- one definition of "a lapsed session raises the modal
  // and never reads as a logout".
  sessionExpired: false,
}

function withExpiry(state, action) {
  return { ...state, ...sessionExpiryReducer({ sessionExpired: state.sessionExpired }, action) }
}

export function authReducer(state, action) {
  switch (action.type) {
    case 'config-loaded':
      return { ...state, config: action.config, configError: null }
    case 'config-failed':
      return { ...state, config: null, configError: action.error }
    case 'auth-event':
      return applyAuthEvent(state, action)
    // Cleared eagerly rather than waiting on signOut()'s SIGNED_OUT, so
    // logout is instant and unconditional in the UI even when the network
    // call revoking the refresh token fails.
    case 'logout-requested':
      return { ...state, session: null, hadSession: false, sessionExpired: false }
    // A 401 from any request, and the modal's own sign-in resolving. Both are
    // T47's, and both are delegated so the rule has one definition. Note that
    // nothing supabase-js emits closes the modal: SIGNED_IN is re-emitted when
    // a backgrounded tab regains focus, so it does not mean the user signed
    // in, and acting on it would wipe a half-typed password off the screen.
    case 'unauthorized':
    case 'reauthenticated':
      return withExpiry(state, action)
    default:
      return state
  }
}

// supabase-js announces every session change through one callback. Only a
// handful of its events change anything here.
function applyAuthEvent(state, { event, session }) {
  // ANY event bootstraps us, not just INITIAL_SESSION. supabase-js does emit
  // INITIAL_SESSION first on subscribe today, but hanging the restoring
  // spinner on that ordering means a session arriving by any other route sits
  // behind a spinner while we hold a valid token for it.
  const heard = { ...state, bootstrapped: true }

  if (event === 'SIGNED_OUT') {
    // A deliberate logout has already cleared hadSession, so reaching here
    // still holding one means the session went away on its own -- the refresh
    // token lapsed, or the account was banned. That is a T47 unauthorized, and
    // the session is deliberately left in place so the view survives it.
    return state.hadSession ? withExpiry(heard, { type: 'unauthorized' }) : heard
  }
  if (!session) return heard
  return { ...heard, session, hadSession: true }
}

/** Which screen the app owes the user right now. */
export function authView(state) {
  if (state.configError) return 'config-error'
  if (!state.config) return 'config-loading'
  if (!state.bootstrapped) return 'restoring'
  return state.hadSession ? 'app' : 'login'
}

/** Whether T47's in-place "sign in to continue" modal is up. */
export function showReauthModal(state) {
  return state.sessionExpired
}

/** The bearer token to send with requests, or null if there is none. */
export function accessTokenOf(state) {
  return state.session?.access_token ?? null
}

// The stable half of a session. App.jsx keys the authenticated tree on this
// rather than on the access token, which auto-refresh rotates on a timer.
export function userIdOf(state) {
  return state.session?.user?.id ?? null
}

/**
 * One line of copy for a failed sign-in. supabase-js rejects with an
 * AuthApiError whose `message` is written for a developer reading a console,
 * so it is mapped rather than shown.
 */
export function describeAuthError(error) {
  const code = error?.code
  const status = error?.status

  if (code === 'invalid_credentials' || status === 400) {
    // Deliberately does not say which half was wrong -- that distinction is an
    // account-enumeration oracle, and it does not help someone who mistyped.
    return 'Invalid email or password.'
  }
  if (code === 'email_not_confirmed') {
    return 'This account has not been confirmed yet. Ask an administrator to confirm it.'
  }
  if (code === 'over_request_rate_limit' || status === 429) {
    return 'Too many sign-in attempts. Wait a moment and try again.'
  }
  // Decision 6 has the browser talking to supabase.co directly, so a failure
  // carrying no real HTTP status means the request never arrived -- supabase-js
  // spells that 0 (AuthRetryableFetchError) or leaves it absent. Reporting it
  // as a bad password sends the user off to reset one that works.
  if (!status) {
    return 'Could not reach the sign-in service. Check your connection and try again.'
  }
  return error?.message || 'Sign-in failed. Try again.'
}
