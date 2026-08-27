import test from 'node:test'
import assert from 'node:assert/strict'

import {
  readAuthConfig,
  initialAuthState,
  authReducer,
  authView,
  accessTokenOf,
  userIdOf,
  showReauthModal,
  describeAuthError,
} from '../src/utils/authSession.js'

const reduce = (state, ...actions) => actions.reduce(authReducer, state)
const sessionFor = (token, id = 'user-uuid-1') => ({
  access_token: token,
  user: { id, email: 'researcher@example.gov' },
})
const configured = () => reduce(initialAuthState, { type: 'config-loaded', config: CONFIG })

const CONFIG = { supabaseUrl: 'https://project-ref.supabase.co', publishableKey: 'sb_publishable_test' }

// T61 decision 11: the identity provider's coordinates arrive at runtime from
// GET /config/auth. This pins the wire shape from the client side; the backend
// test in test_chat_endpoint.py pins the same names from the server side.
// Only both halves together turn a rename into a test failure rather than a
// silent login failure in production.
test('readAuthConfig reads the exact shape GET /config/auth serves', () => {
  assert.deepEqual(
    readAuthConfig({
      supabase_url: 'https://project-ref.supabase.co',
      supabase_publishable_key: 'sb_publishable_test',
    }),
    { supabaseUrl: 'https://project-ref.supabase.co', publishableKey: 'sb_publishable_test' },
  )
})

test('nothing is decided until the config arrives', () => {
  // The app cannot show a login screen before it knows where to log in to, so
  // this is a real view and not a placeholder for one.
  assert.equal(authView(initialAuthState), 'config-loading')
})

test('an unreachable /config/auth is a screen the user can act on, not a blank page', () => {
  // T17 degrade-don't-die: the backend being down must not leave the browser
  // showing nothing, with no indication of what failed or what to do.
  const state = reduce(initialAuthState, { type: 'config-failed', error: 'HTTP 503' })
  assert.equal(authView(state), 'config-error')
  assert.equal(state.configError, 'HTTP 503')
})

test('config in hand, the login screen still waits for Supabase to restore the stored session', () => {
  // supabase-js reads localStorage asynchronously and announces the outcome
  // through INITIAL_SESSION. Rendering 'login' in this window is what makes an
  // already-signed-in user watch the sign-in form flash on every reload.
  const state = reduce(initialAuthState, { type: 'config-loaded', config: CONFIG })
  assert.equal(authView(state), 'restoring')
  assert.deepEqual(state.config, CONFIG)
})

test('no stored session means the sign-in screen, and no token to offer', () => {
  const state = reduce(configured(), { type: 'auth-event', event: 'INITIAL_SESSION', session: null })
  assert.equal(authView(state), 'login')
  assert.equal(accessTokenOf(state), null)
})

test('a restored session goes straight into the app with its token', () => {
  const state = reduce(configured(), {
    type: 'auth-event', event: 'INITIAL_SESSION', session: sessionFor('jwt-restored'),
  })
  assert.equal(authView(state), 'app')
  assert.equal(accessTokenOf(state), 'jwt-restored')
})

test('signing in from the login screen enters the app', () => {
  const state = reduce(
    configured(),
    { type: 'auth-event', event: 'INITIAL_SESSION', session: null },
    { type: 'auth-event', event: 'SIGNED_IN', session: sessionFor('jwt-fresh') },
  )
  assert.equal(authView(state), 'app')
  assert.equal(accessTokenOf(state), 'jwt-fresh')
})

test('a background refresh swaps the token but leaves the user identity alone', () => {
  // App.jsx keys the authenticated tree on the user id for exactly this
  // reason. Keyed on the token, an auto-refresh -- which happens on a timer,
  // every 45 minutes, decision 10 -- would change the key and make React throw
  // the whole tree away mid-analysis: messages, focused output, compare, map,
  // jobs, discovery. That is the T47 rug-pull arriving on a schedule.
  const signedIn = reduce(configured(), {
    type: 'auth-event', event: 'SIGNED_IN', session: sessionFor('jwt-first'),
  })
  const refreshed = reduce(signedIn, {
    type: 'auth-event', event: 'TOKEN_REFRESHED', session: sessionFor('jwt-second'),
  })

  assert.equal(accessTokenOf(refreshed), 'jwt-second')
  assert.equal(userIdOf(refreshed), userIdOf(signedIn))
  assert.equal(authView(refreshed), 'app')
})

test('losing the session mid-analysis raises the re-auth modal over the preserved view', () => {
  // What T47 exists to prevent: a mid-session auth failure used to wipe the
  // token and drop the researcher on a blank login screen, 40 minutes of
  // context gone. The refresh token lapsing is the same event by another name.
  const state = reduce(
    configured(),
    { type: 'auth-event', event: 'SIGNED_IN', session: sessionFor('jwt-live') },
    { type: 'auth-event', event: 'SIGNED_OUT', session: null },
  )

  assert.equal(authView(state), 'app')
  assert.equal(showReauthModal(state), true)
  // The token is dead either way -- but nulling it sends useChat and useJobs
  // down their `if (!accessToken)` branches, which call setSessions([]) and
  // setJobs([]). The sidebar and jobs panel would blank out UNDERNEATH the
  // modal, which is the rug-pull returning by a new route. Keeping the stale
  // token is what preserves the view.
  assert.equal(accessTokenOf(state), 'jwt-live')
})

test('signing out deliberately returns to the login screen, carrying nothing over', () => {
  const state = reduce(
    configured(),
    { type: 'auth-event', event: 'SIGNED_IN', session: sessionFor('jwt-live') },
    { type: 'logout-requested' },
  )

  assert.equal(authView(state), 'login')
  assert.equal(accessTokenOf(state), null)
  // Signing out is not a session failure. Offering "sign in to continue" over
  // a view the user just chose to leave would be nonsense.
  assert.equal(showReauthModal(state), false)
})

test("the SIGNED_OUT that follows a deliberate logout does not raise the modal", () => {
  // signOut() fires SIGNED_OUT a moment after the click. The state is cleared
  // eagerly so logout is instant even when the network call fails, which means
  // this event always arrives at an already-signed-out state. hadSession being
  // false is what distinguishes it -- no separate intent flag needed.
  const state = reduce(
    configured(),
    { type: 'auth-event', event: 'SIGNED_IN', session: sessionFor('jwt-live') },
    { type: 'logout-requested' },
    { type: 'auth-event', event: 'SIGNED_OUT', session: null },
  )

  assert.equal(authView(state), 'login')
  assert.equal(showReauthModal(state), false)
})

const signedIn = (token = 'jwt-live') => reduce(
  configured(),
  { type: 'auth-event', event: 'SIGNED_IN', session: sessionFor(token) },
)

test('a 401 from any request raises the modal without disturbing the view', () => {
  const state = reduce(signedIn(), { type: 'unauthorized' })
  assert.equal(showReauthModal(state), true)
  assert.equal(authView(state), 'app')
  assert.equal(accessTokenOf(state), 'jwt-live')
})

test('a successful re-login closes the modal and resumes on the new token', () => {
  const state = reduce(
    signedIn(),
    { type: 'unauthorized' },
    { type: 'auth-event', event: 'SIGNED_IN', session: sessionFor('jwt-reauth') },
    { type: 'reauthenticated' },
  )
  assert.equal(showReauthModal(state), false)
  assert.equal(accessTokenOf(state), 'jwt-reauth')
  assert.equal(authView(state), 'app')
})

test('a stray SIGNED_IN does not close the modal on its own', () => {
  // supabase-js v2 re-emits SIGNED_IN when a backgrounded tab regains focus
  // and refreshes, so the event does not mean "the user just signed in". Only
  // the modal's own signInWithPassword resolving dispatches 'reauthenticated'.
  // Dismissing the modal on the event would let a tab-focus wipe a
  // half-typed password off the screen.
  const state = reduce(
    signedIn(),
    { type: 'unauthorized' },
    { type: 'auth-event', event: 'SIGNED_IN', session: sessionFor('jwt-refocus') },
  )
  assert.equal(showReauthModal(state), true)
})

test('a background refresh does not close the modal either', () => {
  const state = reduce(
    signedIn(),
    { type: 'unauthorized' },
    { type: 'auth-event', event: 'TOKEN_REFRESHED', session: sessionFor('jwt-refreshed') },
  )
  assert.equal(showReauthModal(state), true)
})

test('an unrecognised action changes nothing at all', () => {
  const before = signedIn()
  assert.equal(authReducer(before, { type: 'something-else' }), before)
})

// supabase-js rejects with an AuthApiError carrying `code`, `status` and a
// developer-facing `message`. The login form shows one line, so the mapping is
// pinned here rather than left to whatever string the library happens to send.
test('a rejected password says so, without leaking which half was wrong', () => {
  const err = Object.assign(new Error('Invalid login credentials'), {
    code: 'invalid_credentials', status: 400,
  })
  assert.equal(describeAuthError(err), 'Invalid email or password.')
})

test('an unreachable auth service is not reported as a bad password', () => {
  // Decision 6 puts the browser on supabase.co directly, so this is a distinct
  // and reachable failure. Telling the user their password is wrong when the
  // network never answered sends them to reset a password that works fine.
  const err = new TypeError('Failed to fetch')
  const message = describeAuthError(err)
  assert.match(message, /connection|reach/i)
  assert.doesNotMatch(message, /password/i)
})

test('a retryable fetch failure is a network failure, status 0 and all', () => {
  // supabase-js wraps a dead connection in AuthRetryableFetchError, which
  // carries status 0 rather than no status at all. Testing only for an absent
  // status would let this fall through to the raw library message.
  const err = Object.assign(new Error('Failed to fetch'), { status: 0 })
  assert.match(describeAuthError(err), /connection|reach/i)
})

test('rate limiting tells the user to wait rather than to retype', () => {
  const err = Object.assign(new Error('Request rate limit reached'), {
    code: 'over_request_rate_limit', status: 429,
  })
  assert.match(describeAuthError(err), /moment|wait/i)
})

test('an unmapped failure still says something, never nothing', () => {
  assert.ok(describeAuthError(Object.assign(new Error(''), { status: 500 })).length > 0)
  assert.ok(describeAuthError(undefined).length > 0)
})
