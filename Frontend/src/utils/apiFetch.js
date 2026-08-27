// Every authenticated request in the app goes through here.
//
// The point is that the token is read at call time rather than captured. A
// token threaded through props is a snapshot of whenever the tree last
// rendered; supabase-js rotates the real one on its own schedule, and the gap
// between those two is a 401 the user did nothing to earn.
//
// It depends on an injected `auth` rather than reaching for the Supabase client
// itself: the only thing it needs is somewhere to ask for a session and
// somewhere to ask for a fresh one. Keeping that seam narrow is also what makes
// it testable in a repo with no jsdom.
import { shouldPromptReauth } from './sessionExpiry.js'
import { isUnreachable } from './authSession.js'

let auth = null
let unauthorized = null
let lastToken = null

export function configureApiFetch(config) {
  auth = config.auth
  unauthorized = config.onUnauthorized ?? null
}

// A bearer header, or nothing. Never `Bearer undefined`: an absent credential
// and a malformed one are different things to the server.
function bearer(session) {
  return session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {}
}

function send(path, options, session) {
  // One writer for the snapshot, so a request and a session event cannot
  // disagree about how a token is recorded.
  noteSession(session)
  return fetch(path, {
    ...options,
    headers: { ...options.headers, ...bearer(session) },
  })
}

/**
 * Record the session supabase-js just announced.
 *
 * Requests alone are not enough to keep the snapshot below correct. supabase-js
 * refreshes on its own timer, at EXPIRY_MARGIN_MS -- 90 seconds before expiry
 * -- so the token a rotation replaces is dead almost immediately, while the
 * snapshot would still be handing it out until something happened to issue a
 * request. An idle app issues none: useJobs polls only while a job is
 * progressing. Feeding this from onAuthStateChange makes the snapshot exactly
 * as current as supabase-js itself.
 *
 * A null session clears it. No header is the honest answer once there is no
 * session, and better than presenting a credential the server has stopped
 * accepting.
 */
export function noteSession(session) {
  lastToken = session?.access_token ?? null
}

/**
 * The most recent access token, or null.
 *
 * For the one caller that cannot await a session: maplibre's `transformRequest`
 * is synchronous. Kept current by every request and by every session change,
 * so it trails the live session by nothing. Anything that can await should call
 * apiFetch instead.
 */
export function currentAccessToken() {
  return lastToken
}

export async function apiFetch(path, options = {}) {
  if (!auth) {
    throw new Error(`apiFetch(${path}) ran before configureApiFetch(); App must configure it first.`)
  }
  const { data } = await auth.getSession()
  const res = await send(path, options, data?.session)
  if (!shouldPromptReauth(res.status)) return res

  // The token was stale rather than the session dead -- the common case for a
  // tab that sat in the background while its refresh timer was throttled. Ask
  // for a fresh one and try the same request again.
  const { data: refreshed, error } = await auth.refreshSession()

  // Told no, or unable to ask? supabase-js reports both as session: null, and
  // they mean opposite things. The browser talks to supabase.co directly while
  // these requests go to our own origin (decision 6), so the auth host can be
  // briefly unreachable while the backend is fine -- and the session is then
  // still perfectly alive. Retrying it without a credential could only 401
  // again, the retry would wipe the snapshot maplibre reads on its way past,
  // and the modal would tell a signed-in user their session had expired. Hand
  // the original 401 back and let the caller treat it as the transient failure
  // it is; loadHistory already keeps the messages on one.
  if (isUnreachable(error)) return res

  const retried = await send(path, options, refreshed?.session)

  // A second 401 is the refresh token itself being gone, which is the only
  // thing T47's modal is for. Every caller reaches it from here, so a lapsed
  // session during a jobs poll or a CSV download is no longer a silent failure.
  if (shouldPromptReauth(retried.status)) unauthorized?.()
  return retried
}
