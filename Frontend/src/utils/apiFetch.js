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
  lastToken = session?.access_token ?? null
  return fetch(path, {
    ...options,
    headers: { ...options.headers, ...bearer(session) },
  })
}

/**
 * The most recent token this module has sent, or null.
 *
 * For the one caller that cannot await a session: maplibre's `transformRequest`
 * is synchronous. It is a snapshot, but a fresher one than the prop it
 * replaces -- that closed over whatever token was in scope when the map was
 * built and never updated, whereas this follows every request and every
 * refresh. Anything that can await should call apiFetch instead.
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
  const { data: refreshed } = await auth.refreshSession()
  const retried = await send(path, options, refreshed?.session)

  // A second 401 is the refresh token itself being gone, which is the only
  // thing T47's modal is for. Every caller reaches it from here, so a lapsed
  // session during a jobs poll or a CSV download is no longer a silent failure.
  if (shouldPromptReauth(retried.status)) unauthorized?.()
  return retried
}
