/**
 * What belongs in "Recent analyses", when, and in what order.
 *
 * A thread's row exists on the server from the moment its message is posted —
 * it is the only record of who owns the thread, and the reattach and stop
 * endpoints refuse without one — but it is not evidence the conversation has
 * anything in it. The server lists a thread once its turn has produced a
 * frame; this is the same rule applied locally, so the sidebar shows the same
 * threads between reloads.
 *
 * The same frame decides order: the server sorts by when a thread last
 * narrated, not when it was created, so a thread returned to today sits
 * above one started yesterday and abandoned.
 */

const MAX_TITLE_LENGTH = 60

/** The turn's narration, as opposed to the follower's bookkeeping.
 *
 * `cursor` is synthesized by the follower, and `stopped`/`interrupted` are
 * written by the registry around a turn that was cut short or found
 * abandoned. The server's stamp wraps the turn's own frame generator and so
 * never sees any of them — listing on one would add a row that the next
 * /sessions fetch takes straight back out.
 */
const FOLLOWER_FRAMES = new Set(['cursor', 'stopped', 'interrupted'])

export function isTurnFrame(event) {
  return !FOLLOWER_FRAMES.has(event)
}

/** A row for a thread whose title the server has not sent back yet.
 *
 * Titled exactly as generate_session_title does it, so the row does not
 * rename itself the next time the list is fetched.
 */
export function localSessionFor(id, message) {
  const title = (message || '').trim().replace(/\s+/g, ' ')
  return {
    id,
    title: title.length > MAX_TITLE_LENGTH
      ? `${title.slice(0, MAX_TITLE_LENGTH - 3).trim()}...`
      : title,
    created_at: new Date().toISOString(),
  }
}

/** `sessions` with this thread at the top, or `sessions` itself if it is
 * already there.
 *
 * A thread already in the list is *moved*, not left alone: the server now
 * orders by when a thread was last used, and this is the same rule applied
 * locally. Without the move, replying in an old thread would leave it
 * wherever it was until the next reload silently rearranged the sidebar —
 * `/sessions` is fetched in full only on mount, and the background poll
 * merges rows without reordering them.
 *
 * The row that moves is the one already there, not a fresh one: it carries
 * the server's title, and rebuilding it from this turn's message would
 * rename the thread after every question.
 *
 * Returning the same array when the thread is already on top matters. That
 * is the common case — you usually reply in the thread you are reading —
 * and a new array re-renders the sidebar for nothing.
 */
export function sessionsWithThread(sessions, id, message) {
  if (!id || !(message || '').trim()) return sessions
  const at = sessions.findIndex(session => sessionId(session) === id)
  if (at === 0) return sessions
  if (at === -1) return [localSessionFor(id, message), ...sessions]
  return [sessions[at], ...sessions.slice(0, at), ...sessions.slice(at + 1)]
}

function sessionId(session) {
  return typeof session === 'string' ? session : session?.id
}

/** `sessions` with every entry from `incoming` this list does not already
 * have, added at the top; `sessions` itself if there is nothing new.
 *
 * For a thread whose first frame arrived after this tab stopped reading its
 * stream -- the user sent the message, then switched away before anything
 * came back, so `sessionsWithThread`'s own optimistic add never ran here.
 * The server lists the thread the moment its turn narrates regardless of
 * who is watching; this is what a background poll of `/sessions` uses to
 * catch that listing up, so the row (and the badge it carries) appears on
 * this client without waiting for a reload.
 */
export function mergeSessions(sessions, incoming) {
  const known = new Set(sessions.map(sessionId))
  const additions = incoming.filter(session => !known.has(sessionId(session)))
  return additions.length ? [...additions, ...sessions] : sessions
}
