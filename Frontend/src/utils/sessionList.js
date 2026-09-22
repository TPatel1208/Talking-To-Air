/**
 * What belongs in "Recent analyses", and when.
 *
 * A thread's row exists on the server from the moment its message is posted —
 * it is the only record of who owns the thread, and the reattach and stop
 * endpoints refuse without one — but it is not evidence the conversation has
 * anything in it. The server lists a thread once its turn has produced a
 * frame; this is the same rule applied locally, so the sidebar shows the same
 * threads between reloads.
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
 * Returning the same array matters: this is called on every frame of every
 * turn, and a new one would re-render the sidebar about ten times a second
 * through an answer.
 */
export function sessionsWithThread(sessions, id, message) {
  if (!id || !(message || '').trim()) return sessions
  const listed = sessions.some(session => (
    (typeof session === 'string' ? session : session?.id) === id
  ))
  return listed ? sessions : [localSessionFor(id, message), ...sessions]
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
