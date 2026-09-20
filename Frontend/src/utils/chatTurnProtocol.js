// The 202-then-GET chat protocol, as pure decisions (T63 Phase 5).
//
// A turn no longer belongs to the connection that started it. `POST /chat`
// accepts the message and answers 202 with a turn id; everything the turn
// narrates leaves over `GET /chat/{thread}/stream`, which any tab on any
// replica can attach to and re-attach to from a cursor.
//
// Everything here is side-effect-free (storage is injected) because this repo
// has no jsdom: what the client believes about the wire has to be testable
// without rendering useChat.

// Where a thread's in-flight turn is remembered across a remount. Sits
// alongside `tta.activeThreadId` rather than inside a session store, because
// T62 moves thread identity into the URL and would have to rewrite anything
// larger.
export const CHAT_TURN_STORAGE_KEY = 'tta.chatTurn'

const TERMINAL_EVENTS = new Set(['done', 'error', 'stopped', 'interrupted'])

/**
 * What the chat POST answered, and what the caller should do about it.
 *
 * Deliberately reads the response rather than a flag the frontend was told
 * about: the two protocols are distinguishable on their own (the old one
 * streams from a 200, the new one hands back a 202 and a JSON body), so one
 * bundle serves both and rolling the backend flag back needs no frontend
 * rebuild -- which matters, because the frontend image is a static bundle.
 */
export function classifyChatPost(status, body) {
  const turnId = body?.turn_id
  const threadId = body?.thread_id

  if (status === 200) return { kind: 'legacy-stream' }
  // D12: the 409 names the turn in flight, so a second tab joins it rather
  // than forking a second `astream` onto one LangGraph thread. Without an id
  // there is nothing to join and it is only a refusal.
  if (status === 409) {
    return turnId ? { kind: 'joined', turnId, threadId } : { kind: 'failed' }
  }
  // D15: Redis is the transport for every event, the active-turn lock and the
  // stop signal, so chat is down rather than degraded. Phase 4 adds the
  // draining replica to the same status.
  if (status === 503) return { kind: 'unavailable' }
  if (status === 202 && turnId) return { kind: 'started', turnId, threadId }
  return { kind: 'failed' }
}

/**
 * Whether this event ends the stream, and what kind of event it is.
 *
 * Four names, not one. The old reader watched only `done` and treated
 * anything else as a lost connection -- which was already wrong, because the
 * generic failure path emits `error` and no `done` at all.
 */
export function classifyStreamEvent(event) {
  if (event === 'cursor') return { terminal: false, kind: 'cursor' }
  return { terminal: TERMINAL_EVENTS.has(event), kind: event }
}

const STOPPED_WITH_NOTHING = 'Stopped.'
const INTERRUPTION_NOTICES = {
  shutdown: 'The server restarted before this answer finished. Reload the session to see anything that was saved, then ask again.',
  stale: 'Lost contact with the server before this answer finished. Reload the session to see anything that was saved, then ask again.',
}
const INTERRUPTION_FALLBACK = INTERRUPTION_NOTICES.stale

/**
 * How a terminal event ends the assistant bubble.
 *
 * `previous` is the message as it stands, because both endings here are
 * partial rather than empty: a stopped turn keeps whatever it finished (D11
 * -- its charts are already persisted, and blanking the bubble would report
 * data loss that did not happen), and an interrupted one keeps its partial
 * answer above the notice.
 */
export function terminalMessagePatch(kind, data, previous) {
  const content = previous?.content || ''

  if (kind === 'stopped') {
    return {
      content: content || STOPPED_WITH_NOTHING,
      isLoading: false,
      isCancelled: true,
      statusMessage: '',
    }
  }

  // Both interruptions offer the same thing -- reload the session, which
  // rehydrates history and reattaches if the thread still has a turn -- but
  // "the server restarted" and "we lost contact" are not the same sentence,
  // and only one of them knows the answer is really gone.
  const notice = INTERRUPTION_NOTICES[data?.reason] || INTERRUPTION_FALLBACK
  return {
    content: content ? `${content}\n\n${notice}` : notice,
    isLoading: false,
    isError: true,
    // Routed into T41's existing affordance rather than a second recovery
    // path: `isConnectionLost` already renders "Reload session", which runs
    // switchSession -> loadHistory + reattach. A parallel recovery path is
    // the same shape as this repo's recorded "dual SSE loops" defect.
    isConnectionLost: true,
    statusMessage: '',
  }
}

function recordKey(threadId) {
  return `${CHAT_TURN_STORAGE_KEY}.${threadId}`
}

/**
 * The in-flight turn remembered for this thread, or null.
 *
 * Turn id and cursor travel together in one record, and are only ever used
 * together: a cursor is a position in one turn's stream, and handing it to
 * the server alongside a different turn's id would ask for a resume point
 * that means nothing there.
 */
export function readTurnRecord(storage, threadId) {
  try {
    const raw = storage.getItem(recordKey(threadId))
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed && parsed.turnId ? parsed : null
  } catch {
    // Shared with every other tab and surviving deploys, so whatever is in
    // there is untrusted -- and the accessor itself throws in a private
    // window or with site data blocked.
    return null
  }
}

export function writeTurnRecord(storage, threadId, record) {
  try {
    storage.setItem(recordKey(threadId), JSON.stringify(record))
  } catch {
    // A reader that cannot remember where it got to restarts the stream
    // instead of resuming it. Worth nothing to crash over.
  }
}

export function clearTurnRecord(storage, threadId) {
  try {
    storage.removeItem(recordKey(threadId))
  } catch {
    // As above.
  }
}

/**
 * Where this thread's turn streams from.
 *
 * Naming a `turnId` says "I am coming back to a turn I already know about",
 * and is what gets a reader the tail of one that has just ended -- the answer
 * is not in history until the write-back lands. A probe that names none is
 * only asking whether anything is running, and is answered 404 by a thread
 * whose turn is over, so an ordinary session switch does not replay a
 * finished answer on top of the history it just loaded.
 */
export function streamPath(apiBase, threadId, { cursor = null, turnId = null } = {}) {
  const path = `${apiBase}/chat/${encodeURIComponent(threadId)}/stream`
  const query = new URLSearchParams()
  if (cursor) query.set('from', cursor)
  if (turnId) query.set('turn', turnId)
  const search = query.toString()
  return search ? `${path}?${search}` : path
}
