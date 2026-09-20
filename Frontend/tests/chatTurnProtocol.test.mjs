import test from 'node:test'
import assert from 'node:assert/strict'

import {
  CHAT_TURN_STORAGE_KEY,
  StreamError,
  isStreamError,
  classifyChatPost,
  classifyStreamEvent,
  clearTurnRecord,
  readTurnRecord,
  streamPath,
  terminalMessagePatch,
  writeTurnRecord,
} from '../src/utils/chatTurnProtocol.js'

/* ── classifyChatPost: which protocol answered, and what it said ──
   The kill switch deliberately does not reach the frontend. The two
   protocols are distinguishable from the response alone, so one bundle
   serves both and a rollback is a backend env change. */

test('a 202 with a turn id means this message started the turn', () => {
  const got = classifyChatPost(202, { turn_id: 't-1', thread_id: 'th-1' })
  assert.equal(got.kind, 'started')
  assert.equal(got.turnId, 't-1')
  assert.equal(got.threadId, 'th-1')
})

test('a 409 names the turn already running, so the caller joins it rather than forking', () => {
  const got = classifyChatPost(409, { turn_id: 't-running', thread_id: 'th-1' })
  assert.equal(got.kind, 'joined')
  assert.equal(got.turnId, 't-running')
  assert.equal(got.threadId, 'th-1')
})

test('a 409 without a turn id cannot be joined and is a plain refusal', () => {
  // The route always sends one, but a proxy-generated 409 would not -- and
  // "join turn undefined" would build a stream URL that 404s forever.
  const got = classifyChatPost(409, { detail: 'conflict' })
  assert.equal(got.kind, 'failed')
})

test('a 503 is Redis or a draining replica, and says so without offering history', () => {
  const got = classifyChatPost(503, { detail: 'Agent is not ready' })
  assert.equal(got.kind, 'unavailable')
})

test('a 200 is the old protocol streaming from the POST itself', () => {
  // CHAT_DETACHED_TURNS_ENABLED off. Branching on the response rather than
  // plumbing the flag is what makes a rollback need no frontend rebuild.
  const got = classifyChatPost(200, null)
  assert.equal(got.kind, 'legacy-stream')
})

test('any other status is a failure, and a 202 with no turn id is too', () => {
  assert.equal(classifyChatPost(500, { detail: 'boom' }).kind, 'failed')
  assert.equal(classifyChatPost(404, null).kind, 'failed')
  assert.equal(classifyChatPost(202, {}).kind, 'failed')
})

/* ── classifyStreamEvent: which events end a stream ──
   The old guard watched only `done`, which was already wrong: the generic
   failure path emits `error` and no `done` at all. The server's closing set
   is now four names wide. */

test('all four terminal events are recognised as ending the stream', () => {
  for (const event of ['done', 'error', 'stopped', 'interrupted']) {
    assert.equal(classifyStreamEvent(event).terminal, true, event)
  }
})

test('narration events do not end the stream', () => {
  for (const event of ['text', 'status', 'chart', 'artifact', 'tool_call', 'job_progress', 'image']) {
    assert.equal(classifyStreamEvent(event).terminal, false, event)
  }
})

test('the cursor event is the follower talking to the reader, not the turn talking', () => {
  // Rendered by the follower after the page it accounts for. It is neither
  // terminal nor something the bubble should ever render.
  const got = classifyStreamEvent('cursor')
  assert.equal(got.terminal, false)
  assert.equal(got.kind, 'cursor')
})

/* ── terminalMessagePatch: what the bubble says at each ending ── */

test('a stopped turn keeps the work it finished rather than reading as a blank cancel', () => {
  // D11: charts already persisted stand, so replacing the content would
  // report data loss that did not happen.
  const patch = terminalMessagePatch('stopped', {}, { content: 'Here are three granules.' })
  assert.equal(patch.content, 'Here are three granules.')
  assert.equal(patch.isLoading, false)
  assert.equal(patch.isCancelled, true)
  assert.equal(patch.isConnectionLost, undefined)
})

test('a stopped turn that had said nothing yet still says something', () => {
  const patch = terminalMessagePatch('stopped', {}, { content: '' })
  assert.match(patch.content, /stopped/i)
  assert.equal(patch.isCancelled, true)
})

test('an interrupted turn offers the retry affordance, and names which interruption', () => {
  const shutdown = terminalMessagePatch('interrupted', { reason: 'shutdown' }, { content: '' })
  const stale = terminalMessagePatch('interrupted', { reason: 'stale' }, { content: '' })

  assert.equal(shutdown.isConnectionLost, true)
  assert.equal(stale.isConnectionLost, true)
  // Same offer, different sentence: "the server restarted" and "we lost
  // contact" are not the same thing to tell someone.
  assert.notEqual(shutdown.content, stale.content)
  assert.match(shutdown.content, /restart/i)
  assert.match(stale.content, /contact/i)
})

test('an interruption keeps the partial answer above its notice', () => {
  const patch = terminalMessagePatch('interrupted', { reason: 'stale' }, { content: 'Partial text.' })
  assert.match(patch.content, /^Partial text\./)
  assert.match(patch.content, /contact/i)
  assert.equal(patch.isLoading, false)
})

test('an interruption with no reason still ends the bubble and still offers retry', () => {
  const patch = terminalMessagePatch('interrupted', {}, { content: '' })
  assert.equal(patch.isLoading, false)
  assert.equal(patch.isConnectionLost, true)
  assert.ok(patch.content.length > 0)
})

/* ── the persisted turn record ──
   Persisted so a remount resumes the turn rather than restarting it, and so
   it can show what was asked: the event log holds only what the turn said,
   and the exchange does not reach history until the write-back lands.
   localStorage is shared with every other tab and outlives deploys, so it is
   read as untrusted. */

function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial))
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    _dump: () => Object.fromEntries(map),
  }
}

test('a written record round-trips for the thread it was written for', () => {
  const storage = fakeStorage()
  writeTurnRecord(storage, 'th-1', { turnId: 't-1', cursor: '17-0', userMessage: 'plot NO2' })

  assert.deepEqual(readTurnRecord(storage, 'th-1'), {
    turnId: 't-1', cursor: '17-0', userMessage: 'plot NO2',
  })
  assert.equal(readTurnRecord(storage, 'th-2'), null)
})

test('clearing a record leaves nothing behind for the next turn to resume from', () => {
  const storage = fakeStorage()
  writeTurnRecord(storage, 'th-1', { turnId: 't-1', cursor: '17-0' })
  clearTurnRecord(storage, 'th-1')
  assert.equal(readTurnRecord(storage, 'th-1'), null)
})

test('a corrupt or half-written record reads as no record, never as a throw', () => {
  // localStorage is shared with every other tab and survives deploys, so
  // whatever is in it has to be treated as untrusted.
  const storage = fakeStorage({ [`${CHAT_TURN_STORAGE_KEY}.th-1`]: '{not json' })
  assert.equal(readTurnRecord(storage, 'th-1'), null)
})

test('storage that throws is survivable -- private mode, blocked site data', () => {
  const hostile = {
    getItem() { throw new Error('denied') },
    setItem() { throw new Error('denied') },
    removeItem() { throw new Error('denied') },
  }
  assert.equal(readTurnRecord(hostile, 'th-1'), null)
  assert.doesNotThrow(() => writeTurnRecord(hostile, 'th-1', { turnId: 't', cursor: 'c' }))
  assert.doesNotThrow(() => clearTurnRecord(hostile, 'th-1'))
})

/* ── the stream URL ── */

test('the stream path carries whichever of cursor and turn the reader has', () => {
  assert.equal(
    streamPath('/api', 'th-1', { cursor: '17-0', turnId: 't-1' }),
    '/api/chat/th-1/stream?from=17-0&turn=t-1',
  )
  assert.equal(streamPath('/api', 'th-1', { turnId: 't-1' }), '/api/chat/th-1/stream?turn=t-1')
  assert.equal(streamPath('/api', 'th-1', {}), '/api/chat/th-1/stream')
  assert.equal(streamPath('/api', 'th-1'), '/api/chat/th-1/stream')
})

test('a bare probe names no turn, which is what makes it a probe', () => {
  // Naming one asks for the tail of a turn that may already have ended.
  // Naming none asks only whether anything is running, and is answered 404
  // on a thread whose turn is over -- so switching sessions does not replay
  // a finished answer over the history just loaded.
  assert.doesNotMatch(streamPath('/api', 'th-1', { cursor: '17-0' }), /turn=/)
})

test('a thread id is encoded, so it cannot forge a query of its own', () => {
  assert.equal(streamPath('/api', 'th 1/../x', {}), '/api/chat/th%201%2F..%2Fx/stream')
})

test('an ending the turn reported is told apart from the transport dying', () => {
  // Both arrive at the same `catch` as a thrown Error, and they are opposite
  // situations: a turn that reported an `error` frame got its message all the
  // way to the reader and has nothing left to say, while a severed body means
  // the turn is very likely still running and reattaching will find it.
  // Mistaking the first for the second replaces a real explanation with
  // "Connection lost", and offers a reload that cannot help.
  const reported = new StreamError('The request hit an internal error.')
  assert.ok(isStreamError(reported))
  assert.equal(reported.message, 'The request hit an internal error.')

  // Everything else is the transport. `TypeError: network error` is what a
  // severed response body actually throws.
  assert.equal(isStreamError(new TypeError('network error')), false)
  assert.equal(isStreamError(new Error('HTTP 502')), false)
  assert.equal(isStreamError(null), false)
  assert.equal(isStreamError(undefined), false)
})

test('a stream error is recognised across module instances', () => {
  // Tagged by name rather than by `instanceof`: a bundle that ends up with
  // two copies of this module would make `instanceof` silently false, and the
  // symptom would be the "Connection lost" confusion above rather than an
  // error anyone could see.
  const lookalike = new Error('boom')
  lookalike.name = 'StreamError'
  assert.ok(isStreamError(lookalike))
})
