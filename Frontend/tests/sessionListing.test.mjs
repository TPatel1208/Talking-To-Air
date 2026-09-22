import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

import { isTurnFrame, localSessionFor, mergeSessions, sessionsWithThread } from '../src/utils/sessionList.js'

/* ── When a thread joins "Recent analyses" ──
   The backend lists a thread once its turn has produced a frame. The sidebar
   is the same rule applied locally, so a thread does not appear in this tab
   and then vanish on the next reload — or, as before, appear only when the
   answer completes, leaving a three-minute turn invisible in the list it is
   running in. */

test('a thread the list does not have is added at the top, titled by the message', () => {
  const got = sessionsWithThread([{ id: 'old' }], 'th-1', '  How is the   air in Newark? ')
  assert.equal(got.length, 2)
  assert.equal(got[0].id, 'th-1')
  assert.equal(got[0].title, 'How is the air in Newark?')
  assert.equal(got[1].id, 'old')
})

test('a long message is truncated the same way the server titles it', () => {
  // generate_session_title: >60 chars becomes 57 + "...". A different rule
  // here means the row renames itself on the next reload.
  const message = 'a'.repeat(80)
  assert.equal(localSessionFor('th-1', message).title, `${'a'.repeat(57)}...`)
  assert.equal(localSessionFor('th-1', 'a'.repeat(60)).title, 'a'.repeat(60))
})

test('a thread already in the list leaves it untouched, array and all', () => {
  // Called on every frame of every turn, so the no-op case has to be one:
  // a fresh array would re-render the sidebar ten times a second.
  const sessions = [{ id: 'th-1', title: 'How is the air' }]
  assert.equal(sessionsWithThread(sessions, 'th-1', 'anything else'), sessions)
})

test('a thread with no id and no message is not invented', () => {
  const sessions = [{ id: 'th-1' }]
  assert.equal(sessionsWithThread(sessions, null, 'hello'), sessions)
  assert.equal(sessionsWithThread(sessions, 'th-2', ''), sessions)
})

/* ── mergeSessions: catching up a row this tab never optimistically added ──
   For a thread whose local stream was aborted (the user switched away)
   before its first frame arrived here -- the server lists it anyway, and a
   background poll of /sessions uses this to bring the row in without a
   reload. */

test('a thread the server has listed but this tab has not is added at the top', () => {
  const got = mergeSessions([{ id: 'old' }], [{ id: 'old' }, { id: 'new' }])
  assert.deepEqual(got.map(s => s.id), ['new', 'old'])
})

test('a thread this tab already has -- optimistic or not -- is not duplicated', () => {
  const sessions = [{ id: 'th-1', title: 'local optimistic title' }]
  const got = mergeSessions(sessions, [{ id: 'th-1', title: 'server title' }])
  // The local entry wins: this is a catch-up for missing rows, not a refresh
  // of ones already there, so it cannot flicker a title mid-stream.
  assert.equal(got, sessions)
})

test('nothing new returns the same array, not a copy', () => {
  const sessions = [{ id: 'th-1' }]
  assert.equal(mergeSessions(sessions, [{ id: 'th-1' }]), sessions)
  assert.equal(mergeSessions(sessions, []), sessions)
})

test('mergeSessions works with bare string ids too', () => {
  const got = mergeSessions(['old'], ['old', 'new'])
  assert.deepEqual(got, ['new', 'old'])
})

/* ── Which frames mean the turn is talking ──
   Not all of them come from the turn. The follower synthesizes `cursor`
   frames, and writes `stopped`/`interrupted` itself when a turn is cut short
   or found abandoned — the server's stamp wraps the turn's own frame
   generator and never sees those. Listing on one would put a row in the
   sidebar that the next /sessions fetch takes straight back out. */

test('the events the turn itself produces list the thread', () => {
  for (const event of ['status', 'text', 'tool_call', 'chart', 'job_progress', 'done', 'error']) {
    assert.equal(isTurnFrame(event), true, event)
  }
})

test('the follower\'s own frames do not', () => {
  for (const event of ['cursor', 'stopped', 'interrupted']) {
    assert.equal(isTurnFrame(event), false, event)
  }
})

/* ── Wiring: the rule has to be applied where the frames arrive ──
   No jsdom here, so the hook's callbacks are readable but not runnable. */

const here = dirname(fileURLToPath(import.meta.url))
const USE_CHAT = readFileSync(join(here, '..', 'src', 'hooks', 'useChat.js'), 'utf8')

function branchBody(source, marker) {
  const start = source.indexOf(marker)
  if (start === -1) return null
  const end = source.indexOf('\n      } else if (', start)
  return end === -1 ? null : source.slice(start, end)
}

test('the sidebar is no longer driven from the end of the turn', () => {
  // The old insert lived here and only here, so a turn that never finished
  // never appeared — and every finished one appeared all at once.
  const done = branchBody(USE_CHAT, "} else if (event === 'done') {")
  assert.ok(done, 'the done branch moved; this guard needs rewriting')
  assert.equal(
    /setSessions\(/.test(done) && !/!ctx\.threadId/.test(done),
    false,
    'the done branch may only list a thread whose id was unknown until it ended — '
    + 'the legacy protocol, where the POST itself streams',
  )
})

test('a frame lists the thread as it arrives', () => {
  assert.match(USE_CHAT, /sessionsWithThread\(/)
  assert.match(USE_CHAT, /isTurnFrame\(/)
})

test('restoring the active thread does not wait for it to be listed', () => {
  // A thread whose turn has not produced yet is deliberately absent from
  // /sessions. Gating the restore on that list would drop the stored thread
  // on a reload during the first ten seconds of a silent turn — exactly the
  // reattach T63 exists for. A deleted thread is still caught: loadHistory
  // 404s and the restore clears it.
  assert.equal(/nextSessions\.some\(/.test(USE_CHAT), false)
})
