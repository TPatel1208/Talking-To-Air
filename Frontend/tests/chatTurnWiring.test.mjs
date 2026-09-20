import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, relative, sep } from 'node:path'

// T63 Phase 5 invariants that live in JSX and in callbacks.
//
// The whole phase is a split: one function that both stopped the turn and
// walked away from it becomes two, and which of them each of six call sites
// gets is the entire behavioural change. There is no jsdom here, so a
// callback passed as a prop is readable but not runnable -- and the failure
// mode of getting one wrong is silent, because `detach()` and `stopTurn()`
// look identical from the outside until a turn is running.
//
// Same idiom, and the same "ban the shape, not today's instances" and
// "mutation-check every guard" rules, as authWiring.test.mjs.

const here = dirname(fileURLToPath(import.meta.url))
const srcDir = join(here, '..', 'src')

function collectSources(dir) {
  const out = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...collectSources(full))
    else if (/\.(js|jsx)$/.test(entry.name)) out.push(full)
  }
  return out
}

const SOURCES = collectSources(srcDir).map(path => ({
  path: relative(srcDir, path).split(sep).join('/'),
  text: readFileSync(path, 'utf8'),
}))
const sourceOf = (name) => SOURCES.find(f => f.path === name).text
const APP = sourceOf('App.jsx')
const USE_CHAT = sourceOf('hooks/useChat.js')

const offenders = (pattern) =>
  SOURCES.filter(f => pattern.test(f.text)).map(f => f.path)

// Everything between `marker` and the line that closes the useCallback.
function callbackBody(source, marker) {
  const start = source.indexOf(marker)
  if (start === -1) return null
  const end = source.indexOf('\n  }, [', start)
  return end === -1 ? null : source.slice(start, end)
}

test('the combined abort/stop function is gone, not merely unused', () => {
  // The trap: `abortActiveRequest` took a `markCancelled` flag that *was*
  // the stop/detach distinction, so a leftover call site reads as
  // deliberate. It is also destructured from a hook that no longer returns
  // it, which fails at the call rather than at the import.
  // Matches a call or a destructure, not the name in prose -- the comments
  // that explain the split are allowed to say what they replaced.
  assert.deepEqual(offenders(/abortActiveRequest\s*[(,]/), [])
})

test('stopping a turn is reachable only from the Stop affordance', () => {
  // Every other path that used to abort -- unmount, newSession,
  // switchSession, logout -- must now detach. A `stopTurn` anywhere else
  // silently reintroduces "switching away kills your turn", which is the
  // defect this whole series exists to fix, and it cannot be seen in a diff
  // of the hook alone.
  const callers = SOURCES.filter(f => /\bstopTurn\b/.test(f.text)).map(f => f.path)
  assert.deepEqual(callers.sort(), ['App.jsx', 'hooks/useChat.js'])

  const appUses = APP.match(/\bstopTurn\b/g) || []
  assert.equal(appUses.length, 2, 'App.jsx should name stopTurn twice: the destructure and the Stop button')
  assert.match(APP, /onAbort=\{stopTurn\}/)
})

test('logging out detaches rather than stopping the turn', () => {
  // A product call, not a mechanical one: the turn keeps running, finishes,
  // and its answer is in history at the next sign-in. Stopping would also
  // need the credential the very next line revokes.
  const body = callbackBody(APP, 'const handleLogout = useCallback(')
  assert.ok(body, 'handleLogout is no longer a useCallback -- re-point this guard')
  assert.match(body, /\bdetach\(\)/)
  assert.doesNotMatch(body, /\bstopTurn\b/)
})

test('every way of walking away from a turn detaches', () => {
  for (const marker of [
    'const newSession = useCallback(',
    'const switchSession = useCallback(',
  ]) {
    const body = callbackBody(USE_CHAT, marker)
    assert.ok(body, `${marker} is no longer a useCallback -- re-point this guard`)
    assert.match(body, /\bdetach\(\)/, marker)
    assert.doesNotMatch(body, /\bstopTurn\b/, marker)
  }
})

test('closing the reader happens in exactly one place', () => {
  // `detach` is the only thing allowed to end this client's read. A second
  // `controller.abort()` elsewhere would be a second teardown path with its
  // own idea of what state to leave behind -- and the refs it forgets to
  // clear are what make a later Stop target the wrong turn.
  const aborts = USE_CHAT.match(/\.abort\(\)/g) || []
  assert.equal(aborts.length, 1, 'exactly one controller.abort() belongs in useChat')
  const body = callbackBody(USE_CHAT, 'const detach = useCallback(')
  assert.match(body, /\.abort\(\)/)
})

test('the client no longer tracks provider jobs in order to cancel them', () => {
  // D10. A reader that joined a turn mid-flight never saw the job_progress
  // events the old client-side set was built from, so its Stop would have
  // leaked exactly the jobs the tracking existed to catch. The authoritative
  // set is server-side, and keeping a second one here would make the two
  // disagree precisely on reattach.
  assert.doesNotMatch(USE_CHAT, /activeJobHandlesRef/)
  assert.doesNotMatch(USE_CHAT, /TERMINAL_STATUSES|TERMINAL_JOB_STATUSES/)
  // And nothing in the Stop path cancels a job from the client any more.
  const appStop = APP.slice(APP.indexOf('onAbort='), APP.indexOf('onAbort=') + 200)
  assert.doesNotMatch(appStop, /cancelJob/)
})

test('the send path never streams from the chat POST under the new protocol', () => {
  // D6: all streaming happens on the GET, so the reattach path is exercised
  // by every turn and cannot rot. The one `res.body` read left is the
  // explicitly-labelled legacy branch, kept so a backend rollback needs no
  // frontend rebuild.
  const legacyBranches = USE_CHAT.match(/'legacy-stream'/g) || []
  assert.equal(legacyBranches.length, 1)
  // The accepted turn is streamed from the GET, not from the POST that
  // accepted it. Matched on the call rather than its exact arguments, so
  // this guards the route and not the spelling.
  assert.match(USE_CHAT, /streamPath\(API_BASE, acceptedThread[,)]/)
  // Every read of a response body belongs to consumeStream, which both
  // protocols share. A `res.body` outside it is a send site streaming the
  // POST for itself, which is how the reattach path stops being exercised on
  // every turn and starts rotting.
  const reader = callbackBody(USE_CHAT, 'const consumeStream = useCallback(')
  assert.ok(reader, 'consumeStream is no longer a useCallback -- re-point this guard')
  const insideReader = (reader.match(/res\.body/g) || []).length
  const everywhere = (USE_CHAT.match(/res\.body/g) || []).length
  assert.ok(insideReader > 0)
  assert.equal(everywhere, insideReader)
})

test('the reattach probe runs on mount and on every session switch', () => {
  // Item 4 of the phase, and the half that is invisible in the hook's own
  // tests: without both, "switch away and come back" still loses the turn,
  // just server-side-politely instead of by aborting it.
  const switchBody = callbackBody(USE_CHAT, 'const switchSession = useCallback(')
  assert.match(switchBody, /attachToThread\(/)
  const fetchBody = callbackBody(USE_CHAT, 'const fetchSessions = useCallback(')
  assert.match(fetchBody, /attachToThread\(/)
})
