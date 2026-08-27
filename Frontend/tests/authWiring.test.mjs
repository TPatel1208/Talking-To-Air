import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, relative, sep } from 'node:path'

// T61 Phase 4 invariants that live in JSX and in callbacks, where this repo
// cannot reach them: there is no jsdom, so a `key` prop, a subscription
// callback and a fetch call inside a component are readable but not runnable.
// Source text is a weak instrument, so it is used only where the invariant is
// genuinely textual, and every guard below is mutation-checked -- a regex that
// matches nothing looks exactly like a regex that matches nothing bad.
//
// Same idiom, and same "ban the shape, not today's instances" rule, as
// noAutoDeleteSession.test.mjs and outputPanelLayout.test.mjs.

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

const offenders = (pattern) =>
  SOURCES.filter(f => pattern.test(f.text)).map(f => f.path)

// The text of a call including its arguments, matched by balancing parens.
// Blind to parens inside strings and comments, which is fine for locating a
// callback body but is the reason this is not used for anything subtler.
function callText(source, marker) {
  const start = source.indexOf(marker)
  if (start === -1) return null
  let depth = 0
  for (let i = start + marker.length - 1; i < source.length; i += 1) {
    if (source[i] === '(') depth += 1
    else if (source[i] === ')') {
      depth -= 1
      if (depth === 0) return source.slice(start, i + 1)
    }
  }
  return null
}

test('the authenticated tree is keyed on the user, never on the access token', () => {
  // The trap this exists for: `key={accessToken}` is harmless while the token
  // changes only at login, and lethal the moment supabase-js auto-refreshes.
  // A refresh mints a new token string every 45 minutes (decision 10), so the
  // key changes on a timer and React throws away the entire authenticated
  // tree -- messages, focused output, compare, map, jobs, discovery -- in the
  // middle of an analysis. That is the T47 rug-pull returning by another
  // route, and it would look like a random crash, not a keying bug.
  const tag = APP.match(/<AuthenticatedApp\b([\s\S]*?)\/?>/)
  assert.ok(tag, 'App must still render AuthenticatedApp')

  const key = tag[1].match(/key=\{([^}]*)\}/)
  assert.ok(key, 'AuthenticatedApp must carry an explicit key, not fall back to position')
  assert.doesNotMatch(key[1], /token/i,
    'the remount key must not be derived from the access token')
  assert.match(key[1], /userId/i,
    'the user id is the stable half of a session and the only safe remount key')

  // Banned as a shape, everywhere -- one reintroduction on a later branch is
  // the whole bug back.
  assert.deepEqual(offenders(/key=\{[^}]*[Tt]oken[^}]*\}/), [])
})

test('the auth-state callback returns immediately and awaits nothing', () => {
  // Supabase's own troubleshooting docs record a deadlock: awaiting a Supabase
  // call inside onAuthStateChange hangs every later call on the client. There
  // is no way to observe that without a browser and a live client, so the
  // source is the only place it can be caught.
  assert.match(APP, /onAuthStateChange\(/,
    'the session must be driven by the subscription, not by polling or an effect')

  const subscription = callText(APP, 'onAuthStateChange(')
  assert.ok(subscription)
  assert.doesNotMatch(subscription, /\bawait\b/,
    'never await inside onAuthStateChange -- it deadlocks the client')
})

test('no call survives to a backend auth route that Phase 2 deleted', () => {
  // Those routes are gone and the middleware is fail-closed, so a leftover
  // call now returns 401 -- which shouldPromptReauth turns into "your
  // session expired". A stale logout call would raise the re-auth modal as
  // the response to clicking sign out.
  assert.deepEqual(offenders(/\/auth\/(register|login|logout)\b/), [])
})

test('supabase-js owns session persistence, and nothing writes a rival copy', () => {
  // Two stores disagreeing about who is signed in, with the stale one winning
  // on reload.
  assert.deepEqual(offenders(/tta\.accessToken/), [])

  // But the active thread key must survive: T47 resuming the same
  // conversation after a re-auth is what that key is for.
  assert.ok(offenders(/tta\.activeThreadId/).includes('hooks/useChat.js'))
})

test('the Supabase client has exactly one construction site', () => {
  // Decision 11 means it cannot be built at import time -- the url and key do
  // not exist until GET /config/auth resolves. Confining the import to the
  // holder is what makes a module-scope client impossible to write by
  // accident anywhere else.
  assert.deepEqual(offenders(/@supabase\/supabase-js/), ['utils/supabaseClient.js'])

  const holder = sourceOf('utils/supabaseClient.js')
  assert.match(holder, /import \{ createClient \}/)
  assert.doesNotMatch(holder, /createClient\(/,
    'createClient is passed as an injected factory and never called by name -- that is both what makes the holder testable and what keeps construction in one place')
})

test('the sign-in form takes an email, and offers no way to register', () => {
  // Decision 8: signup is invite-only and disabled in Supabase, so a register
  // tab could only ever produce a rejection. Username goes with it -- it was
  // read by the login form and a JWT claim, and displayed nowhere.
  assert.match(APP, /type="email"/)
  assert.doesNotMatch(APP, /setUsername|allowRegister/)
})
