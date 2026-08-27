import test from 'node:test'
import assert from 'node:assert/strict'

// Node keys its ESM cache on the full specifier, query string included, so each
// test gets a module with its own configuration state -- no test-only reset
// export has to ship in production code. Same technique as supabaseClient.test.mjs.
let caseId = 0
const load = () => import(`../src/utils/apiFetch.js?case=${++caseId}`)

// A stand-in for supabase-js's auth object. apiFetch depends on exactly two of
// its methods, and that narrow shape is the whole contract between them.
function authHolding(...tokens) {
  const queue = [...tokens]
  let current = queue.shift() ?? null
  const sessionOf = () => (current ? { access_token: current } : null)
  return {
    calls: { getSession: 0, refreshSession: 0 },
    async getSession() {
      this.calls.getSession += 1
      return { data: { session: sessionOf() }, error: null }
    },
    async refreshSession() {
      this.calls.refreshSession += 1
      current = queue.length ? queue.shift() : current
      return { data: { session: sessionOf() }, error: null }
    },
  }
}

function serving(...statuses) {
  const queue = [...statuses]
  const calls = []
  const fn = async (url, options) => {
    calls.push({ url, options })
    return new Response('{}', { status: queue.shift() ?? 200 })
  }
  fn.calls = calls
  globalThis.fetch = fn
  return fn
}

const authOf = (call) => call.options.headers.Authorization

test('a request carries the token the session holds right now', async () => {
  const { configureApiFetch, apiFetch } = await load()
  const auth = authHolding('jwt-current')
  const server = serving(200)
  configureApiFetch({ auth })

  const res = await apiFetch('/api/jobs')

  assert.equal(res.status, 200)
  assert.equal(server.calls[0].url, '/api/jobs')
  assert.equal(authOf(server.calls[0]), 'Bearer jwt-current')
})

test('with no session there is no Authorization header at all', async () => {
  // The public endpoints share this path. A header reading "Bearer undefined"
  // is not an absent header -- it is a malformed credential the backend must
  // then decide what to do with, and the honest answer is not to send one.
  const { configureApiFetch, apiFetch } = await load()
  const server = serving(200)
  configureApiFetch({ auth: authHolding() })

  await apiFetch('/api/capabilities/starters')

  assert.equal('Authorization' in server.calls[0].options.headers, false)
})

test('a 401 refreshes the session and retries once with the new token', async () => {
  // The failure this exists to absorb: a tab left in the background long enough
  // that supabase-js's refresh timer was throttled past the expiry. The first
  // request goes out on a token the server has stopped accepting. Asking for a
  // fresh one and trying again turns that into something the user never sees.
  const { configureApiFetch, apiFetch } = await load()
  const auth = authHolding('jwt-stale', 'jwt-refreshed')
  const server = serving(401, 200)
  configureApiFetch({ auth })

  const res = await apiFetch('/api/jobs')

  assert.equal(res.status, 200)
  assert.equal(server.calls.length, 2)
  assert.equal(authOf(server.calls[0]), 'Bearer jwt-stale')
  assert.equal(authOf(server.calls[1]), 'Bearer jwt-refreshed')
  assert.equal(auth.calls.refreshSession, 1)
})

test('a 401 that survives the refresh raises the re-auth modal, once', async () => {
  // Only the second 401 means the refresh token itself is gone. Raising T47's
  // modal on the first would flash it at a user whose session was about to
  // recover on its own.
  const { configureApiFetch, apiFetch } = await load()
  const raised = []
  const server = serving(401, 401)
  configureApiFetch({ auth: authHolding('jwt-dead'), onUnauthorized: () => raised.push('modal') })

  const res = await apiFetch('/api/jobs')

  // Handed back rather than thrown: every call site already branches on res.ok,
  // and turning a 401 into an exception would rewrite all of them.
  assert.equal(res.status, 401)
  assert.equal(server.calls.length, 2)
  assert.equal(raised.length, 1)
})

test('a request made before configuration names what is missing', async () => {
  // Reachable by mis-wiring rather than by user action: any of the fifteen
  // migrated files could issue a request before App has configured this module.
  // "Cannot read properties of null" would send the next reader into apiFetch;
  // the fix is always at the call to configureApiFetch.
  const { apiFetch } = await load()
  serving(200)

  await assert.rejects(() => apiFetch('/api/jobs'), /configureApiFetch/)
})

test('a 401 the refresh recovers from raises nothing', async () => {
  // The regression that would hurt most: a modal appearing on every routine
  // token rotation. Recovering silently is the entire point of the retry.
  const { configureApiFetch, apiFetch } = await load()
  const raised = []
  serving(401, 200)
  configureApiFetch({ auth: authHolding('jwt-stale', 'jwt-fresh'), onUnauthorized: () => raised.push('modal') })

  const res = await apiFetch('/api/jobs')

  assert.equal(res.status, 200)
  assert.deepEqual(raised, [])
})

test('a failure that is not a 401 is passed straight back', async () => {
  // T41: a 5xx is transient and a 404 is an honest empty state. Refreshing on
  // either would spend a token rotation on a problem it cannot fix, and the
  // caller is the one that knows what its own 404 means.
  const { configureApiFetch, apiFetch } = await load()
  const raised = []
  const auth = authHolding('jwt-fine')
  const server = serving(503)
  configureApiFetch({ auth, onUnauthorized: () => raised.push('modal') })

  const res = await apiFetch('/api/jobs')

  assert.equal(res.status, 503)
  assert.equal(server.calls.length, 1)
  assert.equal(auth.calls.refreshSession, 0)
  assert.deepEqual(raised, [])
})

test('the caller keeps its method, body, signal and its own headers', async () => {
  const { configureApiFetch, apiFetch } = await load()
  const server = serving(200)
  const controller = new AbortController()
  configureApiFetch({ auth: authHolding('jwt-current') })

  await apiFetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{"message":"hi"}',
    signal: controller.signal,
  })

  const sent = server.calls[0].options
  assert.equal(sent.method, 'POST')
  assert.equal(sent.body, '{"message":"hi"}')
  assert.equal(sent.signal, controller.signal)
  // Adding a credential must not cost the caller the header that says how to
  // parse its body -- /chat is a POST whose Content-Type is load-bearing.
  assert.equal(sent.headers['Content-Type'], 'application/json')
  assert.equal(sent.headers.Authorization, 'Bearer jwt-current')
})

test('a streaming response arrives with its body unread', async () => {
  // /chat is consumed as a ReadableStream. Anything here that touched the body
  // -- reading it to inspect an error, cloning it -- would break the stream the
  // whole chat UI is driven by.
  const { configureApiFetch, apiFetch } = await load()
  globalThis.fetch = async () => new Response(
    new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode('data: token')); c.close() } }),
    { status: 200 },
  )
  configureApiFetch({ auth: authHolding('jwt-current') })

  const res = await apiFetch('/api/chat', { method: 'POST' })

  assert.equal(res.bodyUsed, false)
  assert.equal(await new Response(res.body).text(), 'data: token')
})

test('a token rotated between calls is picked up with no re-render', async () => {
  // Phase 5 in one test. The prop this replaces was a snapshot of whenever the
  // tree last rendered; reading the session per call is what closes the gap.
  const { configureApiFetch, apiFetch } = await load()
  const auth = authHolding('jwt-first', 'jwt-rotated')
  const server = serving(200, 200)
  configureApiFetch({ auth })

  await apiFetch('/api/jobs')
  await auth.refreshSession()
  await apiFetch('/api/jobs')

  assert.equal(authOf(server.calls[0]), 'Bearer jwt-first')
  assert.equal(authOf(server.calls[1]), 'Bearer jwt-rotated')
})

test('the most recent token is readable synchronously, refreshes included', async () => {
  // maplibre's transformRequest is synchronous and cannot await a session, so
  // it needs a snapshot. This is not a regression: that callback already closed
  // over a token captured when the map was built (its effect depends on
  // [payload], not on the token), so it went stale after one rotation and
  // stayed stale. A value refreshed by every request the app makes is fresher.
  const { configureApiFetch, apiFetch, currentAccessToken } = await load()
  const auth = authHolding('jwt-first', 'jwt-rotated')
  serving(200, 401, 200)
  configureApiFetch({ auth })

  assert.equal(currentAccessToken(), null)
  await apiFetch('/api/jobs')
  assert.equal(currentAccessToken(), 'jwt-first')

  // The second call 401s and recovers; the snapshot must follow the refresh,
  // not stay pinned to the token that just failed.
  await apiFetch('/api/jobs')
  assert.equal(currentAccessToken(), 'jwt-rotated')
})

test('a background rotation updates the snapshot with no request in between', async () => {
  // The gap this closes. supabase-js refreshes on its own timer and announces
  // it through onAuthStateChange; nothing calls apiFetch on that path. Left to
  // requests alone the snapshot goes wrong 90 seconds after every rotation --
  // auth-js refreshes at EXPIRY_MARGIN_MS, so the token it replaces dies almost
  // immediately -- and stays wrong until something happens to issue a request.
  // An idle app issues none: useJobs only polls while a job is progressing.
  const { configureApiFetch, apiFetch, currentAccessToken, noteSession } = await load()
  serving(200)
  configureApiFetch({ auth: authHolding('jwt-first') })

  await apiFetch('/api/jobs')
  assert.equal(currentAccessToken(), 'jwt-first')

  noteSession({ access_token: 'jwt-rotated' })
  assert.equal(currentAccessToken(), 'jwt-rotated')
})

test('losing the session clears the snapshot rather than leaving a dead token', async () => {
  // No header is the honest answer once there is no session. maplibre would
  // otherwise keep presenting a credential the server has stopped accepting.
  const { configureApiFetch, apiFetch, currentAccessToken, noteSession } = await load()
  serving(200)
  configureApiFetch({ auth: authHolding('jwt-live') })

  await apiFetch('/api/jobs')
  noteSession(null)

  assert.equal(currentAccessToken(), null)
})
