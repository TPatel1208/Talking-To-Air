import test from 'node:test'
import assert from 'node:assert/strict'

// The Supabase client cannot be a module-level singleton: its URL and
// publishable key arrive from GET /config/auth at runtime (decision 11), so it
// is constructed after that resolves. But it must still be constructed exactly
// ONCE -- two GoTrueClients sharing one storage key race each other's refreshes
// and each clobbers the other's tokens. `ensureSupabaseClient` is therefore an
// idempotent holder, and these tests pin that.
//
// Node keys its ESM cache on the full specifier including the query string, so
// a distinct `?case=` gives each test its own module instance -- a fresh
// holder, with no reset hook shipped in production code just to support tests.
let moduleCase = 0
const freshModule = () => import(`../src/utils/supabaseClient.js?case=${++moduleCase}`)

const CONFIG = {
  supabaseUrl: 'https://project-ref.supabase.co',
  publishableKey: 'sb_publishable_test',
}

test('the same config builds the client once, however many times it is asked for', async () => {
  const { ensureSupabaseClient } = await freshModule()
  let built = 0
  const factory = () => ({ instance: ++built })

  const first = ensureSupabaseClient(CONFIG, factory)
  // A DIFFERENT object carrying the SAME values -- which is what actually
  // happens, because this is called during render and the config object is
  // rebuilt each time. Identity has to be by value or StrictMode's double
  // render alone is enough to produce two clients.
  const second = ensureSupabaseClient({ ...CONFIG }, factory)

  assert.equal(built, 1)
  assert.equal(first, second)
})

test('the client is handed the url and key positionally, with the options this app depends on', async () => {
  const { ensureSupabaseClient } = await freshModule()
  const seen = []
  ensureSupabaseClient(CONFIG, (...args) => { seen.push(args); return {} })

  const [url, key, options] = seen[0]
  assert.equal(url, CONFIG.supabaseUrl)
  assert.equal(key, CONFIG.publishableKey)

  // A reload must not force a re-login.
  assert.equal(options.auth.persistSession, true)
  // The point of the 45-minute TTL (decision 10). Without this every session
  // dies at 45 minutes and T47's re-auth modal becomes a routine interruption
  // instead of the rare recovery path it is meant to be.
  assert.equal(options.auth.autoRefreshToken, true)
  // OAuth is a T61 non-goal. Left on, supabase-js parses and REWRITES the URL
  // hash on every load, which would fight T62's incoming router for ownership
  // of the URL.
  assert.equal(options.auth.detectSessionInUrl, false)
})

test('getSupabaseClient answers null before the config lands rather than throwing', async () => {
  // Phase 5's apiFetch reaches for the client on every request, including
  // whatever is in flight while /config/auth is still resolving. It needs a
  // defined "not yet" answer to branch on.
  const { getSupabaseClient, ensureSupabaseClient } = await freshModule()
  assert.equal(getSupabaseClient(), null)

  const client = ensureSupabaseClient(CONFIG, () => ({ ok: true }))
  assert.equal(getSupabaseClient(), client)
})

test('a second, different config is refused instead of silently building a rival client', async () => {
  const { ensureSupabaseClient } = await freshModule()
  ensureSupabaseClient(CONFIG, () => ({}))

  assert.throws(
    () => ensureSupabaseClient({ ...CONFIG, supabaseUrl: 'https://other.supabase.co' }, () => ({})),
    /already configured/i,
    'switching projects mid-session is a bug, not a supported operation -- returning the stale client would hide it and building a second would race refreshes',
  )
})

test('an incomplete config is refused at the seam, naming the backend variable', async () => {
  const { ensureSupabaseClient } = await freshModule()
  // GET /config/auth serves JSON null when the backend env var is unset, so
  // this is a real reachable state, and it must fail here with something
  // actionable rather than deep inside supabase-js.
  assert.throws(() => ensureSupabaseClient({ supabaseUrl: null, publishableKey: 'k' }, () => ({})), /SUPABASE_URL/)
  assert.throws(() => ensureSupabaseClient({ supabaseUrl: 'https://x.supabase.co', publishableKey: '' }, () => ({})), /SUPABASE_PUBLISHABLE_KEY/)
})
