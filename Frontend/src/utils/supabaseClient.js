import { createClient } from '@supabase/supabase-js'

// The identity provider's coordinates are served at runtime by
// GET /config/auth rather than baked into the bundle, so one frontend image
// runs against either the dev or the prod Supabase project. The cost is that
// the client cannot be built at import time -- it has nothing to build from
// until that request resolves.
//
// What it must NOT become is a client built per render. Two GoTrueClients
// sharing one storage key both run their own refresh timers against the same
// refresh token; whichever loses the race writes a token the other has already
// rotated away, and the session dies for no visible reason. React's StrictMode
// double-render is enough to produce that on its own. So: an idempotent holder,
// safe to call during render precisely because a repeat call is a no-op.
let client = null
let configuredWith = null

// Pinned rather than left to defaults because each one is load-bearing; see
// tests/supabaseClient.test.mjs for what breaks without it.
export const SUPABASE_AUTH_OPTIONS = {
  persistSession: true,
  autoRefreshToken: true,
  detectSessionInUrl: false,
}

function identityOf(config) {
  return `${config.supabaseUrl}\n${config.publishableKey}`
}

/**
 * Build the Supabase client, or hand back the one already built.
 *
 * `factory` exists so this is testable without reaching the network or a
 * browser -- the same injectable-key-source shape Phase 1 used on the backend
 * verifier (decision 9).
 */
export function ensureSupabaseClient(config, factory = createClient) {
  // /config/auth serves JSON null when the backend env var is unset, so an
  // incomplete config is reachable, not defensive. Name the backend variable:
  // the fix is always in the backend's environment, never in the browser.
  if (!config?.supabaseUrl) {
    throw new Error('Supabase is not configured: the backend served no SUPABASE_URL.')
  }
  if (!config?.publishableKey) {
    throw new Error('Supabase is not configured: the backend served no SUPABASE_PUBLISHABLE_KEY.')
  }

  const identity = identityOf(config)
  if (client) {
    // Compared by value, not by reference: this is called during render with a
    // config object that is rebuilt each time.
    if (identity !== configuredWith) {
      throw new Error('Supabase client is already configured for a different project.')
    }
    return client
  }

  configuredWith = identity
  client = factory(config.supabaseUrl, config.publishableKey, { auth: SUPABASE_AUTH_OPTIONS })
  return client
}

// Null until the config lands. Phase 5's apiFetch reaches for the client on
// every request, including ones in flight while /config/auth is still
// resolving, and needs a "not yet" it can branch on.
export function getSupabaseClient() {
  return client
}
