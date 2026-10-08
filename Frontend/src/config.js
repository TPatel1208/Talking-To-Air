// Where the SPA sends API requests. Every backend URL in src/ is built from
// API_BASE, so this is the one place that decides it.
//
// The default '/api' is same-origin: nginx (and the Vite dev proxy) forward
// /api/* to the backend. Set VITE_API_URL at build time -- e.g.
// `docker build --build-arg VITE_API_URL=https://api.example.com` -- to serve
// the SPA from a different origin than the backend. Vite inlines the value
// when it builds, so changing it means rebuilding the image.

export function resolveApiBase(raw) {
  const value = typeof raw === 'string' ? raw.trim() : ''
  if (!value) return '/api'
  // Callers append paths that start with '/', so a trailing slash here would
  // produce '//chart/...'.
  return value.replace(/\/+$/, '') || '/api'
}

// `?.` because import.meta.env only exists under Vite; the node test runner
// imports modules that depend on this one.
export const API_BASE = resolveApiBase(import.meta.env?.VITE_API_URL)
