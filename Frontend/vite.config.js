import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// Imported rather than used as a global: eslint.config.js applies
// `globals.browser` to every .js file, this one included, so a bare `process`
// is a `no-undef` error even though Vite evaluates this file in Node.
import process from 'node:process'

// Where `npm run dev` sends /api. Defaults to the nginx edge, because that is
// the stack's only published entrance: the backend deliberately publishes no
// host port (see docker-compose.yml), so the old 'http://localhost:8000' target
// now refuses the connection and every /api call from the dev server 502s.
//
// Going through nginx is also the more faithful target -- the dev server then
// exercises the same rate limits and forwarded-header handling as production.
// To bypass it, bring up docker-compose.debug.yml and set
// VITE_DEV_API_TARGET=http://localhost:8000.
const apiTarget = process.env.VITE_DEV_API_TARGET || 'https://localhost'

// nginx already maps /api/* to the backend's /*, so the prefix must survive the
// hop; a direct-to-backend target has no such mapping and needs it stripped.
const stripApiPrefix = /:8000(\/|$)/.test(apiTarget)

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        // The local edge serves a mkcert certificate, which Node does not
        // trust the way an mkcert-installed browser does. Only relaxed for the
        // dev proxy; nothing here ships.
        secure: false,
        ...(stripApiPrefix
          ? { rewrite: (path) => path.replace(/^\/api/, '') }
          : {}),
      },
    },
  },
  resolve: {
    alias: {
      // Redirect any import of the full plotly bundle to the lightweight build.
      // Without this, react-plotly.js's own entry point pulls plotly.js/dist/plotly
      // (~11 MB), creating a second Plotly instance alongside plotly.js-dist-min
      // and triggering React error #130.
      'plotly.js/dist/plotly': 'plotly.js-dist-min',
    },
  },
  optimizeDeps: {
    // Both of these must be pre-bundled so Vite serves them as ESM chunks.
    //
    // react-plotly.js/factory is CommonJS (Babel output: `__esModule: true`
    // plus `exports.default = plotComponentFactory`). It used to be listed
    // under `exclude`, on the theory that pre-bundling would wrap it in an
    // ESM namespace object. That backfired as of Vite 8: an excluded dep is
    // served raw, with no CJS-to-ESM interop, so the browser parsed
    // `exports.default = ...` as a bare assignment and the module really did
    // export nothing -- "does not provide an export named 'default'", thrown
    // at link time, before the interop shim in ChartMessage.jsx could run.
    //
    // Pre-bundled, the default import is the CJS exports object
    // (`{__esModule: true, default: fn}`); the shim in ChartMessage.jsx
    // unwraps that last hop, so both pieces are load-bearing.
    include: ['plotly.js-dist-min', 'react-plotly.js/factory'],
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
})