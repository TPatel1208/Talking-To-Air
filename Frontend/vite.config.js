import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev server has no nginx in front of it, so proxy /api straight to the
    // backend container (see docker-compose.yml) the way nginx.conf does in prod.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
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