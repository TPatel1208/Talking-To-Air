import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
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
    // plotly.js-dist-min must be pre-bundled so Vite serves it as a single ESM chunk.
    include: ['plotly.js-dist-min'],
    // Exclude the factory from pre-bundling so Vite doesn't wrap the CJS
    // module.exports = fn in an ESM namespace object.  The interop shim in
    // ChartMessage.jsx handles the remaining edge cases across Vite versions.
    exclude: ['react-plotly.js/factory'],
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
})