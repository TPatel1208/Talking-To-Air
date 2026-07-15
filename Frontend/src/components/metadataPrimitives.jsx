/**
 * components/metadataPrimitives.jsx
 * -----------------------------------
 * The label/value row shared by the Metadata tab's JSX shells
 * (OutputPanel.jsx) and MetadataOverview.jsx. Split out on its own so
 * MetadataOverview.jsx doesn't need to import OutputPanel.jsx (and vice
 * versa) just to reach it -- that mutual import was a circular dependency
 * between OutputPanel.jsx and CompareGrid.jsx (T34 review fix).
 *
 * smallButtonStyle/copyToClipboard live in utils/metadataUiHelpers.js
 * instead of here: a file mixing component and non-component exports
 * breaks Vite's fast-refresh (react-refresh/only-export-components).
 */
import { fmt } from '../utils/metadataDisplay'

export function MetaField({ label, value }) {
  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {label}
      </div>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', overflowWrap: 'anywhere', lineHeight: 1.45 }}>
        {fmt(value)}
      </div>
    </div>
  )
}
