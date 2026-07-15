// Non-component helpers shared by the Metadata tab's JSX shells
// (OutputPanel.jsx, MetadataOverview.jsx). Kept out of
// components/metadataPrimitives.jsx because a file mixing component and
// non-component exports breaks Vite's fast-refresh
// (react-refresh/only-export-components).

export const smallButtonStyle = {
  border: '1px solid var(--border)', background: 'var(--bg-card)',
  color: 'var(--text-secondary)', borderRadius: '7px', padding: '4px 9px',
  fontSize: '11px', fontFamily: 'var(--font)', cursor: 'pointer',
}

export async function copyToClipboard(text, setState) {
  try {
    await navigator.clipboard.writeText(text)
    setState('Copied')
  } catch {
    setState('Copy failed')
  }
  window.setTimeout(() => setState(''), 1600)
}
