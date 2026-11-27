// The validated / draft / exploratory badge on an answer, chart or artifact.

const STYLES = {
  validated: { color: 'var(--teal-text)', background: 'var(--teal-light)', border: 'var(--teal)' },
  draft: { color: 'var(--text-primary)', background: 'var(--amber-bg, #fff8e6)', border: 'var(--amber, #d99100)' },
  exploratory: { color: 'var(--text-secondary)', background: 'var(--bg-secondary)', border: 'var(--border)' },
}

export default function ModeBadge({ badge }) {
  if (!badge) return null
  const style = STYLES[badge.kind] || STYLES.exploratory
  return (
    <span
      data-mode={badge.kind}
      style={{
        display: 'inline-block', fontSize: '10.5px', fontWeight: 700, lineHeight: 1.4,
        color: style.color, background: style.background, border: `1px solid ${style.border}`,
        borderRadius: '5px', padding: '1px 7px', width: 'fit-content',
      }}
    >
      {badge.text}
    </span>
  )
}
