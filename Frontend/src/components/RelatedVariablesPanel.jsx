import { relatedVariableSections, hasRelatedVariables, roleLabel, ROLE_META } from '../utils/variableRoles'

// Chart-page related-variables panel (PRD T35): the plotted variable's role
// plus its QA / uncertainty / context siblings — links only, not a re-render of
// the whole inventory, so the chart page stays focused while pointing at what's
// relevant. Built from the classification directly (chart provenance's
// `related_variables`), the thin edge of T36. Renders nothing when a product
// carries no companions (e.g. MODIS AOD) — no invented context.

function SiblingChips({ names }) {
  return (
    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
      {names.map(name => (
        <span key={name} style={{
          fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)',
          background: 'var(--bg-secondary)', border: '1px solid var(--border)',
          borderRadius: '6px', padding: '3px 8px', fontFamily: 'var(--font-mono)',
        }}>
          {name}
        </span>
      ))}
    </div>
  )
}

export default function RelatedVariablesPanel({ related }) {
  if (!hasRelatedVariables(related)) return null
  const view = relatedVariableSections(related)
  const accent = ROLE_META[view.role]?.accent || 'var(--text-hint)'

  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: '10px', padding: '12px 14px',
      display: 'flex', flexDirection: 'column', gap: '10px',
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '12px', fontWeight: 800, color: 'var(--text-primary)' }}>
          Related variables
        </span>
        {view.role && (
          <span style={{ fontSize: '11px', fontWeight: 700, color: accent }}>
            plotted: {roleLabel(view.role)}
          </span>
        )}
      </div>

      {view.sections.length === 0 ? (
        <div style={{ fontSize: '11px', color: 'var(--text-hint)', fontStyle: 'italic' }}>
          This product carries no companion variables.
        </div>
      ) : (
        view.sections.map(section => (
          <div key={section.key} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
              {section.label}
            </span>
            <SiblingChips names={section.names} />
          </div>
        ))
      )}
    </div>
  )
}
