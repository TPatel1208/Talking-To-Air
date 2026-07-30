import { relatedVariableSections, hasRelatedVariables, roleLabel, ROLE_META } from '../utils/variableRoles'
import { buildRelatedVariableActions } from '../utils/companionActions'

// Chart-page related-variables panel (PRD T35/T36): the plotted variable's
// role plus its QA / uncertainty / context siblings, each with one-click
// actions that dispatch a prefilled prompt through the same `sendMessage`
// path as T22's FollowupChips. Actions are capability-gated in
// utils/companionActions.js -- this component only renders whatever that
// gate returns, it never invents a button. Renders nothing when a product
// carries no companions (e.g. MODIS AOD) -- no invented context.

function ActionButton({ label, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '4px 10px', borderRadius: '100px',
        fontSize: '11px', fontWeight: 600, fontFamily: 'var(--font)',
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        color: 'var(--teal-text)', cursor: 'pointer',
        transition: 'border-color 0.15s, background 0.15s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--teal)'
        e.currentTarget.style.background = 'var(--teal-light)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.background = 'var(--bg-card)'
      }}
    >
      {label}
    </button>
  )
}

function SiblingSection({ section, onSend }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
        {section.label}
      </span>
      {section.items.map(({ name, actions }) => (
        <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          <span style={{
            fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)',
            background: 'var(--bg-secondary)', border: '1px solid var(--border)',
            borderRadius: '6px', padding: '3px 8px', fontFamily: 'var(--font-mono)',
          }}>
            {name}
          </span>
          {onSend && actions.map(action => (
            <ActionButton key={action.key} label={action.label} onClick={() => onSend(action.prompt)} />
          ))}
        </div>
      ))}
    </div>
  )
}

export default function RelatedVariablesPanel({ chart, onSend }) {
  const related = chart?.provenance?.related_variables
  if (!hasRelatedVariables(related)) return null
  const view = relatedVariableSections(related)
  const accent = ROLE_META[view.role]?.accent || 'var(--text-hint)'
  const { sections, qaAction } = buildRelatedVariableActions(chart)

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

      {sections.length === 0 ? (
        <div style={{ fontSize: '11px', color: 'var(--text-hint)', fontStyle: 'italic' }}>
          This product carries no companion variables.
        </div>
      ) : (
        sections.map(section => (
          <SiblingSection key={section.key} section={section} onSend={onSend} />
        ))
      )}

      {onSend && qaAction && (
        <div style={{ display: 'flex', flexWrap: 'wrap' }}>
          <ActionButton label={qaAction.label} onClick={() => onSend(qaAction.prompt)} />
        </div>
      )}
    </div>
  )
}
