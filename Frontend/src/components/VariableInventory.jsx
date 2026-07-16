import { groupInventoryByRole, confidenceHint } from '../utils/variableRoles'

// Grouped, name-first variable inventory (PRD T35). Renders the backend's
// classified `inventory` as role sections — science first, one clean
// Unclassified group last when non-empty — so a scientist can see at a glance
// what analyses a dataset makes possible ("does this product carry cloud
// fraction?" is a name question, not a count question). A Low/None-confidence
// entry carries a visible hedge rather than asserting a keyword guess.

function VariableRow({ entry }) {
  const hint = confidenceHint(entry.confidence)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', padding: '3px 0' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
          {entry.leaf || entry.name}
        </span>
        {entry.units && (
          <span style={{ fontSize: '10px', color: 'var(--text-hint)' }}>{entry.units}</span>
        )}
        {hint && (
          <span style={{
            fontSize: '9.5px', fontWeight: 600, color: 'var(--text-hint)',
            border: '1px solid var(--border)', borderRadius: '5px', padding: '0 4px',
          }}>
            {hint}
          </span>
        )}
      </div>
      {entry.long_name && entry.long_name !== entry.name && (
        <span style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.35 }}>
          {entry.long_name}
        </span>
      )}
    </div>
  )
}

export default function VariableInventory({ inventory }) {
  const groups = groupInventoryByRole(inventory)
  if (!groups.length) {
    return (
      <div style={{ fontSize: '11px', color: 'var(--text-hint)', fontStyle: 'italic', padding: '4px 0' }}>
        No variable metadata available for this dataset.
      </div>
    )
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {groups.map(group => (
        <div key={group.role} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{
              width: '7px', height: '7px', borderRadius: '2px', flexShrink: 0,
              background: group.accent, alignSelf: 'center',
            }} />
            <span style={{ fontSize: '11px', fontWeight: 800, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
              {group.label}
            </span>
            <span style={{ fontSize: '10px', color: 'var(--text-hint)' }}>
              {group.variables.length}
            </span>
          </div>
          <div style={{ borderLeft: `2px solid ${group.accent}`, paddingLeft: '9px', marginLeft: '3px' }}>
            {group.variables.map(entry => (
              <VariableRow key={entry.name} entry={entry} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
