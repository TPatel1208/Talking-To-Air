import { useState } from 'react'
import {
  variableChoiceUsesModal,
  groupCandidatesByCategory,
  filterCandidates,
  formatValidFraction,
} from '../utils/variableChoice'

// T49: the deterministic variable-choice picker. When the T48 resolver's
// confidence was medium or low, the backend attaches a `variableChoice` payload
// to the turn; this renders it so the researcher picks the variable themselves
// instead of trusting (or re-typing over) a guess. Clicking a candidate sends
// its pre-composed prompt through the same `onSend` path as FollowupChips —
// the prompt reconstructs the full original request with that variable
// substituted in, so a click never depends on cross-turn context carryover.
//
// <=6 candidates render as inline chips (visually like FollowupChips); a wide
// file (the 400-variable AERDA case that motivated T49) opens a searchable,
// category-grouped modal instead of an unscrollable wall of chips. All logic
// lives in utils/variableChoice.js; this is the thin rendering shell.

function CandidateChip({ candidate, onSend }) {
  const meta = [candidate.units, formatValidFraction(candidate.validFraction)].filter(Boolean).join(' · ')
  return (
    <button
      onClick={() => onSend(candidate.prompt)}
      title={candidate.reasons?.length ? candidate.reasons[0] : undefined}
      style={{
        padding: '6px 12px', borderRadius: '100px',
        fontSize: '12px', fontFamily: 'var(--font)',
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
      <span style={{ fontFamily: 'var(--font-mono)' }}>{candidate.name}</span>
      {meta && <span style={{ color: 'var(--text-hint)', marginLeft: '6px' }}>{meta}</span>}
    </button>
  )
}

function CandidateRow({ candidate, onSelect }) {
  const meta = [candidate.units, formatValidFraction(candidate.validFraction)].filter(Boolean).join(' · ')
  return (
    <button
      onClick={() => onSelect(candidate.prompt)}
      style={{
        display: 'flex', flexDirection: 'column', gap: '2px', textAlign: 'left',
        width: '100%', padding: '8px 10px', borderRadius: '8px',
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        cursor: 'pointer', fontFamily: 'var(--font)',
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
      <span style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
          {candidate.name}
        </span>
        {meta && <span style={{ fontSize: '11px', color: 'var(--text-hint)' }}>{meta}</span>}
      </span>
      {candidate.reasons?.length > 0 && (
        <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{candidate.reasons[0]}</span>
      )}
    </button>
  )
}

function CandidateModal({ variableChoice, onSend, onClose }) {
  const [query, setQuery] = useState('')
  const groups = groupCandidatesByCategory(filterCandidates(variableChoice.candidates, query))

  const select = (prompt) => {
    onSend(prompt)
    onClose()
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.45)', display: 'flex',
        alignItems: 'center', justifyContent: 'center', padding: '16px',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--bg-primary)', border: '1px solid var(--border)',
          borderRadius: '12px', width: 'min(560px, 94vw)', maxHeight: '80vh',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}
      >
        <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', flex: 1 }}>
              {variableChoice.message}
            </span>
            <button
              onClick={onClose}
              aria-label="Close"
              style={{ background: 'none', border: 'none', color: 'var(--text-hint)', cursor: 'pointer', fontSize: '18px', lineHeight: 1 }}
            >
              ×
            </button>
          </div>
          <input
            autoFocus
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search variables…"
            style={{
              padding: '8px 10px', borderRadius: '8px', fontSize: '12px',
              fontFamily: 'var(--font)', background: 'var(--bg-secondary)',
              border: '1px solid var(--border)', color: 'var(--text-primary)',
            }}
          />
        </div>

        <div style={{ padding: '12px 16px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {groups.length === 0 ? (
            <div style={{ fontSize: '12px', color: 'var(--text-hint)', fontStyle: 'italic' }}>
              No variables match “{query}”.
            </div>
          ) : (
            groups.map(group => (
              <div key={group.category} style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                  {group.label} ({group.items.length})
                </span>
                {group.items.map(candidate => (
                  <CandidateRow key={candidate.name} candidate={candidate} onSelect={select} />
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default function VariableChoicePicker({ variableChoice, onSend }) {
  const [modalOpen, setModalOpen] = useState(false)
  if (!variableChoice?.candidates?.length || !onSend) return null

  const useModal = variableChoiceUsesModal(variableChoice.candidates)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', padding: '4px 2px 0' }}>
      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{variableChoice.message}</span>
      {useModal ? (
        <div>
          <button
            onClick={() => setModalOpen(true)}
            style={{
              padding: '6px 12px', borderRadius: '100px', fontSize: '12px', fontWeight: 600,
              fontFamily: 'var(--font)', background: 'var(--bg-card)', border: '1px solid var(--teal)',
              color: 'var(--teal-text)', cursor: 'pointer',
            }}
          >
            Choose a variable ({variableChoice.candidates.length} options)
          </button>
          {modalOpen && (
            <CandidateModal
              variableChoice={variableChoice}
              onSend={onSend}
              onClose={() => setModalOpen(false)}
            />
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          {variableChoice.candidates.map(candidate => (
            <CandidateChip key={candidate.name} candidate={candidate} onSend={onSend} />
          ))}
        </div>
      )}
    </div>
  )
}
