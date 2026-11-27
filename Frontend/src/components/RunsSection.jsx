import { useEffect, useRef, useState } from 'react'
import ModeBadge from './ModeBadge'
import QuestionPicker from './QuestionPicker'
import { runActions, runBadge, runQuestion, runStatusLabel } from '../utils/pausedRuns'

// The conversation's paused runs, listed in the jobs panel beside the
// earthdata jobs. Which buttons a row shows comes from utils/pausedRuns.js.

const CONFIRM_TIMEOUT_MS = 4000

const buttonBase = {
  fontSize: '11px', padding: '4px 10px', borderRadius: '6px', cursor: 'pointer', fontWeight: 600,
}
const ghostButtonStyle = { ...buttonBase, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-secondary)' }
const primaryButtonStyle = { ...buttonBase, border: 'none', background: 'var(--teal)', color: 'white' }
const dangerButtonStyle = { ...buttonBase, border: 'none', background: 'var(--error)', color: 'white' }

function AnswerModal({ question, hidden, onAnswer, onClose }) {
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.45)',
        display: hidden ? 'none' : 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: '12px',
          width: 'min(520px, 94vw)', maxHeight: '80vh', overflowY: 'auto', padding: '14px 16px',
          display: 'flex', flexDirection: 'column', gap: '10px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            aria-label="Close"
            style={{ background: 'none', border: 'none', color: 'var(--text-hint)', cursor: 'pointer', fontSize: '18px', lineHeight: 1 }}
          >
            ×
          </button>
        </div>
        <QuestionPicker question={question} onAnswer={onAnswer} />
      </div>
    </div>
  )
}

function RunRow({ run, busy, onCancel, onContinue, onAnswer }) {
  const [confirming, setConfirming] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  // 'open' while the user edits; 'sent' while an answer is being sent, the
  // modal hidden but mounted so a refused send keeps the edits; null when closed.
  const [answering, setAnswering] = useState(null)
  const confirmTimeoutRef = useRef(null)

  useEffect(() => () => clearTimeout(confirmTimeoutRef.current), [])

  const actions = runActions(run)
  const question = runQuestion(run)

  async function handleAnswer(request) {
    setAnswering('sent')
    const sent = await onAnswer(request)
    setAnswering(sent === false ? 'open' : null)
  }

  function handleCancelClick() {
    if (!confirming) {
      setConfirming(true)
      confirmTimeoutRef.current = setTimeout(() => setConfirming(false), CONFIRM_TIMEOUT_MS)
      return
    }
    clearTimeout(confirmTimeoutRef.current)
    setConfirming(false)
    setCancelling(true)
    Promise.resolve(onCancel(run.run_id)).finally(() => setCancelling(false))
  }

  function handleCancelBlur() {
    clearTimeout(confirmTimeoutRef.current)
    setConfirming(false)
  }

  return (
    <div style={{
      padding: '10px 12px', borderRadius: '10px', background: 'var(--bg-card)', boxShadow: 'var(--shadow-sm)',
      display: 'flex', flexDirection: 'column', gap: '6px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <ModeBadge badge={runBadge(run)} />
        <span style={{ fontSize: '11px', fontWeight: 500, color: 'var(--warning)', textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>
          {cancelling ? 'Cancelling…' : runStatusLabel(run)}
        </span>
      </div>
      {run.reason && (
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{run.reason}</div>
      )}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '6px' }}>
        {actions.answer && !cancelling && (
          <button onClick={() => setAnswering('open')} disabled={busy} style={primaryButtonStyle}>Answer</button>
        )}
        {actions.continue && !cancelling && (
          <button onClick={() => onContinue(run)} disabled={busy} style={primaryButtonStyle}>Continue</button>
        )}
        {actions.cancel && !cancelling && (
          <button
            onClick={handleCancelClick}
            onBlur={handleCancelBlur}
            disabled={busy}
            style={confirming ? dangerButtonStyle : ghostButtonStyle}
          >
            {confirming ? 'Confirm cancel?' : 'Cancel'}
          </button>
        )}
      </div>
      {answering && question && (
        <AnswerModal
          question={question}
          hidden={answering === 'sent'}
          onAnswer={handleAnswer}
          onClose={() => setAnswering(null)}
        />
      )}
    </div>
  )
}

export default function RunsSection({ runs, error, busy, onCancel, onContinue, onAnswer }) {
  if (runs.length === 0 && !error) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
        Paused runs
      </span>
      {error && (
        <div style={{ color: 'var(--error)', fontSize: '12px', padding: '4px 2px' }}>{error}</div>
      )}
      {runs.map(run => (
        <RunRow
          key={run.run_id}
          run={run}
          busy={busy}
          onCancel={onCancel}
          onContinue={onContinue}
          onAnswer={onAnswer}
        />
      ))}
    </div>
  )
}
