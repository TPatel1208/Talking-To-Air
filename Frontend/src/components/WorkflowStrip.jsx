import { useEffect, useState } from 'react'

const STAGE_LABELS = {
  search: 'Searching',
  aoi: 'Resolving area',
  coverage: 'Checking coverage',
  estimate: 'Estimating size',
  submit: 'Submitting retrieval',
  progress: 'Retrieving data',
  open: 'Opening data',
  render: 'Rendering',
  working: 'Still working',
}

function formatElapsed(seconds) {
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${minutes}m ${remainder}s`
}

/* T19: a live strip above the streaming answer — current stage label,
 * progress/detail when present, elapsed time. Collapses (returns null) the
 * moment the strip's `active` flag goes false (the first answer token
 * arrived); on error, shows the stage that was in progress when it failed. */
export default function WorkflowStrip({ workflowStage, startedAt }) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!workflowStage?.active) return undefined
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [workflowStage?.active])

  if (!workflowStage) return null
  const { stage, label, detail, failedStage, active } = workflowStage
  if (!active && !failedStage) return null

  if (failedStage) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: '6px',
        fontSize: '11.5px', color: 'var(--error, #c0392b)', padding: '2px 0',
      }}>
        <span>Failed at: {STAGE_LABELS[failedStage] || failedStage}</span>
      </div>
    )
  }

  const elapsedSeconds = startedAt != null ? Math.max(0, Math.round((now - startedAt) / 1000)) : null

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '8px',
      fontSize: '12px', color: 'var(--text-muted)', padding: '2px 0', lineHeight: 1.4,
    }}>
      <span style={{
        width: '5px', height: '5px', borderRadius: '50%',
        background: 'var(--teal)', flexShrink: 0,
        animation: 'wm-bounce 1.2s ease-in-out infinite',
      }} />
      <span>{label || (stage ? STAGE_LABELS[stage] || stage : '')}</span>
      {detail != null && <span style={{ opacity: 0.75 }}>{String(detail)}</span>}
      {elapsedSeconds != null && (
        <span style={{ opacity: 0.6, marginLeft: 'auto' }}>{formatElapsed(elapsedSeconds)}</span>
      )}
    </div>
  )
}
