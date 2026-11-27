import { useEffect, useState } from 'react'
import { cleanStatusMessage, runRowView, stageLabel } from '../utils/workflowStage'

const DOT = {
  width: '5px', height: '5px', borderRadius: '50%',
  background: 'var(--teal)', flexShrink: 0,
  animation: 'wm-bounce 1.2s ease-in-out infinite',
}

/* One row per sub-agent the supervisor runs in parallel, so
 * their narration reads apart instead of overwriting one line. */
function LaneRows({ lanes, elapsedSeconds }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '12px', lineHeight: 1.4 }}>
      {lanes.map((row, index) => (
        <div key={row.lane} style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
          {row.done
            ? <span style={{ width: '5px', flexShrink: 0, color: 'var(--teal)', fontSize: '10px' }}>✓</span>
            : <span style={DOT} />}
          <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{row.label}</span>
          <span style={{ opacity: row.done ? 0.6 : 1 }}>
            {row.message || (row.stage ? stageLabel(row.stage) || row.stage : '')}
          </span>
          {row.detail != null && !row.done && <span style={{ opacity: 0.75 }}>{String(row.detail)}</span>}
          {index === 0 && elapsedSeconds != null && (
            <span style={{ opacity: 0.6, marginLeft: 'auto' }}>{formatElapsed(elapsedSeconds)}</span>
          )}
        </div>
      ))}
    </div>
  )
}

/* One row per workflow run or exploration task. */
function RunRows({ runs, elapsedSeconds }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '12px', lineHeight: 1.4 }}>
      {runs.map((row, index) => {
        const { title, position, activity, current } = runRowView(row)
        return (
          <div key={row.runId} style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
            {row.done
              ? <span style={{ width: '5px', flexShrink: 0, color: 'var(--teal)', fontSize: '10px' }}>✓</span>
              : <span style={DOT} />}
            <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{title}</span>
            {position && <span style={{ opacity: 0.8 }}>{position}</span>}
            {activity && <span style={{ opacity: row.done ? 0.6 : 1 }}>{activity}</span>}
            {current && !row.done && <span style={{ opacity: 0.75 }}>{current}</span>}
            {index === 0 && elapsedSeconds != null && (
              <span style={{ opacity: 0.6, marginLeft: 'auto' }}>{formatElapsed(elapsedSeconds)}</span>
            )}
          </div>
        )
      })}
    </div>
  )
}

function formatElapsed(seconds) {
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${minutes}m ${remainder}s`
}

/* A live strip above the streaming answer — current stage label,
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
  const { stage, label, detail, failedStage, active, lanes, runs } = workflowStage
  if (!active && !failedStage) return null

  if (failedStage) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: '6px',
        fontSize: '11.5px', color: 'var(--error, #c0392b)', padding: '2px 0',
      }}>
        <span>
          Failed at: {workflowStage.failedLane ? `${workflowStage.failedLane} · ` : ''}
          {stageLabel(failedStage) || failedStage}
        </span>
      </div>
    )
  }

  const elapsedSeconds = startedAt != null ? Math.max(0, Math.round((now - startedAt) / 1000)) : null
  const displayLabel = cleanStatusMessage(label) || (stage ? stageLabel(stage) || stage : '')

  let displayDetail = null
  if (detail != null && stage !== 'working') {
    if (typeof detail === 'string') {
      const trimmed = detail.trim()
      if (trimmed && !displayLabel.includes(trimmed)) {
        displayDetail = trimmed
      }
    }
  }

  if (runs?.length) return <RunRows runs={runs} elapsedSeconds={elapsedSeconds} />
  if (lanes?.length) return <LaneRows lanes={lanes} elapsedSeconds={elapsedSeconds} />

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '8px',
      fontSize: '12px', color: 'var(--text-muted)', padding: '2px 0', lineHeight: 1.4,
    }}>
      <span style={DOT} />
      <span>{displayLabel}</span>
      {displayDetail != null && <span style={{ opacity: 0.75 }}>{displayDetail}</span>}
      {elapsedSeconds != null && (
        <span style={{ opacity: 0.6, marginLeft: 'auto' }}>{formatElapsed(elapsedSeconds)}</span>
      )}
    </div>
  )
}
