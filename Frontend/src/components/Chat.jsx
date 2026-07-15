import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import WorkflowStrip from './WorkflowStrip'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { starterMessage } from '../utils/starterPrompts'
import { compareBadgeLabel, isChartComparable, isSelectionFull, slotIndexOf } from '../utils/compareMode'
import { reachableArtifacts } from '../utils/artifactReachability'

const TYPE_LABEL = { map: 'Map', comparison: 'Comparison', timeseries: 'Time series', table: 'Table' }

function compactDate(value) {
  if (!value) return ''
  return String(value).replace('T00:00:00', '').replace('T23:59:59', '').replace(/Z$/, '')
}

function outputLabel(item) {
  if (item.kind === 'chart') {
    const chart = item.data
    const p = chart.provenance || {}
    const dateRange = [compactDate(p.start_date), compactDate(p.end_date)].filter(Boolean).join(' – ')
    return {
      title: chart.title || p.variable || chart.variable || 'Output',
      subtitle: [p.region_name, dateRange].filter(Boolean).join(' · '),
    }
  }
  const artifact = item.data
  return { title: artifact.title || 'Output', subtitle: TYPE_LABEL[artifact.type] || artifact.type }
}

const API_BASE = '/api'

function toImageUrl(path) {
  if (!path) return null
  if (path.startsWith('http')) return path
  if (path.startsWith('/outputs/')) return `/api${path}`
  return path
}

/* ── One step inside the collapsed tool-call card ── */
function ToolStep({ tc }) {
  const [showArgs, setShowArgs] = useState(false)
  const argStr  = tc.args ? JSON.stringify(tc.args, null, 2) : ''
  const hasArgs = argStr && argStr !== '{}'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '7px', fontSize: '12px', color: 'var(--text-secondary)' }}>
        <span style={{ color: 'var(--teal)', fontWeight: 700, flexShrink: 0 }}>✓</span>
        <span
          onClick={() => hasArgs && setShowArgs(v => !v)}
          style={{ fontFamily: 'var(--font-mono)', cursor: hasArgs ? 'pointer' : 'default', userSelect: 'none' }}
        >
          {tc.name}()
        </span>
      </div>
      {showArgs && (
        <pre style={{
          margin: '0 0 0 20px', padding: '6px 8px',
          background: 'var(--bg-card)', borderRadius: '6px',
          fontSize: '10px', overflowX: 'auto',
          color: 'var(--text-secondary)', border: '1px solid var(--border)',
        }}>
          {argStr}
        </pre>
      )}
    </div>
  )
}

/* ── Collapsible "ask_earthdata_agent() · N steps" card ── */
function ToolStepsCard({ toolCalls, error }) {
  const [expanded, setExpanded] = useState(false)
  if (!toolCalls?.length) return null

  return (
    <div style={{
      border: '1px solid var(--border)', borderRadius: '10px',
      background: 'var(--bg-secondary)', overflow: 'hidden',
    }}>
      <div
        onClick={() => setExpanded(v => !v)}
        style={{ display: 'flex', alignItems: 'center', gap: '7px', padding: '9px 12px', cursor: 'pointer' }}
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--teal)" strokeWidth="2" style={{ flexShrink: 0 }}>
          <path d="M14.5 3.5a4 4 0 0 0-5.4 4.7L3 14.3v3.2h3.2l6.1-6.1a4 4 0 0 0 4.7-5.4l-2.9 2.9-2-2z" strokeLinejoin="round" />
        </svg>
        <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-secondary)', flex: 1 }}>
          ask_earthdata_agent() · {error ? 'error' : `${toolCalls.length} step${toolCalls.length === 1 ? '' : 's'}`}
        </span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          style={{ color: 'var(--text-muted)', transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform .15s' }}>
          <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      {expanded && (
        <div style={{ padding: '0 12px 11px 34px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {toolCalls.map((tc, i) => <ToolStep key={i} tc={tc} />)}
        </div>
      )}
    </div>
  )
}

/* ── Compact output card — click opens the full view in the central
    OutputPanel, or (while compare mode is active) toggles slot membership.
    `inert` cards (wrong chart type while comparing) are dimmed and
    unclickable; `slotBadge` shows which comparison slot a card occupies. ── */
function OutputCard({ item, active, inert, slotBadge, onClick }) {
  const { title, subtitle } = outputLabel(item)
  return (
    <div
      onClick={inert ? undefined : onClick}
      style={{
        display: 'flex', gap: '11px', alignItems: 'center',
        border: `1px solid ${active || slotBadge ? 'var(--teal)' : 'var(--border)'}`,
        borderRadius: '10px', padding: '10px', background: 'var(--bg-card)',
        cursor: inert ? 'not-allowed' : 'pointer',
        opacity: inert ? 0.45 : 1,
        transition: 'border-color 0.15s, opacity 0.15s',
      }}
    >
      <div style={{
        width: '44px', height: '44px', borderRadius: '7px', flexShrink: 0,
        background: 'linear-gradient(135deg, oklch(0.55 0.15 240), oklch(0.75 0.18 80), oklch(0.6 0.2 30))',
      }} />
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{
          fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {title}
        </div>
        {subtitle && (
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '1px' }}>{subtitle}</div>
        )}
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '5px' }}>
          <div style={{
            display: 'inline-block', fontSize: '10.5px', fontWeight: 700,
            color: 'var(--teal-text)', background: 'var(--teal-light)',
            borderRadius: '5px', padding: '2px 7px',
          }}>
            ✓ Completed
          </div>
          {slotBadge && (
            <div style={{
              display: 'inline-block', fontSize: '10.5px', fontWeight: 700,
              color: 'var(--teal-text)', background: 'var(--bg-card)',
              border: '1px solid var(--teal)',
              borderRadius: '5px', padding: '2px 7px',
            }}>
              {slotBadge}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── Loading indicator ── */
function LoadingMessage({ toolCalls, statusMessage, workflowStage, startedAt }) {
  const hasWorkflowStrip = workflowStage?.active
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'flex-start', gap: '10px' }}>
      {/* Avatar */}
      <div style={{
        width: '28px', height: '28px', borderRadius: '50%',
        background: 'var(--teal-light)', display: 'flex',
        alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: '2px',
      }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--teal)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"/><path d="M2 12h4M18 12h4M12 2v4M12 18v4"/>
        </svg>
      </div>

      <div style={{
        maxWidth: '78%', padding: '10px 14px',
        borderRadius: '14px 14px 14px 4px',
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        fontSize: '13px', lineHeight: '1.6', minWidth: '72px',
      }}>
        {toolCalls?.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {toolCalls.map((tc, i) => (
              <ToolStep key={i} tc={tc} />
            ))}
            {hasWorkflowStrip ? (
              <div style={{ marginTop: '5px', paddingLeft: '3px' }}>
                <WorkflowStrip workflowStage={workflowStage} startedAt={startedAt} />
              </div>
            ) : statusMessage && (
              <div style={{
                marginTop: '5px',
                paddingLeft: '3px',
                color: 'var(--text-muted)',
                fontSize: '12px',
                lineHeight: 1.45,
              }}>
                {statusMessage}
              </div>
            )}
            <div style={{ display: 'flex', gap: '4px', paddingTop: '8px' }}>
              {[0, 1, 2].map(i => (
                <div key={i} style={{
                  width: '4px', height: '4px', borderRadius: '50%',
                  background: 'var(--text-muted)',
                  animation: `wm-bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
                }}/>
              ))}
            </div>
          </div>
        ) : hasWorkflowStrip ? (
          <WorkflowStrip workflowStage={workflowStage} startedAt={startedAt} />
        ) : statusMessage ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '12px', lineHeight: 1.45 }}>
            {statusMessage}
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{
                width: '5px', height: '5px', borderRadius: '50%',
                background: 'var(--text-muted)',
                animation: `wm-bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
              }}/>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Inline image with lightbox ── */
function InlineImage({ url, accessToken }) {
  const [lightbox, setLightbox] = useState(false)
  const [blobUrl, setBlobUrl] = useState(null)
  const src = toImageUrl(url)

  useEffect(() => {
    if (!src || !accessToken || !src.startsWith('/api/outputs/')) {
      setBlobUrl(null)
      return undefined
    }

    let cancelled = false
    let objectUrl = null

    fetch(src, { headers: { Authorization: `Bearer ${accessToken}` } })
      .then(response => response.ok ? response.blob() : null)
      .then(blob => {
        if (!blob || cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setBlobUrl(objectUrl)
      })
      .catch(() => {
        if (!cancelled) setBlobUrl(null)
      })

    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [src, accessToken])

  if (!src) return null
  const displaySrc = blobUrl || src

  return (
    <>
      <div style={{ margin: '8px 0' }}>
        <img
          src={displaySrc}
          alt="output"
          onClick={() => setLightbox(true)}
          style={{
            maxWidth: '100%', maxHeight: '400px',
            borderRadius: '10px', border: '1px solid var(--border)',
            display: 'block', cursor: 'zoom-in', objectFit: 'contain',
          }}
        />
        <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
          Click to enlarge
        </div>
      </div>

      {lightbox && (
        <div
          onClick={() => setLightbox(false)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(44,42,40,0.82)',
            zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'zoom-out',
          }}
        >
          <img
            src={displaySrc} alt="fullscreen"
            style={{ maxWidth: '92vw', maxHeight: '92vh', borderRadius: '10px', objectFit: 'contain' }}
          />
        </div>
      )}
    </>
  )
}

/* ── Follow-up suggestion chips (T22) ── */
function FollowupChips({ suggestions, onSend }) {
  if (!suggestions?.length) return null

  return (
    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', padding: '2px 2px 0' }}>
      {suggestions.map((text, i) => (
        <button
          key={i}
          onClick={() => onSend(text)}
          style={{
            padding: '6px 12px', borderRadius: '100px',
            fontSize: '12px', fontFamily: 'var(--font)',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            color: 'var(--teal-text)', cursor: 'pointer',
            transition: 'border-color 0.15s, background 0.15s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = 'var(--teal)'
            e.currentTarget.style.background  = 'var(--teal-light)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'var(--border)'
            e.currentTarget.style.background  = 'var(--bg-card)'
          }}
        >
          {text}
        </button>
      ))}
    </div>
  )
}

/* ── Message bubble ── */
function MessageBubble({
  msg, accessToken, onFollowupClick, focusedOutput, onFocusOutput,
  compareMode, compareSelection, onToggleCompareSlot, onCompareCapFull,
}) {
  const isUser = msg.role === 'user'
  const comparing = compareMode === 'active'

  const handleChartCardClick = (chart) => {
    if (!comparing) {
      onFocusOutput({ kind: 'chart', data: chart })
      return
    }
    if (!isChartComparable(chart, compareSelection)) return
    const alreadyIn = slotIndexOf(compareSelection, chart) !== -1
    if (!alreadyIn && isSelectionFull(compareSelection)) {
      onCompareCapFull()
      return
    }
    onToggleCompareSlot(chart)
  }

  const handleArtifactCardClick = (artifact) => {
    if (comparing) return // table artifacts aren't comparable in T28
    onFocusOutput({ kind: 'artifact', data: artifact })
  }

  return (
    <div style={{
      display: 'flex', gap: '10px',
      flexDirection: isUser ? 'row-reverse' : 'row',
      alignItems: 'flex-start',
    }}>
      {/* Avatar */}
      <div style={{
        width: '28px', height: '28px', borderRadius: '50%', flexShrink: 0, marginTop: '2px',
        background: isUser ? 'var(--bg-tertiary)' : 'var(--teal-light)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {isUser ? (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
          </svg>
        ) : (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--teal)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><path d="M2 12h4M18 12h4M12 2v4M12 18v4"/>
          </svg>
        )}
      </div>

      {/* Bubble content */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxWidth: '78%', minWidth: 0 }}>
        {/* Tool calls above bubble */}
        {!isUser && msg.toolCalls?.length > 0 && (
          <ToolStepsCard toolCalls={msg.toolCalls} error={msg.isError} />
        )}

        {(msg.content || msg.imageUrls?.length > 0 || isUser) && (
          <div
            className="msg-bubble"
            style={{
              padding:      '10px 14px',
              borderRadius: isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
              background:   isUser
                ? 'var(--accent)'
                : msg.isError ? 'var(--error-bg)' : 'var(--bg-card)',
              color:        isUser
                ? 'var(--accent-text)'
                : msg.isError ? 'var(--error)' : 'var(--text-primary)',
              border:       isUser
                ? 'none'
                : msg.isError ? '1px solid var(--error-border)' : '1px solid var(--border)',
              fontSize:     '13.5px',
              lineHeight:   '1.65',
              wordBreak:    'break-word',
              overflow:     'hidden',
            }}
          >
            {isUser ? (
              <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
            ) : (
              <>
                {msg.isError && msg.workflowStage?.failedStage && (
                  <WorkflowStrip workflowStage={msg.workflowStage} />
                )}
                {msg.content && (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                      p:      ({ children }) => <p style={{ margin: '0 0 8px' }}>{children}</p>,
                      h1:     ({ children }) => <p style={{ margin: '0 0 6px', fontWeight: 700, fontSize: '16px' }}>{children}</p>,
                      h2:     ({ children }) => <p style={{ margin: '0 0 6px', fontWeight: 500, fontSize: '14px' }}>{children}</p>,
                      h3:     ({ children }) => <p style={{ margin: '0 0 4px', fontWeight: 500 }}>{children}</p>,
                      ul:     ({ children }) => <ul style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ul>,
                      ol:     ({ children }) => <ol style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ol>,
                      li:     ({ children }) => <li style={{ marginBottom: '3px' }}>{children}</li>,
                      strong: ({ children }) => <strong style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{children}</strong>,
                      // ── FIX: react-markdown v10 removed the `inline` prop.
                      // Detect inline vs block via className: fenced blocks get
                      // className="language-*"; bare backtick spans do not.
                      // We also override `pre` so the outer wrapper is ours.
                      pre: ({ children }) => (
                        <pre style={{
                          background: 'var(--bg-secondary)', borderRadius: '8px',
                          padding: '10px 12px', overflowX: 'auto', fontSize: '12px',
                          fontFamily: 'var(--font-mono, monospace)', margin: '4px 0 8px',
                          border: '1px solid var(--border)',
                        }}>{children}</pre>
                      ),
                      code: ({ className, children }) => {
                        // Fenced code blocks have a className like "language-python".
                        // Inline backtick spans have no className.
                        const isBlock = Boolean(className)
                        if (isBlock) {
                          // Rendered inside our custom <pre> above — just emit <code>
                          return (
                            <code style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: '12px' }}>
                              {children}
                            </code>
                          )
                        }
                        // Inline code
                        return (
                          <code style={{
                            background: 'var(--bg-secondary)', borderRadius: '4px',
                            padding: '1px 5px', fontFamily: 'var(--font-mono, monospace)',
                            fontSize: '12px', color: 'var(--teal-text)',
                          }}>{children}</code>
                        )
                      },
                      a: ({ href, children }) => (
                        <a href={href} target="_blank" rel="noreferrer"
                          style={{ color: 'var(--teal-text)', textDecoration: 'underline', textUnderlineOffset: '2px' }}>
                          {children}
                        </a>
                      ),
                      table: ({ children }) => (
                        <div style={{ overflowX: 'auto', margin: '8px 0' }}>
                          <table style={{ borderCollapse: 'collapse', fontSize: '12px', width: '100%' }}>{children}</table>
                        </div>
                      ),
                      th: ({ children }) => (
                        <th style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', textAlign: 'left', fontWeight: 500, color: 'var(--text-secondary)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{children}</th>
                      ),
                      td: ({ children }) => (
                        <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)' }}>{children}</td>
                      ),
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                )}
                {msg.imageUrls?.filter(Boolean).map((url, i) => (
                  <InlineImage key={i} url={url} accessToken={accessToken} />
                ))}
              </>
            )}
          </div>
        )}

        {/* Output cards — click opens the full map/chart/table in the central
            OutputPanel. Chart-backed artifact types (map/comparison/chart-backed
            timeseries) duplicate what msg.charts already covers, so only 'table'
            and ground-validation 'timeseries' artifacts (no matching chart, T33)
            get their own card here. */}
        {!isUser && (msg.charts?.length > 0 || reachableArtifacts(msg).length > 0) && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            {msg.charts?.map((chart, i) => (
              <OutputCard
                key={`chart-${i}`}
                item={{ kind: 'chart', data: chart }}
                active={!comparing && focusedOutput?.kind === 'chart' && focusedOutput.data === chart}
                inert={comparing && !isChartComparable(chart, compareSelection)}
                slotBadge={comparing ? compareBadgeLabel(compareSelection, chart) : null}
                onClick={() => handleChartCardClick(chart)}
              />
            ))}
            {reachableArtifacts(msg).map((artifact, i) => (
              <OutputCard
                key={artifact.id || `artifact-${i}`}
                item={{ kind: 'artifact', data: artifact }}
                active={!comparing && focusedOutput?.kind === 'artifact' && focusedOutput.data === artifact}
                inert={comparing}
                onClick={() => handleArtifactCardClick(artifact)}
              />
            ))}
          </div>
        )}

        {/* T22: follow-up suggestions grounded in this turn's answer —
            clicking one sends it through the same path as typing. */}
        {!isUser && !msg.isLoading && (
          <FollowupChips suggestions={msg.suggestedFollowups} onSend={onFollowupClick} />
        )}
      </div>
    </div>
  )
}

/* ── Empty state ── */
function EmptyState({ onChipClick }) {
  // T22: fetched from the backend's own starter-prompt constant (config/
  // starter_prompts.py) rather than hardcoded here, so every example on
  // screen is one the eval suite proves works end-to-end (story #4/#11).
  const [starters, setStarters] = useState([])

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/capabilities/starters`)
      .then(res => (res.ok ? res.json() : []))
      .then(data => { if (!cancelled) setStarters(Array.isArray(data) ? data : []) })
      .catch(() => { if (!cancelled) setStarters([]) })
    return () => { cancelled = true }
  }, [])

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', flex: 1, padding: '40px 24px 80px',
      animation: 'wm-fadein 0.35s ease both',
    }}>
      <div style={{
        width: '52px', height: '52px', borderRadius: '50%',
        background: 'var(--teal-light)', display: 'flex',
        alignItems: 'center', justifyContent: 'center', marginBottom: '18px',
      }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--teal)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <path d="M2 12h4M18 12h4M12 2v4M12 18v4"/>
          <path d="M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M19.07 4.93l-2.83 2.83M7.76 16.24l-2.83 2.83"/>
        </svg>
      </div>

      <h1 style={{
        fontWeight: '800', fontSize: '20px', color: 'var(--text-primary)',
        marginBottom: '8px', letterSpacing: '-0.01em',
      }}>
        Talking to Air
      </h1>
      <p style={{
        fontSize: '13px', color: 'var(--text-muted)',
        textAlign: 'center', lineHeight: '1.7', marginBottom: '28px',
        maxWidth: '320px',
      }}>
        Ask about NO₂, ozone, HCHO, aerosol optical depth,<br/>
        or any other air quality data you can think of.
      </p>

      {/* Starter prompts — span the app's workflow types (story #2) */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
        {starters.map(starter => (
          <button
            key={starter.id}
            onClick={() => onChipClick(starterMessage(starter))}
            style={{
              padding: '7px 14px', borderRadius: '100px',
              fontSize: '12px', fontFamily: 'var(--font)',
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              color: 'var(--text-secondary)', cursor: 'pointer',
              transition: 'border-color 0.15s, background 0.15s, color 0.15s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--teal)'
              e.currentTarget.style.background  = 'var(--teal-light)'
              e.currentTarget.style.color       = 'var(--teal-text)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--border)'
              e.currentTarget.style.background  = 'var(--bg-card)'
              e.currentTarget.style.color       = 'var(--text-secondary)'
            }}
          >
            {starter.label}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ── Main Chat component ── */
export default function Chat({
  messages, loading, error, accessToken, chatTitle, onSend, onAbort, onClearError,
  focusedOutput, onFocusOutput,
  compareMode = 'off', compareSelection = [], onToggleCompareSlot,
  onCollapse,
}) {
  const [input, setInput] = useState('')
  const scrollContainerRef = useRef(null)
  const textareaRef = useRef(null)

  // Transient "grid full" hint (T28) -- clicking an unselected, matching
  // card while every slot is already taken is inert except for this cue.
  const [capFullHint, setCapFullHint] = useState(false)
  const capFullTimeoutRef = useRef(null)
  const showCapFullHint = () => {
    setCapFullHint(true)
    if (capFullTimeoutRef.current) clearTimeout(capFullTimeoutRef.current)
    capFullTimeoutRef.current = setTimeout(() => setCapFullHint(false), 2400)
  }
  useEffect(() => () => {
    if (capFullTimeoutRef.current) clearTimeout(capFullTimeoutRef.current)
  }, [])

  useEffect(() => {
    const el = scrollContainerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, loading])

  const handleSend = (text) => {
    const msg = (text || input).trim()
    if (!msg) return
    onSend(msg)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (loading) return
      handleSend()
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'
  }

  const isEmpty = messages.length === 0
  const canSend = Boolean(input.trim())
  const visibleError = error?.startsWith('Failed to delete session')
    ? error
    : `Request failed: ${error}`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Top bar */}
      <div style={{
        padding: '16px 20px 14px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-card)', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
      }}>
        <span style={{
          fontSize: '15px', fontWeight: '800', color: 'var(--text-primary)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {chatTitle || 'New analysis'}
        </span>
        {onCollapse && (
          <button
            type="button"
            onClick={onCollapse}
            title="Collapse chat"
            aria-label="Collapse chat"
            style={{
              background: 'transparent', border: 'none', padding: '2px', cursor: 'pointer',
              display: 'flex', flexShrink: 0, color: 'var(--text-muted)',
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 6 9 12 15 18" />
            </svg>
          </button>
        )}
      </div>

      {/* Message area */}
      <div
        ref={scrollContainerRef}
        style={{
          flex: 1, minHeight: 0, overflowY: 'auto',
          display: 'flex', flexDirection: 'column',
        }}
      >
        {error && (
          <div style={{
            margin: '14px 20px 0',
            padding: '10px 12px',
            border: '1px solid var(--error-border)',
            borderRadius: '8px',
            background: 'var(--error-bg)',
            color: 'var(--error)',
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '12px',
            fontSize: '13px',
            lineHeight: 1.45,
          }}>
            <span>{visibleError}</span>
            <button
              onClick={onClearError}
              aria-label="Dismiss error"
              style={{
                border: 'none',
                background: 'transparent',
                color: 'var(--error)',
                cursor: 'pointer',
                fontSize: '16px',
                lineHeight: 1,
                padding: '0 2px',
              }}
            >
              X
            </button>
          </div>
        )}
        {compareMode === 'active' && capFullHint && (
          <div style={{
            margin: '14px 20px 0',
            padding: '9px 12px',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            background: 'var(--bg-secondary)',
            color: 'var(--text-secondary)',
            fontSize: '12.5px',
          }}>
            Compare grid full — remove one to add another.
          </div>
        )}
        {isEmpty ? (
          <EmptyState onChipClick={handleSend} />
        ) : (
          <div style={{ padding: '20px 20px 8px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {messages.map((msg, i) =>
              msg.isLoading ? (
                <LoadingMessage
                  key={i}
                  toolCalls={msg.toolCalls}
                  statusMessage={msg.statusMessage}
                  workflowStage={msg.workflowStage}
                  startedAt={msg.startedAt}
                />
              ) : (
                <MessageBubble
                  key={i}
                  msg={msg}
                  accessToken={accessToken}
                  onFollowupClick={handleSend}
                  focusedOutput={focusedOutput}
                  onFocusOutput={onFocusOutput}
                  compareMode={compareMode}
                  compareSelection={compareSelection}
                  onToggleCompareSlot={onToggleCompareSlot}
                  onCompareCapFull={showCapFullHint}
                />
              )
            )}
            <div style={{ height: '8px' }} />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div style={{
        padding: '12px 16px 16px',
        background: 'var(--bg-primary)',
        borderTop: isEmpty ? 'none' : '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <div style={{
          display: 'flex', alignItems: 'flex-end', gap: '8px',
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          borderRadius: '18px', padding: '8px 8px 8px 18px',
          boxShadow: 'var(--shadow-md)',
          maxWidth: '760px', margin: '0 auto',
          transition: 'border-color 0.15s',
        }}
          onFocusCapture={e => e.currentTarget.style.borderColor = 'var(--border-hover)'}
          onBlurCapture={e => e.currentTarget.style.borderColor = 'var(--border)'}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKey}
            placeholder="Ask about air quality data…"
            rows={1}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text-primary)', fontSize: '13.5px',
              fontFamily: 'var(--font)', lineHeight: '1.6',
              resize: 'none', minHeight: '24px', maxHeight: '140px',
              overflowY: 'auto', paddingTop: '2px', paddingBottom: '2px',
            }}
          />
          <button
            onClick={() => loading ? onAbort() : handleSend()}
            disabled={!loading && !canSend}
            aria-label={loading ? 'Stop request' : 'Send message'}
            style={{
              width: '34px', height: '34px', borderRadius: '50%', flexShrink: 0,
              background: loading || canSend ? 'var(--accent)' : 'var(--bg-tertiary)',
              color: loading || canSend ? 'var(--accent-text)' : 'var(--text-hint)',
              border: 'none', cursor: loading || canSend ? 'pointer' : 'not-allowed',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.15s, color 0.15s, transform 0.1s',
              fontSize: '16px',
            }}
            onMouseDown={e => { if (loading || canSend) e.currentTarget.style.transform = 'scale(0.93)' }}
            onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}
          >
            {loading ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="2"/>
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>
              </svg>
            )}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes wm-bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
          40%            { transform: translateY(-5px); opacity: 1; }
        }
        @keyframes wm-fadein {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes wm-spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        textarea::placeholder { color: var(--text-hint); }
        .msg-bubble p:last-child,
        .msg-bubble ul:last-child,
        .msg-bubble ol:last-child { margin-bottom: 0; }
        .katex { font-size: 1em; }
      `}</style>
    </div>
  )
}
