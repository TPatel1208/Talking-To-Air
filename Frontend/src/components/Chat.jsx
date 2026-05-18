import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

function toImageUrl(path) {
  if (!path) return null
  if (path.startsWith('http')) return path
  if (path.startsWith('/outputs/')) return `/api${path}`
  return path
}

/* ── Tool call badge (inline, in chat bubble) ── */
function InlineToolBadge({ name, args }) {
  const [expanded, setExpanded] = useState(false)
  const argStr  = args ? JSON.stringify(args, null, 2) : ''
  const hasArgs = argStr && argStr !== '{}'

  return (
    <div style={{
      display:    'flex',
      alignItems: 'flex-start',
      gap:        '6px',
      fontSize:   '11px',
      color:      'var(--text-muted)',
      padding:    '2px 0',
    }}>
      <span style={{
        width: '5px', height: '5px', borderRadius: '50%',
        background: 'var(--teal)', flexShrink: 0, marginTop: '4px',
      }}/>
      <div>
        <span
          onClick={() => hasArgs && setExpanded(e => !e)}
          style={{
            fontFamily: 'var(--font-mono, monospace)',
            color:      'var(--teal-text)',
            cursor:     hasArgs ? 'pointer' : 'default',
            userSelect: 'none',
            fontSize:   '11px',
          }}
        >
          {name}()
        </span>
        {hasArgs && (
          <span style={{ marginLeft: '4px', opacity: 0.5, fontSize: '10px' }}>
            {expanded ? '▲' : '▼'}
          </span>
        )}
        {expanded && (
          <pre style={{
            margin: '4px 0 0', padding: '6px 8px',
            background: 'var(--bg-secondary)', borderRadius: '6px',
            fontSize: '10px', overflowX: 'auto',
            color: 'var(--text-secondary)', border: '1px solid var(--border)',
          }}>
            {argStr}
          </pre>
        )}
      </div>
    </div>
  )
}

/* ── Loading indicator ── */
function LoadingMessage({ toolCalls }) {
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
              <InlineToolBadge key={i} name={tc.name} args={tc.args} />
            ))}
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
function InlineImage({ url }) {
  const [lightbox, setLightbox] = useState(false)
  const src = toImageUrl(url)
  if (!src) return null

  return (
    <>
      <div style={{ margin: '8px 0' }}>
        <img
          src={src}
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
            src={src} alt="fullscreen"
            style={{ maxWidth: '92vw', maxHeight: '92vh', borderRadius: '10px', objectFit: 'contain' }}
          />
        </div>
      )}
    </>
  )
}

/* ── Message bubble ── */
function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'

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
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxWidth: '78%' }}>
        {/* Tool calls above bubble */}
        {!isUser && msg.toolCalls?.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', padding: '0 2px' }}>
            {msg.toolCalls.map((tc, i) => (
              <InlineToolBadge key={i} name={tc.name} args={tc.args} />
            ))}
          </div>
        )}

        {(msg.content || msg.imageUrls?.length > 0 || isUser) && (
          <div
            className="msg-bubble"
            style={{
              padding:      '10px 14px',
              borderRadius: isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
              background:   isUser
                ? 'var(--accent)'
                : msg.isError ? '#fff0f0' : 'var(--bg-card)',
              color:        isUser
                ? 'var(--accent-text)'
                : msg.isError ? 'var(--error)' : 'var(--text-primary)',
              border:       isUser
                ? 'none'
                : msg.isError ? '1px solid #f5c6c6' : '1px solid var(--border)',
              fontSize:     '13.5px',
              lineHeight:   '1.65',
              wordBreak:    'break-word',
            }}
          >
            {isUser ? (
              <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
            ) : (
              <>
                {msg.content && (
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                      p:      ({ children }) => <p style={{ margin: '0 0 8px' }}>{children}</p>,
                      h1:     ({ children }) => <p style={{ margin: '0 0 6px', fontWeight: 500, fontSize: '16px', fontFamily: 'var(--font-serif)' }}>{children}</p>,
                      h2:     ({ children }) => <p style={{ margin: '0 0 6px', fontWeight: 500, fontSize: '14px' }}>{children}</p>,
                      h3:     ({ children }) => <p style={{ margin: '0 0 4px', fontWeight: 500 }}>{children}</p>,
                      ul:     ({ children }) => <ul style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ul>,
                      ol:     ({ children }) => <ol style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ol>,
                      li:     ({ children }) => <li style={{ marginBottom: '3px' }}>{children}</li>,
                      strong: ({ children }) => <strong style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{children}</strong>,
                      code: ({ inline, children }) => inline ? (
                        <code style={{
                          background: 'var(--bg-secondary)', borderRadius: '4px',
                          padding: '1px 5px', fontFamily: 'var(--font-mono, monospace)',
                          fontSize: '12px', color: 'var(--teal-text)',
                        }}>{children}</code>
                      ) : (
                        <pre style={{
                          background: 'var(--bg-secondary)', borderRadius: '8px',
                          padding: '10px 12px', overflowX: 'auto', fontSize: '12px',
                          fontFamily: 'var(--font-mono, monospace)', margin: '4px 0 8px',
                          border: '1px solid var(--border)',
                        }}><code>{children}</code></pre>
                      ),
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
                  <InlineImage key={i} url={url} />
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Empty state ── */
function EmptyState({ onChipClick }) {
  const chips = [
    'NO₂ levels today',
    'Ozone data',
    'Aerosol depth trends',
    'HCHO concentration',
    'Plot AQI on a map',
  ]

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
        fontFamily: 'var(--font-serif)', fontWeight: '400',
        fontSize: '22px', color: 'var(--text-primary)',
        marginBottom: '8px', letterSpacing: '0.01em',
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

      {/* Suggestion chips */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
        {chips.map(chip => (
          <button
            key={chip}
            onClick={() => onChipClick(chip)}
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
            {chip}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ── Main Chat component ── */
export default function Chat({ messages, loading, error, onSend, onClear }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = (text) => {
    const msg = (text || input).trim()
    if (!msg || loading) return
    onSend(msg)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    // Auto-grow textarea
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'
  }

  const isEmpty = messages.length === 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 20px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-primary)', flexShrink: 0,
      }}>
        <span style={{
          fontFamily: 'var(--font-serif)', fontSize: '17px',
          fontWeight: '400', color: 'var(--text-primary)', letterSpacing: '0.01em',
        }}>
          Talking to Air
        </span>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {!isEmpty && (
            <button
              onClick={onClear}
              style={{
                background: 'transparent', border: '1px solid var(--border)',
                borderRadius: '8px', color: 'var(--text-muted)',
                padding: '4px 12px', cursor: 'pointer', fontSize: '12px',
                fontFamily: 'var(--font)', transition: 'border-color 0.15s, color 0.15s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--border-hover)'; e.currentTarget.style.color = 'var(--text-secondary)' }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)' }}
            >
              New chat
            </button>
          )}
        </div>
      </div>

      {/* Message area */}
      <div style={{
        flex: 1, minHeight: 0, overflowY: 'auto',
        display: 'flex', flexDirection: 'column',
      }}>
        {isEmpty ? (
          <EmptyState onChipClick={handleSend} />
        ) : (
          <div style={{ padding: '20px 20px 8px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {messages.map((msg, i) =>
              msg.isLoading ? (
                <LoadingMessage key={i} toolCalls={msg.toolCalls} />
              ) : (
                <MessageBubble key={i} msg={msg} />
              )
            )}
            <div ref={bottomRef} style={{ height: '8px' }} />
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
            disabled={loading}
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
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            style={{
              width: '34px', height: '34px', borderRadius: '50%', flexShrink: 0,
              background: loading || !input.trim() ? 'var(--bg-tertiary)' : 'var(--accent)',
              color: loading || !input.trim() ? 'var(--text-hint)' : 'var(--accent-text)',
              border: 'none', cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.15s, color 0.15s, transform 0.1s',
              fontSize: '16px',
            }}
            onMouseDown={e => { if (!loading && input.trim()) e.currentTarget.style.transform = 'scale(0.93)' }}
            onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}
          >
            {loading ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10" strokeDasharray="32" strokeDashoffset="8" style={{ animation: 'wm-spin 1s linear infinite' }}/>
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
        textarea:disabled     { opacity: 0.6; cursor: not-allowed; }
        .msg-bubble p:last-child,
        .msg-bubble ul:last-child,
        .msg-bubble ol:last-child { margin-bottom: 0; }
        .katex { font-size: 1em; }
      `}</style>
    </div>
  )
}
