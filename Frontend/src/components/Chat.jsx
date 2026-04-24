import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

function ToolCallBadge({ name, args }) {
  const [expanded, setExpanded] = useState(false)
  const argStr = args ? JSON.stringify(args, null, 2) : ''
  const hasArgs = argStr && argStr !== '{}'

  return (
    <div style={{
      display:    'flex',
      alignItems: 'flex-start',
      gap:        '6px',
      fontSize:   '11px',
      color:      'var(--text-secondary)',
      padding:    '3px 0',
    }}>
      <span style={{
        width:        '6px',
        height:       '6px',
        borderRadius: '50%',
        background:   'var(--accent)',
        flexShrink:   0,
        marginTop:    '3px',
        opacity:      0.7,
      }}/>
      <div>
        <span
          onClick={() => hasArgs && setExpanded(e => !e)}
          style={{
            fontFamily: 'var(--font-mono, monospace)',
            color:      'var(--accent)',
            cursor:     hasArgs ? 'pointer' : 'default',
            userSelect: 'none',
          }}
        >
          {name}()
        </span>
        {hasArgs && (
          <span style={{ marginLeft: '4px', opacity: 0.5 }}>
            {expanded ? '▲' : '▼'}
          </span>
        )}
        {expanded && (
          <pre style={{
            margin:       '4px 0 0',
            padding:      '6px 8px',
            background:   'var(--bg-primary)',
            borderRadius: '6px',
            fontSize:     '10px',
            overflowX:    'auto',
            color:        'var(--text-secondary)',
            border:       '1px solid var(--border)',
          }}>
            {argStr}
          </pre>
        )}
      </div>
    </div>
  )
}

function LoadingMessage({ toolCalls }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
      <div style={{
        maxWidth:     '80%',
        padding:      '10px 14px',
        borderRadius: '16px 16px 16px 4px',
        background:   'var(--bg-tertiary)',
        fontSize:     '13px',
        lineHeight:   '1.6',
        minWidth:     '80px',
      }}>
        {toolCalls?.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            {toolCalls.map((tc, i) => (
              <ToolCallBadge key={i} name={tc.name} args={tc.args} />
            ))}
            <div style={{ display: 'flex', gap: '4px', paddingTop: '6px' }}>
              {[0, 1, 2].map(i => (
                <div key={i} style={{
                  width:        '4px',
                  height:       '4px',
                  borderRadius: '50%',
                  background:   'var(--text-secondary)',
                  animation:    `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
                }}/>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '4px' }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{
                width:        '6px',
                height:       '6px',
                borderRadius: '50%',
                background:   'var(--text-secondary)',
                animation:    `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
              }}/>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function InlineImage({ url }) {
  const [lightbox, setLightbox] = useState(false)

  return (
    <>
      <div style={{ margin: '8px 0' }}>
        <img
          src={url}
          alt="output"
          onClick={() => setLightbox(true)}
          style={{
            maxWidth:     '100%',
            maxHeight:    '420px',
            borderRadius: '8px',
            border:       '1px solid var(--border)',
            display:      'block',
            cursor:       'zoom-in',
            objectFit:    'contain',
          }}
        />
        <div style={{
          fontSize:  '10px',
          color:     'var(--text-secondary)',
          marginTop: '4px',
          opacity:   0.6,
        }}>
          Click to enlarge
        </div>
      </div>

      {lightbox && (
        <div
          onClick={() => setLightbox(false)}
          style={{
            position:       'fixed',
            inset:          0,
            background:     'rgba(0,0,0,0.88)',
            zIndex:         1000,
            display:        'flex',
            alignItems:     'center',
            justifyContent: 'center',
            cursor:         'zoom-out',
          }}
        >
          <img
            src={url}
            alt="output fullscreen"
            style={{
              maxWidth:     '92vw',
              maxHeight:    '92vh',
              borderRadius: '8px',
              objectFit:    'contain',
            }}
          />
        </div>
      )}
    </>
  )
}

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'

  return (
    <div style={{
      display:       'flex',
      flexDirection: 'column',
      alignItems:    isUser ? 'flex-end' : 'flex-start',
      gap:           '4px',
    }}>
      {/* Tool call badges shown above assistant bubble */}
      {!isUser && msg.toolCalls?.length > 0 && (
        <div style={{
          maxWidth:      '80%',
          padding:       '4px 10px',
          display:       'flex',
          flexDirection: 'column',
          gap:           '1px',
        }}>
          {msg.toolCalls.map((tc, i) => (
            <ToolCallBadge key={i} name={tc.name} args={tc.args} />
          ))}
        </div>
      )}

      {/* Bubble */}
      {(msg.content || msg.imageUrls?.length > 0 || isUser) && (
        <div
          className="msg-bubble"
          style={{
            maxWidth:     '80%',
            padding:      '10px 14px',
            borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
            background:   isUser
              ? 'var(--accent)'
              : msg.isError ? '#ff5f5720' : 'var(--bg-tertiary)',
            color:        msg.isError ? 'var(--error)' : 'var(--text-primary)',
            border:       msg.isError ? '1px solid var(--error)' : 'none',
            fontSize:     '13px',
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
                  components={{
                    p:      ({ children }) => <p style={{ margin: '0 0 8px' }}>{children}</p>,
                    h1:     ({ children }) => <p style={{ margin: '0 0 6px', fontWeight: 700, fontSize: '15px' }}>{children}</p>,
                    h2:     ({ children }) => <p style={{ margin: '0 0 6px', fontWeight: 600, fontSize: '14px' }}>{children}</p>,
                    h3:     ({ children }) => <p style={{ margin: '0 0 4px', fontWeight: 600 }}>{children}</p>,
                    ul:     ({ children }) => <ul style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ul>,
                    ol:     ({ children }) => <ol style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ol>,
                    li:     ({ children }) => <li style={{ marginBottom: '3px' }}>{children}</li>,
                    strong: ({ children }) => <strong style={{ fontWeight: 600 }}>{children}</strong>,
                    code: ({ inline, children }) => inline ? (
                      <code style={{
                        background:   'var(--bg-primary)',
                        borderRadius: '4px',
                        padding:      '1px 5px',
                        fontFamily:   'var(--font-mono, monospace)',
                        fontSize:     '12px',
                      }}>{children}</code>
                    ) : (
                      <pre style={{
                        background:   'var(--bg-primary)',
                        borderRadius: '6px',
                        padding:      '8px 10px',
                        overflowX:    'auto',
                        fontSize:     '12px',
                        fontFamily:   'var(--font-mono, monospace)',
                        margin:       '4px 0 8px',
                      }}><code>{children}</code></pre>
                    ),
                    a: ({ href, children }) => (
                      <a href={href} target="_blank" rel="noreferrer"
                        style={{ color: 'var(--accent)', textDecoration: 'underline' }}>
                        {children}
                      </a>
                    ),
                    table: ({ children }) => (
                      <div style={{ overflowX: 'auto', margin: '8px 0' }}>
                        <table style={{ borderCollapse: 'collapse', fontSize: '12px', width: '100%' }}>
                          {children}
                        </table>
                      </div>
                    ),
                    th: ({ children }) => (
                      <th style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)' }}>
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td style={{ padding: '5px 10px', borderBottom: '1px solid var(--border)' }}>
                        {children}
                      </td>
                    ),
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              )}

              {/* Inline images rendered after text */}
              {msg.imageUrls?.map((url, i) => (
                <InlineImage key={i} url={url} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default function Chat({ messages, loading, error, onSend, onClear }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = () => {
    if (!input.trim() || loading) return
    onSend(input.trim())
    setInput('')
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={{
      display:       'flex',
      flexDirection: 'column',
      height:        '100%',
      overflow:      'hidden',
    }}>
      {/* Header */}
      <div style={{
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'space-between',
        padding:        '10px 16px',
        borderBottom:   '1px solid var(--border)',
        flexShrink:     0,
        background:     'var(--bg-secondary)',
      }}>
        <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)', letterSpacing: '0.02em' }}>
          Talking to Air
        </span>
        <button onClick={onClear} style={{
          background:   'transparent',
          border:       '1px solid var(--border)',
          borderRadius: '6px',
          color:        'var(--text-secondary)',
          padding:      '3px 10px',
          cursor:       'pointer',
          fontSize:     '11px',
        }}>
          New chat
        </button>
      </div>

      {/* Messages */}
      <div style={{
        flex:          1,
        overflowY:     'auto',
        padding:       '16px',
        display:       'flex',
        flexDirection: 'column',
        gap:           '12px',
      }}>
        {messages.length === 0 && (
          <div style={{
            textAlign:  'center',
            color:      'var(--text-secondary)',
            marginTop:  '60px',
            fontSize:   '13px',
            lineHeight: '1.8',
          }}>
            <div style={{ fontSize: '32px', marginBottom: '10px' }}>🌍</div>
            <div style={{ fontWeight: 500, marginBottom: '4px', color: 'var(--text-primary)' }}>Talking to Air</div>
            <div style={{ opacity: 0.7 }}>
              Ask about NO₂, ozone, HCHO, aerosol optical depth,<br/>
              or any NASA satellite dataset.
            </div>
          </div>
        )}

        {messages.map((msg, i) =>
          msg.isLoading ? (
            <LoadingMessage key={i} toolCalls={msg.toolCalls} />
          ) : (
            <MessageBubble key={i} msg={msg} />
          )
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div style={{
        display:    'flex',
        gap:        '8px',
        padding:    '12px 16px',
        borderTop:  '1px solid var(--border)',
        background: 'var(--bg-secondary)',
        flexShrink: 0,
      }}>
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask about air quality data…"
          disabled={loading}
          rows={1}
          style={{
            flex:         1,
            background:   'var(--bg-tertiary)',
            border:       '1px solid var(--border)',
            borderRadius: '10px',
            color:        'var(--text-primary)',
            padding:      '9px 13px',
            fontSize:     '13px',
            resize:       'none',
            outline:      'none',
            fontFamily:   'var(--font)',
            lineHeight:   '1.6',
            minHeight:    '40px',
            maxHeight:    '140px',
            overflowY:    'auto',
          }}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            background:   loading || !input.trim() ? 'var(--bg-tertiary)' : 'var(--accent)',
            border:       'none',
            borderRadius: '10px',
            color:        loading || !input.trim() ? 'var(--text-secondary)' : '#fff',
            padding:      '0 18px',
            cursor:       loading || !input.trim() ? 'not-allowed' : 'pointer',
            fontSize:     '13px',
            fontWeight:   '500',
            flexShrink:   0,
            transition:   'background 0.15s',
          }}
        >
          {loading ? '…' : 'Send'}
        </button>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40%            { transform: translateY(-6px); }
        }
        textarea:focus { border-color: var(--accent) !important; }
        .msg-bubble p:last-child  { margin-bottom: 0; }
        .msg-bubble ul:last-child,
        .msg-bubble ol:last-child { margin-bottom: 0; }
      `}</style>
    </div>
  )
}