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
      padding:    '4px 0',
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
        borderRadius: '12px 12px 12px 2px',
        background:   'var(--bg-tertiary)',
        fontSize:     '13px',
        lineHeight:   '1.6',
        minWidth:     '120px',
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

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'

  return (
    <div style={{
      display:       'flex',
      flexDirection: 'column',
      alignItems:    isUser ? 'flex-end' : 'flex-start',
      gap:           '6px',
    }}>
      {!isUser && msg.toolCalls?.length > 0 && (
        <div style={{
          maxWidth:      '80%',
          padding:       '6px 12px',
          fontSize:      '12px',
          color:         'var(--text-secondary)',
          display:       'flex',
          flexDirection: 'column',
          gap:           '2px',
        }}>
          {msg.toolCalls.map((tc, i) => (
            <ToolCallBadge key={i} name={tc.name} args={tc.args} />
          ))}
        </div>
      )}

      {(msg.content || isUser) && (
        <div
          className="msg-bubble"
          style={{
            maxWidth:     '80%',
            padding:      '8px 12px',
            borderRadius: isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
            background:   isUser
              ? 'var(--accent)'
              : msg.isError ? '#ff5f5720' : 'var(--bg-tertiary)',
            color:        msg.isError ? 'var(--error)' : 'var(--text-primary)',
            border:       msg.isError ? '1px solid var(--error)' : 'none',
            fontSize:     '13px',
            lineHeight:   '1.6',
            wordBreak:    'break-word',
          }}
        >
          {isUser ? (
            <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
          ) : (
            <ReactMarkdown
              components={{
                p:    ({ children }) => (
                  <p style={{ margin: '0 0 8px' }}>{children}</p>
                ),
                h1:   ({ children }) => (
                  <p style={{ margin: '0 0 6px', fontWeight: 600, fontSize: '15px' }}>{children}</p>
                ),
                h2:   ({ children }) => (
                  <p style={{ margin: '0 0 6px', fontWeight: 600, fontSize: '14px' }}>{children}</p>
                ),
                h3:   ({ children }) => (
                  <p style={{ margin: '0 0 4px', fontWeight: 600 }}>{children}</p>
                ),
                ul:   ({ children }) => (
                  <ul style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ul>
                ),
                ol:   ({ children }) => (
                  <ol style={{ margin: '0 0 8px', paddingLeft: '18px' }}>{children}</ol>
                ),
                li:   ({ children }) => (
                  <li style={{ marginBottom: '2px' }}>{children}</li>
                ),
                code: ({ inline, children }) => inline ? (
                  <code style={{
                    background:   'var(--bg-primary)',
                    borderRadius: '4px',
                    padding:      '1px 5px',
                    fontFamily:   'var(--font-mono, monospace)',
                    fontSize:     '12px',
                  }}>
                    {children}
                  </code>
                ) : (
                  <pre style={{
                    background:   'var(--bg-primary)',
                    borderRadius: '6px',
                    padding:      '8px 10px',
                    overflowX:    'auto',
                    fontSize:     '12px',
                    fontFamily:   'var(--font-mono, monospace)',
                    margin:       '4px 0 8px',
                  }}>
                    <code>{children}</code>
                  </pre>
                ),
                a: ({ href, children }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: 'var(--accent)' }}
                  >
                    {children}
                  </a>
                ),
              }}
            >
              {msg.content}
            </ReactMarkdown>
          )}
        </div>
      )}

      {!isUser && msg.imageUrls?.length > 0 && (
        <div style={{
          display:  'flex',
          flexWrap: 'wrap',
          gap:      '8px',
          maxWidth: '80%',
        }}>
          {msg.imageUrls.map((url, i) => (
            <a key={i} href={url} target="_blank" rel="noreferrer" style={{ display: 'block' }}>
              <img
                src={url}
                alt={`result-${i}`}
                style={{
                  maxWidth:     '100%',
                  maxHeight:    '480px',
                  borderRadius: '8px',
                  border:       '1px solid var(--border)',
                  display:      'block',
                  cursor:       'zoom-in',
                  objectFit:    'contain',
                }}
              />
            </a>
          ))}
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
      <div style={{
        flex:          1,
        overflowY:     'auto',
        padding:       '12px 16px',
        display:       'flex',
        flexDirection: 'column',
        gap:           '10px',
      }}>
        {messages.length === 0 && (
          <div style={{
            textAlign: 'center',
            color:     'var(--text-secondary)',
            marginTop: '24px',
            fontSize:  '13px',
          }}>
            Start a conversation with your environmental data assistant
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
          placeholder="Ask about air quality data..."
          disabled={loading}
          rows={1}
          style={{
            flex:         1,
            background:   'var(--bg-tertiary)',
            border:       '1px solid var(--border)',
            borderRadius: '8px',
            color:        'var(--text-primary)',
            padding:      '8px 12px',
            fontSize:     '13px',
            resize:       'none',
            outline:      'none',
            fontFamily:   'var(--font)',
            lineHeight:   '1.6',
            minHeight:    '38px',
            maxHeight:    '120px',
            overflowY:    'auto',
          }}
        />

        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            background:   loading || !input.trim() ? 'var(--bg-tertiary)' : 'var(--accent)',
            border:       'none',
            borderRadius: '8px',
            color:        loading || !input.trim() ? 'var(--text-secondary)' : '#fff',
            padding:      '0 16px',
            cursor:       loading || !input.trim() ? 'not-allowed' : 'pointer',
            fontSize:     '13px',
            fontWeight:   '500',
            flexShrink:   0,
            transition:   'background 0.15s',
          }}
        >
          {loading ? '...' : 'Send'}
        </button>

        <button
          onClick={onClear}
          title="New session"
          style={{
            background:   'transparent',
            border:       '1px solid var(--border)',
            borderRadius: '8px',
            color:        'var(--text-secondary)',
            padding:      '0 12px',
            cursor:       'pointer',
            fontSize:     '12px',
            flexShrink:   0,
          }}
        >
          New
        </button>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40%            { transform: translateY(-6px); }
        }
        textarea:focus {
          border-color: var(--accent) !important;
        }
        .msg-bubble p:last-child {
          margin-bottom: 0;
        }
        .msg-bubble ul:last-child,
        .msg-bubble ol:last-child {
          margin-bottom: 0;
        }
      `}</style>
    </div>
  )
}