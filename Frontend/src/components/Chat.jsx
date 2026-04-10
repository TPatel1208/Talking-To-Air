import { useState, useRef, useEffect } from 'react'

export default function Chat({ messages, loading, error, onSend, onClear }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  // Auto-scroll to latest message
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
      {/* Message thread */}
      <div style={{
        flex:      1,
        overflowY: 'auto',
        padding:   '12px 16px',
        display:   'flex',
        flexDirection: 'column',
        gap:       '10px',
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

        {messages.map((msg, i) => (
          <div key={i} style={{
            display:       'flex',
            justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
          }}>
            <div style={{
              maxWidth:     '80%',
              padding:      '8px 12px',
              borderRadius: msg.role === 'user'
                ? '12px 12px 2px 12px'
                : '12px 12px 12px 2px',
              background:   msg.role === 'user'
                ? 'var(--accent)'
                : msg.isError
                ? '#ff5f5720'
                : 'var(--bg-tertiary)',
              color:        msg.isError ? 'var(--error)' : 'var(--text-primary)',
              border:       msg.isError ? '1px solid var(--error)' : 'none',
              fontSize:     '13px',
              lineHeight:   '1.6',
              whiteSpace:   'pre-wrap',
              wordBreak:    'break-word',
            }}>
              {msg.content}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div style={{ display: 'flex', gap: '4px', padding: '8px 4px' }}>
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
          placeholder="Ask about air quality data..."
          disabled={loading}
          rows={1}
          style={{
            flex:        1,
            background:  'var(--bg-tertiary)',
            border:      '1px solid var(--border)',
            borderRadius:'8px',
            color:       'var(--text-primary)',
            padding:     '8px 12px',
            fontSize:    '13px',
            resize:      'none',
            outline:     'none',
            fontFamily:  'var(--font)',
            lineHeight:  '1.6',
            minHeight:   '38px',
            maxHeight:   '120px',
            overflowY:   'auto',
          }}
        />

        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            background:   loading || !input.trim()
              ? 'var(--bg-tertiary)'
              : 'var(--accent)',
            border:       'none',
            borderRadius: '8px',
            color:        loading || !input.trim()
              ? 'var(--text-secondary)'
              : '#fff',
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
          40%           { transform: translateY(-6px); }
        }
        textarea:focus {
          border-color: var(--accent) !important;
        }
      `}</style>
    </div>
  )
}