import { useState } from 'react'
import Dashboard from './components/Dashboard'
import Chat from './components/Chat'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, images, toolCalls, loading, error, sendMessage, clearSession } = useChat()

  return (
    <div style={{
      display:       'flex',
      flexDirection: 'column',
      height:        '100vh',
      width:         '100vw',
      overflow:      'hidden',
      background:    'var(--bg-primary)',
    }}>
      {/* Dashboard — top 65% */}
      <div style={{
        flex:        '0 0 65%',
        borderBottom: '1px solid var(--border)',
        overflow:    'hidden',
        display:     'flex',
        flexDirection: 'column',
      }}>
        <Dashboard images={images} toolCalls={toolCalls} />
      </div>

      {/* Chat — bottom 35% */}
      <div style={{
        flex:     '0 0 35%',
        overflow: 'hidden',
        display:  'flex',
        flexDirection: 'column',
      }}>
        <Chat
          messages={messages}
          loading={loading}
          error={error}
          onSend={sendMessage}
          onClear={clearSession}
        />
      </div>
    </div>
  )
}