import Chat from './components/Chat'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, loading, error, sendMessage, clearSession } = useChat()

  return (
    <div style={{
      display:        'flex',
      flexDirection:  'column',
      height:         '100vh',
      width:          '100vw',
      overflow:       'hidden',
      background:     'var(--bg-primary)',
    }}>
      <Chat
        messages={messages}
        loading={loading}
        error={error}
        onSend={sendMessage}
        onClear={clearSession}
      />
    </div>
  )
}