import Chat from './components/Chat'
import SessionSidebar from './components/SessionSidebar'
import { useChat } from './hooks/useChat'

export default function App() {
  const {
    messages,
    loading,
    error,
    threadId,
    sessions,
    sendMessage,
    newSession,
    switchSession,
    deleteSession,
    abortActiveRequest,
    clearError,
  } = useChat()

  return (
    <div style={{
      display:    'flex',
      height:     '100%',
      width:      '100%',
      overflow:   'hidden',
      background: 'var(--bg-primary)',
    }}>
      <SessionSidebar
        sessions={sessions}
        threadId={threadId}
        onSwitch={switchSession}
        onNew={newSession}
        onDelete={deleteSession}
      />

      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Chat
          messages={messages}
          loading={loading}
          error={error}
          onSend={sendMessage}
          onAbort={() => abortActiveRequest(true)}
          onClear={newSession}
          onClearError={clearError}
        />
      </div>
    </div>
  )
}
