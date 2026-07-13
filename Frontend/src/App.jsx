import Chat from './components/Chat'
import OutputPanel from './components/OutputPanel'
import RightPanel from './components/RightPanel'
import SessionSidebar from './components/SessionSidebar'
import { useChat } from './hooks/useChat'
import { useDiscovery } from './hooks/useDiscovery'
import { useJobs } from './hooks/useJobs'
import { createEmptySelection, toggleSlot } from './utils/compareMode'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

// Thin clickable rail standing in for a side column while it's manually
// collapsed -- keeps a one-click way back rather than the column just
// vanishing.
function CollapsedRail({ label, onExpand }) {
  return (
    <button
      type="button"
      onClick={onExpand}
      title={`Show ${label}`}
      aria-label={`Show ${label}`}
      style={{
        width: '28px', flexShrink: 0, border: 'none', cursor: 'pointer',
        background: 'var(--bg-card)', color: 'var(--text-muted)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        borderLeft: '1px solid var(--border)', borderRight: '1px solid var(--border)',
      }}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="9 18 15 12 9 6" />
      </svg>
    </button>
  )
}

const API_BASE = '/api'
const AUTH_STORAGE_KEY = 'tta.accessToken'
const ACTIVE_THREAD_STORAGE_KEY = 'tta.activeThreadId'

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError(null)

    try {
      if (mode === 'register') {
        const registerRes = await fetch(`${API_BASE}/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        })
        if (!registerRes.ok && registerRes.status !== 409) {
          throw new Error(registerRes.status === 422 ? 'Enter a username and password.' : `HTTP ${registerRes.status}`)
        }
        if (registerRes.status === 409) {
          throw new Error('That username is already taken.')
        }
      }

      const loginRes = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!loginRes.ok) {
        throw new Error(loginRes.status === 401 ? 'Invalid username or password.' : `HTTP ${loginRes.status}`)
      }
      const data = await loginRes.json()
      onAuthenticated(data.access_token)
    } catch (err) {
      setError(err.message || 'Authentication failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      minHeight: '100%',
      display: 'grid',
      placeItems: 'center',
      background: 'var(--bg-primary)',
      color: 'var(--text-primary)',
      padding: '24px',
    }}>
      <form onSubmit={submit} style={{
        width: 'min(100%, 360px)',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
        padding: '22px',
        border: '1px solid var(--border)',
        borderRadius: '8px',
        background: 'var(--bg-card)',
      }}>
        <div>
          <h1 style={{ margin: '0 0 6px', fontSize: '22px', letterSpacing: 0 }}>Talking to Air</h1>
          <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '13px' }}>
            {mode === 'login' ? 'Sign in to continue.' : 'Create an account to continue.'}
          </p>
        </div>

        <label style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
          Username
          <input
            value={username}
            onChange={event => setUsername(event.target.value)}
            autoComplete="username"
            required
            style={{
              height: '38px',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              padding: '0 10px',
            }}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
          Password
          <input
            value={password}
            onChange={event => setPassword(event.target.value)}
            type="password"
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            required
            style={{
              height: '38px',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              padding: '0 10px',
            }}
          />
        </label>

        {error && <div style={{ color: 'var(--danger, #b42318)', fontSize: '13px' }}>{error}</div>}

        <button
          type="submit"
          disabled={loading}
          style={{
            height: '38px',
            border: 0,
            borderRadius: '6px',
            background: 'var(--teal)',
            color: 'white',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.75 : 1,
          }}
        >
          {loading ? 'Please wait...' : mode === 'login' ? 'Sign in' : 'Register'}
        </button>

        <button
          type="button"
          onClick={() => {
            setMode(mode === 'login' ? 'register' : 'login')
            setError(null)
          }}
          style={{
            height: '34px',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            background: 'transparent',
            color: 'var(--text-primary)',
            cursor: 'pointer',
          }}
        >
          {mode === 'login' ? 'Create account' : 'Use existing account'}
        </button>
      </form>
    </div>
  )
}

function AuthenticatedApp({ accessToken, onLogout, onUnauthorized }) {
  const {
    jobs,
    error: jobsError,
    fetchJobs,
    applyJobProgress,
    cancelJob,
  } = useJobs(accessToken)

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
  } = useChat(accessToken, onUnauthorized, applyJobProgress)

  const discovery = useDiscovery(accessToken)

  // The card's retrieve action hands off to the standard agent flow (safe_retrieve
  // gates included) rather than bypassing it — one retrieval pipeline, two entry points.
  const handleRetrieve = useCallback((dataset, location, timeRange) => {
    const label = dataset.summary || dataset.dataset_handle
    const parts = [`Retrieve ${label} (${dataset.dataset_handle})`]
    if (location.trim()) parts.push(`over ${location.trim()}`)
    if (timeRange.trim()) parts.push(`for ${timeRange.trim()}`)
    sendMessage(`${parts.join(' ')}.`)
  }, [sendMessage])

  // A job card's "View result" action hands off to the agent rather than
  // rendering the obs_handle itself — same one-pipeline principle as
  // handleRetrieve, and no new backend surface for opening a handle (T27).
  const handleViewResult = useCallback((job) => {
    if (!job.obs_handle) return
    const label = job.short_name || job.dataset_handle || 'this retrieval'
    sendMessage(`Show me the result of ${label} (${job.obs_handle}).`)
  }, [sendMessage])

  // The central OutputPanel shows whichever chart/artifact is "focused" —
  // the newest one from a completed reply, or whatever the user clicked in
  // the chat history.
  const [focusedOutput, setFocusedOutput] = useState(null)
  const wasLoadingRef = useRef(false)

  useEffect(() => {
    if (wasLoadingRef.current && !loading) {
      const last = messages[messages.length - 1]
      if (last?.role === 'assistant') {
        if (last.charts?.length) {
          setFocusedOutput({ kind: 'chart', data: last.charts[last.charts.length - 1] })
        } else {
          const tableArtifact = (last.artifacts || []).find(a => a.type === 'table')
          if (tableArtifact) setFocusedOutput({ kind: 'artifact', data: tableArtifact })
        }
      }
    }
    wasLoadingRef.current = loading
  }, [loading, messages])

  // Compare mode (T28): off | choosing-count | active. Pure in-memory state,
  // owned here alongside focusedOutput -- no new store, no persistence, and
  // it resets on reload or session switch exactly like focusedOutput does.
  const [compareMode, setCompareMode] = useState('off')
  const [compareCount, setCompareCount] = useState(2)
  const [compareSelection, setCompareSelection] = useState([])

  // Sessions, Chat, and Jobs/Discover collapse independently and only on
  // explicit user action -- they used to auto-collapse together when compare
  // mode started, but that made the layout jump around outside the user's
  // control. Now it's just a manual, per-panel toggle that persists across
  // mode changes.
  const [sessionsCollapsed, setSessionsCollapsed] = useState(false)
  const [chatCollapsed, setChatCollapsed] = useState(false)
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false)
  const toggleSessionsCollapsed = useCallback(() => setSessionsCollapsed(v => !v), [])
  const toggleChatCollapsed = useCallback(() => setChatCollapsed(v => !v), [])
  const toggleRightPanelCollapsed = useCallback(() => setRightPanelCollapsed(v => !v), [])

  const resetCompare = useCallback(() => {
    setCompareMode('off')
    setCompareSelection([])
  }, [])

  const startChoosingCompare = useCallback(() => setCompareMode('choosing-count'), [])
  const cancelChoosingCompare = useCallback(() => setCompareMode('off'), [])

  const enterCompare = useCallback((count) => {
    setCompareCount(count)
    setCompareSelection(createEmptySelection(count))
    setCompareMode('active')
  }, [])

  const exitCompare = useCallback(() => resetCompare(), [resetCompare])

  const toggleCompareSlot = useCallback((chart) => {
    setCompareSelection(prev => toggleSlot(prev, chart).selection)
  }, [])

  const handleNewSession = useCallback(() => {
    setFocusedOutput(null)
    resetCompare()
    newSession()
  }, [newSession, resetCompare])

  const handleSwitchSession = useCallback((id) => {
    setFocusedOutput(null)
    resetCompare()
    switchSession(id)
  }, [switchSession, resetCompare])

  const chatTitle = useMemo(() => {
    const active = sessions.find(session => (typeof session === 'string' ? session : session?.id) === threadId)
    return (active && typeof active === 'object' ? active.title : null) || (messages.length ? 'Chat' : 'New analysis')
  }, [sessions, threadId, messages.length])

  const { images, artifacts } = useMemo(() => {
    const seenArtifactIds = new Set()
    const dedupedArtifacts = []
    const allImages = []
    for (const msg of messages) {
      for (const url of msg.imageUrls || []) allImages.push(url)
      for (const artifact of msg.artifacts || []) {
        const key = artifact.id || JSON.stringify(artifact)
        if (seenArtifactIds.has(key)) continue
        seenArtifactIds.add(key)
        dedupedArtifacts.push(artifact)
      }
    }
    return { images: allImages, artifacts: dedupedArtifacts }
  }, [messages])

  const handleLogout = useCallback(async () => {
    abortActiveRequest(true)
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
      })
    } catch {
      // Local cleanup still matters if the network drops.
    } finally {
      onLogout()
    }
  }, [abortActiveRequest, accessToken, onLogout])

  return (
    <div style={{
      display:    'flex',
      height:     '100%',
      width:      '100%',
      overflow:   'hidden',
      background: 'var(--bg-primary)',
    }}>
      {sessionsCollapsed ? (
        <CollapsedRail label="sessions" onExpand={toggleSessionsCollapsed} />
      ) : (
        <SessionSidebar
          sessions={sessions}
          threadId={threadId}
          onSwitch={handleSwitchSession}
          onNew={handleNewSession}
          onDelete={deleteSession}
          onLogout={handleLogout}
          images={images}
          artifacts={artifacts}
          accessToken={accessToken}
          onCollapse={toggleSessionsCollapsed}
        />
      )}

      {chatCollapsed ? (
        <CollapsedRail label="chat" onExpand={toggleChatCollapsed} />
      ) : (
        <div style={{ width: '380px', flexShrink: 0, borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Chat
            messages={messages}
            loading={loading}
            error={error}
            accessToken={accessToken}
            chatTitle={chatTitle}
            onSend={sendMessage}
            onAbort={() => abortActiveRequest(true)}
            onClearError={clearError}
            focusedOutput={focusedOutput}
            onFocusOutput={setFocusedOutput}
            compareMode={compareMode}
            compareSelection={compareSelection}
            onToggleCompareSlot={toggleCompareSlot}
            onCollapse={toggleChatCollapsed}
          />
        </div>
      )}

      <OutputPanel
        focusedOutput={focusedOutput}
        accessToken={accessToken}
        compareMode={compareMode}
        compareCount={compareCount}
        compareSelection={compareSelection}
        onStartCompare={startChoosingCompare}
        onCancelChooseCompare={cancelChoosingCompare}
        onEnterCompare={enterCompare}
        onExitCompare={exitCompare}
      />

      {rightPanelCollapsed ? (
        <CollapsedRail label="jobs and discover" onExpand={toggleRightPanelCollapsed} />
      ) : (
        <RightPanel
          discovery={discovery}
          jobs={jobs}
          jobsError={jobsError}
          onCancelJob={cancelJob}
          onRefreshJobs={fetchJobs}
          onRetrieve={handleRetrieve}
          onViewResult={handleViewResult}
          onCollapse={toggleRightPanelCollapsed}
        />
      )}
    </div>
  )
}

export default function App() {
  const [accessToken, setAccessToken] = useState(() => window.localStorage.getItem(AUTH_STORAGE_KEY))

  const clearAuthState = useCallback(() => {
    window.localStorage.removeItem(AUTH_STORAGE_KEY)
    window.localStorage.removeItem(ACTIVE_THREAD_STORAGE_KEY)
    setAccessToken(null)
  }, [])

  const handleAuthenticated = useCallback((token) => {
    window.localStorage.setItem(AUTH_STORAGE_KEY, token)
    window.localStorage.removeItem(ACTIVE_THREAD_STORAGE_KEY)
    setAccessToken(token)
  }, [])

  if (!accessToken) {
    return <AuthScreen onAuthenticated={handleAuthenticated} />
  }

  return (
    <AuthenticatedApp
      key={accessToken}
      accessToken={accessToken}
      onLogout={clearAuthState}
      onUnauthorized={clearAuthState}
    />
  )
}
