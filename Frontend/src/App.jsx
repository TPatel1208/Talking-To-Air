import Chat from './components/Chat'
import Dashboard from './components/Dashboard'
import DiscoveryPane from './components/DiscoveryPane'
import JobsPanel from './components/JobsPanel'
import SessionSidebar from './components/SessionSidebar'
import { useChat } from './hooks/useChat'
import { useDiscovery } from './hooks/useDiscovery'
import { useJobs } from './hooks/useJobs'
import { useCallback, useMemo, useState } from 'react'

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
      <SessionSidebar
        sessions={sessions}
        threadId={threadId}
        onSwitch={switchSession}
        onNew={newSession}
        onDelete={deleteSession}
        onLogout={handleLogout}
      />

      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Chat
          messages={messages}
          loading={loading}
          error={error}
          accessToken={accessToken}
          onSend={sendMessage}
          onAbort={() => abortActiveRequest(true)}
          onClear={newSession}
          onClearError={clearError}
        />
      </div>

      <div style={{ width: '320px', flexShrink: 0, borderLeft: '1px solid var(--border)', overflow: 'hidden' }}>
        <Dashboard images={images} artifacts={artifacts} accessToken={accessToken} />
      </div>

      <JobsPanel
        jobs={jobs}
        error={jobsError}
        onCancel={cancelJob}
        onRefresh={fetchJobs}
      />

      <DiscoveryPane
        query={discovery.query}
        setQuery={discovery.setQuery}
        location={discovery.location}
        setLocation={discovery.setLocation}
        timeRange={discovery.timeRange}
        setTimeRange={discovery.setTimeRange}
        results={discovery.results}
        loading={discovery.loading}
        error={discovery.error}
        previews={discovery.previews}
        coverages={discovery.coverages}
        onSearch={discovery.search}
        onPreview={discovery.preview}
        onCoverage={discovery.checkCoverage}
        onRetrieve={handleRetrieve}
      />
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
