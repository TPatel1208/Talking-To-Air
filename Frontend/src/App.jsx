import Chat from './components/Chat'
import OutputPanel from './components/OutputPanel'
import RightPanel from './components/RightPanel'
import SessionSidebar from './components/SessionSidebar'
import { useChat } from './hooks/useChat'
import { useDiscovery } from './hooks/useDiscovery'
import { useJobs } from './hooks/useJobs'
import { createEmptySelection, toggleSlot } from './utils/compareMode'
import { authTransition } from './utils/sessionExpiry'
import { turnCompletionFocus } from './utils/turnFocus'
import { ensureSupabaseClient, getSupabaseClient } from './utils/supabaseClient'
import { configureApiFetch, noteSession } from './utils/apiFetch'
import {
  authReducer,
  authView,
  describeAuthError,
  initialAuthState,
  isUnreachable,
  readAuthConfig,
  showReauthModal,
  userIdOf,
} from './utils/authSession'
import { useCallback, useEffect, useMemo, useReducer, useState } from 'react'

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
// There is deliberately no token key here. supabase-js owns session
// persistence and refresh (decision 6); a second copy of the token in our own
// storage would be a rival source of truth, and the stale one wins on reload.
const ACTIVE_THREAD_STORAGE_KEY = 'tta.activeThreadId'

const FIELD_STYLE = {
  height: '38px',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  background: 'var(--bg-primary)',
  color: 'var(--text-primary)',
  padding: '0 10px',
}

// Sign-in only. There is no register tab: signup is invite-only and disabled
// in the Supabase project (decision 8), so a register button could produce
// nothing but a rejection. The username field went with it -- it was read by
// this form and by a JWT claim, and displayed nowhere in the product, so email
// replaces it at no cost.
function LoginForm({ onSubmit, heading = 'Talking to Air', subtitle = 'Sign in to continue.' }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    try {
      await onSubmit(email, password)
      // No setLoading(false) here: a successful sign-in unmounts this form,
      // either into the app or by closing the re-auth modal over it.
    } catch (err) {
      setError(describeAuthError(err))
      setLoading(false)
    }
  }

  return (
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
        <h1 style={{ margin: '0 0 6px', fontSize: '22px', letterSpacing: 0 }}>{heading}</h1>
        <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '13px' }}>{subtitle}</p>
      </div>

      <label style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
        Email
        <input
          value={email}
          onChange={event => setEmail(event.target.value)}
          type="email"
          autoComplete="email"
          required
          style={FIELD_STYLE}
        />
      </label>

      <label style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
        Password
        <input
          value={password}
          onChange={event => setPassword(event.target.value)}
          type="password"
          autoComplete="current-password"
          required
          style={FIELD_STYLE}
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
        {loading ? 'Signing in...' : 'Sign in'}
      </button>
    </form>
  )
}

// Shown before there is an app to show: while the runtime auth config is in
// flight, while supabase-js restores a stored session, and when the config
// could not be fetched at all.
function Splash({ heading, message, actionLabel, onAction }) {
  return (
    <div style={{
      minHeight: '100%',
      display: 'grid',
      placeItems: 'center',
      background: 'var(--bg-primary)',
      color: 'var(--text-primary)',
      padding: '24px',
    }}>
      <div style={{ textAlign: 'center', maxWidth: '420px' }}>
        {heading && <h1 style={{ margin: '0 0 8px', fontSize: '20px' }}>{heading}</h1>}
        <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '13px' }}>{message}</p>
        {onAction && (
          <button
            type="button"
            onClick={onAction}
            style={{
              marginTop: '16px', height: '34px', padding: '0 16px',
              border: '1px solid var(--border)', borderRadius: '6px',
              background: 'transparent', color: 'var(--text-primary)', cursor: 'pointer',
            }}
          >
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  )
}

// Full-screen sign-in shown when there's no session at all.
function AuthScreen({ onSignIn }) {
  return (
    <div style={{
      minHeight: '100%',
      display: 'grid',
      placeItems: 'center',
      background: 'var(--bg-primary)',
      color: 'var(--text-primary)',
      padding: '24px',
    }}>
      <LoginForm onSubmit={onSignIn} />
    </div>
  )
}

// T47: an auth failure mid-analysis raises this over the preserved view rather
// than dropping the researcher to a blank login screen. Re-login resumes the
// same thread in place (the active thread id is deliberately not cleared), so
// a 40-minute session survives a lapsed token.
//
// It now fires far less often than it used to: a 401 means the *refresh* token
// lapsed, not the access token, which supabase-js renews on its own.
function SessionExpiredModal({ onSignIn }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Session expired"
      style={{
        position: 'fixed',
        inset: 0,
        display: 'grid',
        placeItems: 'center',
        background: 'rgba(0, 0, 0, 0.55)',
        backdropFilter: 'blur(2px)',
        padding: '24px',
        zIndex: 1000,
      }}
    >
      <LoginForm
        onSubmit={onSignIn}
        heading="Session expired"
        subtitle="Sign in to continue — your work is right where you left it."
      />
    </div>
  )
}

function AuthenticatedApp({ onLogout }) {
  const {
    jobs,
    error: jobsError,
    fetchJobs,
    applyJobProgress,
    cancelJob,
  } = useJobs()

  const {
    messages,
    loading,
    error,
    historyError,
    threadId,
    sessions,
    sendMessage,
    newSession,
    switchSession,
    reloadSession,
    retryHistory,
    deleteSession,
    abortActiveRequest,
    clearError,
  } = useChat(applyJobProgress)

  const discovery = useDiscovery()

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

  // Focus follows the newest output of a turn the moment that turn settles,
  // but a chart the user clicked in the history keeps focus until the *next*
  // turn settles. That is a transition in React's own state, not a
  // synchronization with anything outside it, so it belongs in the render
  // pass (react.dev, "adjusting state when a prop changes") rather than in an
  // effect: `prevLoading` carries the previous render's value, the comparison
  // is only ever true on the transition render, and the re-render React does
  // in response happens before this one is committed -- no cascading render,
  // and no frame painted with the stale focus.
  const [prevLoading, setPrevLoading] = useState(loading)
  if (prevLoading !== loading) {
    const settledFocus = turnCompletionFocus(prevLoading, loading, messages)
    setPrevLoading(loading)
    // A null result means "leave focus alone" -- see turnFocus.js.
    if (settledFocus) setFocusedOutput(settledFocus)
  }

  // Compare mode (T28): off | choosing-count | active. Pure in-memory state,
  // owned here alongside focusedOutput -- no new store, no persistence, and
  // it resets on reload or session switch exactly like focusedOutput does.
  const [compareMode, setCompareMode] = useState('off')
  const [compareCount, setCompareCount] = useState(2)
  const [compareSelection, setCompareSelection] = useState([])
  // Bumped on every enterCompare so OutputPanel can tell "still this compare
  // session" from "a fresh one just started" without an effect/ref -- e.g.
  // to re-arm a dismissed collapse-panel hint per session.
  const [compareSessionId, setCompareSessionId] = useState(0)

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
    setCompareSessionId(id => id + 1)
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

  // Stop the stream first, then hand off. Revoking the session upstream is the
  // root component's job now -- there is no backend auth route left to call.
  const handleLogout = useCallback(() => {
    abortActiveRequest(true)
    onLogout()
  }, [abortActiveRequest, onLogout])

  return (
    // Horizontally scrollable, not clipped. The three side panels are all
    // fixed-width and flexShrink: 0 (232 + 380 + 308 = 920 px), so a viewport
    // narrower than that plus the output panel's own floor has genuinely more
    // content than width. `overflow: hidden` answered that by making the
    // overflowing panel invisible AND unreachable -- measured at 556 px, the
    // output column sat at left: 634 with a 0 px map inside it. Admitting the
    // overflow means the user can reach it, and the three collapse toggles
    // remain the way to make it fit.
    //
    // Deliberately NOT solved by auto-collapsing a panel at some breakpoint:
    // that is what this layout used to do, and it was removed because it moved
    // the layout around outside the user's control (see the collapse state
    // above). Scrolling leaves the choice with them.
    <div style={{
      display:    'flex',
      height:     '100%',
      width:      '100%',
      overflowX:  'auto',
      overflowY:  'hidden',
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
            historyError={historyError}
            onRetryHistory={retryHistory}
            onReloadSession={reloadSession}
            chatTitle={chatTitle}
            onSend={sendMessage}
            onAbort={() => {
              // Stop must mean stop: abort the stream AND cancel whatever
              // retrieval jobs this turn reported as still in flight —
              // locally and (best-effort) upstream at the provider.
              const inFlightJobs = abortActiveRequest(true) || []
              inFlightJobs.forEach(handle => cancelJob(handle))
            }}
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
        onFocusOutput={setFocusedOutput}
        onSend={sendMessage}
        compareMode={compareMode}
        compareCount={compareCount}
        compareSelection={compareSelection}
        compareSessionId={compareSessionId}
        onStartCompare={startChoosingCompare}
        onCancelChooseCompare={cancelChoosingCompare}
        onEnterCompare={enterCompare}
        onExitCompare={exitCompare}
        sessionsCollapsed={sessionsCollapsed}
        chatCollapsed={chatCollapsed}
        rightPanelCollapsed={rightPanelCollapsed}
        onCollapseSessions={toggleSessionsCollapsed}
        onCollapseRightPanel={toggleRightPanelCollapsed}
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
  const [state, dispatch] = useReducer(authReducer, initialAuthState)
  // Bumped by the retry button on the config-error screen.
  const [configAttempt, setConfigAttempt] = useState(0)

  // T47: a 401 during an active session raises the re-auth modal instead of
  // wiping the session and dumping the user to the login screen.
  const handleUnauthorized = useCallback(() => dispatch({ type: 'unauthorized' }), [])

  // The app's first request, and everything waits behind it: the identity
  // provider's coordinates are served at runtime rather than baked into this
  // bundle (decision 11), so one image runs against either project. Failure is
  // a screen with a retry rather than a blank page -- T17's degrade-don't-die.
  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/config/auth`)
      .then(res => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((body) => {
        const config = readAuthConfig(body)
        // Built here, before anything reports success, so a backend serving a
        // blank key surfaces as the config-error screen naming the missing
        // variable instead of throwing during a later render.
        ensureSupabaseClient(config)
        // Wired here rather than in an effect of its own, and deliberately
        // before config-loaded is dispatched. React runs child effects before
        // parent effects, so an effect in App could lose the race to a hook
        // inside a freshly mounted AuthenticatedApp. Configuring before the
        // dispatch that can first produce a view means no authenticated tree
        // can exist un-wired, whatever React batches together.
        configureApiFetch({ auth: getSupabaseClient().auth, onUnauthorized: handleUnauthorized })
        if (!cancelled) dispatch({ type: 'config-loaded', config })
      })
      .catch((err) => {
        if (!cancelled) dispatch({ type: 'config-failed', error: err.message || 'unreachable' })
      })
    return () => { cancelled = true }
  }, [configAttempt, handleUnauthorized])

  // supabase-js announces the restored session, every refresh, and every
  // sign-out through this one callback. It is the only writer of session state.
  useEffect(() => {
    if (!state.config) return undefined
    const { data } = getSupabaseClient().auth.onAuthStateChange((event, session) => {
      // Nothing may be awaited in here. Awaiting a Supabase call inside this
      // callback deadlocks every later call on the client -- documented in
      // Supabase's own troubleshooting guide. Both of these are synchronous.
      //
      // noteSession keeps apiFetch's snapshot -- the one maplibre's synchronous
      // transformRequest reads -- level with the live session. Requests alone
      // cannot: a rotation happens on supabase-js's timer, 90 seconds before
      // expiry, and an idle app may issue no request for hours.
      noteSession(session)
      dispatch({ type: 'auth-event', event, session })
    })
    return () => data.subscription.unsubscribe()
  }, [state.config])

  // supabase-js cannot say why a restore came back empty: it swallows an
  // unreachable service into the same null INITIAL_SESSION it emits for a
  // genuinely absent session. So ask again, where the reason survives -- a
  // status-0 AuthRetryableFetchError is a blip worth retrying, anything else
  // (a missing session is status 400) is a real sign-out. Without this a wifi
  // blip on reload reads as a sign-out, and the sign-in that follows clears
  // the active thread.
  useEffect(() => {
    if (!state.restoreUnresolved) return undefined
    let cancelled = false
    getSupabaseClient().auth.refreshSession()
      // A session recovered here arrives on its own through onAuthStateChange;
      // this only has to report whether the question got an answer.
      .then(({ error }) => (isUnreachable(error) ? 'the sign-in service is unreachable' : null))
      .catch(() => 'the sign-in service is unreachable')
      .then((error) => { if (!cancelled) dispatch({ type: 'restore-settled', error }) })
    return () => { cancelled = true }
  }, [state.restoreUnresolved])

  const signIn = useCallback(async (email, password, kind) => {
    // Cleared BEFORE the sign-in, not after. onAuthStateChange fires as soon
    // as the session lands and mounts the authenticated tree, whose useChat
    // reads this key on mount -- clearing afterwards is a race the previous
    // user's thread can win. A re-auth deliberately keeps it, which is what
    // resumes the exact conversation the researcher was in (T47).
    if (authTransition(kind).clearActiveThread) {
      window.localStorage.removeItem(ACTIVE_THREAD_STORAGE_KEY)
    }
    const { error } = await getSupabaseClient().auth.signInWithPassword({ email, password })
    if (error) throw error
    // Only an explicit re-login closes the modal. supabase-js re-emits
    // SIGNED_IN when a backgrounded tab regains focus, so the event itself
    // cannot be read as "the user just signed in".
    if (kind === 'reauth') dispatch({ type: 'reauthenticated' })
  }, [])

  const handleSignIn = useCallback((email, password) => signIn(email, password, 'login'), [signIn])
  const handleReauth = useCallback((email, password) => signIn(email, password, 'reauth'), [signIn])

  const handleLogout = useCallback(() => {
    // Cleared eagerly, so signing out is instant and unconditional even when
    // the call revoking the refresh token upstream never lands.
    dispatch({ type: 'logout-requested' })
    window.localStorage.removeItem(ACTIVE_THREAD_STORAGE_KEY)
    getSupabaseClient()?.auth.signOut().catch(() => {})
  }, [])

  const retryConfig = useCallback(() => setConfigAttempt(n => n + 1), [])
  const retryRestore = useCallback(() => dispatch({ type: 'restore-retry' }), [])

  const view = authView(state)
  if (view === 'config-loading') return <Splash message="Starting up..." />
  if (view === 'config-error') {
    return (
      <Splash
        heading="Cannot reach the server"
        message={`Sign-in is unavailable until the app can load its configuration (${state.configError}).`}
        actionLabel="Try again"
        onAction={retryConfig}
      />
    )
  }
  // Not the same as signed out: supabase-js reads its stored session
  // asynchronously, and showing the login form in this window is what makes an
  // already-signed-in user watch it flash by on every single reload.
  if (view === 'restoring') return <Splash message="Restoring your session..." />
  // Not signed out -- unasked. Dropping to the login form here would clear the
  // active thread on the sign-in that follows, which is the one thing T47
  // exists to prevent.
  if (view === 'restore-error') {
    return (
      <Splash
        heading="Could not restore your session"
        message={`Your saved session could not be restored because ${state.restoreError}. Check your connection and try again.`}
        actionLabel="Try again"
        onAction={retryRestore}
      />
    )
  }
  if (view === 'login') return <AuthScreen onSignIn={handleSignIn} />

  return (
    <>
      <AuthenticatedApp
        // Keyed on the user, never on the access token. A token key would
        // change on every auto-refresh -- every 45 minutes, decision 10 -- and
        // React would discard this entire tree mid-analysis.
        key={userIdOf(state)}
        onLogout={handleLogout}
      />
      {showReauthModal(state) && <SessionExpiredModal onSignIn={handleReauth} />}
    </>
  )
}
