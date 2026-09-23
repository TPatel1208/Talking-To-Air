import { useState, useCallback, useEffect, useRef } from 'react'
import { createSseParser } from '../utils/sseParser'
import { applyWorkflowEvent, INITIAL_WORKFLOW_STATE } from '../utils/workflowStage'
import { extractSuggestedFollowups } from '../utils/followups'
import { extractVariableChoice } from '../utils/variableChoice'
import { classifyHistoryFetchFailure, historyStateReducer } from '../utils/historyLoad'
import { apiFetch } from '../utils/apiFetch.js'
import {
  classifyChatPost,
  classifyStreamEvent,
  classifyTurnStatus,
  clearTurnRecord,
  isStreamError,
  readTurnRecord,
  StreamError,
  streamPath,
  terminalMessagePatch,
  turnRecordThreadIds,
  writeTurnRecord,
} from '../utils/chatTurnProtocol.js'
import { isTurnFrame, mergeSessions, sessionsWithThread } from '../utils/sessionList.js'

const API_BASE = '/api'
const ACTIVE_THREAD_STORAGE_KEY = 'tta.activeThreadId'

// D12: one turn per thread. The server answers a second send with a 409, so
// there is nothing to gain by sending it — and aborting the running turn to
// make room would throw away an answer the user is waiting for.
const ALREADY_RUNNING_MESSAGE = 'This conversation already has a turn running. Wait for it to finish, or press Stop.'
// D15: Redis carries every event, the per-thread claim and the stop signal,
// so chat is down rather than degraded. Reloading history would not help.
const UNAVAILABLE_MESSAGE = 'Chat is temporarily unavailable. Try again in a moment.'
// How often a reader's resume point is written down while a turn streams.
const CURSOR_PERSIST_MS = 1000
const CONNECTION_LOST_MESSAGE = 'Connection lost before the response finished. The backend may still be working — reload this session to see any results.'
// How often a thread the user is not looking at is checked for whether its
// turn is still going. Only threads the sidebar is badging are polled at
// all, so this trades a little latency on the badge turning green for not
// opening a request per second per background thread.
const BACKGROUND_STATUS_POLL_MS = 3000

function newIdempotencyKey() {
  // D13: the 202 handshake makes the retry window real, so a resent POST has
  // to be answerable with the turn it already bought.
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  return `send-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function useChat(onJobProgress) {
  const [messages, setMessages] = useState([])
  const [threadId, setThreadId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [historyError, setHistoryError] = useState(null)
  // Sidebar badge state per thread: 'running' | 'done' | 'error', absent for
  // a thread with nothing to report. Keyed independently of `messages` and
  // `threadId` because its whole point is to outlive the user switching away
  // from the thread it describes (T63's detached turns keep running there).
  const [turnStatus, setTurnStatus] = useState({})

  const abortControllerRef = useRef(null)
  const activeRequestIdRef = useRef(0)
  const activeStreamIdRef = useRef(null)
  const frameRef = useRef(null)
  const loadingRef = useRef(false)
  const pendingAssistantUpdatesRef = useRef([])
  const threadIdRef = useRef(null)
  const turnStatusRef = useRef({})
  const sessionsRef = useRef([])
  const didRestoreRef = useRef(false)
  // Held rather than closed over. The stream reader now sits between the
  // mount effect and this callback — effect -> fetchSessions ->
  // attachToThread -> consumeStream — so depending on it directly would make
  // an unstable `onJobProgress` re-run the session fetch on every render.
  const onJobProgressRef = useRef(onJobProgress)

  useEffect(() => {
    onJobProgressRef.current = onJobProgress
  }, [onJobProgress])

  useEffect(() => {
    loadingRef.current = loading
  }, [loading])

  useEffect(() => {
    threadIdRef.current = threadId
  }, [threadId])

  useEffect(() => {
    turnStatusRef.current = turnStatus
  }, [turnStatus])

  useEffect(() => {
    sessionsRef.current = sessions
  }, [sessions])

  const persistActiveThread = useCallback((id) => {
    if (id) {
      window.localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, id)
    } else {
      window.localStorage.removeItem(ACTIVE_THREAD_STORAGE_KEY)
    }
  }, [])

  const isCurrentRequest = useCallback((requestId) => {
    return activeRequestIdRef.current === requestId
  }, [])

  const getSessionId = useCallback((session) => (
    typeof session === 'string' ? session : session?.id
  ), [])

  // A thread joins the sidebar — and returns to the top of it — when its
  // turn starts narrating, which is what the server both lists and orders
  // by. So the row appears while the answer is being produced rather than
  // when it lands, a turn that produces nothing leaves no empty
  // conversation behind, and a thread picked back up after a week sorts
  // where it was last used rather than where it was started.
  const listThread = useCallback((id, message) => {
    setSessions(prev => sessionsWithThread(prev, id, message))
  }, [])

  const markTurnStatus = useCallback((id, status) => {
    if (!id) return
    setTurnStatus(prev => (prev[id] === status ? prev : { ...prev, [id]: status }))
  }, [])

  const clearTurnStatus = useCallback((id) => {
    if (!id) return
    setTurnStatus(prev => {
      if (!(id in prev)) return prev
      const next = { ...prev }
      delete next[id]
      return next
    })
  }, [])

  const flushAssistantUpdates = useCallback(() => {
    frameRef.current = null

    const updates = pendingAssistantUpdatesRef.current
    pendingAssistantUpdatesRef.current = []
    if (!updates.length) return

    setMessages(prev => {
      let next = prev

      updates.forEach(({ streamId, updater }) => {
        const idx = next.findIndex(msg => msg.streamId === streamId)
        if (idx === -1 || next[idx].role !== 'assistant') return

        if (next === prev) next = [...prev]
        next[idx] = { ...next[idx], ...updater(next[idx]) }
      })

      return next
    })
  }, [])

  const queueAssistantUpdate = useCallback((streamId, updater) => {
    pendingAssistantUpdatesRef.current.push({ streamId, updater })
    if (frameRef.current !== null) return

    // setTimeout, not requestAnimationFrame: rAF is paused by the browser
    // when the tab is backgrounded, which would freeze streamed replies at
    // isLoading: true until the user switches back.
    frameRef.current = window.setTimeout(flushAssistantUpdates, 16)
  }, [flushAssistantUpdates])

  const cancelScheduledFlush = useCallback(() => {
    if (frameRef.current === null) return
    window.clearTimeout(frameRef.current)
    frameRef.current = null
  }, [])

  // Close this client's reader, and nothing else.
  //
  // The half of the old `abortActiveRequest` that switching sessions,
  // unmounting and logging out want. Under T63 the turn does not belong to
  // this connection: it keeps running, keeps writing to its event log, and
  // is there to attach to again. Killing it here is the bug the whole series
  // exists to fix.
  const detach = useCallback(() => {
    const controller = abortControllerRef.current
    if (controller && !controller.signal.aborted) {
      controller.abort()
    }
    abortControllerRef.current = null
    activeStreamIdRef.current = null
    loadingRef.current = false
    setLoading(false)
  }, [])

  // The local ending, for a turn the server will not be ending for us.
  const markCancelledLocally = useCallback((streamId) => {
    if (streamId === null) return
    pendingAssistantUpdatesRef.current = pendingAssistantUpdatesRef.current
      .filter(update => update.streamId !== streamId)

    setMessages(prev => prev.map(msg => (
      // Still loading, or there is nothing to cancel. Stop pressed a moment
      // before the answer landed would otherwise stamp "cancelled" on a
      // reply that arrived in full.
      msg.streamId === streamId && msg.role === 'assistant' && msg.isLoading
        ? {
            ...msg,
            content: msg.content || 'Request cancelled.',
            isLoading: false,
            isCancelled: true,
            statusMessage: '',
          }
        : msg
    )))
  }, [])

  // The other half of the old `abortActiveRequest`: stop the *turn*, not just
  // this reader.
  //
  // The stop is server-side (D10), and so is the ending: the server cancels
  // the turn, cancels the provider retrievals it orphaned, and writes a
  // `stopped` terminal entry that every tab watching this thread receives.
  // Ending the bubble here instead would give the tab that pressed Stop a
  // different story from the tab beside it.
  const stopTurn = useCallback(async () => {
    const id = threadIdRef.current
    const streamId = activeStreamIdRef.current
    if (id) {
      try {
        const res = await apiFetch(`${API_BASE}/chat/${encodeURIComponent(id)}/stop`, {
          method: 'POST',
        })
        if (res.ok) return
      } catch {
        // Fall through: an unreachable backend cannot end the turn for us.
      }
    }
    // No detached turn to stop — the old streaming POST, or a stop that could
    // not be delivered. Closing the reader is then the only ending available.
    detach()
    markCancelledLocally(streamId)
  }, [detach, markCancelledLocally])

  useEffect(() => {
    return () => {
      detach()
      cancelScheduledFlush()
    }
  }, [detach, cancelScheduledFlush])

  // Returns 'loaded' | 'not-found' | 'failed'. On a transient failure the
  // current messages are left in place (T41) -- a blip in the connection
  // must not read as "your conversation was deleted." Only a genuine 404
  // (the session doesn't exist) clears the view.
  const loadHistory = useCallback(async (id) => {
    let action
    try {
      const res = await apiFetch(`${API_BASE}/session/${id}/history`)
      if (!res.ok) {
        action = classifyHistoryFetchFailure(res.status) === 'not-found'
          ? { type: 'not-found' }
          : { type: 'failed' }
      } else {
        const data = await res.json()
        const hydrated = (data.messages || []).map(m => ({
          ...m,
          artifacts: m.artifacts || [],
          imageUrls: (m.imageUrls || []).map(u =>
            u.startsWith('http') ? u : `${API_BASE}${u}`
          ),
        }))
        action = { type: 'loaded', messages: hydrated }
      }
    } catch {
      action = { type: 'failed' }
    }

    setMessages(prev => historyStateReducer({ messages: prev, historyError: null }, action).messages)
    setHistoryError(historyStateReducer({ messages: [], historyError: null }, action).historyError)

    return action.type === 'loaded' ? 'loaded' : action.type
  }, [])

  const retryHistory = useCallback(() => loadHistory(threadIdRef.current), [loadHistory])

  /**
   * Read an SSE body to its end, applying every frame to `ctx.streamId`.
   *
   * One handler for both protocols. The streaming POST and the GET carry
   * byte-identical frames (D5) — the event log stores what `sse()` rendered
   * — so two handlers would have nothing to gain and somewhere to drift.
   *
   * Returns whether the stream ended the way it is supposed to. It is not
   * enough to watch for `done`: the closing set is four names wide, and even
   * before Phase 5 the generic failure path emitted `error` and no `done` at
   * all, so the old single-name guard already mislabelled a real backend
   * error as a lost connection.
   */
  const consumeStream = useCallback(async (res, ctx) => {
    if (!res.body) throw new Error('Streaming response was empty')

    const { requestId, streamId } = ctx
    const state = { sawTerminal: false, reconcile: false }
    const decoder = new TextDecoder()
    const reader = res.body.getReader()
    // The follower emits a cursor after every page it delivers, which during
    // an answer is about ten a second. localStorage.setItem is synchronous,
    // and this is the streaming hot path. Being a second behind costs a
    // resume a second of replay into a bubble that is empty anyway.
    let cursorWrittenAt = 0
    // Whether this thread has been put in the sidebar yet. One check per
    // frame, one setSessions per stream.
    let listed = false

    const parser = createSseParser(({ event, data: rawData }) => {
      if (!isCurrentRequest(requestId)) return

      let data
      try {
        data = JSON.parse(rawData)
      } catch {
        queueAssistantUpdate(streamId, () => ({
          content: 'Error: Received malformed stream data.',
          isError: true,
          isLoading: false,
        }))
        throw new Error('Malformed stream data')
      }

      const { terminal, kind } = classifyStreamEvent(event)
      if (terminal) {
        state.sawTerminal = true
        // Independent of whether this reader is attached because it sent
        // the message or because it reattached to a thread the sidebar was
        // badging: either way, the badge for this thread should now say how
        // the turn ended rather than that it is still running.
        markTurnStatus(ctx.threadId, classifyTurnStatus(kind))
      }

      // The turn has started narrating, so this conversation has something in
      // it and belongs in the sidebar (utils/sessionList.js). The server
      // stamps the same moment from its side of the stream, which is why the
      // follower's own frames are excluded: nothing stamps for those, and a
      // row listed on one would disappear on the next /sessions fetch.
      if (!listed && isTurnFrame(event)) {
        listed = true
        listThread(ctx.threadId, ctx.userMessage)
      }

      if (kind === 'cursor') {
        // The follower's own frame, emitted *after* the page it accounts
        // for — so a stored cursor only ever covers frames already
        // rendered. Persisted per thread so a remount resumes the turn
        // instead of restarting it.
        const now = Date.now()
        if (data.turn_id && data.cursor && ctx.threadId && now - cursorWrittenAt >= CURSOR_PERSIST_MS) {
          cursorWrittenAt = now
          writeTurnRecord(window.localStorage, ctx.threadId, {
            turnId: data.turn_id,
            cursor: data.cursor,
            // Carried so a remount can show what was asked. The event log
            // holds only what the turn *said*, and the exchange does not
            // reach history until the turn is written back.
            userMessage: ctx.userMessage,
          })
        }
      } else if (event === 'tool_call') {
        queueAssistantUpdate(streamId, msg => ({
          toolCalls: [...(msg.toolCalls || []), { name: data.name, args: data.args }],
        }))
      } else if (event === 'status') {
        queueAssistantUpdate(streamId, msg => ({
          statusMessage: data.message || '',
          workflowStage: applyWorkflowEvent(msg.workflowStage || INITIAL_WORKFLOW_STATE, 'status', data),
        }))
      } else if (event === 'image') {
        queueAssistantUpdate(streamId, msg => ({
          imageUrls: [...(msg.imageUrls || []), `${API_BASE}${data.url}`],
        }))
      } else if (event === 'chart') {
        if (!data || typeof data !== 'object' || !data.type) {
          console.warn('[useChat] Ignoring non-object chart event:', data)
        } else {
          queueAssistantUpdate(streamId, msg => ({
            charts: [...(msg.charts || []), data],
          }))
        }
      } else if (event === 'artifact') {
        if (!data || typeof data !== 'object' || !data.id || !data.type) {
          console.warn('[useChat] Ignoring non-object artifact event:', data)
        } else {
          queueAssistantUpdate(streamId, msg => ({
            artifacts: [...(msg.artifacts || []), data],
          }))
        }
      } else if (event === 'job_progress') {
        onJobProgressRef.current?.(data)
        queueAssistantUpdate(streamId, msg => ({
          workflowStage: applyWorkflowEvent(msg.workflowStage || INITIAL_WORKFLOW_STATE, 'job_progress', data),
        }))
      } else if (event === 'text') {
        const chunk = typeof data === 'string' ? data : data.content
        if (chunk) {
          queueAssistantUpdate(streamId, msg => ({
            content: `${msg.content || ''}${chunk}`,
            // User story #6: narration stops cleanly the moment the
            // answer starts streaming — progress never talks over
            // results.
            workflowStage: applyWorkflowEvent(msg.workflowStage || INITIAL_WORKFLOW_STATE, 'text', data),
          }))
        }
      } else if (event === 'done') {
        const newId = data.thread_id || ctx.threadId
        setThreadId(newId)
        threadIdRef.current = newId
        persistActiveThread(newId)
        clearTurnRecord(window.localStorage, newId)
        // A turn that ended is history's to tell, not this stream's (D8).
        // A reader that only *joined* this turn — a remount, a second tab —
        // is sitting on top of a history fetch that may already contain the
        // same answer, so it reloads history and lets that replace what it
        // rendered. The write-back happens before the `done` frame, so by
        // now history has it.
        //
        // The bubble is still completed first, and deliberately: the reload
        // can fail transiently, and T41 keeps the current messages when it
        // does — which would leave this bubble spinning on an answer that
        // had already arrived.
        if (ctx.reattached) state.reconcile = newId
        queueAssistantUpdate(streamId, msg => ({
          content: data.response || msg.content || '',
          imageUrls: (data.image_urls || []).map(u => `${API_BASE}${u}`),
          charts: msg.charts || [],
          artifacts: msg.artifacts?.length ? msg.artifacts : (data.artifacts || []),
          suggestedFollowups: extractSuggestedFollowups(data),
          // T49: the deterministic variable-choice picker, when the resolver
          // couldn't confidently choose. null the vast majority of turns.
          variableChoice: extractVariableChoice(data),
          statusMessage: '',
          workflowStage: INITIAL_WORKFLOW_STATE,
          isLoading: false,
        }))
        // The legacy protocol only. Its POST streams the turn itself and
        // carries the thread id in this frame, so a brand-new thread has no
        // id to list under until here. Under the detached protocol the 202
        // named the thread before the first frame arrived and it is already
        // listed — which is the point: a turn that never finishes still
        // shows up in the sidebar it is running in.
        if (!ctx.threadId) listThread(newId, ctx.userMessage)
      } else if (event === 'stopped' || event === 'interrupted') {
        // The two endings the user did not ask for and did not get an answer
        // from. Both leave whatever the turn had already produced in place:
        // a stopped turn's charts are persisted (D11), and an interrupted
        // one's partial text is still the most it managed to say.
        if (ctx.threadId) clearTurnRecord(window.localStorage, ctx.threadId)
        queueAssistantUpdate(streamId, msg => terminalMessagePatch(event, data, msg))
      } else if (event === 'error') {
        // The turn's own report of how it failed, and it arrived -- so the
        // reader's catch must not mistake it for the transport going away.
        throw new StreamError(data.detail || 'Stream error')
      }
    })

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parser.feed(decoder.decode(value, { stream: true }))
    }

    const finalChunk = decoder.decode()
    if (finalChunk) parser.feed(finalChunk)
    parser.end()

    return state
  }, [isCurrentRequest, listThread, markTurnStatus, persistActiveThread, queueAssistantUpdate])

  const markConnectionLost = useCallback((streamId) => {
    // The stream stopped without saying why — a proxy idle timeout, a dropped
    // socket, a laptop that slept. Keep any partial text: the turn is
    // detached, so it is very likely still running, and "Reload session"
    // rehydrates history and reattaches to it.
    setError('Connection lost before the response finished.')
    queueAssistantUpdate(streamId, prevMsg => ({
      content: prevMsg.content ? `${prevMsg.content}\n\n${CONNECTION_LOST_MESSAGE}` : CONNECTION_LOST_MESSAGE,
      isError: true,
      isConnectionLost: true,
      isLoading: false,
      statusMessage: '',
      workflowStage: applyWorkflowEvent(prevMsg.workflowStage || INITIAL_WORKFLOW_STATE, 'error', {}),
    }))
  }, [queueAssistantUpdate])

  const beginLocalTurn = useCallback(() => {
    const requestId = activeRequestIdRef.current + 1
    const streamId = `stream-${requestId}`
    const controller = new AbortController()

    activeRequestIdRef.current = requestId
    activeStreamIdRef.current = streamId
    abortControllerRef.current = controller

    return { requestId, streamId, controller }
  }, [])

  const assistantPlaceholder = useCallback((streamId) => ({
    role: 'assistant',
    content: '',
    toolCalls: [],
    statusMessage: '',
    workflowStage: INITIAL_WORKFLOW_STATE,
    startedAt: Date.now(),
    imageUrls: [],
    charts: [],
    artifacts: [],
    suggestedFollowups: [],
    variableChoice: null,
    isLoading: true,
    streamId,
  }), [])

  const releaseIfCurrent = useCallback((requestId) => {
    if (!isCurrentRequest(requestId)) return
    abortControllerRef.current = null
    activeStreamIdRef.current = null
    loadingRef.current = false
    setLoading(false)
  }, [isCurrentRequest])

  /**
   * Watch the turn already running on this thread, if there is one.
   *
   * Runs on mount and on every session switch, which is what makes "switch
   * away and come back" and "reload the page mid-turn" work at all — and
   * what lets a second tab join a turn rather than forking one (D12).
   *
   * The 404 is the ordinary case: most threads have no turn in flight.
   */
  const attachToThread = useCallback(async (id) => {
    if (!id || loadingRef.current) return
    const record = readTurnRecord(window.localStorage, id)
    const { requestId, streamId, controller } = beginLocalTurn()
    // Nothing is shown until there is a turn to show. Most of the time this
    // is a probe that comes back 404, and a probe must be invisible: an
    // unreachable backend on a routine session switch is the history fetch's
    // story to tell, not a lost-connection notice on a bubble that was never
    // rendered.
    let rendered = false

    try {
      const res = await apiFetch(streamPath(API_BASE, id, {
        cursor: record?.cursor, turnId: record?.turnId,
      }), {
        signal: controller.signal,
      })
      if (res.status === 404) {
        releaseIfCurrent(requestId)
        // Nothing is running, so nothing this client remembers about a turn
        // on this thread is worth keeping -- including any badge it was
        // showing for it.
        clearTurnRecord(window.localStorage, id)
        clearTurnStatus(id)
        return
      }
      if (!res.ok) {
        releaseIfCurrent(requestId)
        return
      }

      // Confirmed against the server rather than assumed from the stored
      // record: this path also runs for a thread the sidebar only guessed
      // was still running (seeded from a leftover record on mount), and a
      // 200 here is what turns that guess into fact.
      markTurnStatus(id, 'running')
      setMessages(prev => [
        ...prev,
        // What was asked, if this client is the one that asked it. A tab
        // joining someone else's turn has no record and shows the answer
        // alone, which is the honest thing for it to show.
        ...(record?.userMessage ? [{ role: 'user', content: record.userMessage }] : []),
        assistantPlaceholder(streamId),
      ])
      setLoading(true)
      loadingRef.current = true
      rendered = true

      const state = await consumeStream(res, {
        requestId, streamId, threadId: id, reattached: true, userMessage: record?.userMessage,
      })
      if (!isCurrentRequest(requestId)) return
      if (state.reconcile) {
        await loadHistory(state.reconcile)
      } else if (!state.sawTerminal) {
        markConnectionLost(streamId)
      }
      if (state.sawTerminal) {
        // Whatever this attach just showed -- a replay of an ending that had
        // already happened, or one that landed while the user watched -- they
        // have now seen it directly. The badge exists to say "go look at
        // this"; visiting discharges it, so it does not reappear until
        // something new actually happens on this thread.
        clearTurnStatus(id)
      }
    } catch (err) {
      if (err.name === 'AbortError') return
      if (!isCurrentRequest(requestId) || !rendered) return
      markConnectionLost(streamId)
    } finally {
      releaseIfCurrent(requestId)
    }
  }, [
    assistantPlaceholder, beginLocalTurn, clearTurnStatus, consumeStream,
    isCurrentRequest, loadHistory, markConnectionLost, markTurnStatus,
    releaseIfCurrent,
  ])

  const fetchSessions = useCallback(async () => {
    const restore = async () => {
      if (didRestoreRef.current) return
      didRestoreRef.current = true
      const storedThreadId = window.localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY)
      if (!storedThreadId) return
      setThreadId(storedThreadId)
      threadIdRef.current = storedThreadId
      const loaded = await loadHistory(storedThreadId)
      if (loaded === 'not-found') {
        persistActiveThread(null)
        clearTurnRecord(window.localStorage, storedThreadId)
        clearTurnStatus(storedThreadId)
        return
      }
      await attachToThread(storedThreadId)
    }

    try {
      const res = await apiFetch(`${API_BASE}/sessions`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      const nextSessions = data.sessions || []
      setSessions(nextSessions)

      // Deliberately not gated on the list: a thread whose turn has not
      // produced a frame yet is not in it, and dropping the stored thread
      // there would abandon a running turn on a reload — the reattach this
      // whole protocol exists for. A thread that is genuinely gone is still
      // caught, one request later, by restore()'s own not-found handling.
      await restore()
    } catch {
      // Non-fatal; the active chat can continue without the sidebar list.
      await restore()
    }
  }, [attachToThread, clearTurnStatus, loadHistory, persistActiveThread])

  useEffect(() => { fetchSessions() }, [fetchSessions])

  // Seeds the badge for every thread this browser still has a turn record
  // for, before anything has confirmed whether those turns are still going.
  // Optimistic on purpose: `attachToThread` (above, for the restored active
  // thread) and the poll below (for every other one) each correct their own
  // entry within one round trip, and the alternative -- waiting for that
  // round trip before showing anything -- is the "can I navigate back to it"
  // question arriving late on exactly the reload this is for.
  useEffect(() => {
    const ids = turnRecordThreadIds(window.localStorage)
    if (!ids.length) return
    setTurnStatus(prev => {
      const next = { ...prev }
      let changed = false
      for (const id of ids) {
        if (next[id]) continue
        next[id] = 'running'
        changed = true
      }
      return changed ? next : prev
    })
  }, [])

  // Watches every thread the badge calls 'running' that is not the one on
  // screen right now -- the active thread's own status comes from the live
  // stream above, which is cheaper and more current than a poll could be.
  useEffect(() => {
    const tick = async () => {
      const active = threadIdRef.current
      const badged = Object.keys(turnStatusRef.current)

      // A thread the badge already knows about but the sidebar does not: the
      // user sent the message and switched away before its first frame got
      // here, so `sessionsWithThread`'s own optimistic add never ran on this
      // tab (see useChat's send/reattach paths). The server lists the thread
      // the moment its turn narrates regardless of who is watching, so this
      // is what brings the row -- and the badge sitting on it -- in without
      // the user having to reload.
      const known = new Set(sessionsRef.current.map(getSessionId))
      if (badged.some(id => !known.has(id))) {
        try {
          const res = await apiFetch(`${API_BASE}/sessions`)
          if (res.ok) {
            const data = await res.json()
            setSessions(prev => mergeSessions(prev, data.sessions || []))
          }
        } catch {
          // Best-effort; the next tick tries again.
        }
      }

      const pending = Object.entries(turnStatusRef.current)
        .filter(([id, status]) => status === 'running' && id !== active)
        .map(([id]) => id)
      if (!pending.length) return
      await Promise.all(pending.map(async (id) => {
        try {
          const res = await apiFetch(`${API_BASE}/chat/${encodeURIComponent(id)}/status`)
          if (res.status === 404) {
            // Nothing this replica remembers any more -- past the ten-minute
            // window `last_turn` covers, most likely. Nothing to badge: an
            // unknown ending is worth less than no badge at all.
            clearTurnStatus(id)
            return
          }
          if (!res.ok) return
          const body = await res.json()
          markTurnStatus(id, classifyTurnStatus(body.terminal))
        } catch {
          // Transient; the next tick tries again rather than guessing.
        }
      }))
    }
    const interval = window.setInterval(tick, BACKGROUND_STATUS_POLL_MS)
    return () => window.clearInterval(interval)
  }, [clearTurnStatus, getSessionId, markTurnStatus])

  const sendMessage = useCallback(async (text) => {
    const message = text.trim()
    if (!message) return

    if (loadingRef.current) {
      // Refused here rather than sent and refused there. Under D12 the
      // server answers this with a 409 whatever the client does first, and
      // the thing the old code did first — abort the running turn — now
      // throws away an answer that is still being produced.
      setError(ALREADY_RUNNING_MESSAGE)
      return false
    }

    const { requestId, streamId, controller } = beginLocalTurn()
    // Whether the answer had started arriving when something threw. Past
    // this point a failure is the connection, not the request.
    let streaming = false

    setMessages(prev => [
      ...prev,
      { role: 'user', content: text },
      assistantPlaceholder(streamId),
    ])
    setLoading(true)
    loadingRef.current = true
    setError(null)

    try {
      const res = await apiFetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Idempotency-Key': newIdempotencyKey(),
        },
        body: JSON.stringify({ message: text, thread_id: threadIdRef.current }),
        signal: controller.signal,
      })

      // The kill switch deliberately does not reach the frontend: 200 +
      // text/event-stream and 202 + JSON are already distinguishable, so one
      // bundle serves both and rolling the backend flag back needs no
      // frontend rebuild.
      const body = res.status === 200 ? null : await res.json().catch(() => null)
      const outcome = classifyChatPost(res.status, body)

      if (outcome.kind === 'unavailable') {
        throw new Error(body?.detail || UNAVAILABLE_MESSAGE)
      }
      if (outcome.kind === 'failed') {
        throw new Error(`HTTP ${res.status}`)
      }

      if (outcome.kind === 'legacy-stream') {
        const state = await consumeStream(res, {
          requestId, streamId, threadId: threadIdRef.current, userMessage: text,
        })
        if (!state.sawTerminal && isCurrentRequest(requestId)) markConnectionLost(streamId)
        return
      }

      const acceptedThread = outcome.threadId || threadIdRef.current
      setThreadId(acceptedThread)
      threadIdRef.current = acceptedThread
      persistActiveThread(acceptedThread)

      if (outcome.kind === 'joined') {
        // This message was not accepted — the id names a turn that was
        // already running. Joining it is what the user with two tabs
        // expects; the bubbles staged for a send that did not happen are
        // not, so they come back out.
        setMessages(prev => {
          const idx = prev.findIndex(msg => msg.streamId === streamId)
          if (idx === -1) return prev
          const from = idx > 0 && prev[idx - 1].role === 'user' ? idx - 1 : idx
          return [...prev.slice(0, from), ...prev.slice(idx + 1)]
        })
        setError(ALREADY_RUNNING_MESSAGE)
        releaseIfCurrent(requestId)
        // Started, not awaited. Joining lasts as long as the turn being
        // joined, and the caller is only asking whether its message was
        // accepted -- which the 409 has already answered. Awaiting this held
        // that answer for minutes, so the composer took the user's text back
        // when somebody else's turn ended, into whichever conversation they
        // were looking at by then. attachToThread handles its own failures
        // and never rejects.
        attachToThread(acceptedThread)
        // Refused, so the caller keeps what the user typed. Nothing was
        // sent, and there is nowhere else for that text to have gone.
        return false
      }

      // D6: the POST carried no events at all. Everything this turn narrates
      // arrives over the GET, so the reattach path runs on every turn and
      // cannot rot.
      writeTurnRecord(window.localStorage, acceptedThread, {
        turnId: outcome.turnId, cursor: null, userMessage: text,
      })
      // The moment the sidebar has something to badge: this send bought a
      // turn, whether or not the user is still looking at this thread by the
      // time it finishes.
      markTurnStatus(acceptedThread, 'running')
      const stream = await apiFetch(streamPath(API_BASE, acceptedThread, {
        turnId: outcome.turnId,
      }), {
        signal: controller.signal,
      })
      if (!stream.ok) throw new Error(`HTTP ${stream.status}`)

      streaming = true
      const state = await consumeStream(stream, {
        requestId, streamId, threadId: acceptedThread, userMessage: text,
      })
      if (!state.sawTerminal && isCurrentRequest(requestId)) markConnectionLost(streamId)
    } catch (err) {
      if (err.name === 'AbortError') return
      if (!isCurrentRequest(requestId)) return

      if (streaming && !isStreamError(err)) {
        // The answer was already arriving and the turn did not say why it
        // stopped, so this is the transport going away under it -- a backend
        // rolled mid-turn severs the response body, and reading it throws
        // rather than returning, which is why the `sawTerminal` check above
        // never runs. The turn itself is very likely still there: it outlives
        // this connection by design, and "Reload session" reattaches to it.
        // Same handling as the reattach path, which is the same situation
        // reached the other way round.
        markConnectionLost(streamId)
        return
      }
      const msg = err.message || 'Request failed'
      setError(msg)
      queueAssistantUpdate(streamId, prevMsg => ({
        content: `Error: ${msg}`,
        isError: true,
        isLoading: false,
        statusMessage: '',
        // User story #9: the strip shows which stage failed, so the error
        // answer has visible context instead of the progress trail just
        // vanishing.
        workflowStage: applyWorkflowEvent(prevMsg.workflowStage || INITIAL_WORKFLOW_STATE, 'error', {}),
      }))
    } finally {
      releaseIfCurrent(requestId)
    }
  }, [
    assistantPlaceholder, attachToThread, beginLocalTurn, consumeStream,
    isCurrentRequest, markConnectionLost, markTurnStatus, persistActiveThread,
    queueAssistantUpdate, releaseIfCurrent,
  ])

  const newSession = useCallback(() => {
    detach()
    pendingAssistantUpdatesRef.current = []
    cancelScheduledFlush()
    setMessages([])
    setThreadId(null)
    threadIdRef.current = null
    persistActiveThread(null)
    setError(null)
    setHistoryError(null)
  }, [cancelScheduledFlush, detach, persistActiveThread])

  const switchSession = useCallback(async (id) => {
    detach()
    pendingAssistantUpdatesRef.current = []
    cancelScheduledFlush()
    setError(null)
    setThreadId(id)
    threadIdRef.current = id
    persistActiveThread(id)
    const loaded = await loadHistory(id)
    if (loaded === 'not-found') {
      persistActiveThread(null)
      clearTurnRecord(window.localStorage, id)
      clearTurnStatus(id)
      return
    }
    await attachToThread(id)
  }, [attachToThread, cancelScheduledFlush, clearTurnStatus, detach, loadHistory, persistActiveThread])

  // The reload affordance on a connection-lost or interrupted message (T41,
  // T63): reuses switchSession on the same thread so there's exactly one
  // history-hydration code path, and — since switchSession now reattaches —
  // one recovery path for a turn that is still running.
  const reloadSession = useCallback(() => switchSession(threadIdRef.current), [switchSession])

  const deleteSession = useCallback(async (id) => {
    try {
      const res = await apiFetch(`${API_BASE}/session/${id}`, { method: 'DELETE' })
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      setSessions(prev => prev.filter(session => getSessionId(session) !== id))
      clearTurnRecord(window.localStorage, id)
      clearTurnStatus(id)
      if (id === threadIdRef.current) newSession()
    } catch (err) {
      setError(err.message ? `Failed to delete session: ${err.message}` : 'Failed to delete session. Please try again.')
    }
  }, [clearTurnStatus, getSessionId, newSession])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const clearHistoryError = useCallback(() => {
    setHistoryError(null)
  }, [])

  return {
    messages,
    loading,
    error,
    historyError,
    threadId,
    sessions,
    turnStatus,
    sendMessage,
    newSession,
    switchSession,
    reloadSession,
    retryHistory,
    deleteSession,
    stopTurn,
    detach,
    clearError,
    clearHistoryError,
  }
}
