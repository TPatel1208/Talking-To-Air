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
  clearTurnRecord,
  readTurnRecord,
  streamPath,
  terminalMessagePatch,
  writeTurnRecord,
} from '../utils/chatTurnProtocol.js'

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

  const abortControllerRef = useRef(null)
  const activeRequestIdRef = useRef(0)
  const activeStreamIdRef = useRef(null)
  const frameRef = useRef(null)
  const loadingRef = useRef(false)
  const pendingAssistantUpdatesRef = useRef([])
  const threadIdRef = useRef(null)
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

  const makeLocalSession = useCallback((id, message) => {
    const title = message.trim().replace(/\s+/g, ' ')
    return {
      id,
      title: title.length > 60 ? `${title.slice(0, 57).trim()}...` : title,
      created_at: new Date().toISOString(),
    }
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
      if (terminal) state.sawTerminal = true

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
        if (ctx.userMessage) {
          setSessions(prev => (
            prev.some(session => getSessionId(session) === newId)
              ? prev
              : [makeLocalSession(newId, ctx.userMessage), ...prev]
          ))
        }
      } else if (event === 'stopped' || event === 'interrupted') {
        // The two endings the user did not ask for and did not get an answer
        // from. Both leave whatever the turn had already produced in place:
        // a stopped turn's charts are persisted (D11), and an interrupted
        // one's partial text is still the most it managed to say.
        if (ctx.threadId) clearTurnRecord(window.localStorage, ctx.threadId)
        queueAssistantUpdate(streamId, msg => terminalMessagePatch(event, data, msg))
      } else if (event === 'error') {
        throw new Error(data.detail || 'Stream error')
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
  }, [getSessionId, isCurrentRequest, makeLocalSession, persistActiveThread, queueAssistantUpdate])

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
        // on this thread is worth keeping.
        clearTurnRecord(window.localStorage, id)
        return
      }
      if (!res.ok) {
        releaseIfCurrent(requestId)
        return
      }

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
    } catch (err) {
      if (err.name === 'AbortError') return
      if (!isCurrentRequest(requestId) || !rendered) return
      markConnectionLost(streamId)
    } finally {
      releaseIfCurrent(requestId)
    }
  }, [
    assistantPlaceholder, beginLocalTurn, consumeStream, isCurrentRequest,
    loadHistory, markConnectionLost, releaseIfCurrent,
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

      if (!didRestoreRef.current) {
        const storedThreadId = window.localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY)
        if (storedThreadId && !nextSessions.some(session => getSessionId(session) === storedThreadId)) {
          didRestoreRef.current = true
          persistActiveThread(null)
          return
        }
        await restore()
      }
    } catch {
      // Non-fatal; the active chat can continue without the sidebar list.
      await restore()
    }
  }, [attachToThread, getSessionId, loadHistory, persistActiveThread])

  useEffect(() => { fetchSessions() }, [fetchSessions])

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
        await attachToThread(acceptedThread)
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
      const stream = await apiFetch(streamPath(API_BASE, acceptedThread, {
        turnId: outcome.turnId,
      }), {
        signal: controller.signal,
      })
      if (!stream.ok) throw new Error(`HTTP ${stream.status}`)

      const state = await consumeStream(stream, {
        requestId, streamId, threadId: acceptedThread, userMessage: text,
      })
      if (!state.sawTerminal && isCurrentRequest(requestId)) markConnectionLost(streamId)
    } catch (err) {
      if (err.name === 'AbortError') return
      if (!isCurrentRequest(requestId)) return

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
    isCurrentRequest, markConnectionLost, persistActiveThread,
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
      return
    }
    await attachToThread(id)
  }, [attachToThread, cancelScheduledFlush, detach, loadHistory, persistActiveThread])

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
      if (id === threadIdRef.current) newSession()
    } catch (err) {
      setError(err.message ? `Failed to delete session: ${err.message}` : 'Failed to delete session. Please try again.')
    }
  }, [getSessionId, newSession])

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
