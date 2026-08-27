import { useState, useCallback, useEffect, useRef } from 'react'
import { createSseParser } from '../utils/sseParser'
import { applyWorkflowEvent, INITIAL_WORKFLOW_STATE } from '../utils/workflowStage'
import { extractSuggestedFollowups } from '../utils/followups'
import { extractVariableChoice } from '../utils/variableChoice'
import { TERMINAL_STATUSES as TERMINAL_JOB_STATUSES } from '../utils/jobCard'
import { classifyHistoryFetchFailure, historyStateReducer } from '../utils/historyLoad'
import { shouldPromptReauth } from '../utils/sessionExpiry'

const API_BASE = '/api'
const ACTIVE_THREAD_STORAGE_KEY = 'tta.activeThreadId'

export function useChat(accessToken, onUnauthorized, onJobProgress) {
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
  // Retrieval jobs the active stream has reported as still in flight
  // (via job_progress events). "Stop request" returns these from
  // abortActiveRequest so the caller can cancel them — stopping the chat
  // turn alone leaves the job running at the provider and the jobs panel
  // stuck on a row nothing will ever finish.
  const activeJobHandlesRef = useRef(new Set())

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

  const authHeaders = useCallback((extra = {}) => ({
    ...extra,
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
  }), [accessToken])

  const handleUnauthorized = useCallback((res) => {
    // T47's rule has one definition, and sessionExpiry.test.mjs guards it --
    // an inlined `=== 401` here would leave that test guarding nothing.
    if (shouldPromptReauth(res.status) && onUnauthorized) onUnauthorized()
  }, [onUnauthorized])

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

  const abortActiveRequest = useCallback((markCancelled = false) => {
    const controller = abortControllerRef.current
    const streamId = activeStreamIdRef.current
    const inFlightJobHandles = Array.from(activeJobHandlesRef.current)
    activeJobHandlesRef.current = new Set()

    if (controller && !controller.signal.aborted) {
      controller.abort()
    }

    abortControllerRef.current = null
    activeStreamIdRef.current = null
    loadingRef.current = false
    setLoading(false)

    if (markCancelled && streamId !== null) {
      pendingAssistantUpdatesRef.current = pendingAssistantUpdatesRef.current
        .filter(update => update.streamId !== streamId)

      setMessages(prev => prev.map(msg => (
        msg.streamId === streamId && msg.role === 'assistant'
          ? {
              ...msg,
              content: msg.content || 'Request cancelled.',
              isLoading: false,
              isCancelled: true,
              statusMessage: '',
            }
          : msg
      )))
    }

    return inFlightJobHandles
  }, [])

  useEffect(() => {
    return () => {
      abortActiveRequest()
      cancelScheduledFlush()
    }
  }, [abortActiveRequest, cancelScheduledFlush])

  // Returns 'loaded' | 'not-found' | 'failed'. On a transient failure the
  // current messages are left in place (T41) -- a blip in the connection
  // must not read as "your conversation was deleted." Only a genuine 404
  // (the session doesn't exist) clears the view.
  const loadHistory = useCallback(async (id) => {
    if (!accessToken) return 'failed'
    let action
    try {
      const res = await fetch(`${API_BASE}/session/${id}/history`, {
        headers: authHeaders(),
      })
      if (!res.ok) {
        handleUnauthorized(res)
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
  }, [accessToken, authHeaders, handleUnauthorized])

  const retryHistory = useCallback(() => loadHistory(threadIdRef.current), [loadHistory])

  const fetchSessions = useCallback(async () => {
    if (!accessToken) {
      setSessions([])
      return
    }
    try {
      const res = await fetch(`${API_BASE}/sessions`, { headers: authHeaders() })
      if (!res.ok) {
        handleUnauthorized(res)
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      const nextSessions = data.sessions || []
      setSessions(nextSessions)

      if (!didRestoreRef.current) {
        didRestoreRef.current = true
        const storedThreadId = window.localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY)
        if (storedThreadId && nextSessions.some(session => getSessionId(session) === storedThreadId)) {
          setThreadId(storedThreadId)
          threadIdRef.current = storedThreadId
          const loaded = await loadHistory(storedThreadId)
          if (loaded === 'not-found') persistActiveThread(null)
        } else if (storedThreadId) {
          persistActiveThread(null)
        }
      }
    } catch {
      // Non-fatal; the active chat can continue without the sidebar list.
      if (!didRestoreRef.current) {
        didRestoreRef.current = true
        const storedThreadId = window.localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY)
        if (storedThreadId) {
          setThreadId(storedThreadId)
          threadIdRef.current = storedThreadId
          const loaded = await loadHistory(storedThreadId)
          if (loaded === 'not-found') persistActiveThread(null)
        }
      }
    }
  }, [accessToken, authHeaders, getSessionId, handleUnauthorized, loadHistory, persistActiveThread])

  useEffect(() => { fetchSessions() }, [fetchSessions])

  const sendMessage = useCallback(async (text) => {
    if (!accessToken) {
      setError('Please sign in to continue.')
      return
    }
    const message = text.trim()
    if (!message) return

    if (loadingRef.current) {
      abortActiveRequest(true)
    }

    const requestId = activeRequestIdRef.current + 1
    const streamId = `stream-${requestId}`
    const controller = new AbortController()

    activeRequestIdRef.current = requestId
    activeStreamIdRef.current = streamId
    abortControllerRef.current = controller
    // Fresh turn, fresh in-flight-job tracking: handles from a completed
    // earlier stream must not be cancelled by a later Stop click.
    activeJobHandlesRef.current = new Set()

    setMessages(prev => [
      ...prev,
      { role: 'user', content: text },
      {
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
      },
    ])
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ message: text, thread_id: threadIdRef.current }),
        signal: controller.signal,
      })

      if (!res.ok) {
        handleUnauthorized(res)
        throw new Error(`HTTP ${res.status}`)
      }
      if (!res.body) throw new Error('Streaming response was empty')

      const decoder = new TextDecoder()
      const reader = res.body.getReader()
      // The stream must end with a `done` (or `error`) event. If it ends
      // without one — backend killed mid-turn, proxy idle timeout, container
      // restart — reader.read() completes cleanly, nothing throws, and the
      // assistant bubble would stay isLoading forever.
      let sawDone = false
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

        if (event === 'tool_call') {
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
          if (data && data.job_handle) {
            if (TERMINAL_JOB_STATUSES.has(data.status)) {
              activeJobHandlesRef.current.delete(data.job_handle)
            } else {
              activeJobHandlesRef.current.add(data.job_handle)
            }
          }
          if (onJobProgress) onJobProgress(data)
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
          sawDone = true
          const newId = data.thread_id
          setThreadId(newId)
          threadIdRef.current = newId
          persistActiveThread(newId)
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
          setSessions(prev => (
            prev.some(session => getSessionId(session) === newId)
              ? prev
              : [makeLocalSession(newId, message), ...prev]
          ))
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

      if (!sawDone && isCurrentRequest(requestId)) {
        // Stream ended without a terminal event: surface it instead of
        // leaving the bubble spinning. Keep any partial text — the backend
        // may still be working server-side, and its results (charts, jobs)
        // land in history/the Jobs panel.
        const note = 'Connection lost before the response finished. The backend may still be working — reload this session to see any results.'
        setError('Connection lost before the response finished.')
        queueAssistantUpdate(streamId, prevMsg => ({
          content: prevMsg.content ? `${prevMsg.content}\n\n${note}` : note,
          isError: true,
          isConnectionLost: true,
          isLoading: false,
          statusMessage: '',
          workflowStage: applyWorkflowEvent(prevMsg.workflowStage || INITIAL_WORKFLOW_STATE, 'error', {}),
        }))
      }
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
      if (isCurrentRequest(requestId)) {
        abortControllerRef.current = null
        activeStreamIdRef.current = null
        activeJobHandlesRef.current = new Set()
        loadingRef.current = false
        setLoading(false)
      }
    }
  }, [abortActiveRequest, accessToken, authHeaders, getSessionId, handleUnauthorized, isCurrentRequest, makeLocalSession, onJobProgress, persistActiveThread, queueAssistantUpdate])

  const newSession = useCallback(() => {
    abortActiveRequest()
    pendingAssistantUpdatesRef.current = []
    cancelScheduledFlush()
    setMessages([])
    setThreadId(null)
    threadIdRef.current = null
    persistActiveThread(null)
    setError(null)
    setHistoryError(null)
  }, [abortActiveRequest, cancelScheduledFlush, persistActiveThread])

  const switchSession = useCallback(async (id) => {
    abortActiveRequest()
    pendingAssistantUpdatesRef.current = []
    cancelScheduledFlush()
    setError(null)
    setThreadId(id)
    threadIdRef.current = id
    persistActiveThread(id)
    const loaded = await loadHistory(id)
    if (loaded === 'not-found') persistActiveThread(null)
  }, [abortActiveRequest, cancelScheduledFlush, loadHistory, persistActiveThread])

  // The reload affordance on a connection-lost message (T41): reuses
  // switchSession on the same thread so there's exactly one
  // history-hydration code path, rather than a second bespoke reload.
  const reloadSession = useCallback(() => switchSession(threadIdRef.current), [switchSession])

  const deleteSession = useCallback(async (id) => {
    try {
      const res = await fetch(`${API_BASE}/session/${id}`, {
        method: 'DELETE',
        headers: authHeaders(),
      })
      if (!res.ok) {
        handleUnauthorized(res)
        throw new Error(`HTTP ${res.status}`)
      }
      setSessions(prev => prev.filter(session => getSessionId(session) !== id))
      if (id === threadIdRef.current) newSession()
    } catch (err) {
      setError(err.message ? `Failed to delete session: ${err.message}` : 'Failed to delete session. Please try again.')
    }
  }, [authHeaders, getSessionId, handleUnauthorized, newSession])

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
    abortActiveRequest,
    clearError,
    clearHistoryError,
  }
}
