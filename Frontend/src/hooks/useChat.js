import { useState, useCallback, useEffect, useRef } from 'react'
import { createSseParser } from '../utils/sseParser'

const API_BASE = '/api'
const ACTIVE_THREAD_STORAGE_KEY = 'tta.activeThreadId'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [threadId, setThreadId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const abortControllerRef = useRef(null)
  const activeRequestIdRef = useRef(0)
  const activeStreamIdRef = useRef(null)
  const frameRef = useRef(null)
  const loadingRef = useRef(false)
  const pendingAssistantUpdatesRef = useRef([])
  const threadIdRef = useRef(null)
  const didRestoreRef = useRef(false)

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

    frameRef.current = window.requestAnimationFrame
      ? window.requestAnimationFrame(flushAssistantUpdates)
      : window.setTimeout(flushAssistantUpdates, 16)
  }, [flushAssistantUpdates])

  const cancelScheduledFlush = useCallback(() => {
    if (frameRef.current === null) return

    if (window.cancelAnimationFrame) {
      window.cancelAnimationFrame(frameRef.current)
    } else {
      window.clearTimeout(frameRef.current)
    }
    frameRef.current = null
  }, [])

  const abortActiveRequest = useCallback((markCancelled = false) => {
    const controller = abortControllerRef.current
    const streamId = activeStreamIdRef.current

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
  }, [])

  useEffect(() => {
    return () => {
      abortActiveRequest()
      cancelScheduledFlush()
    }
  }, [abortActiveRequest, cancelScheduledFlush])

  const loadHistory = useCallback(async (id) => {
    try {
      const res = await fetch(`${API_BASE}/session/${id}/history`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const hydrated = (data.messages || []).map(m => ({
        ...m,
        imageUrls: (m.imageUrls || []).map(u =>
          u.startsWith('http') ? u : `${API_BASE}${u}`
        ),
      }))
      setMessages(hydrated)
      return true
    } catch {
      setMessages([])
      return false
    }
  }, [])

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`)
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
          if (!loaded) persistActiveThread(null)
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
          if (!loaded) persistActiveThread(null)
        }
      }
    }
  }, [getSessionId, loadHistory, persistActiveThread])

  useEffect(() => { fetchSessions() }, [fetchSessions])

  const sendMessage = useCallback(async (text) => {
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

    setMessages(prev => [
      ...prev,
      { role: 'user', content: text },
      {
        role: 'assistant',
        content: '',
        toolCalls: [],
        statusMessage: '',
        imageUrls: [],
        charts: [],
        isLoading: true,
        streamId,
      },
    ])
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, thread_id: threadIdRef.current }),
        signal: controller.signal,
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      if (!res.body) throw new Error('Streaming response was empty')

      const decoder = new TextDecoder()
      const reader = res.body.getReader()
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
          queueAssistantUpdate(streamId, () => ({
            statusMessage: data.message || '',
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
        } else if (event === 'done') {
          const newId = data.thread_id
          setThreadId(newId)
          threadIdRef.current = newId
          persistActiveThread(newId)
          queueAssistantUpdate(streamId, msg => ({
            content: data.response,
            imageUrls: (data.image_urls || []).map(u => `${API_BASE}${u}`),
            charts: msg.charts || [],
            statusMessage: '',
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
    } catch (err) {
      if (err.name === 'AbortError') return
      if (!isCurrentRequest(requestId)) return

      const msg = err.message || 'Request failed'
      setError(msg)
      queueAssistantUpdate(streamId, () => ({
        content: `Error: ${msg}`,
        isError: true,
        isLoading: false,
        statusMessage: '',
      }))
    } finally {
      if (isCurrentRequest(requestId)) {
        abortControllerRef.current = null
        activeStreamIdRef.current = null
        loadingRef.current = false
        setLoading(false)
      }
    }
  }, [abortActiveRequest, getSessionId, isCurrentRequest, makeLocalSession, persistActiveThread, queueAssistantUpdate])

  const newSession = useCallback(() => {
    abortActiveRequest()
    pendingAssistantUpdatesRef.current = []
    cancelScheduledFlush()
    setMessages([])
    setThreadId(null)
    threadIdRef.current = null
    persistActiveThread(null)
    setError(null)
  }, [abortActiveRequest, cancelScheduledFlush, persistActiveThread])

  const switchSession = useCallback(async (id) => {
    abortActiveRequest()
    pendingAssistantUpdatesRef.current = []
    cancelScheduledFlush()
    setError(null)
    setThreadId(id)
    threadIdRef.current = id
    persistActiveThread(id)
    await loadHistory(id)
  }, [abortActiveRequest, cancelScheduledFlush, loadHistory, persistActiveThread])

  const deleteSession = useCallback(async (id) => {
    try {
      const res = await fetch(`${API_BASE}/session/${id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setSessions(prev => prev.filter(session => getSessionId(session) !== id))
      if (id === threadIdRef.current) newSession()
    } catch (err) {
      setError(err.message ? `Failed to delete session: ${err.message}` : 'Failed to delete session. Please try again.')
    }
  }, [getSessionId, newSession])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
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
  }
}
