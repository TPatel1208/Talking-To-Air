import { useState, useCallback, useEffect } from 'react'

const API_BASE = 'http://localhost:8000'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [threadId, setThreadId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  // ── Fetch session list ──────────────────────────────────────────────────────
  const fetchSessions = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/sessions`)
      const data = await res.json()
      setSessions(data.sessions || [])
    } catch { /* non-fatal */ }
  }, [])

  useEffect(() => { fetchSessions() }, [fetchSessions])

  // ── Load history for a thread and hydrate messages ──────────────────────────
  const loadHistory = useCallback(async (id) => {
    try {
      const res  = await fetch(`${API_BASE}/session/${id}/history`)
      const data = await res.json()
      const hydrated = (data.messages || []).map(m => ({
        ...m,
        // Prepend API base to image URLs so they resolve correctly
        imageUrls: (m.imageUrls || []).map(u =>
          u.startsWith('http') ? u : `${API_BASE}${u}`
        ),
      }))
      setMessages(hydrated)
    } catch {
      setMessages([])
    }
  }, [])

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const updateLastAssistant = (updater) => {
    setMessages(prev => {
      const next = [...prev]
      const idx  = next.length - 1
      if (idx >= 0 && next[idx].role === 'assistant') {
        next[idx] = { ...next[idx], ...updater(next[idx]) }
      }
      return next
    })
  }

  // ── Send a message ──────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return

    // Only append — don't re-add history that's already shown
    setMessages(prev => [
      ...prev,
      { role: 'user',      content: text },
      { role: 'assistant', content: '', toolCalls: [], imageUrls: [], isLoading: true },
    ])
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text, thread_id: threadId }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()

        for (const part of parts) {
          const eventMatch = part.match(/^event:\s*(.+)$/m)
          const dataMatch  = part.match(/^data:\s*(.+)$/m)
          if (!eventMatch || !dataMatch) continue

          const event = eventMatch[1].trim()
          let   data
          try { data = JSON.parse(dataMatch[1]) } catch { continue }

          if (event === 'tool_call') {
            updateLastAssistant(msg => ({
              toolCalls: [...msg.toolCalls, { name: data.name, args: data.args }],
            }))
          }

          else if (event === 'image') {
            updateLastAssistant(msg => ({
              imageUrls: [...msg.imageUrls, `${API_BASE}${data.url}`],
            }))
          }

          else if (event === 'done') {
            const newId = data.thread_id
            setThreadId(newId)
            updateLastAssistant(() => ({
              content:   data.response,
              imageUrls: (data.image_urls || []).map(u => `${API_BASE}${u}`),
              isLoading: false,
            }))
            setSessions(prev => prev.includes(newId) ? prev : [...prev, newId])
          }

          else if (event === 'error') {
            throw new Error(data.detail || 'Stream error')
          }
        }
      }

    } catch (err) {
      const msg = err.message || 'Request failed'
      setError(msg)
      updateLastAssistant(() => ({
        content:   `Error: ${msg}`,
        isError:   true,
        isLoading: false,
      }))
    } finally {
      setLoading(false)
    }
  }, [threadId, loading])

  // ── Session management ──────────────────────────────────────────────────────
  const newSession = useCallback(() => {
    setMessages([])
    setThreadId(null)
    setError(null)
  }, [])

  const switchSession = useCallback(async (id) => {
    setError(null)
    setThreadId(id)
    await loadHistory(id)   // hydrate UI with Postgres history
  }, [loadHistory])

  const deleteSession = useCallback(async (id) => {
    try {
      await fetch(`${API_BASE}/session/${id}`, { method: 'DELETE' })
      setSessions(prev => prev.filter(s => s !== id))
      if (id === threadId) newSession()
    } catch { /* non-fatal */ }
  }, [threadId, newSession])

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
  }
}