import { useState, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

function normalizeImageUrl(rawUrl) {
  if (!rawUrl) return null
  // Already a clean relative path
  if (rawUrl.startsWith('/outputs/')) return `${API_BASE}${rawUrl}`
  // Full file:/// or absolute OS path — extract just the filename
  const filename = rawUrl.replace(/\\/g, '/').split('/').pop()
  return `${API_BASE}/outputs/${filename}`
}

export function useChat() {
  const [messages, setMessages]   = useState([])
  const [threadId, setThreadId]   = useState(null)
  const [loading,  setLoading]    = useState(false)
  const [error,    setError]      = useState(null)

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

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return

    setMessages(prev => [
      ...prev,
      { role: 'user', content: text },
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
            const url = normalizeImageUrl(data.url)
            if (url) {
              updateLastAssistant(msg => ({
                imageUrls: [...msg.imageUrls, url],
              }))
            }
          }

          else if (event === 'done') {
            setThreadId(data.thread_id)
            const normalizedUrls = (data.image_urls || [])
              .map(normalizeImageUrl)
              .filter(Boolean)
            updateLastAssistant(() => ({
              content:   data.response,
              imageUrls: normalizedUrls,
              isLoading: false,
            }))
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

  const clearSession = useCallback(async () => {
    if (threadId) {
      await fetch(`${API_BASE}/session/${threadId}`, { method: 'DELETE' }).catch(() => {})
    }
    setMessages([])
    setThreadId(null)
    setError(null)
  }, [threadId])

  // No longer exposing images/toolCalls separately — everything lives in messages
  return { messages, loading, error, sendMessage, clearSession, threadId }
}