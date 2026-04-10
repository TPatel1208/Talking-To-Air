import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

export function useChat() {
  const [messages, setMessages]     = useState([])
  const [images, setImages]         = useState([])
  const [toolCalls, setToolCalls]   = useState([])
  const [threadId, setThreadId]     = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)

  const sendMessage = useCallback(async (text) => {
    if (!text.trim()) return

    // Add user message immediately
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)
    setError(null)

    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        message:   text,
        thread_id: threadId,
      })

      const { thread_id, response, image_urls, tool_calls } = res.data

      // Persist thread for conversation memory
      setThreadId(thread_id)

      // Add assistant response
      setMessages(prev => [...prev, { role: 'assistant', content: response }])

      // Update dashboard
      if (image_urls?.length)  setImages(image_urls)
      if (tool_calls?.length)  setToolCalls(tool_calls)

    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Request failed'
      setError(msg)
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${msg}`, isError: true }])
    } finally {
      setLoading(false)
    }
  }, [threadId])

  const clearSession = useCallback(async () => {
    if (threadId) {
      await axios.delete(`${API_BASE}/session/${threadId}`).catch(() => {})
    }
    setMessages([])
    setImages([])
    setToolCalls([])
    setThreadId(null)
    setError(null)
  }, [threadId])

  return { messages, images, toolCalls, loading, error, sendMessage, clearSession, threadId }
}