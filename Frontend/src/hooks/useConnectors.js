import { useState, useCallback, useEffect } from 'react'
import { apiFetch } from '../utils/apiFetch.js'

const API_BASE = '/api'

export function useConnectors() {
  const [connectors, setConnectors] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [notConfigured, setNotConfigured] = useState(false)

  const fetchConnectors = useCallback(async () => {
    setLoading(true)
    try {
      const res = await apiFetch(`${API_BASE}/connectors`)
      if (res.status === 503) {
        setNotConfigured(true)
        setConnectors([])
        setError(null)
        return
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setNotConfigured(false)
      setConnectors(data.connectors || [])
      setError(null)
    } catch (err) {
      setError(err.message || 'Failed to load connectors')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchConnectors() }, [fetchConnectors])

  const setToken = useCallback(async (connectorType, token) => {
    const res = await apiFetch(`${API_BASE}/connectors/${connectorType}/token`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
    const data = await res.json().catch(() => null)
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`)
    setConnectors(prev => prev.map(c => (c.connector_type === connectorType ? data : c)))
    return data
  }, [])

  const disconnect = useCallback(async (connectorType) => {
    const res = await apiFetch(`${API_BASE}/connectors/${connectorType}`, { method: 'DELETE' })
    const data = await res.json().catch(() => null)
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`)
    setConnectors(prev => prev.map(c => (c.connector_type === connectorType ? data : c)))
    return data
  }, [])

  return { connectors, loading, error, notConfigured, fetchConnectors, setToken, disconnect }
}
