import { useState, useCallback, useEffect } from 'react'

const API_BASE = '/api'

export function useConnectors(accessToken) {
  const [connectors, setConnectors] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [notConfigured, setNotConfigured] = useState(false)

  const authHeaders = useCallback((extra = {}) => ({
    ...extra,
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
  }), [accessToken])

  const fetchConnectors = useCallback(async () => {
    if (!accessToken) {
      setConnectors([])
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/connectors`, { headers: authHeaders() })
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
  }, [accessToken, authHeaders])

  useEffect(() => { fetchConnectors() }, [fetchConnectors])

  const setToken = useCallback(async (connectorType, token) => {
    const res = await fetch(`${API_BASE}/connectors/${connectorType}/token`, {
      method: 'PUT',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ token }),
    })
    const data = await res.json().catch(() => null)
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`)
    setConnectors(prev => prev.map(c => (c.connector_type === connectorType ? data : c)))
    return data
  }, [authHeaders])

  const disconnect = useCallback(async (connectorType) => {
    const res = await fetch(`${API_BASE}/connectors/${connectorType}`, {
      method: 'DELETE',
      headers: authHeaders(),
    })
    const data = await res.json().catch(() => null)
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`)
    setConnectors(prev => prev.map(c => (c.connector_type === connectorType ? data : c)))
    return data
  }, [authHeaders])

  return { connectors, loading, error, notConfigured, fetchConnectors, setToken, disconnect }
}
