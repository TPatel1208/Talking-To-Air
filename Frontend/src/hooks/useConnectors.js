import { useState, useCallback, useEffect } from 'react'
import { apiFetch } from '../utils/apiFetch.js'

const API_BASE = '/api'

// 503 means the server has no connector store configured, not a failure.
async function requestConnectors() {
  const res = await apiFetch(`${API_BASE}/connectors`)
  if (res.status === 503) return { notConfigured: true, connectors: [] }
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return { notConfigured: false, connectors: data.connectors || [] }
}

export function useConnectors() {
  const [connectors, setConnectors] = useState([])
  // True from the start: the mount effect below loads straight away.
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [notConfigured, setNotConfigured] = useState(false)

  // State is set only in promise callbacks, so the mount effect can call this
  // (react-hooks/set-state-in-effect flags a direct setState anywhere in a
  // function an effect calls, even one after an await).
  const loadConnectors = useCallback(() => (
    requestConnectors()
      .then(({ notConfigured: off, connectors: next }) => {
        setNotConfigured(off)
        setConnectors(next)
        setError(null)
      })
      .catch(err => setError(err.message || 'Failed to load connectors'))
      .finally(() => setLoading(false))
  ), [])

  const fetchConnectors = useCallback(() => {
    setLoading(true)
    return loadConnectors()
  }, [loadConnectors])

  useEffect(() => { loadConnectors() }, [loadConnectors])

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
