import { useCallback, useState } from 'react'

const API_BASE = '/api'

export function useDiscovery(accessToken) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [location, setLocation] = useState('')
  const [timeRange, setTimeRange] = useState('')
  // Per-dataset_handle quick-look/coverage results, so multiple cards can
  // hold their own state at once without clobbering each other.
  const [previews, setPreviews] = useState({})
  const [coverages, setCoverages] = useState({})

  const authHeaders = useCallback((extra = {}) => ({
    ...extra,
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
  }), [accessToken])

  // T18: every discovery endpoint failure now arrives as
  // {"error": {"category", "message", "suggestion"}} (api.py's
  // MCPToolError handler) instead of a bare status code — show the human
  // message (and suggestion, when present) instead of "HTTP 422".
  const readErrorMessage = async (res) => {
    try {
      const body = await res.json()
      const message = body?.error?.message
      if (message) return body.error.suggestion ? `${message} ${body.error.suggestion}` : message
    } catch {
      // Body wasn't JSON (or had no error envelope) — fall through.
    }
    return `HTTP ${res.status}`
  }

  const search = useCallback(async () => {
    const trimmed = query.trim()
    if (!trimmed) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/discovery/search`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ query: trimmed }),
      })
      if (!res.ok) throw new Error(await readErrorMessage(res))
      const data = await res.json()
      setResults(data.results || [])
    } catch (err) {
      setError(err.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }, [query, authHeaders])

  const preview = useCallback(async (datasetHandle, layer) => {
    setPreviews(prev => ({ ...prev, [datasetHandle]: { loading: true, error: null } }))
    try {
      const res = await fetch(`${API_BASE}/discovery/dataset/${datasetHandle}/preview`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          location: location.trim() || undefined,
          time_range: timeRange.trim() || undefined,
          layer: layer || undefined,
        }),
      })
      if (!res.ok) throw new Error(await readErrorMessage(res))
      const data = await res.json()
      setPreviews(prev => ({ ...prev, [datasetHandle]: { loading: false, error: null, ...data } }))
    } catch (err) {
      setPreviews(prev => ({ ...prev, [datasetHandle]: { loading: false, error: err.message || 'Quick-look failed' } }))
    }
  }, [location, timeRange, authHeaders])

  const checkCoverage = useCallback(async (datasetHandle) => {
    if (!location.trim() || !timeRange.trim()) {
      setCoverages(prev => ({
        ...prev,
        [datasetHandle]: { loading: false, error: 'Set an area and time window above first.' },
      }))
      return
    }
    setCoverages(prev => ({ ...prev, [datasetHandle]: { loading: true, error: null } }))
    try {
      const res = await fetch(`${API_BASE}/discovery/dataset/${datasetHandle}/coverage`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ location: location.trim(), time_range: timeRange.trim() }),
      })
      if (!res.ok) throw new Error(await readErrorMessage(res))
      const data = await res.json()
      setCoverages(prev => ({ ...prev, [datasetHandle]: { loading: false, error: null, ...data } }))
    } catch (err) {
      setCoverages(prev => ({ ...prev, [datasetHandle]: { loading: false, error: err.message || 'Coverage check failed' } }))
    }
  }, [location, timeRange, authHeaders])

  return {
    query, setQuery,
    location, setLocation,
    timeRange, setTimeRange,
    results, loading, error,
    previews, coverages,
    search, preview, checkCoverage,
  }
}
