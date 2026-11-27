import { useEffect, useState } from 'react'
import { apiFetch } from '../utils/apiFetch.js'
import { API_BASE } from '../config.js'
import { provenanceUrl } from '../utils/chartProvenance.js'

const cache = new Map()

// The `/chart/{id}/provenance` body for a chart, or null until it loads (and
// when the chart has no id or the request fails).
export function useChartProvenance(chartId) {
  const [loaded, setLoaded] = useState({ id: null, data: null })

  useEffect(() => {
    if (!chartId || cache.has(chartId)) return undefined
    let cancelled = false
    apiFetch(provenanceUrl(API_BASE, chartId))
      .then(res => (res.ok ? res.json() : null))
      .catch(() => null)
      .then(data => {
        if (data) cache.set(chartId, data)
        if (!cancelled) setLoaded({ id: chartId, data })
      })
    return () => { cancelled = true }
  }, [chartId])

  if (!chartId) return null
  return cache.get(chartId) ?? (loaded.id === chartId ? loaded.data : null)
}
