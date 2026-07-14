import { useState, useCallback, useEffect } from 'react'
import { sortJobs } from '../utils/jobCard.js'

const API_BASE = '/api'

export function useJobs(accessToken) {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const authHeaders = useCallback((extra = {}) => ({
    ...extra,
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
  }), [accessToken])

  const fetchJobs = useCallback(async () => {
    if (!accessToken) {
      setJobs([])
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/jobs`, { headers: authHeaders() })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setJobs(data.jobs || [])
      setError(null)
    } catch (err) {
      setError(err.message || 'Failed to load jobs')
    } finally {
      setLoading(false)
    }
  }, [accessToken, authHeaders])

  // Populated from the backend on mount (and whenever the access token
  // changes) so reloading the page never loses running jobs — the panel
  // never relies on chat history to know what's in flight.
  useEffect(() => { fetchJobs() }, [fetchJobs])

  const applyJobProgress = useCallback((data) => {
    if (!data || !data.job_handle) return
    setJobs(prev => {
      const idx = prev.findIndex(job => job.job_handle === data.job_handle)
      const next = idx === -1 ? [...prev, data] : prev.map(job => (job.job_handle === data.job_handle ? { ...job, ...data } : job))
      return sortJobs(next)
    })
  }, [])

  const cancelJob = useCallback(async (jobHandle) => {
    try {
      const res = await fetch(`${API_BASE}/jobs/${jobHandle}/cancel`, {
        method: 'POST',
        headers: authHeaders(),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      applyJobProgress(data)
    } catch (err) {
      setError(err.message || 'Failed to cancel job')
    }
  }, [authHeaders, applyJobProgress])

  return { jobs, loading, error, fetchJobs, applyJobProgress, cancelJob }
}
