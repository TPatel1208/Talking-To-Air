import { useState, useCallback, useEffect } from 'react'
import { sortJobs, hasProgressingJob } from '../utils/jobCard.js'
import { apiFetch } from '../utils/apiFetch.js'

const API_BASE = '/api'

// While any job is non-terminal the panel re-fetches on this cadence. Its
// only other live update channel is the chat stream's job_progress events —
// once that stream ends (or the user hits Stop) a row would otherwise freeze
// at its last-seen status ("Processing — 0%") until a manual Refresh.
const ACTIVE_JOB_POLL_MS = 15000

export function useJobs() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchJobs = useCallback(async () => {
    setLoading(true)
    try {
      const res = await apiFetch(`${API_BASE}/jobs`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setJobs(data.jobs || [])
      setError(null)
    } catch (err) {
      setError(err.message || 'Failed to load jobs')
    } finally {
      setLoading(false)
    }
  }, [])

  // Populated from the backend on mount so reloading the page never loses
  // running jobs — the panel never relies on chat history to know what's in
  // flight. Stable now that no token is threaded in, so this runs once rather
  // than again on every silent token rotation.
  useEffect(() => { fetchJobs() }, [fetchJobs])

  // Keep in-flight rows live even when no chat stream is feeding
  // job_progress events (stopped request, reloaded page, job started in
  // another session). Stops itself once no job is still progressing — a
  // paused job (needs user action) or a terminal one never keeps it running,
  // so a stuck workspace can't poll (and re-fan-out to the MCP) forever.
  const hasActiveJobs = hasProgressingJob(jobs)
  useEffect(() => {
    if (!hasActiveJobs) return undefined
    const id = setInterval(() => { fetchJobs() }, ACTIVE_JOB_POLL_MS)
    return () => clearInterval(id)
  }, [hasActiveJobs, fetchJobs])

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
      const res = await apiFetch(`${API_BASE}/jobs/${jobHandle}/cancel`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      applyJobProgress(data)
    } catch (err) {
      setError(err.message || 'Failed to cancel job')
    }
  }, [applyJobProgress])

  return { jobs, loading, error, fetchJobs, applyJobProgress, cancelJob }
}
