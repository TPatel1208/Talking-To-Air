import { useState, useCallback, useEffect, useRef } from 'react'
import { apiFetch } from '../utils/apiFetch.js'
import { normalizeRuns, withoutRun } from '../utils/pausedRuns.js'
import { API_BASE } from '../config.js'

// The runs the open conversation has paused on a question or on a retrieval.
// Loaded when a conversation is opened and after each turn ends (`turnsDone`
// changes), and again on the panel's Refresh. Opening a conversation also
// lets the backend mark runs whose retrievals finished as ready.
export function useRuns(threadId, turnsDone) {
  // The listing and the thread it is for, so another thread's runs never show.
  const [listing, setListing] = useState({ thread: null, runs: [] })
  // The failure and the thread it is for, so it clears when the conversation changes.
  const [failure, setFailure] = useState({ thread: null, message: null })
  // The thread the last request was for, so a slow answer for a conversation
  // already left never replaces the one on screen.
  const wantedRef = useRef(threadId)

  useEffect(() => {
    wantedRef.current = threadId
  }, [threadId])

  const loadRuns = useCallback((thread) => {
    if (!thread) return Promise.resolve()
    return apiFetch(`${API_BASE}/runs?thread_id=${encodeURIComponent(thread)}`)
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return normalizeRuns(await res.json())
      })
      .then(next => {
        if (wantedRef.current !== thread) return
        setListing({ thread, runs: next })
        setFailure({ thread, message: null })
      })
      .catch(err => {
        if (wantedRef.current === thread) setFailure({ thread, message: err.message || 'Failed to load paused runs' })
      })
  }, [])

  const fetchRuns = useCallback(() => loadRuns(threadId), [loadRuns, threadId])

  // State is set only in promise callbacks, as useJobs does.
  useEffect(() => { loadRuns(threadId) }, [loadRuns, threadId, turnsDone])

  const cancelRun = useCallback(async (runId) => {
    try {
      const res = await apiFetch(`${API_BASE}/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setListing(prev => ({ ...prev, runs: withoutRun(prev.runs, runId) }))
      setFailure({ thread: wantedRef.current, message: null })
    } catch (err) {
      setFailure({ thread: wantedRef.current, message: err.message || 'Failed to cancel run' })
      // The run may have finished or been cancelled elsewhere.
      loadRuns(wantedRef.current)
    }
  }, [loadRuns])

  const shown = listing.thread === threadId ? listing.runs : []

  const error = failure.thread === threadId ? failure.message : null

  return { runs: shown, error, fetchRuns, cancelRun }
}
