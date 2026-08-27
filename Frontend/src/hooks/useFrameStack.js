import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../utils/apiFetch.js'
import { decodeFrameStack } from '../utils/frameStack.js'

const API_BASE = '/api'

/**
 * Fetches and decodes a chart's frame blob (T59 D13).
 *
 * The blob is served `Content-Encoding: gzip` with an ETag and
 * `private, immutable, max-age`, so the browser inflates it on the wire and
 * re-opens of the same chart come out of the HTTP cache. There is deliberately
 * no JS decompressor here.
 *
 * Fetches only when the caller asks (`enabled`), because entering scrubber
 * mode is what makes the values worth paying for -- the axis, the coverage,
 * the QA rates and the pooled scale all ride the chart's jsonb row and are on
 * screen before this resolves.
 *
 * 404 is `expired`, not `failed`: the entry was evicted or its pipeline
 * version superseded, and D8 forbids regenerating it. That is a terminal state
 * the UI names, not an error it retries. A transport failure is not terminal
 * and is retried when the user reopens the scrubber.
 *
 * `block` is the payload's own `frames` block -- the layout (`shape`, `dtype`,
 * `period_index`) is read off it rather than off the bytes, so a truncated
 * blob is refused instead of reshaping cleanly into fewer frames and rendering
 * as a plausible map with the tail of the scrub silently absent.
 */
export function useFrameStack(block, enabled) {
  const [result, setResult] = useState({ url: null, loadState: 'idle', stack: null })
  // The request in flight, so a chart switched mid-fetch can never have the
  // previous chart's pixels land on it.
  const requestRef = useRef(0)
  // The url whose fetch reached a terminal answer, so reopening the scrubber
  // does not re-issue a request the HTTP cache would serve anyway.
  const settledRef = useRef(null)
  const url = block?.url || null

  useEffect(() => {
    if (!enabled || !url || settledRef.current === url) return undefined

    const request = ++requestRef.current
    const controller = new AbortController()

    apiFetch(`${API_BASE}${url}`, { signal: controller.signal })
      .then(async (res) => {
        if (res.status === 404) return { loadState: 'expired', stack: null }
        if (!res.ok) return { loadState: 'failed', stack: null }
        const stack = decodeFrameStack(await res.arrayBuffer(), block)
        return stack ? { loadState: 'loaded', stack } : { loadState: 'failed', stack: null }
      })
      .catch(() => ({ loadState: 'failed', stack: null }))
      .then((next) => {
        if (request !== requestRef.current) return
        // 'failed' is a transport blip and stays retryable; 'loaded' and
        // 'expired' are both answers, and D8 means neither changes.
        if (next.loadState !== 'failed') settledRef.current = url
        setResult({ url, ...next })
      })

    return () => controller.abort()
  }, [block, url, enabled])

  // 'loading' is DERIVED from "we have no answer for this url yet", never set
  // synchronously in the effect body -- that cascades a render, and this
  // codebase's react-hooks lint rejects it outright.
  if (!enabled || !url) return { loadState: 'idle', stack: null }
  if (result.url !== url) return { loadState: 'loading', stack: null }
  return { loadState: result.loadState, stack: result.stack }
}
