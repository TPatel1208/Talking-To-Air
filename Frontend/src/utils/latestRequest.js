/**
 * Keeping only the newest of overlapping requests. Pure, so it is
 * unit-testable under plain `node --test`.
 */

/** Tokens for requests in the order they start; a response whose token is
 * no longer the latest is stale and should be dropped. */
export function createLatest() {
  let current = 0
  return {
    next: () => ++current,
    isLatest: (token) => token === current,
  }
}

/** `fn` run once, with the last arguments, after `ms` without another call.
 * `.cancel()` drops a pending run. */
export function debounce(fn, ms) {
  let timer = null
  const run = (...args) => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      timer = null
      fn(...args)
    }, ms)
  }
  run.cancel = () => {
    clearTimeout(timer)
    timer = null
  }
  return run
}
