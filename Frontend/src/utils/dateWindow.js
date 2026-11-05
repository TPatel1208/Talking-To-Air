/**
 * The discovery pane's time window. It is held as the `start/end` string the
 * discovery endpoints and the Retrieve message already take (a single day is
 * `day/day`), and picked from calendar inputs rather than typed. Pure, so it is
 * unit-testable under plain `node --test`.
 */

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/

export function windowString(start, end) {
  if (!start) return ''
  return `${start}/${end || start}`
}

export function parseWindow(value) {
  const [start = '', end = start] = String(value || '').split('/').map(part => part.trim())
  if (!ISO_DATE.test(start) || !ISO_DATE.test(end)) return { start: '', end: '' }
  return { start, end }
}

export function windowError(start, end) {
  if (start && end && end < start) return 'The end date is before the start date.'
  return null
}

/** The window as CMR's collection `temporal` filter, or null when unset or invalid. */
export function cmrTemporal(value) {
  const { start, end } = parseWindow(value)
  if (!start || windowError(start, end)) return null
  return `${start}T00:00:00Z,${end}T23:59:59Z`
}

/** The window as words for the chat message Retrieve sends. */
export function windowPhrase(value) {
  const { start, end } = parseWindow(value)
  if (!start) return ''
  return start === end ? `on ${start}` : `from ${start} to ${end}`
}
