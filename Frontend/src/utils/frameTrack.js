// Where the scrub track is empty, and which day each stretch of it belongs to,
// as fractions along the track's own length.
//
// The axis already resolves per-stop state (`frameAxis.bucketState`); this
// turns that into something the track can be painted with, and nothing else.
import { utcDayLabel } from './frameAxis.js'

// Semantic runs, not a finished gradient string. A string is testable only by
// matching the thing you just wrote -- the exact shape of the Phase 9 failure
// where `aggregateAnchor`'s test asserted `/block-meaned/` and held a false
// sentence in place. Runs are data: a test can ask which stops they cover and
// get an answer keyed to the fixture's own state rather than to the output.
export function buildTrackMarks(axis) {
  const stops = axis?.stops
  if (!Array.isArray(stops) || stops.length < 2) return []

  // By RUN, not per stop. Measured in the running app the track is 284 px over
  // 49 stops -- 5.9 px each -- so 49 tick marks is a hatched bar carrying no
  // information. The real axis merges its 22 empty stops into 3 runs of 10, 2
  // and 10, which at that density are 59, 12 and 59 px: legible, and each one
  // is a night rather than a list of hours.
  const marks = []
  let runStart = null
  for (let i = 0; i < stops.length; i++) {
    if (isEmpty(stops[i])) {
      if (runStart === null) runStart = i
      continue
    }
    if (runStart !== null) marks.push(markFor(runStart, i - 1, stops.length))
    runStart = null
  }
  if (runStart !== null) marks.push(markFor(runStart, stops.length - 1, stops.length))
  return marks
}

// ONE empty state on the track, both kinds. Measured on the design target:
// `qa-rejected` lands on stops 2 and 26, each the single LEADING stop of a
// ten-stop run -- it is dusk, the last scan of the day attempted and rejected
// before a night that retrieves nothing. At 5.9 px per stop a per-kind split
// renders that as a 5.9 px sliver, which is precisely the density that rules
// out per-stop ticks in the first place.
//
// D10's split is not lost, it is placed: its own wording puts the distinction
// in the readout ("the readout says which blank this is"), which names both
// kinds at the stop the reader is actually on. The track answers "is there
// anything here"; the readout answers "why not".
function isEmpty(stop) {
  // Stop 0 is the period aggregate, not an interval -- it has no state and is
  // never empty. Keying off `kind` rather than off the index says why.
  if (!stop || stop.kind !== 'interval') return false
  return stop.state === 'qa-rejected' || stop.state === 'not-retrieved'
}

// Each stop owns an equal slice of the track centred on where the thumb parks
// for it, so a mark's edges fall halfway to its neighbours rather than on the
// stop centres -- otherwise a run would visually stop short of the last stop
// it covers by half a slice.
function markFor(from, to, total) {
  const last = total - 1
  return {
    startStop: from,
    endStop: to,
    start: clamp((from - 0.5) / last),
    end: clamp((to + 0.5) / last),
  }
}

// One label per calendar day the axis touches, at the stop that opens it.
//
// The track prints exactly two labels today, and on a real multi-day chart they
// read "Period aggregate" and "15 Jun 23:00 UTC" -- `stops[0]` is the
// aggregate, so the day the axis STARTS on appears nowhere on it. That is what
// makes two marked nights indistinguishable from each other: without the days
// between them they are two identical grey bands.
//
// The opening day counts as a boundary even when it is the only one. On a
// single-day chart there is nothing to divide, but the date is still absent
// from the track, and that is the thing being fixed.
export function buildDayBoundaries(axis) {
  const stops = axis?.stops
  if (!Array.isArray(stops) || stops.length < 2) return []

  const boundaries = []
  let previous = null
  for (let i = 0; i < stops.length; i++) {
    const stop = stops[i]
    if (stop.kind !== 'interval') continue
    const label = utcDayLabel(stop.tStart)
    // A stamp that cannot be read names no day. Skipping it leaves the track
    // unlabelled there rather than labelling it with a guess.
    if (!label || label === previous) continue
    boundaries.push({ stop: i, fraction: clamp(i / (stops.length - 1)), label })
    previous = label
  }
  return boundaries
}

// The key to the marking, or null when there is nothing marked.
//
// A hatched track with no key is grey diagonal stripes a reader has to guess
// at, which is this phase's own failure one level along. It states the tally
// and what the hatching means, and stops there: the REASON a given stop is
// empty belongs to the readout, which names it at the stop the reader is on,
// and to methods.md, which tallies both kinds. A third home for that split is
// how two homes start disagreeing.
export function trackLegend(axis) {
  const stops = axis?.stops
  if (!Array.isArray(stops) || stops.length < 2) return null

  const total = stops.length - 1
  const empty = stops.filter(isEmpty).length
  if (!empty) return null

  // A stop is an INTERVAL only where a frame is one cadence bucket. In the
  // coarsened tier each stop averages several, so counting them as intervals
  // understates the axis by the coarsening factor -- 48 three-hour frames are
  // not 48 hours. `methods.md` splits the same two nouns for the same reason,
  // and this reads the axis's own answer rather than deciding again.
  const noun = axis.coarsened ? 'frames' : 'intervals'
  return (
    `${empty} of the ${total} ${noun} hold nothing — hatched on the track. ` +
    'Land on one to see why it is blank.'
  )
}

function clamp(fraction) {
  return Math.min(1, Math.max(0, fraction))
}
