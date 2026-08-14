process.env.TZ = 'America/New_York'

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { buildScrubAxis } from '../src/utils/frameAxis.js'
import { buildDayBoundaries, buildTrackMarks, trackLegend } from '../src/utils/frameTrack.js'

// The fixtures are the real distributions, measured off the live payloads
// rather than invented, because a marking layer is only as good as the shapes
// it was checked against.
//
//   'O' observed        n_granules > 0, statistics.count > 0
//   'Q' qa-rejected     nothing survived, but QA ran   -> qa_pass_rate 0
//   'N' not-retrieved   nothing was retrieved at all   -> qa_pass_rate null
//
// The two empty kinds are exactly `frameAxis.bucketState`'s split, so building
// the axis through `buildScrubAxis` rather than hand-shaping stops keeps this
// honest: if the axis ever stops resolving a state, these fail too.
function axisFrom(pattern, { cadence = 'hourly', tier = 'cadence', bucketsPerFrame = 1 } = {}) {
  const frames = [...pattern].map((mark, i) => {
    const hour = String(i % 24).padStart(2, '0')
    const day = 14 + Math.floor(i / 24)
    const t_start = `2025-06-${day}T${hour}:00:00`
    const t_end = `2025-06-${day}T${String((i + 1) % 24).padStart(2, '0')}:00:00`
    if (mark === 'O') {
      return {
        t_start, t_end, n_granules: 1, valid_fraction: 0.44, qa_pass_rate: 0.71,
        statistics: { count: 6091, mean: 1.2e15 },
      }
    }
    return {
      t_start, t_end, n_granules: 0, valid_fraction: 0.0,
      // The whole difference between the two empty kinds: QA ran and rejected
      // everything, versus QA never ran because nothing arrived.
      qa_pass_rate: mark === 'Q' ? 0.0 : null,
      statistics: { count: 0 },
    }
  })

  return buildScrubAxis({
    type: 'heatmap',
    units: 'molecules/cm^2',
    frames: {
      frames, period_index: 0, cadence, tier,
      buckets_per_frame: bucketsPerFrame, coarsen_k: [5, 5],
      cells_per_frame: 14124, url: '/chart/c1/frames.f32.gz',
    },
  })
}

// `map_f06a59b99afb` -- 13 hourly stops, every one of them holding a granule.
// The control case, and the reason it is the first test: a marking layer that
// invents a mark on a complete chart is worse than no marking at all, because
// it reports a gap in data that has none.
const POPULATED_13 = 'O'.repeat(13)

// `map_2ea3dd7b34cf` -- the design target, read off the live payload: 48 hourly
// stops, 22 empty, 2 of them QA-rejected and 20 with nothing retrieved. The
// two QA-rejected stops are not scattered; each is the single LEADING stop of a
// ten-stop run, which is dusk -- the last scan of the day attempted and
// rejected, then a night that retrieves nothing.
const TEMPO_48 = 'OQ' + 'N'.repeat(9) + 'O'.repeat(7) + 'NN' + 'O'.repeat(5)
  + 'Q' + 'N'.repeat(9) + 'O'.repeat(13)

// A reference written differently from the implementation, on purpose: one that
// shares its arithmetic asserts only that a thing agrees with itself. This
// walks the axis the util was handed and reports which stop indices are empty,
// so every expectation below is keyed to the fixture's own resolved state
// rather than to output that happened to be written down.
function emptyStopsOf(axis) {
  return axis.stops
    .map((stop, index) => [index, stop])
    .filter(([, stop]) => stop.kind === 'interval'
      && (stop.state === 'qa-rejected' || stop.state === 'not-retrieved'))
    .map(([index]) => index)
}

function coveredBy(marks) {
  const covered = []
  for (const mark of marks) {
    for (let stop = mark.startStop; stop <= mark.endStop; stop++) covered.push(stop)
  }
  return covered
}

test('a fully populated axis is marked nowhere', () => {
  const marks = buildTrackMarks(axisFrom(POPULATED_13))

  assert.deepEqual(marks, [])
})

test('consecutive empty stops become one mark, and separated ones do not', () => {
  // Per-stop ticks do not survive the density -- 5.9 px per stop, measured in
  // the running app -- so the marks have to be by run. This is the assertion
  // that forces that: the real axis has 22 empty stops in 3 runs, and 22 marks
  // would be a hatched bar with no information in it.
  const axis = axisFrom(TEMPO_48)
  const marks = buildTrackMarks(axis)

  // Runs derived from the axis, not transcribed: count the empty stops that
  // are not preceded by another empty stop.
  const empties = emptyStopsOf(axis)
  const runCount = empties.filter((stop) => !empties.includes(stop - 1)).length

  assert.equal(marks.length, runCount)
  assert.ok(marks.length < empties.length,
    'the marks did not merge -- one per empty stop is the density this exists to avoid')
  assert.deepEqual(coveredBy(marks), empties)
})

test('the period aggregate is never swallowed by a run that starts beside it', () => {
  // Stop 0 is the period aggregate, not an interval -- it is the whole span
  // averaged, so it is never "empty" however empty its neighbour is. A run
  // merger that walks backwards from the first empty interval, or that keys
  // off state alone rather than off `kind`, takes stop 0 with it and paints
  // the one stop that always has pixels as a gap.
  const axis = axisFrom('NN' + 'O'.repeat(11))
  const marks = buildTrackMarks(axis)

  assert.equal(axis.stops[0].kind, 'aggregate')
  assert.ok(!coveredBy(marks).includes(0), 'the aggregate was marked empty')
  assert.equal(marks[0].startStop, 1)
  assert.ok(marks[0].start > 0,
    'the mark reaches the start of the track, so it covers the aggregate visually '
    + 'even though it does not cover it numerically')
})

test('a run spans both kinds of empty rather than splitting on the reason', () => {
  // The decision this pins, and the measurement behind it. On the design target
  // the QA-rejected stops are 2 and 26 -- each the single LEADING stop of a
  // ten-stop run, because dusk is one scan attempted and rejected followed by a
  // night that retrieves nothing. Splitting the run on the reason would render
  // that reason as a 5.9 px sliver, which is the density this phase exists to
  // escape. The readout still names both kinds at the stop the reader is on.
  const axis = axisFrom(TEMPO_48)
  const marks = buildTrackMarks(axis)

  const run = marks.find((mark) => mark.startStop <= 2 && mark.endStop >= 2)
  const kinds = new Set()
  for (let stop = run.startStop; stop <= run.endStop; stop++) kinds.add(axis.stops[stop].state)

  // Vacuous unless the fixture really does hold both kinds inside one run.
  assert.deepEqual([...kinds].sort(), ['not-retrieved', 'qa-rejected'])
  assert.equal(run.startStop, 2)
  assert.equal(run.endStop, 11)
})

test('the marks stay inside the track, in order, without overlapping', () => {
  // The component turns these straight into CSS offsets, so a fraction outside
  // [0,1] paints outside the control and a pair out of order paints backwards.
  const marks = buildTrackMarks(axisFrom(TEMPO_48))

  let previousEnd = -1
  for (const mark of marks) {
    assert.ok(mark.start >= 0 && mark.end <= 1, `mark escapes the track: ${JSON.stringify(mark)}`)
    assert.ok(mark.end > mark.start, `mark has no width: ${JSON.stringify(mark)}`)
    assert.ok(mark.start > previousEnd, 'marks overlap or run backwards')
    previousEnd = mark.end
  }
})

test('a run against either end of the track reaches that end exactly', () => {
  // Half a slice past the last stop is off the end of the track. Clamped, or
  // the final night paints past the control and the axis looks longer than the
  // span it covers.
  const axis = axisFrom('O'.repeat(11) + 'NN')
  const marks = buildTrackMarks(axis)

  const last = marks[marks.length - 1]
  assert.equal(last.endStop, axis.stops.length - 1)
  assert.equal(last.end, 1)
})

test('a multi-day axis says which day each stretch of it belongs to', () => {
  // The track prints two labels, and on a real chart they are "Period
  // aggregate" and "15 Jun 23:00 UTC" -- because `stops[0]` IS the aggregate,
  // so the axis never prints the day it STARTS on anywhere. Marking the nights
  // without naming the days leaves a reader with two identical grey bands and
  // no way to tell they are different nights.
  const axis = axisFrom(TEMPO_48)
  const boundaries = buildDayBoundaries(axis)

  // Independent of the implementation's date handling: the payload's stamps are
  // naive ISO, so the calendar day is the first ten characters, no parsing and
  // nothing a viewer's timezone can move.
  const firstStopOfEachDay = []
  const seen = new Set()
  axis.stops.forEach((stop, index) => {
    if (stop.kind !== 'interval') return
    const day = stop.tStart.slice(0, 10)
    if (seen.has(day)) return
    seen.add(day)
    firstStopOfEachDay.push(index)
  })

  assert.ok(firstStopOfEachDay.length > 1, 'fixture does not span a day boundary')
  assert.deepEqual(boundaries.map((b) => b.stop), firstStopOfEachDay)
  assert.deepEqual(boundaries.map((b) => b.label), ['14 Jun', '15 Jun'])
  for (const boundary of boundaries) {
    assert.ok(boundary.fraction >= 0 && boundary.fraction <= 1)
  }
})

test('a single-day axis still names its day once', () => {
  // `map_f06a59b99afb` is 13 hourly stops inside one day. There is no boundary
  // to mark, but the reader still cannot see the date anywhere on the track, so
  // the opening label is the whole value here.
  const boundaries = buildDayBoundaries(axisFrom(POPULATED_13))

  assert.equal(boundaries.length, 1)
  assert.equal(boundaries[0].stop, 1)
  assert.equal(boundaries[0].label, '14 Jun')
})

test('an axis whose stamps are unusable offers no day labels rather than bad ones', () => {
  assert.deepEqual(buildDayBoundaries(null), [])
  assert.deepEqual(buildDayBoundaries({ stops: [] }), [])
})

test('a marked track says what the marking means, in the axis\'s own numbers', () => {
  // Hatching a track without a key leaves a reader looking at grey diagonal
  // stripes with nothing anywhere saying they mean "no data" -- which is the
  // same failure this phase exists to fix, one level along. The sentence lives
  // here rather than in the JSX because with no jsdom a sentence written in a
  // component is a sentence no test can read, which is how "the same field the
  // Map tab shows" survived two phases carrying an unstated block mean.
  const axis = axisFrom(TEMPO_48)
  const legend = trackLegend(axis)

  const empties = emptyStopsOf(axis).length
  const intervals = axis.stops.length - 1
  assert.ok(empties > 0 && intervals > empties, 'fixture is not a partly-empty axis')

  // Keyed to the fixture's own state, never to the sentence that happens to be
  // written: `aggregateAnchor`'s test asserted the literal output and held a
  // false claim in place for a whole phase.
  assert.match(legend, new RegExp(`\\b${empties}\\b`))
  assert.match(legend, new RegExp(`\\b${intervals}\\b`))
})

test('a chart with no gaps is given no key to a marking it does not have', () => {
  assert.equal(trackLegend(axisFrom(POPULATED_13)), null)
})

test('a coarsened track counts frames, because its stops are not intervals', () => {
  // Found live on `map_1531e35a0e18` before it shipped: the legend read "13 of
  // the 48 intervals hold nothing" on a chart whose every stop averages THREE
  // hourly intervals. There are 144 intervals there and 48 frames, and the
  // backend keeps the two nouns apart on purpose -- methods.md says "Empty
  // frames" in this tier and "Empty intervals" in the other, so that three
  // empty 3-hour frames are never reported as three empty hours.
  const coarsened = axisFrom('OQN' + 'O'.repeat(9), { tier: 'coarsened', bucketsPerFrame: 3 })
  const legend = trackLegend(coarsened)

  assert.match(legend, /\bframes\b/)
  assert.doesNotMatch(legend, /\bintervals\b/)
})

test('a cadence track still counts intervals, because there a frame IS one', () => {
  assert.match(trackLegend(axisFrom(TEMPO_48)), /\bintervals\b/)
})

test('the noun follows the same two witnesses the stop labels do', () => {
  // `buildScrubAxis` decides "coarsened" on tier OR buckets_per_frame > 1,
  // deliberately, because deciding it on one field surviving understates a
  // frame's width by the coarsening factor. The legend has to ask the same
  // question the same way or a payload missing `tier` gets one noun on the
  // labels and the other in the key beneath them.
  const noTier = axisFrom('ON' + 'O'.repeat(9), { tier: null, bucketsPerFrame: 3 })

  assert.match(trackLegend(noTier), /\bframes\b/)
})

test('the legend does not restate the reason a stop is empty', () => {
  // The reason belongs to the readout, which names it at the stop the reader is
  // actually on, and to methods.md, which tallies it. A third home for the same
  // split is how two homes start disagreeing -- and the track deliberately
  // carries ONE empty state, so a legend naming two would describe a marking
  // that is not on screen.
  const legend = trackLegend(axisFrom(TEMPO_48))

  assert.doesNotMatch(legend, /QA|quality|retrieved/i)
})

// The only reach this repo has into a component. `MapScrubber` called
// `aggregateAnchor()` with NO argument for a whole build and every util test
// stayed green, because a util test hands itself the argument the component
// forgot. Same idiom as SharedDeltaThresholdTests and
// FinishedRowStatusContractTests: read the source, assert the seam.
const SCRUBBER = readFileSync(
  new URL('../src/components/MapScrubber.jsx', import.meta.url), 'utf-8',
)

test('MapScrubber hands both track utils the axis, not nothing', () => {
  // Two links, because the axis reaches the utils through a prop: the track has
  // to be GIVEN `state.axis`, and it has to pass that on. Breaking either one
  // leaves both utils returning [] forever -- an unmarked track that looks
  // exactly like a chart with no gaps, which is the failure mode with no
  // symptom.
  assert.match(
    SCRUBBER, /<ScrubTrack[^>]*axis=\{state\.axis\}/s,
    'ScrubTrack is not given `state.axis`, so the track has no axis to mark.',
  )
  assert.match(
    SCRUBBER, /buildTrackMarks\(\s*axis\s*\)/,
    'buildTrackMarks is not passed the axis, so the track is marked off nothing.',
  )
  assert.match(
    SCRUBBER, /buildDayBoundaries\(\s*axis\s*\)/,
    'buildDayBoundaries is not passed the axis.',
  )
  assert.match(
    SCRUBBER, /trackLegend\(\s*axis\s*\)/,
    'trackLegend is not passed the axis, so the hatching has no key.',
  )
})

test('MapScrubber renders the legend it computed', () => {
  // Computing a sentence and not rendering it is the same as not having it, and
  // it is invisible to the util test that just checked the sentence is right.
  assert.match(SCRUBBER, /\{legend\}/,
    'the legend is computed but never rendered, so the track is hatched with '
    + 'nothing on screen saying what the hatching means.')
})

test('MapScrubber positions the marks from the fractions it was given', () => {
  // A marks layer that renders without reading `start`/`end` is a layer that
  // draws the same thing everywhere -- correct util, wrong picture, and no test
  // in this repo renders a component to notice.
  assert.match(SCRUBBER, /mark\.start/, 'the marks layer ignores each mark\'s start fraction')
  assert.match(SCRUBBER, /mark\.end/, 'the marks layer ignores each mark\'s end fraction')
})

test('the slider still lets its own track show the marks behind it', () => {
  // The failure this catches is silent and total: the marks layer sits BEHIND
  // the input, so if the input keeps its OS-painted opaque track it covers
  // every mark, and the whole feature renders as nothing while all nine util
  // tests above stay green. The class is what makes the native track
  // transparent, and it must be on the input and defined in the stylesheet.
  assert.match(SCRUBBER, /className="frame-scrub"/,
    'the range input is not marked `frame-scrub`, so its native track stays '
    + 'opaque and paints over the marks layer behind it.')

  const css = readFileSync(new URL('../src/index.css', import.meta.url), 'utf-8')
  assert.match(css, /\.frame-scrub::-webkit-slider-runnable-track/,
    'no webkit track rule for .frame-scrub')
  assert.match(css, /\.frame-scrub::-moz-range-track/,
    'no firefox track rule for .frame-scrub')
})
