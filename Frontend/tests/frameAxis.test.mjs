// Pinned before anything touches Date: the payload's stamps are naive ISO
// ('2025-06-14T01:00:00'), which `new Date()` parses as LOCAL time. A label
// built that way shifts every interval by the viewer's offset, so a reader in
// New York scrubbing TEMPO's 01:00 UTC scan is told they are looking at 21:00
// the previous day. Running the assertions under a non-UTC zone is what makes
// that a failing test rather than a machine-dependent one.
process.env.TZ = 'America/New_York'

import assert from 'node:assert/strict'
import test from 'node:test'
import { buildScrubAxis, resolveFrameState } from '../src/utils/frameAxis.js'

// Shaped from the real jsonb block frame_store._axis_block writes: three
// hourly buckets over a 2x3 frame grid, in the cadence tier. The empty-bucket
// pair is lifted verbatim from the Phase 3 verification's §4 table -- one
// bucket QA rejected everything in, one nothing was retrieved for.
const framesBlock = {
  frames: [
    {
      t_start: '2025-06-14T00:00:00', t_end: '2025-06-14T01:00:00',
      n_granules: 1, valid_fraction: 0.3444, qa_pass_rate: 0.3505,
      statistics: { count: 12000, mean: 1.2e15, min: -3.1e14, max: 9.849e17 },
    },
    {
      t_start: '2025-06-14T01:00:00', t_end: '2025-06-14T02:00:00',
      n_granules: 0, valid_fraction: 0.0, qa_pass_rate: 0.0,
      statistics: { count: 0 },
    },
    {
      t_start: '2025-06-14T02:00:00', t_end: '2025-06-14T03:00:00',
      n_granules: 0, valid_fraction: 0.0, qa_pass_rate: null,
      statistics: { count: 0 },
    },
  ],
  lats: [10, 11], lons: [20, 21, 22],
  shape: [4, 2, 3],
  dtype: 'float32',
  period_index: 0,
  cells_per_frame: 6,
  cadence: 'hourly',
  tier: 'cadence',
  buckets_per_frame: 1,
  coarsen_k: [1, 1],
  delta: null,
  value_range: [-3.14e14, 6.21e15],
  scale_basis: '2nd-98th percentile pooled across every frame and the period mean',
  pipeline_version: 'sha-abc',
  url: '/chart/c1/frames.f32.gz',
}

const chart = { type: 'heatmap', units: 'molecules/cm^2', frames: framesBlock }

test('the aggregate leads the axis and every stop names the plane it draws', () => {
  const axis = buildScrubAxis(chart)

  // 1 + N stops for a [1+N, ny, nx] blob: the period mean plus one per bucket.
  assert.equal(axis.stops.length, 4)
  assert.equal(axis.stops[0].kind, 'aggregate')
  assert.deepEqual(axis.stops.slice(1).map(s => s.kind), ['interval', 'interval', 'interval'])

  // The plane is read off period_index, never assumed. With the aggregate at
  // plane 0 the buckets fill 1..N in order.
  assert.deepEqual(axis.stops.map(s => s.plane), [0, 1, 2, 3])
})

test('the two empty-bucket states stay distinguishable', () => {
  // Phase 3 measured 28 of 54 hourly buckets empty on a real 2.2-day TEMPO
  // span, and D10 turns on these two not collapsing into one blank map:
  // qa_pass_rate 0.0 is "observed, QA rejected all of it"; qa_pass_rate null
  // is "nothing retrieved". A blank map reads as zero, and zero is a
  // measurement.
  const [, observed, rejected, notRetrieved] = buildScrubAxis(chart).stops

  assert.equal(observed.state, 'observed')
  assert.equal(rejected.state, 'qa-rejected')
  assert.equal(notRetrieved.state, 'not-retrieved')
})

test('a null QA rate is carried through as absent, never floored to zero', () => {
  const [, observed, rejected, notRetrieved] = buildScrubAxis(chart).stops

  assert.equal(observed.qaPassRate, 0.3505)
  assert.equal(rejected.qaPassRate, 0)
  assert.equal(notRetrieved.qaPassRate, null)
})

test('interval labels are UTC, and the aggregate says what it is', () => {
  const [aggregate, observed] = buildScrubAxis(chart).stops

  assert.equal(aggregate.label, 'Period aggregate')
  assert.equal(observed.label, '14 Jun 00:00 UTC')
})

test('a daily cadence labels the day, not a spurious midnight', () => {
  const daily = buildScrubAxis({
    ...chart,
    frames: {
      ...framesBlock,
      cadence: 'daily',
      frames: [{ ...framesBlock.frames[0], t_start: '2025-06-14T00:00:00', t_end: '2025-06-15T00:00:00' }],
      shape: [2, 2, 3],
    },
  })

  assert.equal(daily.stops[1].label, '14 Jun 2025')
})

test('a period_index elsewhere in the stack moves the buckets around it', () => {
  // Nothing in the payload promises the aggregate is plane 0 -- the contract
  // is that period_index says which plane it is. A frontend that hardcodes
  // "frames start at 1" draws the wrong hour the day that changes.
  const axis = buildScrubAxis({
    ...chart,
    frames: { ...framesBlock, period_index: 2 },
  })

  assert.equal(axis.stops[0].plane, 2)
  assert.deepEqual(axis.stops.slice(1).map(s => s.plane), [0, 1, 3])
})

// ── Degrading in three distinguishable ways ──────────────────────────────────

test('a payload with no frames block offers no scrubber and says nothing', () => {
  // The normal single-granule case. A missing `frames` key with no
  // `frames_unavailable` beside it is not an error and must not produce a
  // disclosure line about a feature the request never implied.
  const state = resolveFrameState({ type: 'heatmap' }, 'idle')

  assert.equal(state.mode, 'none')
  assert.equal(state.offered, false)
  assert.equal(state.detail, null)
})

test('a disclosed refusal is relayed in the backend\'s own words', () => {
  const state = resolveFrameState({
    type: 'heatmap',
    frames_unavailable: {
      reason: 'extent_too_large',
      detail: 'A frame axis over this region would reduce 17,024,450 cells per interval, above the 4,000,000-cell limit a frame stack is built within.',
    },
  }, 'idle')

  assert.equal(state.mode, 'refused')
  assert.equal(state.offered, false)
  assert.equal(state.reason, 'extent_too_large')
  assert.match(state.detail, /4,000,000-cell limit/)
})

test('frames whose values never landed keep the axis and disable the slider', () => {
  // D8 forbids regeneration, so this is the terminal state, not a retry: the
  // labelled hours are the disclosure, and the slider says it cannot answer.
  const { url, ...noUrl } = framesBlock
  const state = resolveFrameState({ ...chart, frames: noUrl }, 'idle')

  assert.equal(state.mode, 'axis-only')
  assert.equal(state.offered, true)
  assert.equal(state.sliderEnabled, false)
  assert.match(state.detail, /not stored|no longer/i)
})

test('the slider stays disabled and parked at the aggregate until the bytes arrive', () => {
  // Decision 2. An enabled slider showing the aggregate at all 54 stops reads
  // as "nothing happened" to someone scrubbing for an event -- the same
  // false-negative D10 exists to prevent, arriving through the loading state.
  const pending = resolveFrameState(chart, 'loading')

  assert.equal(pending.mode, 'pending')
  assert.equal(pending.offered, true)
  assert.equal(pending.sliderEnabled, false)
})

test('an expired stack is not the same state as one still loading', () => {
  const expired = resolveFrameState(chart, 'expired')
  const pending = resolveFrameState(chart, 'loading')

  assert.equal(expired.mode, 'axis-only')
  assert.equal(expired.sliderEnabled, false)
  assert.notEqual(expired.detail, pending.detail)
  assert.match(expired.detail, /expired|no longer/i)
})

test('a loaded stack is the only state that scrubs', () => {
  const state = resolveFrameState(chart, 'loaded')

  assert.equal(state.mode, 'ready')
  assert.equal(state.sliderEnabled, true)
  assert.equal(state.detail, null)
})

test('the axis carries the block\'s own descriptors, for the control to label itself', () => {
  const axis = buildScrubAxis(chart)

  assert.equal(axis.cadence, 'hourly')
  assert.equal(axis.tier, 'cadence')
  assert.equal(axis.cellsPerFrame, 6)
})
