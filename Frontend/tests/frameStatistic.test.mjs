import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import test from 'node:test'
import { STATISTIC_ORDER, extentOverstatementNote, resolveScrubStop, resolveStatisticChoice, resolveStatisticSource, scrubbableStatistics } from '../src/utils/frameStatistic.js'

const GRID_BASIS = 'cos(latitude)-weighted sum of |mean of the stored frame planes - the stored period plane| over the same weighted sum of |the stored period plane|, on the stored frame grid the browser downloads, over cells finite in both'

// The Phase 14 payload contract, in the shape `test_plot_frames_wiring.py`
// pins it: one shared layout, one block per plane carrying only what differs.
function framesWithPlanes(planes) {
  return {
    frames: [
      { t_start: '2025-06-14T00:00:00', t_end: '2025-06-14T01:00:00', n_granules: 1, valid_fraction: 1, qa_pass_rate: 0.9, statistics: { count: 10, mean: 1e15, min: -1e14, max: 5e15 } },
      { t_start: '2025-06-14T01:00:00', t_end: '2025-06-14T02:00:00', n_granules: 1, valid_fraction: 1, qa_pass_rate: 0.9, statistics: { count: 10, mean: 2e15, min: -1e14, max: 6e15 } },
    ],
    shape: [3, 4, 5],
    dtype: 'float32',
    period_index: 0,
    lats: [40, 40.5, 41, 41.5],
    lons: [-75, -74.5, -74, -73.5, -73],
    cadence: 'hourly',
    tier: 'cadence',
    buckets_per_frame: 1,
    coarsen_k: [2, 2],
    value_range: [-3.14e14, 6.21e15],
    scale_basis: '2nd-98th percentile pooled across every frame and the period mean, at native resolution before the block mean; the scrubber\'s scale only, the Map tab is untouched',
    frame_grid_delta: { headline: 0.018760, max_abs: 2.72e15, basis: GRID_BASIS },
    pipeline_version: 'open-v3',
    _key: 'mean-entry',
    url: '/chart/c1/frames.f32.gz',
    ...(planes ? { planes } : {}),
  }
}

const MAX_PLANE = {
  value_range: [1.0e14, 2.46e16],
  frame_grid_delta: { headline: 0.0, max_abs: 0.0, basis: GRID_BASIS },
  extent_overstatement: { headline: 24.6985, worst_frame: 24.9312, ceiling: 25, basis: 'cos(latitude)-weighted area painted at each block\'s max over the area the native cells holding that max actually cover' },
  _key: 'max-entry',
  url: '/chart/c1/frames.max.f32.gz',
}

const MIN_PLANE = {
  value_range: [-6.49e14, 1.2e15],
  frame_grid_delta: { headline: 0.0, max_abs: 0.0, basis: GRID_BASIS },
  extent_overstatement: null,
  _key: 'min-entry',
  url: '/chart/c1/frames.min.f32.gz',
}

export function chartWithPlanes(planes = { max: MAX_PLANE, min: MIN_PLANE }) {
  return { type: 'heatmap', units: 'molecules/cm^2', frames: framesWithPlanes(planes) }
}

export function chartWithoutPlanes() {
  return { type: 'heatmap', units: 'molecules/cm^2', frames: framesWithPlanes(null) }
}

export { MAX_PLANE, MIN_PLANE, GRID_BASIS }

test('the toggle offers the mean and every plane whose bytes actually landed', () => {
  assert.deepEqual(scrubbableStatistics(chartWithPlanes()), ['mean', 'max', 'min'])
})

test('a plane that failed to store is not offered as a broken option', () => {
  // `store_frame_stack` degrades one statistic at a time, so a block entry
  // with no url is a plane the reader would watch 404. The same rule
  // `_scrubbable_statistics` derives the agent's list under, read off the
  // field the frontend is allowed to hold.
  const stored = { ...MAX_PLANE }
  delete stored.url

  assert.deepEqual(scrubbableStatistics(chartWithPlanes({ max: stored, min: MIN_PLANE })), ['mean', 'min'])
})

test('a chart with no planes key at all offers the mean alone', () => {
  // Phase 13 omits `planes` rather than emitting it empty, so every chart
  // plotted before that phase reads as exactly what it is.
  assert.deepEqual(scrubbableStatistics(chartWithoutPlanes()), ['mean'])
})

test('a chart whose own frame values were never stored offers nothing to toggle', () => {
  const chart = chartWithoutPlanes()
  delete chart.frames.url

  assert.deepEqual(scrubbableStatistics(chart), [])
})

test('the toggle names the same statistics, in the same order, as the backend tells the agent', () => {
  // One vocabulary, two languages -- `SharedDeltaThresholdTests` pointed the
  // other way. `_scrubbable_statistics` orders the agent's list by
  // `PLANE_STATISTICS` "so the agent sees one stable vocabulary", and a
  // researcher reading the answer beside the toggle is looking at both. A
  // comment naming the other file is not a check: Phase 8 measured that
  // exact arrangement drifting with every test in both suites green.
  const path = fileURLToPath(new URL('../../Backend/tta_backend/preprocessing/frame_stack.py', import.meta.url))
  const source = readFileSync(path, 'utf-8')
  const match = source.match(/PLANE_STATISTICS\s*=\s*\(([^)]*)\)/)

  assert.ok(match, 'PLANE_STATISTICS could not be found in frame_stack.py. If it was renamed, re-point this check rather than deleting it — the two lists are still two.')
  const backend = [...match[1].matchAll(/"([a-z_]+)"/g)].map((m) => m[1])

  assert.deepEqual(STATISTIC_ORDER, backend)
})

test('each statistic is fetched from its own url, on the layout every plane shares', () => {
  // Phase 13 decision 2: the per-plane block carries only what DIFFERS. The
  // shape, the dtype and which plane is the aggregate are properties of the
  // chart, and copying them per plane would give a quantity that has one
  // account a second one.
  const chart = chartWithPlanes()

  const mean = resolveStatisticSource(chart, 'mean')
  const max = resolveStatisticSource(chart, 'max')

  assert.equal(mean.url, '/chart/c1/frames.f32.gz')
  assert.equal(max.url, '/chart/c1/frames.max.f32.gz')
  assert.deepEqual(max.shape, mean.shape)
  assert.equal(max.dtype, mean.dtype)
  assert.equal(max.period_index, mean.period_index)
  assert.equal(max.statistic, 'max')
})

test('the source never carries the store key, in any statistic', () => {
  // `_key` addresses the blob store directly and is not the frontend's to
  // hold. It is on the block beside `url` in every plane, so the source is
  // built field by field rather than spread.
  const chart = chartWithPlanes()

  for (const statistic of STATISTIC_ORDER) {
    assert.equal('_key' in resolveStatisticSource(chart, statistic), false, statistic)
  }
})

test('a statistic with no bytes has no source to fetch, rather than the mean’s', () => {
  // Falling back to the mean's url here is the failure the whole phase is
  // written against: a wrong plane renders believably.
  const stored = { ...MAX_PLANE }
  delete stored.url

  assert.equal(resolveStatisticSource(chartWithPlanes({ max: stored, min: MIN_PLANE }), 'max'), null)
  assert.equal(resolveStatisticSource(chartWithPlanes(), 'p99'), null)
})

test('the max mode discloses how much ground a rendered peak claims', () => {
  // Phase 11 G4 asked for this figure "in `methods.md` and beside the max-mode
  // scrubber". `methods.md` is Phase 16's; beside the scrubber is this one's.
  // Block max paints one native cell's peak across all k^2 cells of its block,
  // and G4 measured that as the typical case rather than a tail.
  const note = extentOverstatementNote(chartWithPlanes(), 'max')

  assert.match(note, /24\.7/)
  assert.match(note, /24\.9/)
})

test('the overstatement is this chart’s own measurement, not a constant', () => {
  const gentler = chartWithPlanes({
    max: { ...MAX_PLANE, extent_overstatement: { headline: 4.0000014, worst_frame: 4.31, ceiling: 4, basis: MAX_PLANE.extent_overstatement.basis } },
    min: MIN_PLANE,
  })

  assert.match(extentOverstatementNote(gentler, 'max'), /4\.0/)
  assert.doesNotMatch(extentOverstatementNote(gentler, 'max'), /24\.7/)
})

test('the overstatement is never worded as a bound', () => {
  // Phase 14's finding, which binds this prose: `ceiling` is k^2 and is NOT an
  // upper bound. Both sums in `_overstatement_terms` are cos(latitude)-weighted
  // per cell, so a block whose max sits on its lowest-weighted row exceeds k^2
  // -- measured at 4.0000014 against a ceiling of 4. "Up to k x" and "at most
  // 25x" are both claims this quantity does not support.
  const straddling = chartWithPlanes({
    max: { ...MAX_PLANE, extent_overstatement: { headline: 4.0000014, worst_frame: 4.31, ceiling: 4, basis: MAX_PLANE.extent_overstatement.basis } },
    min: MIN_PLANE,
  })
  const note = extentOverstatementNote(straddling, 'max')

  assert.doesNotMatch(note, /up to/i)
  assert.doesNotMatch(note, /at most/i)
  assert.doesNotMatch(note, /ceiling|no more than|never exceeds/i)
})

test('only the max plane has an overstatement to disclose', () => {
  // The payload carries `extent_overstatement: null` on the min plane, and the
  // mean has no such quantity at all -- a block mean does not paint a peak.
  assert.equal(extentOverstatementNote(chartWithPlanes(), 'min'), null)
  assert.equal(extentOverstatementNote(chartWithPlanes(), 'mean'), null)
  assert.equal(extentOverstatementNote(chartWithoutPlanes(), 'max'), null)
})

test('a max plane that could not be measured says nothing rather than something', () => {
  // `_extent_overstatement` returns None when no block was measurable, and a
  // sentence with a blank in it is worse than no sentence.
  const unmeasured = chartWithPlanes({ max: { ...MAX_PLANE, extent_overstatement: null }, min: MIN_PLANE })

  assert.equal(extentOverstatementNote(unmeasured, 'max'), null)
})

// ── The toggle's own state: what is asked for vs what is on screen ───────────

test('nothing claims to be the new statistic until its bytes have landed', () => {
  // Design tension 3, and Phase 13's statistic-guard argument arriving at the
  // frontend: the mean's pixels must never be on screen under a max label. A
  // wrong plane renders believably, so the label follows the BYTES, not the
  // click.
  const chart = chartWithPlanes()

  const inFlight = resolveStatisticChoice(chart, 'max', 'loading')
  const landed = resolveStatisticChoice(chart, 'max', 'loaded')

  assert.equal(inFlight.selected, 'max')
  assert.equal(inFlight.rendered, 'mean')
  assert.equal(inFlight.pending, true)
  assert.equal(landed.rendered, 'max')
  assert.equal(landed.pending, false)
})

test('a plane that will never arrive leaves the mean on screen and says which failed', () => {
  const chart = chartWithPlanes()

  for (const loadState of ['expired', 'failed', 'idle']) {
    const choice = resolveStatisticChoice(chart, 'max', loadState)
    assert.equal(choice.rendered, 'mean', loadState)
  }
})

test('a statistic that is no longer offered falls back to the mean rather than to nothing', () => {
  // A plane evicted between the payload being written and the chart being
  // reopened. The toggle cannot show a selection it does not offer.
  const gone = { ...MAX_PLANE }
  delete gone.url
  const chart = chartWithPlanes({ max: gone, min: MIN_PLANE })

  const choice = resolveStatisticChoice(chart, 'max', 'loaded')

  assert.equal(choice.selected, 'mean')
  assert.equal(choice.rendered, 'mean')
  assert.deepEqual(choice.options, ['mean', 'min'])
})

test('a chart above the plane ceiling keeps its scrubber and says why it has no toggle', () => {
  // Deliverable 7 / Phase 5 decision 2, one level in: a reader who finds no
  // toggle and no reason is exactly the person the disclosure exists to keep
  // out of that position. `plane_gate` refusing is not the scrubber failing --
  // this chart HAS one, in the mean it always had.
  const chart = chartWithoutPlanes()
  chart.frames.planes_unavailable = {
    reason: 'extent_too_large',
    detail: 'Only the average is browsable for a map this large; the highest and lowest values per interval would need more memory than this service has. Narrow the region to see them.',
  }

  const choice = resolveStatisticChoice(chart, 'mean', 'loaded')

  assert.deepEqual(choice.options, ['mean'])
  assert.equal(choice.offered, true)
  assert.equal(choice.refusal.reason, 'extent_too_large')
  assert.match(choice.refusal.detail, /Narrow the region/)
})

test('a chart that simply has no planes offers no toggle and no explanation', () => {
  // Phase 13 omits `planes` rather than emitting it empty precisely so every
  // chart plotted before it stayed byte-identical. An explanation here would
  // be a disclosure about a capability nothing ever refused.
  const choice = resolveStatisticChoice(chartWithoutPlanes(), 'mean', 'loaded')

  assert.equal(choice.offered, false)
  assert.equal(choice.refusal, null)
})

test('the reader keeps their place across a statistic switch', () => {
  // Design tension 3, decided: the remembered stop SURVIVES and the display
  // parks on the aggregate while the pixels are absent. Finding a peak is the
  // entire reason the max plane exists, so someone who scrubbed to hour 17 and
  // asked for its maximum has done precisely the thing that must not be
  // punished by being sent back to stop 0.
  const stops = [{ kind: 'aggregate', index: 0 }, ...Array.from({ length: 20 }, (_, i) => ({ kind: 'interval', index: i + 1 }))]

  const whileLoading = resolveScrubStop({ stops, index: 17, sliderEnabled: false })
  const onceLanded = resolveScrubStop({ stops, index: 17, sliderEnabled: true })

  assert.equal(whileLoading.kind, 'aggregate')
  assert.equal(onceLanded.index, 17)
})

test('a remembered stop past the end of a shorter axis lands inside it', () => {
  const stops = [{ kind: 'aggregate', index: 0 }, { kind: 'interval', index: 1 }]

  assert.equal(resolveScrubStop({ stops, index: 9, sliderEnabled: true }).index, 1)
  assert.equal(resolveScrubStop({ stops: [], index: 0, sliderEnabled: true }), null)
})
