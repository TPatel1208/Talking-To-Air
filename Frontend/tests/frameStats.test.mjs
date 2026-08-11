import assert from 'node:assert/strict'
import test from 'node:test'
import { buildScrubAxis } from '../src/utils/frameAxis.js'
import { computeChartStats } from '../src/utils/chartStats.js'
import { formatFrameQaRate, statsForStop } from '../src/utils/frameStats.js'

const chart = {
  type: 'heatmap',
  units: 'molecules/cm^2',
  // The map's own numbers, computed backend-side on the full-resolution field.
  statistics: { count: 349000, mean: 2.4e15, min: -3.1e14, max: 9.8e17, valid_fraction: 0.938 },
  provenance: {
    masking: {
      qa_status: 'verified', qa_source: 'main_data_quality_flag',
      qa_pass_rate: 0.7412, qa_checked_pixels: 400000, qa_passing_pixels: 296480,
    },
  },
  frames: {
    frames: [
      {
        t_start: '2025-06-14T00:00:00', t_end: '2025-06-14T01:00:00',
        n_granules: 1, valid_fraction: 0.3444, qa_pass_rate: 0.3505,
        statistics: { count: 12000, mean: 1.2e15, min: -2.0e14, max: 9.849e17 },
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
    shape: [4, 2, 3], dtype: 'float32', period_index: 0, cadence: 'hourly',
    tier: 'cadence', buckets_per_frame: 1, cells_per_frame: 6,
    value_range: [-3.14e14, 6.21e15], url: '/chart/c1/frames.f32.gz',
  },
}

const stops = buildScrubAxis(chart).stops

test('parking at the aggregate reproduces today exactly', () => {
  // D11's whole rule rests on this: frame 0 IS the period aggregate, so the
  // Statistics tab at rest must be byte-for-byte what it shows with no
  // scrubber in the product at all.
  const { qaPassRate, scope, empty, state, ...core } = statsForStop(chart, stops[0])

  assert.deepEqual(core, computeChartStats(chart))
  assert.equal(scope, 'Period aggregate')
  assert.equal(empty, false)
})

test('the aggregate keeps the mask\'s own QA rate', () => {
  assert.equal(statsForStop(chart, stops[0]).qaPassRate, 0.7412)
})

test('an interval reports its own numbers, measured before the block mean', () => {
  const stats = statsForStop(chart, stops[1])

  assert.equal(stats.scope, '14 Jun 00:00 UTC')
  assert.equal(stats.mean, 1.2e15)
  assert.equal(stats.max, 9.849e17)
  assert.equal(stats.count, 12000)
  // D5a: valid_fraction is cos-lat weighted over the ANALYZED REGION and
  // computed natively. Recomputing coverage from the 20,000-cell frame grid
  // would report a sparse hour as almost fully covered.
  assert.equal(stats.validPct, 34.44)
  assert.equal(stats.basis, 'analyzed-region')
  assert.equal(stats.qaPassRate, 0.3505)
})

test('an interval nothing survived in reports no maximum, not a maximum of zero', () => {
  // `{count: 0}` alone is what the backend ships when nothing survived --
  // absent statistics, not zeros. A max of 0 would be a measurement nobody made.
  const stats = statsForStop(chart, stops[2])

  assert.equal(stats.empty, true)
  assert.equal(stats.count, 0)
  assert.equal(stats.mean, null)
  assert.equal(stats.max, null)
  assert.equal(stats.min, null)
  assert.equal(stats.validPct, 0)
})

test('the two empty intervals disagree about QA, and the readout keeps them apart', () => {
  const rejected = statsForStop(chart, stops[2])
  const notRetrieved = statsForStop(chart, stops[3])

  assert.equal(rejected.qaPassRate, 0)
  assert.equal(rejected.state, 'qa-rejected')
  assert.equal(notRetrieved.qaPassRate, null)
  assert.equal(notRetrieved.state, 'not-retrieved')
})

test('a QA rate is floored at both ends so rounding cannot manufacture a verdict', () => {
  // One rejected pixel in 15,000 rounds to 100.0%, and reading that as a clean
  // interval is the same class of error the map-level pass rate is already
  // floored against. The frame payload carries a rate and no pixel counts, so
  // the floor keys off the rate itself.
  assert.equal(formatFrameQaRate(0.999993), '99.9%')
  assert.equal(formatFrameQaRate(0.000007), '0.1%')
  assert.equal(formatFrameQaRate(1), '100.0%')
  assert.equal(formatFrameQaRate(0), '0.0%')
  assert.equal(formatFrameQaRate(0.3512), '35.1%')
  assert.equal(formatFrameQaRate(null), null)
})

test('a chart with no numeric values still has none at the aggregate', () => {
  // The Statistics tab's "No numeric values available" branch is the one that
  // shows masking disclosure on an all-masked scene -- exactly when it matters
  // most. Parking at the aggregate must not manufacture a stat card there.
  const blank = { type: 'heatmap', units: 'K', values: [] }

  assert.equal(statsForStop(blank, null), null)
  assert.equal(statsForStop(blank, buildScrubAxis(chart).stops[0]), null)
})
