import test from 'node:test'
import assert from 'node:assert/strict'

import { computeChartStats } from '../src/utils/chartStats.js'

// Backend heatmap payloads serialize masked pixels (QA-flag drops, fill
// values, out-of-region cells) as null in the 2D `values` grid
// (plot_tools._da_to_heatmap_payload). The "Valid values" stat must count
// those nulls as invalid — not silently drop them first and report 100%.
test('heatmap validPct counts null (masked) cells as invalid', () => {
  const chart = {
    type: 'heatmap',
    units: 'DU',
    lats: [10, 11],
    lons: [20, 21],
    // 2 valid pixels, 2 QA-masked pixels -> 50% valid
    values: [
      [1.0, null],
      [null, 3.0],
    ],
  }
  const stats = computeChartStats(chart)
  assert.ok(stats, 'stats should be computed')
  assert.equal(stats.count, 2)
  assert.equal(stats.validPct, 50)
  assert.equal(stats.mean, 2.0)
})

// The `values` grid is thinned to _MAX_GRID_CELLS before it is serialized, so
// recomputing statistics from it describes the thumbnail, not the analyzed
// region — on a real TEMPO NO2 L3 scene that reported the maximum 13.4x low.
// The backend now ships statistics computed on the full-resolution field.
test('heatmap statistics come from the payload, not the thinned grid', () => {
  const chart = {
    type: 'heatmap',
    units: 'mol/cm2',
    lats: [10, 11],
    lons: [20, 21],
    values: [
      [1.0, 1.0],
      [1.0, null],
    ],
    statistics: {
      count: 3_818_995,
      valid_fraction: 0.167,
      mean: 1.1172e15,
      min: -9.9999e14,
      max: 9.2418e17,
    },
  }
  const stats = computeChartStats(chart)
  assert.equal(stats.max, 9.2418e17, 'peak must survive decimation')
  assert.equal(stats.min, -9.9999e14)
  assert.equal(stats.mean, 1.1172e15)
  assert.equal(stats.count, 3_818_995, 'count is observations, not rendered cells')
  assert.ok(Math.abs(stats.validPct - 16.7) < 1e-9)
  assert.equal(stats.units, 'mol/cm2')
})

// Which extent a number describes is exactly the kind of thing this project
// refuses to leave unstated, and the two cases genuinely differ: a payload
// carrying server-computed statistics describes the analyzed region, while a
// pre-existing artifact can only be recomputed from the grid it shipped.
test('stats declare whether they describe the analyzed region or the rendered grid', () => {
  const grid = {
    type: 'heatmap',
    units: '',
    lats: [10],
    lons: [20, 21],
    values: [[1.0, 2.0]],
  }
  assert.equal(computeChartStats(grid).basis, 'rendered-grid')

  const withStats = {
    ...grid,
    statistics: { count: 40, valid_fraction: 1, mean: 1.5, min: 1, max: 2 },
  }
  assert.equal(computeChartStats(withStats).basis, 'analyzed-region')

  // A timeseries ships every step it plotted — nothing is thinned, so its
  // numbers are exact and must not be labelled as a rendered-grid estimate.
  const series = { type: 'timeseries', units: 'ppb', values: [1.0, null, 3.0] }
  assert.equal(computeChartStats(series).basis, 'series')
})

// The backend stopped emitting `points` — a flattened duplicate of the same
// field — but charts are durable rows in Postgres, so payloads that shipped it
// and predate `statistics` are still out there and still open. For those the
// flattened list is the only value array they carry, and it must keep working.
test('legacy points-only payload still yields stats', () => {
  const chart = {
    type: 'heatmap',
    units: 'mol/cm2',
    points: {
      lats: [10, 10, 11],
      lons: [20, 21, 20],
      values: [1.0, 2.0, 3.0],
    },
  }
  const stats = computeChartStats(chart)
  assert.ok(stats, 'an old persisted chart must still report its own numbers')
  assert.equal(stats.count, 3)
  assert.equal(stats.mean, 2.0)
  assert.equal(stats.min, 1.0)
  assert.equal(stats.max, 3.0)
  assert.equal(stats.basis, 'rendered-grid')
})

// A payload with neither grid, points, nor statistics must return null rather
// than throw — the read path sees whatever shape was persisted, not a contract.
test('payload with no value array at all returns null', () => {
  assert.equal(computeChartStats({ type: 'heatmap', units: 'DU' }), null)
})

test('heatmap_multi validPct uses first panel grid including nulls', () => {
  const chart = {
    type: 'heatmap_multi',
    panels: [
      {
        units: '',
        lats: [10],
        lons: [20, 21, 22, 23],
        values: [[5.0, null, null, null]],
      },
    ],
  }
  const stats = computeChartStats(chart)
  assert.ok(stats, 'stats should be computed')
  assert.equal(stats.validPct, 25)
})

test('timeseries validPct counts null time steps as invalid', () => {
  const chart = {
    type: 'timeseries',
    units: 'ppb',
    values: [1.0, null, 3.0, null],
  }
  const stats = computeChartStats(chart)
  assert.ok(stats, 'stats should be computed')
  assert.equal(stats.validPct, 50)
  assert.equal(stats.count, 2)
})

test('fully valid grid still reports 100%', () => {
  const chart = {
    type: 'heatmap',
    units: '',
    lats: [10],
    lons: [20, 21],
    values: [[1.0, 2.0]],
  }
  const stats = computeChartStats(chart)
  assert.equal(stats.validPct, 100)
})
