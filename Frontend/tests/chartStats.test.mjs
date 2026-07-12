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
