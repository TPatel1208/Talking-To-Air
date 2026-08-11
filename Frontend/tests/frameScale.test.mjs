import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveScrubberScale } from '../src/utils/frameScale.js'

const colormap = { name: 'viridis', lut: [[68, 1, 84, 255], [253, 231, 37, 255]] }

const chart = {
  type: 'heatmap',
  // The Map tab's own percentile clip, which nothing here may disturb.
  vmin: 2.1e14, vmax: 3.3e15,
  colormap,
  scale: { basis: 'percentile', lower_percentile: 2, upper_percentile: 98 },
  frames: {
    frames: [
      // Per-frame extremes deliberately far outside the pooled clip: a union
      // of per-frame ranges is what computeSharedColorScale builds for compare
      // panels, and Phase 3 measured that union at 2.5-3.9x too wide here.
      { t_start: '2025-06-14T00:00:00', t_end: '2025-06-14T01:00:00', n_granules: 1, valid_fraction: 1, qa_pass_rate: 0.9, statistics: { count: 10, mean: 1e15, min: -6.49e14, max: 2.46e16 } },
      { t_start: '2025-06-14T01:00:00', t_end: '2025-06-14T02:00:00', n_granules: 1, valid_fraction: 1, qa_pass_rate: 0.9, statistics: { count: 10, mean: 1e15, min: -1e14, max: 5e15 } },
    ],
    period_index: 0,
    value_range: [-3.14e14, 6.21e15],
    scale_basis: '2nd-98th percentile pooled across every frame and the period mean, at native resolution before the block mean; the scrubber\'s scale only, the Map tab is untouched',
    url: '/chart/c1/frames.f32.gz',
  },
}

test('outside scrubber mode the Map tab keeps its own scale, untouched', () => {
  // D9: deriving the map's scale from the stack would make its colours depend
  // on whether a storage gate happened to fire, with no disclosure surface.
  assert.equal(resolveScrubberScale(chart, false), null)
})

test('entering scrubber mode swaps in the pooled range and says what it is', () => {
  const scale = resolveScrubberScale(chart, true)

  assert.equal(scale.vmin, -3.14e14)
  assert.equal(scale.vmax, 6.21e15)
  assert.equal(scale.colormap, colormap)
  assert.match(scale.basis, /pooled across every frame and the period mean/)
})

test('the pooled range is taken whole, never rebuilt from per-frame extremes', () => {
  // The union of the fixture's per-frame min/max is -6.49e14 .. 2.46e16, the
  // 3.86x-too-wide ramp Phase 3 measured. Landing on it would render every
  // ordinary hour in the bottom third of the colormap.
  const scale = resolveScrubberScale(chart, true)

  assert.notEqual(scale.vmax, 2.46e16)
  assert.ok(scale.vmax < 1e16)
})

test('the aggregate is drawn on the pooled scale too — it is in the pool', () => {
  // The scale is a property of the MODE, not of the stop, so there is one
  // recolour at entry and none mid-scrub. Frame 0 is the period mean, and
  // POOLED_SCALE_BASIS pools it with every frame.
  const atAggregate = resolveScrubberScale(chart, true, 0)
  const atInterval = resolveScrubberScale(chart, true, 2)

  assert.deepEqual(atAggregate, atInterval)
})

test('a stack with nothing valid in it has no pooled scale to offer', () => {
  const empty = { ...chart, frames: { ...chart.frames, value_range: null } }

  assert.equal(resolveScrubberScale(empty, true), null)
})

test('the legend gets a caption it can actually fit, the disclosure gets the whole basis', () => {
  // scale_basis is a ~200-character sentence: correct for a disclosure block,
  // unreadable as an 8px legend caption. Both are carried, because the legend
  // going quiet about which of the two scales is drawn is the legibility
  // hazard Risk 5 accepts and the mode switch is supposed to mitigate.
  const scale = resolveScrubberScale(chart, true)

  assert.ok(scale.legendNote.length < 60)
  assert.match(scale.legendNote, /pooled/)
  assert.ok(scale.basis.length > 100)
})
