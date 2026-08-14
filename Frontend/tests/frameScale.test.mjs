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

// ── T59 Phase 15: each plane's own pooled clip (D9) ───────────────────────────

const planed = {
  ...chart,
  frames: {
    ...chart.frames,
    planes: {
      max: { value_range: [1.0e14, 2.46e16], url: '/chart/c1/frames.max.f32.gz' },
      min: { value_range: [-6.49e14, 1.2e15], url: '/chart/c1/frames.min.f32.gz' },
    },
  },
}

test('the max plane is drawn on its own pooled clip, never the mean’s', () => {
  // D9, one level in. The mean's pooled range tops out at 6.21e15 and the max
  // plane reaches 2.46e16, so drawing the max on the mean's clip saturates it
  // at exactly the stops a reader switched to max to see -- flat colour over
  // every peak, which is the one thing this mode exists to show.
  const max = resolveScrubberScale(planed, true, 'max')

  assert.equal(max.vmin, 1.0e14)
  assert.equal(max.vmax, 2.46e16)
  assert.notEqual(max.vmax, planed.frames.value_range[1])
})

test('the min plane gets its own clip too, and the mean keeps the one it had', () => {
  const min = resolveScrubberScale(planed, true, 'min')
  const mean = resolveScrubberScale(planed, true, 'mean')

  assert.equal(min.vmin, -6.49e14)
  assert.deepEqual([mean.vmin, mean.vmax], planed.frames.value_range)
})

test('the legend caption says which statistic it pooled', () => {
  // Risk 5's mitigation is that the legend's numbers, its caption and the
  // button all change on the same click. This phase adds a third and a fourth
  // ramp to one field, so a caption that names no statistic leaves the reader
  // with four scales and one sentence.
  const max = resolveScrubberScale(planed, true, 'max')
  const min = resolveScrubberScale(planed, true, 'min')

  assert.match(max.legendNote, /max/i)
  assert.match(min.legendNote, /min/i)
  assert.notEqual(max.legendNote, min.legendNote)
  assert.ok(max.legendNote.length < 60)
})

test('the mean’s caption is byte-identical whether or not the chart has planes', () => {
  // D6a decision 5: the mean entry keeps its exact shape and cost. A chart
  // that gained planes must not have its default mode reworded by them.
  assert.equal(
    resolveScrubberScale(planed, true, 'mean').legendNote,
    resolveScrubberScale(chart, true).legendNote,
  )
})

test('a statistic with no pooled range of its own gets no ramp, not the mean’s', () => {
  // Same rule as the source url. Falling back would put the max plane's
  // pixels on the mean's ramp with the legend saying "max" -- a picture that
  // is wrong everywhere and looks entirely plausible.
  const blind = { ...planed, frames: { ...planed.frames, planes: { max: { url: '/x' } } } }

  assert.equal(resolveScrubberScale(blind, true, 'max'), null)
  assert.equal(resolveScrubberScale(planed, true, 'p99'), null)
})
