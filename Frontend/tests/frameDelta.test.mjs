import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveFrameDelta } from '../src/utils/frameDelta.js'

const DELTA_BASIS = 'cos(latitude)-weighted sum of |frame-derived period mean - period map| over the same weighted sum of |period map|, at native resolution over cells finite in both'

function chartWith(frames) {
  return { type: 'heatmap', units: 'molecules/cm^2', frames: { period_index: 0, frames: [], ...frames } }
}

test('the cadence tier states the relationship is identity, and measures nothing', () => {
  // D14 tier one: the frames ARE the product's cadence buckets and the map is
  // derived from them. Saying so is different from measuring a delta and
  // finding nothing -- there is no number here to print.
  const delta = resolveFrameDelta(chartWith({ tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null }))

  assert.equal(delta.kind, 'exact')
  assert.equal(delta.headlinePct, null)
  assert.match(delta.summary, /hourly/)
})

test('a coarsened stack shows its measured disagreement, not an offer to see it', () => {
  // Phase 3 measured 22.27% on a real TEMPO bundle -- not the sub-1% D16's own
  // illustration implies. Risk 4 is this becoming a badge nobody reads.
  const delta = resolveFrameDelta(chartWith({
    tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 5,
    delta: { headline: 0.2227, max_abs: 9.58e16, basis: DELTA_BASIS },
  }))

  assert.equal(delta.kind, 'approximate')
  assert.equal(delta.headlinePct, '22.3%')
  assert.match(delta.summary, /5 hourly intervals/)
  assert.equal(delta.basis, DELTA_BASIS)
})

test('a double-digit delta outranks a single-digit one', () => {
  // Both are real measurements from Phase 3's two bundles. 5.4% is a caveat;
  // 22.3% means the frames are a materially different aggregation from the
  // map, and the two must not render identically.
  const big = resolveFrameDelta(chartWith({
    tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 5,
    delta: { headline: 0.2227, max_abs: 9.58e16, basis: DELTA_BASIS },
  }))
  const small = resolveFrameDelta(chartWith({
    tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 4,
    delta: { headline: 0.0543, max_abs: 3.39e15, basis: DELTA_BASIS },
  }))

  assert.equal(big.severity, 'high')
  assert.equal(small.severity, 'moderate')
})

test('the worst pixel is reported in the field\'s own units', () => {
  const delta = resolveFrameDelta(chartWith({
    tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 5,
    delta: { headline: 0.2227, max_abs: 9.58e16, basis: DELTA_BASIS },
  }))

  assert.match(delta.maxAbs, /9\.580e\+16 molecules\/cm\^2/)
})

test('a delta that could not be computed says so instead of printing 0%', () => {
  const delta = resolveFrameDelta(chartWith({
    tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 5,
    delta: { headline: null, max_abs: null, basis: DELTA_BASIS },
  }))

  assert.equal(delta.kind, 'approximate')
  assert.equal(delta.headlinePct, null)
  assert.equal(delta.severity, 'unknown')
  assert.match(delta.summary, /5 hourly intervals/)
})

test('a chart with no frames has no delta to resolve', () => {
  assert.equal(resolveFrameDelta({ type: 'heatmap' }), null)
})
