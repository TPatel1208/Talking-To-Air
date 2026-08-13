import assert from 'node:assert/strict'
import test from 'node:test'
import { aggregateAnchor, resolveFrameDelta } from '../src/utils/frameDelta.js'

const DELTA_BASIS = 'cos(latitude)-weighted sum of |frame-derived period mean - period map| over the same weighted sum of |period map|, at native resolution over cells finite in both'
const GRID_BASIS = 'cos(latitude)-weighted sum of |mean of the stored frame planes - the stored period plane| over the same weighted sum of |the stored period plane|, on the block-mean frame grid the browser downloads, over cells finite in both'

// Phase 8 measured this on `map_2ea3dd7b34cf`, a real regional TEMPO chart at
// k=(5,5): the CADENCE tier, whose temporal relationship is an exact identity
// and which therefore has no `delta` at all, still misses its own period plane
// by 1.876% on the arrays it hands the browser.
const SHIPPED = { headline: 0.018760, max_abs: 2.72e15, basis: GRID_BASIS }

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

test('the cadence tier no longer claims the map above IS the frames\' average', () => {
  // The sentence Phase 8 falsified. "The map above is their average" is a
  // claim about the two arrays on screen, and it is 1.876% wrong on a real
  // regional chart -- the frames are block-meaned to the rendering grid and
  // the map is not, and those two reductions do not commute where coverage is
  // uneven. The equal weighting is still true of the INTERVALS, which is what
  // it was ever a fact about.
  const delta = resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1,
    delta: null, frame_grid_delta: SHIPPED,
  }))

  assert.match(delta.summary, /equally-weighted average at native resolution/)
  assert.match(delta.summary, /misses the aggregate by 1\.9%/)
  assert.match(delta.summary, /2\.720e\+15 molecules\/cm\^2/)
  assert.equal(delta.gridPct, '1.9%')
  assert.equal(delta.gridBasis, GRID_BASIS)
})

test('a tier-one chart whose blocks were seen evenly still says so plainly', () => {
  // Under 0.1% the screen cannot resolve the number it would be printing --
  // an undownsampled chart agrees to float32 storage noise (Phase 3's
  // 0.000002%) -- and "0.0%" reads as a measurement that found nothing rather
  // than as a check that passed.
  const delta = resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null,
    frame_grid_delta: { headline: 2e-8, max_abs: 1.5e8, basis: GRID_BASIS },
  }))

  assert.equal(delta.gridPct, 'under 0.1%')
  assert.equal(delta.severity, 'low')
})

test('a tier-one chart predating the measurement keeps the old quiet note', () => {
  // Nothing measured is not the same as nothing to say, but it is the same as
  // nothing to print. The scoped sentence carries the row on its own.
  const delta = resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null,
  }))

  assert.equal(delta.gridPct, null)
  assert.equal(delta.severity, 'none')
  assert.doesNotMatch(delta.summary, /misses the aggregate/)
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

test('the coarsened tier reports both disagreements and refuses to merge them', () => {
  // Phase 8 §2 on `map_1531e35a0e18`: 3.74% disclosed at native resolution
  // beside arrays that disagreed by 3.28%. The block mean partly CANCELLED the
  // coarsening term there rather than compounding it, so neither number bounds
  // the other -- and the smaller one was the one on screen.
  const delta = resolveFrameDelta(chartWith({
    tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 3,
    delta: { headline: 0.0374, max_abs: 3.127e15, basis: DELTA_BASIS },
    frame_grid_delta: { headline: 0.0328, max_abs: 1.953e15, basis: GRID_BASIS },
  }))

  assert.equal(delta.headlinePct, '3.7%')
  assert.equal(delta.gridPct, '3.3%')
  assert.match(delta.summary, /They differ from it by 3\.7%/)
  assert.match(delta.summary, /misses the aggregate by 3\.3%/)
  assert.match(delta.summary, /neither bounds the other/)
  assert.equal(delta.basis, DELTA_BASIS)
  assert.equal(delta.gridBasis, GRID_BASIS)
})

test('severity follows the larger disagreement, whichever one it is', () => {
  // One escalation rule over everything disclosed, matching the document's.
  // Keying it to the native figure alone leaves the cadence tier -- which has
  // no native figure at all -- unable to raise its voice, and that is the tier
  // where a double-digit block-mean disagreement is most surprising.
  const shipped = resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null,
    frame_grid_delta: { headline: 0.14, max_abs: 9.0e15, basis: GRID_BASIS },
  }))
  const native = resolveFrameDelta(chartWith({
    tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 5,
    delta: { headline: 0.02, max_abs: 1.0e15, basis: DELTA_BASIS },
    frame_grid_delta: { headline: 0.31, max_abs: 4.0e15, basis: GRID_BASIS },
  }))

  assert.equal(shipped.severity, 'high')
  assert.equal(native.severity, 'high')
})

test("stop 0's anchor names the two ways it is not the Map tab", () => {
  // Phase 8 §9: entering scrubber mode roughly halves the apparent intensity
  // of everything, because the pooled ramp is 1.99-2.09x the Map tab's -- and
  // the plane itself is block-meaned to the frame grid while the Map tab's is
  // not. "The same field the Map tab shows" was carrying both silently, at
  // exactly the stop a reader uses to re-orient.
  const anchor = aggregateAnchor(resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1,
    delta: null, frame_grid_delta: SHIPPED,
  })))

  assert.match(anchor, /block-meaned/)
  assert.match(anchor, /pooled/)
  // The number itself belongs to the delta line directly beneath, and putting
  // it here too would be one measurement with two homes on one screen.
  assert.doesNotMatch(anchor, /1\.9%/)
})

test('the anchor still stands when nothing about the frames was measured', () => {
  assert.match(aggregateAnchor(null), /period aggregate/)
})
