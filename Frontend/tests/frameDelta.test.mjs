import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { aggregateAnchor, resolveFrameDelta } from '../src/utils/frameDelta.js'

const DELTA_BASIS = 'cos(latitude)-weighted sum of |frame-derived period mean - period map| over the same weighted sum of |period map|, at native resolution over cells finite in both'
const GRID_BASIS = 'cos(latitude)-weighted sum of |mean of the stored frame planes - the stored period plane| over the same weighted sum of |the stored period plane|, on the stored frame grid the browser downloads, over cells finite in both'

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

test('an undownsampled chart is never told its frames were block-meaned', () => {
  // Found on live data 2026-08-13, in a sentence written by this phase. A New
  // Jersey retrieval is 10,624 native cells -- under the 20,000-cell ceiling --
  // so coarsen_k is (1,1) and there is NO block mean. The summary said "they
  // are block-meaned to the frame grid and the map is not", which describes a
  // mechanism this figure does not have. Small regions are most regions.
  const delta = resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null,
    coarsen_k: [1, 1],
    frame_grid_delta: { headline: 2.2915e-8, max_abs: 6.6076e8, basis: GRID_BASIS },
  }))

  assert.doesNotMatch(delta.summary, /block-meaned/)
  assert.match(delta.summary, /these frames are not downsampled/)
  assert.match(delta.summary, /misses the aggregate by under 0\.1%/)
})

test('a downsampled chart still names the block mean, because it has one', () => {
  const delta = resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null,
    coarsen_k: [5, 5], frame_grid_delta: SHIPPED,
  }))

  assert.match(delta.summary, /block-meaned to the frame grid and the map is not/)
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
  // the plane itself is on a different grid from the one the Map tab draws.
  // "The same field the Map tab shows" was carrying both silently, at exactly
  // the stop a reader uses to re-orient.
  const anchor = aggregateAnchor(resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1,
    delta: null, coarsen_k: [5, 5], frame_grid_delta: SHIPPED,
  })))

  assert.match(anchor, /block-meaned/)
  assert.match(anchor, /pooled/)
  // The Map tab does not draw its native field either -- `_downsample_grid`
  // strides it to 8,000 cells. Measured live on `map_f06a59b99afb`: the Map
  // tab drew 64x42 = 2,688 cells beside a 128x83 = 10,624-cell frame plane.
  // A sentence naming only the frames' reduction implies the other array had
  // none.
  assert.match(anchor, /strides/)
  // The number itself belongs to the delta line directly beneath, and putting
  // it here too would be one measurement with two homes on one screen.
  assert.doesNotMatch(anchor, /1\.9%/)
})

test('an undownsampled chart\'s anchor does not invent a block mean either', () => {
  // The SAME defect as `shippedSentence` had, in the sentence directly above
  // it -- caught only by rendering both at once on a k=(1,1) chart, where the
  // screen read "block-meaned to the rendering resolution" one line above
  // "these frames are not downsampled". Two adjacent sentences contradicting
  // each other about the same array.
  const anchor = aggregateAnchor(resolveFrameDelta(chartWith({
    tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null,
    coarsen_k: [1, 1],
    frame_grid_delta: { headline: 2.2915e-8, max_abs: 6.6076e8, basis: GRID_BASIS },
  })))

  assert.doesNotMatch(anchor, /block-meaned/)
  assert.match(anchor, /native resolution/)
  // Still not the Map tab's array: at k=(1,1) the frame plane is the FINER of
  // the two, which the sentence must not leave a reader guessing about.
  assert.match(anchor, /strides/)
  assert.match(anchor, /pooled/)
})

test('the anchor still stands when nothing about the frames was measured', () => {
  assert.match(aggregateAnchor(null), /period aggregate/)
})

test('MapScrubber actually hands the anchor its argument', () => {
  // The bug this exists for shipped to a live page: `aggregateAnchor()` was
  // called with no argument, so it always took the "unknown reads as
  // downsampled" branch and told a k=(1,1) chart its frames were block-meaned
  // -- directly above a line saying they were not.
  //
  // A util test cannot see this, because it hands the argument over itself,
  // and there is no jsdom here to render the component. So the source is read
  // instead, the same way test_jobs_service and SharedDeltaThresholdTests read
  // JS constants out of their files: a seam no test can reach is a seam that
  // gets one this way or not at all.
  const source = readFileSync(
    new URL('../src/components/MapScrubber.jsx', import.meta.url), 'utf-8',
  )

  assert.match(
    source, /aggregateAnchor\(\s*delta\s*\)/,
    'MapScrubber calls aggregateAnchor without passing `delta`, so the anchor '
    + 'cannot tell whether this chart was downsampled and will assume it was.',
  )
  assert.match(
    source, /<StopReadout[^>]*delta=\{delta\}/,
    'StopReadout is not given `delta`, so the anchor inside it has nothing to '
    + 'read.',
  )
})

// ── T59 Phase 15: the sentences that stop being true when the toggle flips ────

// A selection plane's own agreement figure, as Phase 11 G5 measured it and
// Phase 14 shipped it: bit-exact on both identities, both bundles.
const EXACT = { headline: 0.0, max_abs: 0.0, basis: GRID_BASIS }

function planedChart(frames, planes) {
  const chart = chartWith(frames)
  chart.frames.planes = planes
  return chart
}

test('a selection plane never prints "under 0.1%" for an identity that holds exactly', () => {
  // D6a: "Printing 'under 0.1 %' here would be exactly the failure
  // `_DELTA_FLOOR` exists to prevent." A measurement that found nothing and an
  // identity that holds exactly are different claims, and this tier's is the
  // second.
  const delta = resolveFrameDelta(planedChart(
    { tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null, frame_grid_delta: SHIPPED },
    { max: { frame_grid_delta: EXACT, url: '/chart/c1/frames.max.f32.gz' } },
  ), 'max')

  assert.doesNotMatch(delta.summary, /under 0\.1%/)
  assert.doesNotMatch(delta.summary, /0\.0%/)
  assert.equal(delta.gridPct, null)
})

test('a selection plane states the identity instead of measuring one', () => {
  // D6a decision 8 asks for one sentence stating the identity. `max` selects
  // a value rather than combining values, so it is associative and invariant
  // to weighting -- there is no number here to print.
  const delta = resolveFrameDelta(planedChart(
    { tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null, frame_grid_delta: SHIPPED },
    { max: { frame_grid_delta: EXACT, url: '/chart/c1/frames.max.f32.gz' } },
  ), 'max')

  assert.match(delta.summary, /exactly/)
  assert.equal(delta.severity, 'none')
})

test('a selection plane does not describe the frames as being averaged', () => {
  // "the map above is their equally-weighted average" is not a relationship a
  // max plane has. Under a selection the relationship is not an average at
  // all, and it is the associativity that makes the figure zero.
  const max = resolveFrameDelta(planedChart(
    { tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null, frame_grid_delta: SHIPPED },
    { max: { frame_grid_delta: EXACT, url: '/x' } },
  ), 'max')
  const min = resolveFrameDelta(planedChart(
    { tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 3, delta: SHIPPED, frame_grid_delta: SHIPPED },
    { min: { frame_grid_delta: EXACT, url: '/x' } },
  ), 'min')

  assert.doesNotMatch(max.summary, /average|averag/i)
  assert.doesNotMatch(min.summary, /average|averag/i)
  // Deleted rather than reworded would leave the max tier saying nothing about
  // a relationship the mean tier explains at length.
  assert.match(max.summary, /hourly/)
  assert.match(min.summary, /3 hourly/)
})

test('the coarsened tier’s temporal delta is not carried onto a selection plane', () => {
  // `delta` and `frame_grid_delta` on the block are the MEAN's. Both are
  // structurally zero for a selection, in both tiers, and reporting the mean's
  // 3.74% beside a max picture would attribute a mean-only disagreement to a
  // plane that does not have it.
  const delta = resolveFrameDelta(planedChart(
    { tier: 'coarsened', cadence: 'hourly', buckets_per_frame: 3, delta: SHIPPED, frame_grid_delta: SHIPPED },
    { max: { frame_grid_delta: EXACT, url: '/x' } },
  ), 'max')

  assert.equal(delta.headlinePct, null)
  assert.doesNotMatch(delta.summary, /1\.9%|3\.7%|1\.876|under 0\.1%/)
})

test('a plane whose own figure is NOT zero reports it, rather than claiming the identity', () => {
  // The identity is asserted from the payload, not from the operation's name.
  // If a plane ever ships a real disagreement, saying "exactly" would be the
  // same class of falsehood Phase 8 caught in this file.
  const delta = resolveFrameDelta(planedChart(
    { tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, delta: null, frame_grid_delta: SHIPPED },
    { max: { frame_grid_delta: { headline: 0.021, max_abs: 3e15, basis: GRID_BASIS }, url: '/x' } },
  ), 'max')

  assert.doesNotMatch(delta.summary, /exactly/)
  assert.match(delta.summary, /2\.1%/)
  assert.equal(delta.severity, 'moderate')
})

test('stop 0 in max mode does not claim to be the field the Map tab shows', () => {
  // D11's explicit exception. `plot_singular` has no statistic parameter, so
  // the Map tab is always a mean and stop 0 in max mode is the period MAX --
  // emphatically not the field above it. This is the same KIND of falsehood
  // Phase 8 §1 caught in this exact function.
  const delta = resolveFrameDelta(planedChart(
    { tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, coarsen_k: [2, 2], delta: null, frame_grid_delta: SHIPPED },
    { max: { frame_grid_delta: EXACT, url: '/x' } },
  ), 'max')

  const anchor = aggregateAnchor(delta)

  assert.doesNotMatch(anchor, /the same field the Map tab shows/)
  assert.match(anchor, /Map tab/)
  assert.match(anchor, /maximum|highest/i)
})

test('the anchor does not call a block MAX a block mean', () => {
  // D6a decision 4: block max is the spatial reducer for a selection plane,
  // and every rendered value is one some native cell actually held. "Block-
  // meaned onto the frame grid" names a mechanism this plane does not have --
  // the same mistake `shippedSentence` was rewritten to stop making one tier
  // along.
  const downsampled = resolveFrameDelta(planedChart(
    { tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, coarsen_k: [5, 5], delta: null, frame_grid_delta: SHIPPED },
    { max: { frame_grid_delta: EXACT, url: '/x' } },
  ), 'max')

  const anchor = aggregateAnchor(downsampled)

  assert.doesNotMatch(anchor, /block-meaned/)
  assert.match(anchor, /block-max/i)
})

test('the mean’s own sentences are untouched by the toggle existing', () => {
  // Deliverable 1: a chart read as the mean renders exactly what shipped
  // before this phase, whether or not planes sit beside it.
  const frames = { tier: 'cadence', cadence: 'hourly', buckets_per_frame: 1, coarsen_k: [1, 1], delta: null, frame_grid_delta: SHIPPED }
  const bare = resolveFrameDelta(chartWith(frames))
  const withPlanes = resolveFrameDelta(planedChart(frames, { max: { frame_grid_delta: EXACT, url: '/x' } }), 'mean')

  assert.equal(withPlanes.summary, bare.summary)
  assert.equal(aggregateAnchor(withPlanes), aggregateAnchor(bare))
  assert.match(aggregateAnchor(bare), /the same field the Map tab shows/)
})
