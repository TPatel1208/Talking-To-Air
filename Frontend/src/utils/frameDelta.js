// What the scrubber's frames are, relative to the map above them (T59 D14/D16).
//
// TWO disagreements, because there are two questions and a reader has both:
//
//   delta            - is the scrubber a different TEMPORAL aggregation from
//                      the map? Measured at native resolution, so a +1.2 and a
//                      -1.2 inside one block cannot cancel. Absent in the
//                      cadence tier, where a frame IS a cadence bucket and the
//                      relationship is identity.
//   frame_grid_delta - if I average the planes I downloaded, do I get the plane
//                      beside them? Measured on the arrays themselves, in both
//                      tiers, because they were block-meaned to the rendering
//                      grid and the map was not.
//
// Phase 8 is why the second one exists. It measured `mean(frames)` against
// plane 0 on a real regional TEMPO chart -- 352,181 native cells, k=(5,5) --
// and got 1.876% in the tier this file used to describe as an exact identity,
// with a worst pixel 2.72e15 against a map ramp of 5.7e14-3.3e15. The block
// mean and the across-frame mean do not commute under partial coverage: the
// period plane is block_mean(mean over intervals) and the frames are
// mean over intervals of block_mean(...), and inside a 5x5 block whose native
// cells were seen in different hours those weight different things.
//
// NEITHER NUMBER BOUNDS THE OTHER. On the coarsened chart Phase 8 measured,
// the native figure was 3.74% and the shipped one 3.28% -- the block mean
// partly cancelled the coarsening term rather than compounding it. That is why
// both are published rather than one standing in for the other, and why they
// never share a `basis` line.
//
// The whole sentence is composed here rather than assembled in JSX: there is no
// jsdom in this project, so a sentence built in a component is a sentence no
// test can read -- which is how the one Phase 8 falsified survived two phases.
//
// The delta is shown, not offered (Risk 4 is that it becomes a badge nobody
// reads), and `severity` exists so a double-digit disagreement cannot render
// as the same quiet footnote a sub-1% one does.
export function resolveFrameDelta(chart, statistic = 'mean') {
  const block = chart?.frames
  if (!block || !block.tier) return null

  const cadence = block.cadence || 'unknown'
  const perFrame = Number.isFinite(block.buckets_per_frame) ? block.buckets_per_frame : 1
  const units = chart.units || ''
  const coarsened = block.tier === 'coarsened'
  const downsampled = isDownsampled(block.coarsen_k)

  // A selection plane is a different disclosure, not a differently-worded one
  // (T59 Phase 15). BOTH of the block's figures belong to the mean: `delta`
  // is its temporal disagreement and `frame_grid_delta` its shipped-array one,
  // and a max plane has neither -- max is associative and invariant to
  // weighting, so Phase 11 G5 measured both identities bit-exact. Passing the
  // mean's numbers through would attribute a mean-only disagreement to a plane
  // that does not have it, and passing the plane's own zero through
  // `formatPct` would print "under 0.1%" for an identity that holds exactly --
  // which is D6a's own example of the failure `_DELTA_FLOOR` exists to prevent.
  if (statistic !== 'mean') {
    return selectionDelta(block, statistic, { cadence, perFrame, coarsened, downsampled, units })
  }

  const grid = figureOf(block.frame_grid_delta, units)
  const native = coarsened ? figureOf(block.delta, units) : null

  return {
    kind: coarsened ? 'approximate' : 'exact',
    statistic,
    severity: severityFor(native, grid, coarsened),
    //: Whether a block mean actually ran. Read by `aggregateAnchor` so the two
    //: adjacent sentences cannot describe the same array differently -- which
    //: they did, live, before this existed.
    downsampled,
    headlinePct: native?.pct ?? null,
    maxAbs: native?.maxAbs ?? null,
    basis: native?.basis ?? null,
    gridPct: grid?.pct ?? null,
    gridMaxAbs: grid?.maxAbs ?? null,
    gridBasis: grid?.basis ?? null,
    summary: coarsened
      ? coarsenedSummary(perFrame, cadence, native, grid, downsampled)
      : cadenceSummary(cadence, grid, downsampled),
  }
}

// How a selection plane relates to the frames beside it, and to the map above.
//
// The figure is ASSERTED FROM THE PAYLOAD, never from the operation's name. If
// a plane ever ships a real disagreement, printing "exactly" because max is
// associative in theory would be the same class of falsehood Phase 8 caught in
// this file -- a sentence that was true of the mechanism and false of the
// arrays. The exact branch is the one the backend measures today (Phase 11 G5,
// `0.0` max abs diff on both identities, both bundles); the other exists so it
// cannot silently become wrong.
const SELECTION_VERB = { max: 'highest', min: 'lowest' }
const SELECTION_NOUN = { max: 'maximum', min: 'minimum' }

function selectionDelta(block, statistic, { cadence, perFrame, coarsened, downsampled, units }) {
  const plane = block.planes?.[statistic] || {}
  const grid = figureOf(plane.frame_grid_delta, units)
  const verb = SELECTION_VERB[statistic] || statistic
  const exact = !grid || grid.headline === 0

  const frameIs = coarsened
    ? `Each frame is the ${verb} value across ${perFrame} ${cadence} intervals of this product.`
    : `Each frame is the ${verb} value in one ${cadence} interval of this product.`
  const relationship = exact
    // The whole point, and why there is no figure: a selection is associative,
    // so combining the frames reproduces the period plane rather than
    // approximating it. Stated about the arrays the browser actually holds,
    // because that is the claim Phase 8 found the mean tier getting wrong.
    ? `Taking the ${verb} of the frames you can download reproduces the period plane at stop 0 exactly — a ${SELECTION_NOUN[statistic] || statistic} selects one of the values it is given rather than combining them, so there is no disagreement here to measure.`
    : `Combining the frames you can download misses the period plane at stop 0 by ${grid.pct}${grid.maxAbs ? `, up to ${grid.maxAbs} at the worst pixel` : ''} — which this statistic is not supposed to be able to do, so read it as a defect rather than as a caveat.`

  return {
    kind: 'selection',
    statistic,
    severity: exact ? 'none' : severityOf(grid.headline),
    downsampled,
    headlinePct: null,
    maxAbs: null,
    basis: null,
    gridPct: exact ? null : grid.pct,
    gridMaxAbs: exact ? null : grid.maxAbs,
    gridBasis: plane.frame_grid_delta?.basis || null,
    summary: `${frameIs} ${relationship}`,
  }
}

// Whether a block mean actually ran. An absent `coarsen_k` reads as
// downsampled: a payload that cannot say is one whose frames may well have
// been, and claiming the stronger "same grid" property for it would invent
// exactly the kind of thing this file exists to stop inventing.
function isDownsampled(factors) {
  if (!Array.isArray(factors) || !factors.length) return true
  return factors.some((k) => Number.isFinite(k) && k > 1)
}

// Stop 0's anchor: the sentence a reader uses to re-orient, and the one place
// both of Phase 8's presentation caveats land at once. It used to read "the
// same field the Map tab shows, on the frame grid", which glossed the grid
// difference in BOTH directions and said nothing about the ramp.
//
// Neither array is native. `_downsample_grid` strides the map payload to 8,000
// cells; the frames block-mean to 20,000, or stay native when the region is
// under the ceiling. Measured live on a New Jersey chart: the Map tab drew
// 64x42 = 2,688 cells beside a 128x83 = 10,624-cell frame plane, so at k=(1,1)
// the scrubber's plane is the FINER of the two. A sentence naming only the
// frames' reduction implies the other array had none.
//
// And the ramp is D9's pooled 2-98, which Phase 8 measured at 1.99-2.09x the
// Map tab's own clip -- entering this mode roughly halves the apparent
// intensity of everything on screen.
//
// The NUMBER is deliberately not here. It belongs to the delta line directly
// beneath this one, and one measurement with two homes on one screen is how
// two homes start disagreeing.
//
// D11 GAINS AN EXPLICIT EXCEPTION in a selection mode (D6a decision 3), and it
// inverts this sentence rather than qualifying it. `plot_singular` has no
// statistic parameter, so the Map tab is always a period mean: stop 0 in max
// mode is the period MAX and is emphatically not the field above it. The Map
// tab does not follow the toggle in any mode, ever, and saying so here is
// cheaper than a reader inferring it wrongly from a picture that changed.
//
// The spatial reducer changes with it. D6a decision 4 makes block MAX the
// reducer for a selection plane -- every rendered value is one some native
// cell actually held, where a block mean invents values none did -- so
// "block-meaned onto the frame grid" would name a mechanism this plane does
// not have. That is `shippedSentence`'s own bug, one tier along.
export function aggregateAnchor(delta) {
  if (delta && delta.kind === 'selection') return selectionAnchor(delta)
  // Unknown reads as downsampled, the same way `isDownsampled` treats an
  // absent `coarsen_k`: claiming the stronger "native resolution" property for
  // a chart that cannot say is inventing the thing this sentence exists to
  // stop inventing.
  const grid = delta && delta.downsampled === false
    ? 'at native resolution on the frame grid'
    : 'block-meaned onto the frame grid'
  return (
    `The period aggregate — the same field the Map tab shows, ${grid} where the ` +
    'Map tab strides onto a coarser one of its own, and coloured on the pooled ' +
    'scale rather than the Map tab’s clip.'
  )
}

function selectionAnchor(delta) {
  const verb = SELECTION_VERB[delta.statistic] || delta.statistic
  const noun = SELECTION_NOUN[delta.statistic] || delta.statistic
  const grid = delta.downsampled === false
    ? 'at native resolution on the frame grid'
    : `block-${verb === 'highest' ? 'maxed' : 'minned'} onto the frame grid, so each rendered cell holds a value some native cell in that block actually had`
  return (
    `The period ${noun} — the ${verb} value any interval holds at each cell, ${grid}. ` +
    'This is NOT the field the Map tab shows: that stays the period mean whichever ' +
    'statistic this toggle is on. Coloured on this plane’s own pooled scale rather ' +
    'than the Map tab’s clip.'
  )
}

// Tier one. The equal weighting is a fact about the product's INTERVALS, and
// it is exactly true of them at the resolution the map is computed at. It was
// never a fact about the two arrays on screen, and saying "the map above is
// their average" told a reader it was.
function cadenceSummary(cadence, grid, downsampled) {
  const lead =
    `Each frame is one ${cadence} interval of this product, and the map above ` +
    'is their equally-weighted average at native resolution.'
  if (!grid) return lead
  return `${lead} ${shippedSentence(grid, downsampled)}`
}

function coarsenedSummary(perFrame, cadence, native, grid, downsampled) {
  const parts = [
    `Each frame averages ${perFrame} ${cadence} intervals, so the frames are a ` +
    'different temporal aggregation from the map above.',
  ]
  parts.push(
    native?.pct
      ? `They differ from it by ${native.pct}${native.maxAbs ? `, up to ${native.maxAbs} at the worst pixel` : ''}.`
      : 'The size of that difference could not be measured for this map.',
  )
  if (grid) {
    parts.push(shippedSentence(grid, downsampled))
    if (native?.pct) {
      parts.push(
        'Those are measured on different arrays and neither bounds the other.',
      )
    }
  }
  return parts.join(' ')
}

// The causal clause follows `coarsen_k`, not the tier. A region under the
// 20,000-cell ceiling does not coarsen at all -- New Jersey is 10,624 native
// cells -- and telling that reader their frames were block-meaned describes a
// mechanism their chart does not have. Found on live data, in this function's
// first draft, which is the same mistake one level along from the one this
// file was rewritten to fix.
function shippedSentence(grid, downsampled) {
  const cause = downsampled
    ? 'they are block-meaned to the frame grid and the map is not'
    // NOT "they sit on the map's own grid" -- the Map tab draws a stride-thinned
    // array, so "the map's grid" is ambiguous between the period map's native
    // one and the one on screen. Name the resolution instead.
    : 'these frames are not downsampled, so both are at native resolution'
  return (
    `Averaging the frames you can download misses the aggregate by ${grid.pct}` +
    `${grid.maxAbs ? `, up to ${grid.maxAbs} at the worst pixel` : ''}: ${cause}.`
  )
}

function figureOf(delta, units) {
  const headline = Number.isFinite(delta?.headline) ? delta.headline : null
  if (headline == null) return null
  const maxAbs = Number.isFinite(delta.max_abs) ? delta.max_abs : null
  return {
    headline,
    pct: formatPct(headline),
    maxAbs: maxAbs == null ? null : `${maxAbs.toExponential(3)} ${units}`.trim(),
    basis: delta.basis || null,
  }
}

// Under 0.1% the screen cannot resolve the number it would be printing: an
// undownsampled chart's two arrays agree to float32 storage noise (Phase 3's
// 0.000002%), and "0.0%" reads as a measurement that found nothing rather than
// as a check that passed. The document's `_pct` draws the floor in the same
// place for the same reason.
function formatPct(headline) {
  return headline >= 0.001 ? `${(headline * 100).toFixed(1)}%` : 'under 0.1%'
}

// One escalation rule over everything disclosed, matching the document's.
// Keying it to the native figure alone would leave the cadence tier -- which
// has no native figure at all -- unable to raise its voice, and that is the
// tier where a double-digit block-mean disagreement is most surprising,
// because the temporal relationship there is an exact identity.
function severityFor(native, grid, coarsened) {
  const measured = [native, grid].filter(Boolean).map((figure) => figure.headline)
  if (measured.length) return severityOf(Math.max(...measured))
  return coarsened ? 'unknown' : 'none'
}

// Thresholds sit where Phase 3's two real measurements fall on either side of
// them: 5.4% is a caveat a reader should carry, 22.3% means the scrubber and
// the map are answering materially different questions.
function severityOf(headline) {
  if (headline == null) return 'unknown'
  if (headline >= 0.1) return 'high'
  if (headline >= 0.01) return 'moderate'
  return 'low'
}
