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
export function resolveFrameDelta(chart) {
  const block = chart?.frames
  if (!block || !block.tier) return null

  const cadence = block.cadence || 'unknown'
  const perFrame = Number.isFinite(block.buckets_per_frame) ? block.buckets_per_frame : 1
  const units = chart.units || ''
  const grid = figureOf(block.frame_grid_delta, units)
  const coarsened = block.tier === 'coarsened'
  const native = coarsened ? figureOf(block.delta, units) : null

  return {
    kind: coarsened ? 'approximate' : 'exact',
    severity: severityFor(native, grid, coarsened),
    headlinePct: native?.pct ?? null,
    maxAbs: native?.maxAbs ?? null,
    basis: native?.basis ?? null,
    gridPct: grid?.pct ?? null,
    gridMaxAbs: grid?.maxAbs ?? null,
    gridBasis: grid?.basis ?? null,
    summary: coarsened
      ? coarsenedSummary(perFrame, cadence, native, grid)
      : cadenceSummary(cadence, grid),
  }
}

// Stop 0's anchor: the sentence a reader uses to re-orient, and the one place
// both of Phase 8's presentation caveats land at once. It used to read "the
// same field the Map tab shows, on the frame grid", which was carrying two
// unstated differences -- the plane is block-meaned to the rendering
// resolution and the Map tab's is not, and the ramp is D9's pooled 2-98, which
// Phase 8 measured at 1.99-2.09x the Map tab's own clip, so entering this mode
// roughly halves the apparent intensity of everything.
//
// The NUMBER is deliberately not here. It belongs to the delta line directly
// beneath this one, and one measurement with two homes on one screen is how
// two homes start disagreeing.
export function aggregateAnchor() {
  return (
    'The period aggregate — the Map tab’s field on the frame grid: block-meaned ' +
    'to the rendering resolution, and coloured on the pooled scale rather than ' +
    'the Map tab’s own.'
  )
}

// Tier one. The equal weighting is a fact about the product's INTERVALS, and
// it is exactly true of them at the resolution the map is computed at. It was
// never a fact about the two arrays on screen, and saying "the map above is
// their average" told a reader it was.
function cadenceSummary(cadence, grid) {
  const lead =
    `Each frame is one ${cadence} interval of this product, and the map above ` +
    'is their equally-weighted average at native resolution.'
  if (!grid) return lead
  return `${lead} ${shippedSentence(grid)}`
}

function coarsenedSummary(perFrame, cadence, native, grid) {
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
    parts.push(shippedSentence(grid))
    if (native?.pct) {
      parts.push(
        'Those are measured on different arrays and neither bounds the other.',
      )
    }
  }
  return parts.join(' ')
}

function shippedSentence(grid) {
  return (
    `Averaging the frames you can download misses the aggregate by ${grid.pct}` +
    `${grid.maxAbs ? `, up to ${grid.maxAbs} at the worst pixel` : ''}: they are ` +
    'block-meaned to the frame grid and the map is not.'
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
