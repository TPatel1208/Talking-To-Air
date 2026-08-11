// What the scrubber's frames are, relative to the map above them (T59 D14/D16).
//
// Two tiers, and they mean different things:
//
//   cadence   - a frame IS one of the product's own cadence buckets, and the
//               period map is derived from them. The relationship is identity;
//               there is no number, and printing "0%" would claim a
//               measurement that was never taken.
//   coarsened - a frame averages several cadence buckets, so the frames are a
//               DIFFERENT temporal aggregation from the map. Phase 3 measured
//               that disagreement at 5.4% and 22.3% on two real TEMPO bundles,
//               and the 60-frame budget is reached at ~2.5 days, so this is
//               the common case, not the edge.
//
// The delta is shown, not offered (Risk 4 is that it becomes a badge nobody
// reads), and `severity` exists so a double-digit disagreement cannot render
// as the same quiet footnote a sub-1% one does.
export function resolveFrameDelta(chart) {
  const block = chart?.frames
  if (!block || !block.tier) return null

  const cadence = block.cadence || 'unknown'
  const perFrame = Number.isFinite(block.buckets_per_frame) ? block.buckets_per_frame : 1

  if (block.tier !== 'coarsened') {
    return {
      kind: 'exact',
      severity: 'none',
      headlinePct: null,
      maxAbs: null,
      basis: null,
      summary: `Each frame is one ${cadence} interval of this product, and the map above is their average.`,
    }
  }

  const delta = block.delta || {}
  const headline = Number.isFinite(delta.headline) ? delta.headline : null
  const maxAbs = Number.isFinite(delta.max_abs) ? delta.max_abs : null

  return {
    kind: 'approximate',
    severity: severityOf(headline),
    headlinePct: headline == null ? null : `${(headline * 100).toFixed(1)}%`,
    maxAbs: maxAbs == null ? null : `${maxAbs.toExponential(3)} ${chart.units || ''}`.trim(),
    basis: delta.basis || null,
    summary: `Each frame averages ${perFrame} ${cadence} intervals, so the frames are a different temporal aggregation from the map above.`,
  }
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
