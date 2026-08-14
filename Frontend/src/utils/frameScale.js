// The scrubber's colour scale (T59 D9), and the one decision D9 left open:
// WHEN the map recolours.
//
// Two scales exist for one field -- the Map tab's own percentile clip and the
// stack's pooled 2-98 (`value_range` + `scale_basis`). The scale is a property
// of the MODE, not of the slider position:
//
//   * The Map tab is untouched. Deriving its scale from the stack would let a
//     storage gate firing change a map's appearance, with nothing on screen
//     saying why.
//   * Entering scrubber mode recolours once, onto the pooled range, and every
//     stop including the aggregate is drawn on it. Frame 0 is the period mean
//     and the pool includes the period mean, so the aggregate already IS its
//     pooled colour -- which is why the recolour can happen at entry, before
//     the blob has even landed, without anything jumping when it does.
//   * Colours therefore never change mid-scrub, and the one jump there is
//     happens on a click, with the legend redrawing in the same instant.
//
// Deliberately NOT `computeSharedColorScale`: that takes min(vmins)/max(vmaxs),
// a union of per-panel clips, which Phase 3 measured at 2.5-3.9x too wide over
// a real frame stack -- every ordinary hour in the bottom third of the colormap.
//
// Returns the same shape MapLibreHeatmapPanel's `colorScaleOverride` already
// takes, plus the basis line the legend discloses instead of the payload's own
// percentile note (which no longer describes what is drawn).
// `statistic` is D6a's toggle (Phase 15). Each plane pools its OWN 2-98 --
// the mean's clip on a max plane saturates at exactly the stops someone
// switched to max to see, which is the one thing that mode exists to show.
export function resolveScrubberScale(chart, scrubbing, statistic = 'mean') {
  if (!scrubbing) return null

  const source = statisticRange(chart?.frames, statistic)
  const range = source?.value_range
  // Null when nothing in the stack survived masking, and null again when the
  // named statistic has no pooled range of its own. There is no clip to show,
  // and inventing one -- from the map's range, or from the mean's -- would
  // label the scrubber with a scale it was not built on.
  if (!Array.isArray(range) || range.length !== 2) return null
  const [vmin, vmax] = range
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) return null

  return {
    vmin,
    vmax,
    colormap: chart.colormap,
    // Two lengths for two places. `scale_basis` is the backend's full sentence
    // and belongs in the disclosure block; the legend gets a caption that fits
    // under a 180px colorbar, because a legend that goes quiet about which of
    // the two scales is drawn is precisely Risk 5's legibility hazard.
    legendNote: legendNoteFor(statistic),
    basis: chart.frames.scale_basis || null,
  }
}

// The pooled range for one statistic. The mean's is the chart's own top-level
// entry (D6a decision 5 -- it is not a `planes` key); a plane's is on its own
// block, because `value_range` is one of the four things Phase 13 decision 2
// found genuinely differ per plane.
function statisticRange(block, statistic) {
  if (!block) return null
  if (statistic === 'mean') return block
  return block.planes?.[statistic] || null
}

// Under a 180px colorbar, so the statistic goes in as a word and the pooling
// stays the clause it always was.
//
// The mean's caption is UNCHANGED, deliberately: D6a decision 5 keeps the mean
// entry exactly as it was, and a chart that gained planes must not have its
// default mode reworded by the existence of modes the reader has not opened.
// What disambiguates the mean is the same click Risk 5's mitigation already
// rests on -- the toggle, the numbers and this caption all move together.
const SCRUBBER_LEGEND_NOTE = '2nd–98th pct, pooled across frames'
const PLANE_LEGEND_NOTE = {
  max: '2nd–98th pct, pooled across the max frames',
  min: '2nd–98th pct, pooled across the min frames',
}

function legendNoteFor(statistic) {
  return PLANE_LEGEND_NOTE[statistic] || SCRUBBER_LEGEND_NOTE
}
