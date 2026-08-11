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
export function resolveScrubberScale(chart, scrubbing) {
  if (!scrubbing) return null

  const range = chart?.frames?.value_range
  // Null when nothing in the stack survived masking. There is no pooled clip
  // to show, and inventing one from the map's range would label the scrubber
  // with a scale it was not built on.
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
    legendNote: SCRUBBER_LEGEND_NOTE,
    basis: chart.frames.scale_basis || null,
  }
}

const SCRUBBER_LEGEND_NOTE = '2nd–98th pct, pooled across frames'
