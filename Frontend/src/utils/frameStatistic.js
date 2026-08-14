// Which statistic the scrubber is showing, and everything that follows from it
// (T59 D6a decision 7 / Phase 15).
//
// A chart under `MAX_PLANE_NATIVE_CELLS` carries a block-max and a block-min
// plane beside its mean, each with its own bytes, its own pooled clip and its
// own disclosure. This module is the one place that decides which of them the
// reader has asked for and which of them is actually on screen -- two facts
// that are the same only after the bytes have landed.

// The vocabulary, in the backend's own order. `PLANE_STATISTICS` is
// ("max", "min") and the mean leads because it is the chart's own top-level
// entry rather than a `planes` key -- a list naming only the extras reads as
// though the default were not among them (`_scrubbable_statistics`' own words).
export const STATISTIC_ORDER = ['mean', 'max', 'min']

export const STATISTIC_LABELS = { mean: 'Mean', max: 'Max', min: 'Min' }

// Which statistics this chart can actually be scrubbed as.
//
// Read off the urls that were MINTED, never off a hardcoded list and never off
// the presence of a `planes` entry: `store_frame_stack` degrades one statistic
// at a time, so a plane whose write failed -- or one evicted since -- has a
// block and no url, and offering it would promise a toggle the reader watches
// 404. That is the same rule `_scrubbable_statistics` derives the agent's list
// under, applied to `url` because `_key` addresses the blob store directly and
// is not the frontend's to hold.
export function scrubbableStatistics(chart) {
  const block = chart?.frames
  if (!block) return []
  const planes = block.planes || {}
  return STATISTIC_ORDER.filter((name) => (
    name === 'mean' ? Boolean(block.url) : Boolean(planes[name]?.url)
  ))
}

// The whole toggle in one read: what is offered, what the reader asked for,
// what is actually on screen, and why there is nothing to ask for.
//
// `selected` and `rendered` are TWO facts, and collapsing them is the failure
// this phase is written against. Phase 13's statistic guard exists because a
// wrong plane renders believably -- the same shape, the same grid, the same
// colours, different numbers -- and the frontend's version of that hazard is
// the interval between the click and the bytes. `useFrameStack` holds one
// result keyed by one url, so during a switch there is no stack at all and the
// canvas falls back to the chart's own period aggregate, which IS the mean.
// Saying "max" over it would be exactly the lie the backend refuses to serve.
//
// Everything that speaks -- the scale, the legend, the anchor, the delta, the
// Statistics tab's scoping -- keys off `rendered`, so the label, the ramp, the
// caption, the sentences and the pixels all change on one event.
export function resolveStatisticChoice(chart, selected, loadState) {
  const options = scrubbableStatistics(chart)
  const refusal = chart?.frames?.planes_unavailable || null
  // A plane evicted between the payload being written and the chart being
  // reopened. The toggle cannot show a selection it does not offer.
  const asked = options.includes(selected) ? selected : 'mean'
  const pending = asked !== 'mean' && loadState !== 'loaded'

  return {
    options,
    // Offered when there is a choice to make, or when there was one and the
    // gate took it away -- Phase 5 decision 2's rule, one level in. A chart
    // that never had planes gets neither, and renders exactly as it did.
    offered: options.length > 1 || Boolean(refusal),
    selected: asked,
    rendered: pending ? 'mean' : asked,
    pending,
    refusal: refusal ? { reason: refusal.reason || null, detail: refusal.detail || null } : null,
  }
}

// Which stop is DISPLAYED, given the stop the reader remembers being on.
//
// The remembered index is kept by the caller across a statistic switch and
// only parked here, never reset: finding a peak is the entire reason the max
// plane exists, so someone who scrubbed to hour 17 and then asked for its
// maximum has done precisely the thing that must not cost them their place.
//
// Parked on the aggregate rather than held at 17 while the pixels are absent,
// because that is Phase 6 decision 2 unchanged -- the aggregate is the only
// stop whose pixels are actually on screen, and a readout naming hour 17 over
// the period map "would read as a quiet two days to someone scrubbing for an
// event". The index comes back on its own when the slider re-enables.
export function resolveScrubStop({ stops, index, sliderEnabled }) {
  if (!Array.isArray(stops) || !stops.length) return null
  if (!sliderEnabled) return stops[0]
  const clamped = Math.min(Math.max(Number.isFinite(index) ? index : 0, 0), stops.length - 1)
  return stops[clamped] || stops[0]
}

// How much ground a rendered peak claims (D6a decision 9 / Phase 11 G4).
//
// Block max paints one native cell's peak across every cell of its block, so
// the area shown at peak level is larger than the area that reached it. G4
// measured ≈24.7-24.75x on both live bundles with the per-frame spread under
// half a percentage point -- not an occasional worst case but what block max
// costs on essentially every finite block, which is why it is disclosed rather
// than caveated.
//
// PHASE 14'S FINDING BINDS THIS WORDING: `ceiling` is k^2 and is NOT an upper
// bound. Both sums in `_overstatement_terms` are cos(latitude)-weighted per
// cell, so a block whose max sits on its lowest-weighted row contributes more
// than k^2, and the pooled figure straddles k^2 rather than approaching it from
// below -- measured at 4.0000014 against a ceiling of 4. Nothing here may say
// "up to k x" or "at most 25x", and `ceiling` is deliberately not rendered at
// all. What the quantity supports is the measured figure itself, per chart.
//
// The mean has no such quantity (a block mean does not paint a peak) and the
// min's is `null` in the payload by construction.
export function extentOverstatementNote(chart, statistic) {
  const measured = chart?.frames?.planes?.[statistic]?.extent_overstatement
  const headline = measured?.headline
  if (!Number.isFinite(headline)) return null
  const worst = Number.isFinite(measured.worst_frame)
    ? ` Its single worst frame stretched one further, to ${measured.worst_frame.toFixed(1)}×.`
    : ''
  return (
    `Every cell of a block is drawn at that block’s highest value, so the ground shown ` +
    `at peak level is about ${headline.toFixed(1)}× the ground the peak was actually ` +
    `measured over — pooled across this chart’s own blocks, cos(latitude)-weighted.${worst}`
  )
}

// What `useFrameStack` fetches and `decodeFrameStack` reshapes for one
// statistic: that plane's own url on the layout every plane of the chart
// shares. Null when the statistic has no bytes -- NEVER the mean's url as a
// fallback, which is the failure Phase 13 decision 4 built a manifest guard
// against on the way out: every plane of one chart has the same shape, grid and
// version, so the wrong one renders a believable field with the wrong numbers.
//
// Built field by field rather than spread off the block, because `_key` sits
// beside `url` in all four of them and addresses the blob store directly.
export function resolveStatisticSource(chart, statistic) {
  const block = chart?.frames
  if (!block || !STATISTIC_ORDER.includes(statistic)) return null
  const url = statistic === 'mean' ? block.url : block.planes?.[statistic]?.url
  if (!url) return null
  return {
    statistic,
    url,
    shape: block.shape,
    dtype: block.dtype,
    period_index: block.period_index,
  }
}
