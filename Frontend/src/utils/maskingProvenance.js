// QA-flag masking provenance rides along on chart payloads from the backend
// (Backend/preprocessing/aggregation_service.resolve_and_mask ->
// datasets/qa_flags.resolve_qa_info). Its qa_status is one of the vocabulary
// values in qa_flags.py: "verified", "cf-deterministic", "inferred, not
// verified", "ambiguous — awaiting classification", "not applied — semantics
// unknown". It lands in a different spot per chart type: top-level `masking`
// for timeseries (plot_tools sets ts_payload["masking"]), and inside
// `provenance.masking` / `aggregation_meta.masking` for heatmaps (plot_tools
// _provenance copies agg_meta["masking"] there). Check all three so the
// Statistics tab can show whether QA masking actually ran, instead of leaving
// the user to infer it from the valid-pixel count.
export function resolveMasking(chart) {
  if (!chart || typeof chart !== 'object') return null

  const masking =
    chart.masking ||
    chart.provenance?.masking ||
    chart.aggregation_meta?.masking ||
    null
  if (!masking || typeof masking !== 'object') return null

  const qaStatus = masking.qa_status
  if (!qaStatus) return null

  return {
    qaStatus,
    qaSource: masking.qa_source || '',
    qaNote: masking.qa_note || '',
    // T55: the pass rate the mask actually produced, counted backend-side from
    // the same boolean condition that discarded the pixels. Every field is
    // normalized to null-when-absent rather than defaulted, because a genuine
    // 0 (a scene QA wiped out, or a scene with nothing retrievable to check)
    // is a real answer and must not read as "not applied".
    qaPassRate: numberOrNull(masking.qa_pass_rate),
    qaCheckedPixels: numberOrNull(masking.qa_checked_pixels),
    qaPassingPixels: numberOrNull(masking.qa_passing_pixels),
    qaFlagMissingPixels: numberOrNull(masking.qa_flag_missing_pixels),
    qaPassRateByTime: Array.isArray(masking.qa_pass_rate_by_time) ? masking.qa_pass_rate_by_time : null,
    qaPassRateTimes: Array.isArray(masking.qa_pass_rate_times) ? masking.qa_pass_rate_times : null,
    qaPassRateBasis: masking.qa_pass_rate_basis || '',
  }
}

function numberOrNull(value) {
  return Number.isFinite(value) ? value : null
}

// The QA pass rate as a display percentage, with both ends floored so rounding
// can never manufacture a verdict the pixels don't support: one discarded pixel
// in 15,000 rounds to 100.0%, and reading that as a clean scene is the same
// class of error as the valid-pct tautology fixed once already here. A rate
// that IS genuinely 1 or 0 renders verbatim -- the floor keys off the pixel
// counts, not off the rounded number. Returns null when no rate was reported,
// so the caller shows "Not applied" rather than a fabricated percentage.
export function formatQaPassRate(masking) {
  const rate = masking?.qaPassRate
  if (rate == null) return null
  const checked = masking.qaCheckedPixels
  const passing = masking.qaPassingPixels
  let pct = rate * 100
  if (passing != null && checked != null) {
    if (passing < checked && pct > 99.9) pct = 99.9
    if (passing > 0 && pct < 0.1) pct = 0.1
  }
  return `${pct.toFixed(1)}%`
}

// Region fidelity (T42): the backend discloses *what kind* of region it
// masked -- region_type is 'polygon' (the named place's real boundary),
// 'bounding_box' (a rectangle that averages in whatever it spans),
// 'point_buffer' (a 0.1° box around a geocoded point with no boundary), or
// 'boundary_cells' (a sub-cell region rescued from an empty mask). Only the
// last three are worth a disclosure line -- a real polygon is the faithful
// case and needs no caveat. display_name is the place the geocoder actually
// resolved, so a wrong-place answer is catchable. Rides on chart.provenance.
//
// T60 D10 adds 'composite_union' (a "+" grammar region, e.g. "NY + NJ") and
// D10b makes it FAITHFUL: it is the exact union of real state boundaries, as
// truthful as 'polygon', so it joins the no-caveat set. Left out, the note
// lookup would fall through to the raw enum and a correct answer would read
// as a defect (Phase 3b gate, V16, traced not assumed).
//
// D10a also splits the two facts region_type was carrying. 'boundary_cells'
// is *rasterization fidelity*; 'composite_union' is *shape provenance*. Both
// can be true at once, so region_origin rides alongside rather than being
// overwritten -- otherwise a small composite that self-heals silently stops
// telling the researcher its shape was a construction.
const FAITHFUL_REGION_TYPES = new Set(['polygon', 'composite_union'])

export function resolveRegionFidelity(chart) {
  if (!chart || typeof chart !== 'object') return null
  const provenance = chart.provenance || {}
  const regionType = provenance.region_type || chart.region_type
  if (!regionType || FAITHFUL_REGION_TYPES.has(regionType)) return null
  return {
    regionType,
    regionOrigin: provenance.region_origin || chart.region_origin || '',
    displayName: provenance.display_name || chart.display_name || provenance.region_name || '',
  }
}

// One-line, plain-language caveat for a region_type that isn't faithful.
// Lives here rather than in OutputPanel so it is testable at all: this repo
// has no jsdom, so a lookup embedded in a component cannot be asserted on --
// and the `|| regionType` fallback it replaces is exactly the kind of thing
// that silently starts leaking raw enums when the backend adds a value.
const REGION_TYPE_NOTE = {
  bounding_box: 'approximated as a bounding box (a rectangle), not the named boundary',
  point_buffer: 'approximated as a small box around a geocoded point (no boundary found)',
  boundary_cells: 'smaller than a grid cell — showing the cells it touches',
  composite_union: 'built by combining the boundaries you named',
}

const REGION_ORIGIN_NOTE = {
  composite_union: 'combined from the boundaries you named',
}

export function regionTypeNote(regionType) {
  return REGION_TYPE_NOTE[regionType] || 'approximated — see the metadata for details'
}

// The D10a second sentence: what the shape *was* before rasterization
// downgraded the slot. Empty when the origin adds nothing (a plain polygon
// that self-healed is already fully described by the note above).
export function regionOriginNote(regionOrigin) {
  return REGION_ORIGIN_NOTE[regionOrigin] || ''
}
