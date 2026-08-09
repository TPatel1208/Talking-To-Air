/**
 * utils/metadataDisplay.js
 * -------------------------
 * Pure field-derivation helpers behind the chart Metadata tab's Overview/
 * Details split (T32, OutputPanel.jsx). Kept separate from the JSX so the
 * "which field goes where, and what renders when a fact is missing" logic
 * is unit-testable without a React render harness (this repo's frontend
 * test runner is plain `node --test`, no jsdom/RTL).
 */
import { resolveMasking, formatQaPassRate } from './maskingProvenance.js'

export const NOT_AVAILABLE = 'Not available'

// Missing metadata is disclosed, not hidden (T32): every field a section
// defines renders even when empty, so the user knows a fact was checked
// for and doesn't exist, rather than wondering if the UI forgot it.
export function fmt(value) {
  if (value === null || value === undefined || value === '') return NOT_AVAILABLE
  if (Array.isArray(value) && value.length === 0) return NOT_AVAILABLE
  return value
}

export function compactMetadataDate(value) {
  if (!value) return ''
  return String(value).replace('T00:00:00', '').replace('T23:59:59', '').replace(/Z$/, '')
}

export function formatBBox(bbox) {
  if (!Array.isArray(bbox)) return bbox || ''
  return bbox.map(value => Number.isFinite(value) ? value.toFixed(4) : value).join(', ')
}

// Some chart types (e.g. comparison_tools.py's heatmap_multi) never attach
// a provenance object at all -- the Overview must show one clear empty
// state for those, not a grid of "Not available" per field that reads as
// a bug rather than "this chart type has no provenance to show".
export function hasProvenance(chart) {
  return Boolean(chart?.provenance)
}

export function dateRangeLabel(provenance) {
  const p = provenance || {}
  const range = [compactMetadataDate(p.start_date), compactMetadataDate(p.end_date)].filter(Boolean).join(' to ')
  return range || null
}

// Overview gets a one-line count + cadence, never the date list (T32
// Overview/Details split -- the full list belongs in Details, not
// cluttering the always-expanded Overview).
export function granuleSummary(chart) {
  const meta = chart?.aggregation_meta
  const provenance = chart?.provenance || {}
  const nGranules = meta?.n_granules ?? provenance.n_granules
  const cadence = meta?.cadence || provenance.cadence || ''
  if (nGranules == null) return null
  return `${nGranules} ${cadence ? `${cadence} ` : ''}granule${nGranules === 1 ? '' : 's'}`
}

// Details' Temporal section gets the full per-granule date list.
export function granuleDates(chart) {
  const meta = chart?.aggregation_meta
  const provenance = chart?.provenance || {}
  return meta?.granule_dates || provenance.granule_dates || []
}

// Green/yellow/red trust signal for the Overview QA line (T32): "verified"/
// "cf-deterministic" mean a mask was applied with a deterministic rule
// (datasets/qa_flags.py), "inferred, not verified" applied one the model
// proposed, and the ambiguous/not-applied statuses mean no reliable QA
// masking ran at all.
export function maskingStatusColor(qaStatus) {
  if (qaStatus === 'verified' || qaStatus === 'cf-deterministic') return 'var(--success, #1a7f4b)'
  if (qaStatus === 'inferred, not verified') return 'var(--warning, #b98900)'
  // A product that publishes no quality flag has nothing anyone can pin or
  // fix, so it reads neutral. Red is reserved for the statuses that mean a
  // mask was expected and did not happen.
  if (qaStatus === 'not applied — this product publishes no quality flag variable') {
    return 'var(--text-muted, #667085)'
  }
  return 'var(--error, #b42318)'
}

// The full masking-provenance record (fill_value_source/valid_range_source/
// applied, alongside qa_status/qa_source/qa_note) -- resolveMasking only
// extracts the QA-status subset the Statistics tab needs; Details'
// Provenance section also wants the fill/valid-range tier that won.
export function resolveMaskingRaw(chart) {
  return chart?.masking || chart?.provenance?.masking || chart?.aggregation_meta?.masking || null
}

export function citationString(provenance) {
  const p = provenance || {}
  const parts = [
    p.dataset,
    p.dataset_description && `(${p.dataset_description})`,
    p.dataset_version && `version ${p.dataset_version}`,
    p.source,
    p.collection_id && `Collection ID: ${p.collection_id}`,
  ].filter(Boolean)
  return parts.join(', ')
}

// Where a product sits in its provider's validation lifecycle, and the caveat
// that comes with it (T57).
//
// Returns null for an unstated maturity, deliberately. "unknown" means nobody
// checked, which is a different thing from a checked clean bill of health --
// rendering it as a field would read like the latter. A `cautionary` level is
// one whose provider tells you not to publish on it yet, and it is flagged so
// the UI can weight it rather than listing it beside the version number.
const CAUTIONARY_MATURITY = new Set(['beta', 'provisional'])

export function maturityFields(chart) {
  const level = chart?.provenance?.maturity
  if (!level || level === 'unknown') return null
  return {
    level,
    note: chart?.provenance?.maturity_note || '',
    cautionary: CAUTIONARY_MATURITY.has(level),
  }
}

// Which layer a physical level request resolved to, and how much of an
// approximation that was (T58 D5).
//
// Two INDEPENDENT facts, because either can be perfect while the other is poor:
// how much of the analyzed region agrees on this layer, and how far the layer
// actually sits from what was asked. The spike measured a 300 hPa request
// landing 40 hPa away at 83% agreement, and an 850 hPa request landing 46 hPa
// away at 100% -- a panel showing only agreement would call the second one
// perfect. Returns null for a chart that selected no physical level, so the
// section is absent rather than a row of "Not available" on every ordinary map.
const LEVEL_DECIMALS = 2

// Below this, 2-dp rounding would print "0" for a value that is not zero --
// and the panel deliberately distinguishes "exactly the level requested" from a
// small error. Rendering 0.004 hPa as "0 hPa from the level requested" is a
// sentence that contradicts itself.
const LEVEL_EPSILON = 10 ** -LEVEL_DECIMALS / 2

function roundedLevel(value) {
  if (!Number.isFinite(value)) return null
  return Number(value.toFixed(LEVEL_DECIMALS))
}

export function levelResolutionFields(chart) {
  const level = chart?.provenance?.level_resolution
  if (!level) return null
  const units = level.units || ''
  const suffix = units ? ` ${units}` : ''
  // Every field guards its own input. This is the one block whose absence takes
  // the whole Metadata tab down rather than leaving a row blank, because it
  // renders outside fmt()/NOT_AVAILABLE -- so a payload missing a field must
  // degrade to a missing line, not a TypeError.
  const withUnits = value => {
    const rounded = roundedLevel(value)
    return rounded === null ? null : `${rounded}${suffix}`
  }
  const pct = fraction =>
    (Number.isFinite(fraction) ? `${Number((fraction * 100).toFixed(1))}%` : null)
  const hasRunnerUp = level.runner_up !== null && level.runner_up !== undefined
  const resolved = withUnits(level.resolved_level)
  const agreement = pct(level.dominant_fraction)
  const count = Number.isFinite(level.n_pixels) ? Number(level.n_pixels).toLocaleString('en-US') : null
  const excluded = pct(level.excluded_fraction)

  return {
    kind: level.kind || null,
    requested: Number.isFinite(level.requested) ? `${Number(level.requested)}${suffix}` : null,
    resolved: resolved === null ? null : `${resolved} (layer ${level.index})`,
    // Stated even when zero: "not shown" and "exact" read identically to
    // someone scanning the panel, and they are different facts. Anything below
    // half a displayed unit is reported as a band rather than rounded to "0",
    // which would collapse the same distinction from the other side.
    levelError: levelError(level, withUnits, suffix),
    // "of the analyzed AREA", not "of N pixels". The fraction is
    // cos(latitude)-weighted and the count is raw, so "91.3% of 40 analyzed
    // pixels" is false whenever 40 equal-count cells span an unequal area --
    // by count that same split can be 50/50. The count travels as a sample
    // size, which is what it actually is.
    agreement: agreement === null ? null
      : `${agreement} of the analyzed area resolves to layer ${level.index}` +
        (count === null ? '' : ` (${count} pixel-columns sampled)`),
    runnerUp: hasRunnerUp && pct(level.runner_up_fraction) !== null
      ? `${pct(level.runner_up_fraction)} resolves to layer ${level.runner_up} instead`
      : null,
    // The denominator behind the sample size. Without it, "100% of 4
    // pixel-columns" sits beside a map built from forty.
    excluded: level.excluded_fraction ? `${excluded} of the region had no usable vertical coordinate` : null,
    spread: level.resolved_level_spread
      ? `This layer itself ranges ${withUnits(level.resolved_level_spread)} across the region`
      : null,
    axisVariable: level.axis_variable || null,
  }
}

function levelError(level, withUnits, suffix) {
  const error = level.level_error
  if (!Number.isFinite(error)) return null
  if (error === 0) return 'exactly the level requested'
  if (error < LEVEL_EPSILON) return `less than ${LEVEL_EPSILON}${suffix} from the level requested`
  return `${withUnits(error)} from the level requested`
}

export function datasetLandingUrl(collectionId) {
  return collectionId ? `https://cmr.earthdata.nasa.gov/search/concepts/${collectionId}.html` : null
}

export function spatialFields(chart) {
  const provenance = chart?.provenance || {}
  const bbox = chart?.bounds || chart?.query?.bbox
  return {
    regionName: provenance.region_name ?? null,
    bbox: bbox ? formatBBox(bbox) : null,
  }
}

// Overview's single-line Region fact: region_name when the backend resolved
// one, else the raw bounding box coordinates -- so a bbox-only selection
// (no named region) still shows something instead of "Not available".
export function regionLabel(chart) {
  const { regionName, bbox } = spatialFields(chart)
  return regionName || bbox || null
}

export function temporalFields(chart) {
  const provenance = chart?.provenance || {}
  const meta = chart?.aggregation_meta
  return {
    dateRange: dateRangeLabel(provenance),
    cadence: meta?.cadence || provenance.cadence || null,
    dates: granuleDates(chart),
  }
}

export function qaMethodologyFields(chart) {
  const methodology = chart?.provenance?.qa_methodology || {}
  const masking = resolveMaskingRaw(chart) || {}
  return {
    qualityFlagVar: methodology.quality_flag_var ?? null,
    qaGoodValues: methodology.qa_good_values ?? null,
    qaBadValues: methodology.qa_bad_values ?? null,
    fillValueSource: masking.fill_value_source ?? null,
    validRangeSource: masking.valid_range_source ?? null,
    // T55: what those settings actually did to this data. The realized rate
    // belongs beside the rule that produced it -- this block is what gets
    // copied out of the Metadata tab, and a QA rule without its outcome is
    // half the story.
    qaPassRate: qaPassRateSummary(chart),
    qaPassRateBasis: masking.qa_pass_rate_basis ?? null,
  }
}

// "75.3% (11,298 of 15,000 checked pixels)" -- the percentage never travels
// without the denominator it was computed over, so it can't be mistaken for
// the Statistics tab's "Valid values %", which answers the different question
// "did we get data at all".
function qaPassRateSummary(chart) {
  const resolved = resolveMasking(chart)
  const pct = formatQaPassRate(resolved)
  if (pct === null) return null
  const { qaPassingPixels: passing, qaCheckedPixels: checked } = resolved
  if (passing == null || checked == null) return pct
  const count = value => value.toLocaleString('en-US')
  return `${pct} (${count(passing)} of ${count(checked)} checked pixels)`
}

export function variableDefinitionFields(chart) {
  const varDef = chart?.provenance?.variable_definition || {}
  const hasRange = varDef.valid_ranges && (varDef.valid_ranges.min != null || varDef.valid_ranges.max != null)
  return {
    longName: varDef.long_name ?? null,
    units: varDef.units ?? null,
    advisoryNotes: varDef.advisory_notes?.length ? varDef.advisory_notes.join('; ') : null,
    validRange: hasRange ? `${varDef.valid_ranges.min ?? '—'} to ${varDef.valid_ranges.max ?? '—'}` : null,
    maskNote: varDef.mask_note ?? null,
  }
}

// The object copied to the clipboard by Details' Reproducibility "copy
// query" action -- query snapshot plus the source handles it was built
// from, so a copied query is enough to re-run or share the exact request.
export function reproducibilityQuery(chart) {
  const query = chart?.query || {}
  const sourceHandles = chart?.provenance?.source_handles || []
  return { ...query, source_handles: sourceHandles }
}

export function reproducibilityFields(chart) {
  const query = chart?.query || {}
  const sourceHandles = chart?.provenance?.source_handles || []
  return {
    dataset: query.dataset ?? null,
    startDate: query.start_date ?? null,
    endDate: query.end_date ?? null,
    bbox: query.bbox ? formatBBox(query.bbox) : null,
    aggregation: query.aggregation ?? null,
    sourceHandles: sourceHandles.length ? sourceHandles.join(', ') : null,
  }
}

// The object rendered verbatim by Details' raw-JSON toggle.
export function rawMetadataJson(chart) {
  return { provenance: chart?.provenance ?? null, aggregation_meta: chart?.aggregation_meta ?? null }
}
