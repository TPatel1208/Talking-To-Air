/**
 * utils/metadataDisplay.js
 * -------------------------
 * Pure field-derivation helpers behind the chart Metadata tab's Overview/
 * Details split (T32, OutputPanel.jsx). Kept separate from the JSX so the
 * "which field goes where, and what renders when a fact is missing" logic
 * is unit-testable without a React render harness (this repo's frontend
 * test runner is plain `node --test`, no jsdom/RTL).
 */
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
  if (!nGranules) return null
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
  }
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
