import test from 'node:test'
import assert from 'node:assert/strict'

import {
  NOT_AVAILABLE, fmt, dateRangeLabel, granuleSummary, granuleDates,
  maskingStatusColor, resolveMaskingRaw, citationString, datasetLandingUrl,
  spatialFields, temporalFields, qaMethodologyFields, variableDefinitionFields,
  reproducibilityQuery, reproducibilityFields, rawMetadataJson, regionLabel,
  hasProvenance,
} from '../src/utils/metadataDisplay.js'

// A fixture provenance object exercising every field the Overview/Details
// sections read (T32 PRD Testing Decisions), mirroring what
// Backend/tools/satellite_tools/plot_tools.py::_provenance now attaches.
function fixtureChart(overrides = {}) {
  return {
    type: 'heatmap',
    bounds: [-75.5, 39.0, -73.5, 41.0],
    provenance: {
      variable: 'vertical_column_troposphere',
      dataset: 'TEMPO_NO2_L3',
      dataset_description: 'TEMPO tropospheric NO2 vertical column',
      dataset_version: 'V04',
      collection_id: 'C3685896708-LARC_CLOUD',
      provider: 'NASA LARC',
      instrument: 'TEMPO',
      source: 'NASA LARC — TEMPO',
      start_date: '2024-01-01T00:00:00',
      end_date: '2024-01-02T00:00:00',
      region_name: 'New Jersey',
      aggregation: 'Daily Mean, 2 hourly granules, 2024-01-01 to 2024-01-02',
      n_granules: 2,
      cadence: 'hourly',
      granule_dates: ['2024-01-01', '2024-01-02'],
      masking: { qa_status: 'verified', qa_source: 'collections_yaml', fill_value_source: 'collections_yaml', valid_range_source: 'collections_yaml' },
      qa_methodology: { quality_flag_var: 'main_data_quality_flag', qa_good_values: [0] },
      variable_definition: {
        long_name: 'NO2 tropospheric column',
        units: 'molecules/cm^2',
        advisory_notes: ['QA-flagged advisory note'],
        valid_ranges: { min: -1e15, max: 1e18 },
        mask_note: 'fill values and a valid range are defined',
      },
      source_handles: ['obs_1'],
    },
    aggregation_meta: {
      aggregation_label: 'Daily Mean, 2 hourly granules, 2024-01-01 to 2024-01-02',
      n_granules: 2,
      cadence: 'hourly',
      granule_dates: ['2024-01-01', '2024-01-02'],
      masking: { qa_status: 'verified', qa_source: 'collections_yaml', fill_value_source: 'collections_yaml', valid_range_source: 'collections_yaml' },
    },
    query: {
      dataset: 'vertical_column_troposphere',
      start_date: '2024-01-01T00:00:00',
      end_date: '2024-01-02T00:00:00',
      bbox: [-75.5, 39.0, -73.5, 41.0],
      aggregation: 'Daily Mean, 2 hourly granules, 2024-01-01 to 2024-01-02',
      chart_parameters: { chart_type: 'heatmap' },
    },
    ...overrides,
  }
}

// ── fmt() -- missing fields render "Not available", not hidden ─────────────

test('fmt() falls back to "Not available" for null/undefined/empty', () => {
  assert.equal(fmt(null), NOT_AVAILABLE)
  assert.equal(fmt(undefined), NOT_AVAILABLE)
  assert.equal(fmt(''), NOT_AVAILABLE)
  assert.equal(fmt([]), NOT_AVAILABLE)
})

// Review fix: comparison_tools.py's heatmap_multi charts never set
// provenance at all -- the Overview must recognize that and show one clear
// empty state, not a fully-populated-looking grid of "Not available" per
// field (which reads as a bug, not "this chart type has no provenance").
test('hasProvenance is false when the chart has no provenance object at all', () => {
  assert.equal(hasProvenance({ type: 'heatmap_multi' }), false)
  assert.equal(hasProvenance({}), false)
  assert.equal(hasProvenance(null), false)
})

test('hasProvenance is true even when individual provenance fields are missing', () => {
  assert.equal(hasProvenance({ provenance: {} }), true)
  assert.equal(hasProvenance(fixtureChart()), true)
})

test('fmt() passes real values through unchanged', () => {
  assert.equal(fmt('TEMPO_NO2_L3'), 'TEMPO_NO2_L3')
  assert.equal(fmt(0), 0)
  assert.deepEqual(fmt(['a']), ['a'])
})

// ── Overview: "This view" facts ─────────────────────────────────────────────

test('dateRangeLabel builds a compact "start to end" range', () => {
  const chart = fixtureChart()
  assert.equal(dateRangeLabel(chart.provenance), '2024-01-01 to 2024-01-02')
})

test('dateRangeLabel returns null when no dates are present', () => {
  assert.equal(dateRangeLabel({}), null)
  assert.equal(dateRangeLabel(null), null)
})

// Overview shows only a count+cadence summary -- never the date list.
test('granuleSummary is a one-line count+cadence string', () => {
  const chart = fixtureChart()
  assert.equal(granuleSummary(chart), '2 hourly granules')
})

test('granuleSummary singularizes a single granule and omits cadence when absent', () => {
  assert.equal(granuleSummary({ provenance: { n_granules: 1, cadence: 'daily' } }), '1 daily granule')
  assert.equal(granuleSummary({ provenance: { n_granules: 3 } }), '3 granules')
})

test('granuleSummary is null when no granule count is available', () => {
  assert.equal(granuleSummary({}), null)
  assert.equal(granuleSummary({ provenance: {} }), null)
})

// Review fix: a request whose masking excluded every timestep can
// legitimately produce n_granules=0 -- that's a real signal ("this
// request returned no valid data"), distinct from the field being absent.
test('granuleSummary surfaces a legitimate zero-granule count instead of treating it as missing', () => {
  assert.equal(granuleSummary({ provenance: { n_granules: 0, cadence: 'daily' } }), '0 daily granules')
  assert.equal(granuleSummary({ aggregation_meta: { n_granules: 0 } }), '0 granules')
})

// ── Overview vs Details: granule dates only appear in Details ──────────────

test('granuleDates returns the full per-granule list for Details, not Overview', () => {
  const chart = fixtureChart()
  assert.deepEqual(granuleDates(chart), ['2024-01-01', '2024-01-02'])
  // Overview's own summary never includes the dates array itself.
  assert.equal(granuleSummary(chart).includes('2024-01-01'), false)
})

test('granuleDates falls back to provenance when aggregation_meta is absent (timeseries-shaped payload)', () => {
  const chart = { provenance: { granule_dates: ['2024-03-01'] } }
  assert.deepEqual(granuleDates(chart), ['2024-03-01'])
})

test('granuleDates is empty, not thrown, for a chart with no granule info', () => {
  assert.deepEqual(granuleDates({}), [])
})

// ── QA/trust status color mapping ───────────────────────────────────────────

test('maskingStatusColor maps verified/cf-deterministic to the green/success color', () => {
  const green = 'var(--success, #1a7f4b)'
  assert.equal(maskingStatusColor('verified'), green)
  assert.equal(maskingStatusColor('cf-deterministic'), green)
})

test('maskingStatusColor maps inferred to the yellow/warning color', () => {
  assert.equal(maskingStatusColor('inferred, not verified'), 'var(--warning, #b98900)')
})

test('maskingStatusColor maps ambiguous/not-applied to the red/error color', () => {
  const red = 'var(--error, #b42318)'
  assert.equal(maskingStatusColor('ambiguous — awaiting classification'), red)
  assert.equal(maskingStatusColor('not applied — semantics unknown'), red)
})

// ── Source dataset / citation ───────────────────────────────────────────────

test('citationString composes dataset, description, version, source, and collection id', () => {
  const chart = fixtureChart()
  assert.equal(
    citationString(chart.provenance),
    'TEMPO_NO2_L3, (TEMPO tropospheric NO2 vertical column), version V04, NASA LARC — TEMPO, Collection ID: C3685896708-LARC_CLOUD',
  )
})

test('citationString omits missing parts rather than leaving empty gaps', () => {
  assert.equal(citationString({ dataset: 'OMI_O3' }), 'OMI_O3')
  assert.equal(citationString({}), '')
  assert.equal(citationString(undefined), '')
})

test('datasetLandingUrl builds a CMR concept-id landing page, or null with no collection id', () => {
  assert.equal(
    datasetLandingUrl('C3685896708-LARC_CLOUD'),
    'https://cmr.earthdata.nasa.gov/search/concepts/C3685896708-LARC_CLOUD.html',
  )
  assert.equal(datasetLandingUrl(undefined), null)
  assert.equal(datasetLandingUrl(''), null)
})

// ── Details: Spatial section ────────────────────────────────────────────────

test('spatialFields reads region name and formats the bounding box', () => {
  const chart = fixtureChart()
  const fields = spatialFields(chart)
  assert.equal(fields.regionName, 'New Jersey')
  assert.equal(fields.bbox, '-75.5000, 39.0000, -73.5000, 41.0000')
})

test('spatialFields falls back to query.bbox when the chart has no top-level bounds (timeseries)', () => {
  const fields = spatialFields({ query: { bbox: [-1, -2, 1, 2] } })
  assert.equal(fields.bbox, '-1.0000, -2.0000, 1.0000, 2.0000')
})

test('spatialFields returns nulls, not throws, with no chart data', () => {
  assert.deepEqual(spatialFields({}), { regionName: null, bbox: null })
})

// Review fix: the Overview's Region field used to show only region_name,
// unlike the pre-T32 code's region_name-or-bbox fallback -- a chart with an
// empty region_name (e.g. a raw bbox selection) has real coordinates that
// should still surface instead of "Not available".
test('regionLabel prefers region_name when present', () => {
  assert.equal(regionLabel(fixtureChart()), 'New Jersey')
})

test('regionLabel falls back to the formatted bounding box when region_name is empty', () => {
  const chart = fixtureChart({ provenance: { ...fixtureChart().provenance, region_name: null } })
  assert.equal(regionLabel(chart), '-75.5000, 39.0000, -73.5000, 41.0000')
})

test('regionLabel is null when neither region_name nor a bbox is available', () => {
  assert.equal(regionLabel({}), null)
})

// ── Details: Temporal section ───────────────────────────────────────────────

test('temporalFields returns date range, cadence, and the full granule date list', () => {
  const chart = fixtureChart()
  assert.deepEqual(temporalFields(chart), {
    dateRange: '2024-01-01 to 2024-01-02',
    cadence: 'hourly',
    dates: ['2024-01-01', '2024-01-02'],
  })
})

// ── Details: Provenance / QA methodology section ────────────────────────────

test('qaMethodologyFields reads the pinned registry rule and the fill/valid-range tier', () => {
  const chart = fixtureChart()
  assert.deepEqual(qaMethodologyFields(chart), {
    qualityFlagVar: 'main_data_quality_flag',
    qaGoodValues: [0],
    qaBadValues: null,
    fillValueSource: 'collections_yaml',
    validRangeSource: 'collections_yaml',
  })
})

test('qaMethodologyFields is all null for an unregistered collection, not thrown', () => {
  assert.deepEqual(qaMethodologyFields({}), {
    qualityFlagVar: null, qaGoodValues: null, qaBadValues: null,
    fillValueSource: null, validRangeSource: null,
  })
})

test('resolveMaskingRaw checks top-level, provenance, then aggregation_meta in order', () => {
  assert.deepEqual(resolveMaskingRaw({ masking: { qa_status: 'verified' } }), { qa_status: 'verified' })
  assert.deepEqual(resolveMaskingRaw({ provenance: { masking: { qa_status: 'verified' } } }), { qa_status: 'verified' })
  assert.deepEqual(resolveMaskingRaw({ aggregation_meta: { masking: { qa_status: 'verified' } } }), { qa_status: 'verified' })
  assert.equal(resolveMaskingRaw({}), null)
})

// ── Details: Variable Definition section ────────────────────────────────────

test('variableDefinitionFields surfaces long_name/units/advisory notes/valid range/mask note', () => {
  const chart = fixtureChart()
  assert.deepEqual(variableDefinitionFields(chart), {
    longName: 'NO2 tropospheric column',
    units: 'molecules/cm^2',
    advisoryNotes: 'QA-flagged advisory note',
    validRange: '-1000000000000000 to 1000000000000000000',
    maskNote: 'fill values and a valid range are defined',
  })
})

test('variableDefinitionFields renders all-null when the backend attached nothing, not an exception', () => {
  assert.deepEqual(variableDefinitionFields({}), {
    longName: null, units: null, advisoryNotes: null, validRange: null, maskNote: null,
  })
})

test('variableDefinitionFields joins multiple advisory notes', () => {
  const fields = variableDefinitionFields({
    provenance: { variable_definition: { advisory_notes: ['note one', 'note two'] } },
  })
  assert.equal(fields.advisoryNotes, 'note one; note two')
})

// ── Details: Reproducibility section ────────────────────────────────────────

test('reproducibilityFields reads the query snapshot and joins source handles', () => {
  const chart = fixtureChart()
  const fields = reproducibilityFields(chart)
  assert.equal(fields.dataset, 'vertical_column_troposphere')
  assert.equal(fields.startDate, '2024-01-01T00:00:00')
  assert.equal(fields.bbox, '-75.5000, 39.0000, -73.5000, 41.0000')
  assert.equal(fields.sourceHandles, 'obs_1')
})

// The "copy query" button writes this object to the clipboard verbatim --
// query snapshot plus the source handles it was built from.
test('reproducibilityQuery is the exact object the copy-query button writes', () => {
  const chart = fixtureChart()
  assert.deepEqual(reproducibilityQuery(chart), {
    dataset: 'vertical_column_troposphere',
    start_date: '2024-01-01T00:00:00',
    end_date: '2024-01-02T00:00:00',
    bbox: [-75.5, 39.0, -73.5, 41.0],
    aggregation: 'Daily Mean, 2 hourly granules, 2024-01-01 to 2024-01-02',
    chart_parameters: { chart_type: 'heatmap' },
    source_handles: ['obs_1'],
  })
})

test('reproducibilityQuery still returns a well-shaped object with no query or handles', () => {
  assert.deepEqual(reproducibilityQuery({}), { source_handles: [] })
})

// ── Details: raw JSON toggle ─────────────────────────────────────────────────

test('rawMetadataJson renders provenance and aggregation_meta verbatim', () => {
  const chart = fixtureChart()
  assert.deepEqual(rawMetadataJson(chart), {
    provenance: chart.provenance,
    aggregation_meta: chart.aggregation_meta,
  })
})

test('rawMetadataJson is null-safe for a chart with neither field', () => {
  assert.deepEqual(rawMetadataJson({}), { provenance: null, aggregation_meta: null })
})
