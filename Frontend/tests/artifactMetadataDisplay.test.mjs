import test from 'node:test'
import assert from 'node:assert/strict'

import { NOT_AVAILABLE } from '../src/utils/metadataDisplay.js'
import {
  tableOverviewFields, tableDetailsFields,
  groundValidationOverviewFields, groundValidationDetailsFields,
  rawArtifactMetadataJson,
} from '../src/utils/artifactMetadataDisplay.js'

// ── Table fixtures ───────────────────────────────────────────────────────

function tableArtifact(overrides = {}) {
  return {
    id: 'tbl_1',
    type: 'table',
    title: 'EPA Daily Summary',
    row_count: 1200,
    metadata: { endpoint: 'dailyData', granularity: 'daily' },
    ...overrides,
  }
}

function tablePage(overrides = {}) {
  return {
    columns: ['date_local', 'arithmetic_mean', 'units_of_measure'],
    total_rows: 1200,
    ...overrides,
  }
}

test('tableOverviewFields reads row/column counts from the fetched page', () => {
  const fields = tableOverviewFields(tableArtifact(), tablePage())
  assert.equal(fields.rowCount, 1200)
  assert.equal(fields.columnCount, 3)
})

test('tableOverviewFields falls back to artifact.row_count when no page is loaded yet', () => {
  const fields = tableOverviewFields(tableArtifact(), null)
  assert.equal(fields.rowCount, 1200)
  assert.equal(fields.columnCount, null)
})

// EPA AQS table tools don't set a "dataset" field on their metadata (verified
// against Backend/tools/ground_sensor_tools/epa_aqs_tools.py) -- Overview
// must render "Not available" for it, not hide the field or throw.
test('tableOverviewFields is null (renders "Not available") for a producing tool that sets no source dataset', () => {
  const fields = tableOverviewFields(tableArtifact(), tablePage())
  assert.equal(fields.sourceDataset, null)
})

test('tableOverviewFields surfaces a source dataset when the producing tool sets one', () => {
  const fields = tableOverviewFields(tableArtifact({ metadata: { dataset: 'EPA_AQS' } }), tablePage())
  assert.equal(fields.sourceDataset, 'EPA_AQS')
})

test('tableDetailsFields returns the full column list', () => {
  const fields = tableDetailsFields(tableArtifact(), tablePage())
  assert.deepEqual(fields.columns, ['date_local', 'arithmetic_mean', 'units_of_measure'])
})

test('tableDetailsFields is empty, not thrown, with no page loaded', () => {
  assert.deepEqual(tableDetailsFields(tableArtifact(), null).columns, [])
})

// ── Ground-validation timeseries fixtures ───────────────────────────────

function validateArtifact(overrides = {}) {
  return {
    id: 'ts_overlay1',
    type: 'timeseries',
    title: 'TEMPO NO2 vs 34-017-0006',
    metadata: {
      series: [
        { label: 'TEMPO NO2 (satellite)', source_kind: 'satellite' },
        { label: 'EPA monitor 34-017-0006', source_kind: 'ground', station_id: '34-017-0006' },
      ],
      source_handles: ['cube_1'],
      stats: { r: 0.87, n: 12, coverage_fraction: 0.6 },
      coverage: { n_total: 20, n_valid: 12, n_excluded: 8, coverage_fraction: 0.6 },
      exceedance_dates: null,
    },
    ...overrides,
  }
}

function exceedanceArtifact() {
  return validateArtifact({
    id: 'ts_exceed1',
    title: 'TEMPO NO2 with 34-017-0006 exceedance days',
    metadata: {
      series: [{ label: 'TEMPO NO2 (satellite)', source_kind: 'satellite' }],
      source_handles: ['cube_2'],
      stats: null,
      coverage: { n_total: 3, n_valid: 3, n_excluded: 0, coverage_fraction: 1.0 },
      exceedance_dates: ['2024-01-02', '2024-01-05'],
    },
  })
}

test('groundValidationOverviewFields reads the satellite variable and ground station(s) from series', () => {
  const fields = groundValidationOverviewFields(validateArtifact())
  assert.equal(fields.satelliteVariable, 'TEMPO NO2 (satellite)')
  assert.equal(fields.groundStations, 'EPA monitor 34-017-0006 (34-017-0006)')
})

test('groundValidationOverviewFields builds a correlation summary from stats', () => {
  const fields = groundValidationOverviewFields(validateArtifact())
  assert.match(fields.correlationSummary, /r = 0\.87/)
  assert.match(fields.correlationSummary, /n=12/)
})

test('groundValidationOverviewFields falls back to a coverage-only summary when stats.r is absent (exceedance_overlay)', () => {
  const fields = groundValidationOverviewFields(exceedanceArtifact())
  assert.match(fields.correlationSummary, /100%/)
})

test('groundValidationOverviewFields is null-safe for a missing series/stats/coverage', () => {
  const fields = groundValidationOverviewFields({ id: 'ts_x', type: 'timeseries', metadata: {} })
  assert.equal(fields.satelliteVariable, null)
  assert.equal(fields.groundStations, null)
  assert.equal(fields.correlationSummary, null)
  assert.equal(fields.qaStatus, null)
})

// Review fix (T34): validate_against_ground/exceedance_overlay now populate
// masking provenance (Backend/services/artifact_registry.py forwards
// ts_payload["masking"] into TimeseriesArtifactMetadata) so the "Data
// quality" field has something real to show.
test('groundValidationOverviewFields reads qaStatus from meta.masking', () => {
  const fields = groundValidationOverviewFields(validateArtifact({
    metadata: {
      ...validateArtifact().metadata,
      masking: { qa_status: 'verified', qa_source: 'collections_yaml', qa_note: '' },
    },
  }))
  assert.equal(fields.qaStatus, 'verified')
})

test('groundValidationDetailsFields exposes the full series breakdown and source handles', () => {
  const fields = groundValidationDetailsFields(validateArtifact())
  assert.equal(fields.series.length, 2)
  assert.equal(fields.sourceHandles, 'cube_1')
})

test('groundValidationDetailsFields carries exceedance dates for exceedance_overlay results', () => {
  const fields = groundValidationDetailsFields(exceedanceArtifact())
  assert.deepEqual(fields.exceedanceDates, ['2024-01-02', '2024-01-05'])
  assert.equal(fields.stats, null)
})

test('groundValidationDetailsFields carries correlation stats for validate_against_ground results', () => {
  const fields = groundValidationDetailsFields(validateArtifact())
  assert.equal(fields.exceedanceDates, null)
  assert.deepEqual(fields.stats, { r: 0.87, n: 12, coverage_fraction: 0.6 })
})

// ── Raw JSON toggle ──────────────────────────────────────────────────────

test('rawArtifactMetadataJson renders artifact.metadata verbatim', () => {
  const artifact = validateArtifact()
  assert.deepEqual(rawArtifactMetadataJson(artifact), artifact.metadata)
})

test('rawArtifactMetadataJson is null-safe', () => {
  assert.equal(rawArtifactMetadataJson({}), null)
  assert.equal(rawArtifactMetadataJson(undefined), null)
})

test('NOT_AVAILABLE constant stays the shared "Not available" string used across metadata surfaces', () => {
  assert.equal(NOT_AVAILABLE, 'Not available')
})
