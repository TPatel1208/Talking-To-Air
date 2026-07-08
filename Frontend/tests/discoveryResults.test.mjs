import assert from 'node:assert/strict'
import test from 'node:test'
import { normalizeSearchResults } from '../src/utils/discoveryResults.js'

// The real MCP shape (live-verified 2026-07-08): datasets/handle/summary-object.
// This is the exact response that used to blank the pane because the code read
// `data.results` and `dataset.dataset_handle`.
const realSearchResponse = {
  count: 2,
  datasets: [
    {
      handle: 'dataset_fbe6db02c44e5f9e',
      summary: {
        concept_id: 'C2872767107-NSIDC_CPRD',
        short_name: 'SPL3SMAP',
        version: '002',
        entry_title: 'SMAP L3 Radar/Radiometer Global Daily 9 km Soil Moisture V002',
        processing_level: '3',
        advisory_notes: [],
      },
    },
    {
      handle: 'dataset_e617f2f2bc3039e4',
      summary: {
        concept_id: 'C2548143472-FEDEO',
        short_name: '057dd6c36f0741d3',
        version: 'NA',
        entry_title: 'ESA Soil Moisture CCI COMBINED Product, Version 05.2',
        processing_level: 'NA',
        advisory_notes: [],
      },
    },
  ],
}

test('maps the real MCP datasets/handle/summary-object shape into renderable cards', () => {
  const rows = normalizeSearchResults(realSearchResponse)
  assert.equal(rows.length, 2)
  assert.equal(rows[0].dataset_handle, 'dataset_fbe6db02c44e5f9e')
  // summary must be a string title (rendering the raw object crashes React).
  assert.equal(typeof rows[0].summary, 'string')
  assert.equal(rows[0].summary, 'SMAP L3 Radar/Radiometer Global Daily 9 km Soil Moisture V002')
  assert.equal(rows[0].short_name, 'SPL3SMAP')
  assert.equal(rows[0].processing_level, '3')
  assert.equal(rows[0].version, '002')
})

test("drops 'NA' placeholder version/level rather than showing them", () => {
  const rows = normalizeSearchResults(realSearchResponse)
  assert.equal(rows[1].version, undefined)
  assert.equal(rows[1].processing_level, undefined)
})

test('still tolerates the older results/dataset_handle/string-summary fixture shape', () => {
  const rows = normalizeSearchResults({
    results: [{ dataset_handle: 'dataset_smap_l3', summary: 'Soil moisture, L3 daily', provider: 'NSIDC' }],
  })
  assert.equal(rows.length, 1)
  assert.equal(rows[0].dataset_handle, 'dataset_smap_l3')
  assert.equal(rows[0].summary, 'Soil moisture, L3 daily')
  assert.equal(rows[0].provider, 'NSIDC')
})

test('falls back to the handle as the title when no summary is present', () => {
  const rows = normalizeSearchResults({ datasets: [{ handle: 'dataset_x' }] })
  assert.equal(rows[0].summary, 'dataset_x')
})

test('returns an empty array for missing/empty/malformed responses', () => {
  assert.deepEqual(normalizeSearchResults(undefined), [])
  assert.deepEqual(normalizeSearchResults({}), [])
  assert.deepEqual(normalizeSearchResults({ datasets: null }), [])
  assert.deepEqual(normalizeSearchResults({ datasets: 'nope' }), [])
})
