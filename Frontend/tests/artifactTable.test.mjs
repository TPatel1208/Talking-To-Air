import assert from 'node:assert/strict'
import test from 'node:test'
import { sortArtifactRows } from '../src/utils/artifactTable.js'

const rows = [
  { date: '2024-01-03', value: 2, site: 'Site 10' },
  { date: '2024-01-01', value: 12, site: 'Site 2' },
  { date: '2024-01-02', value: 4, site: 'Site 1' },
]

test('sorts artifact table rows ascending by numeric value', () => {
  assert.deepEqual(
    sortArtifactRows(rows, 'value', 'asc').map(row => row.value),
    [2, 4, 12],
  )
})

test('sorts artifact table rows descending by numeric value', () => {
  assert.deepEqual(
    sortArtifactRows(rows, 'value', 'desc').map(row => row.value),
    [12, 4, 2],
  )
})

test('sorts artifact table rows with natural string comparison', () => {
  assert.deepEqual(
    sortArtifactRows(rows, 'site', 'asc').map(row => row.site),
    ['Site 1', 'Site 2', 'Site 10'],
  )
})
