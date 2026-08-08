import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveCsvExport, resolveNetcdfExport } from '../src/utils/chartExport.js'

test('a heatmap without full-resolution export metadata offers no CSV at all', () => {
  // The payload's grid is thinned to <=8000 cells for rendering, so a CSV
  // built from it is a stride sample -- 8000 of 22.3M cells on a TEMPO CONUS
  // scene, 0.036% of the field. It used to download under the same filename
  // and report the same "Export complete" as the real thing, so nothing told
  // the reader which one they had. Refusing is the honest answer: the missing
  // values are not recoverable client-side.
  const chart = {
    type: 'heatmap',
    lats: [1, 2],
    lons: [3, 4],
    values: [[1, 2], [3, 4]],
  }

  const resolved = resolveCsvExport(chart)

  assert.equal(resolved.kind, 'unavailable')
  assert.match(resolved.message, /full-resolution/i)
})

test('a heatmap with export metadata exports from the server at full resolution', () => {
  const chart = {
    type: 'heatmap',
    chart_id: 'abc123',
    export: { type: 'heatmap', source_handles: ['cube_xyz'] },
    lats: [1, 2],
    lons: [3, 4],
    values: [[1, 2], [3, 4]],
  }

  const resolved = resolveCsvExport(chart)

  assert.equal(resolved.kind, 'server')
  assert.equal(resolved.url, '/api/chart/abc123/export.csv')
})

test('a timeseries without export metadata still exports client-side, complete', () => {
  // The heatmap fallback was removed because the grid it reads is thinned.
  // A timeseries payload is NOT thinned -- it ships every time and value it
  // has -- so a client-side CSV of one is the whole series, not a sample of
  // it. That difference is the entire reason this branch survives.
  const chart = {
    type: 'timeseries',
    variable: 'NO2',
    units: 'mol/cm2',
    stat: 'mean',
    times: ['2024-06-01', '2024-06-02', '2024-06-03'],
    values: [1.5, 2.5, 3.5],
  }

  const resolved = resolveCsvExport(chart)

  assert.equal(resolved.kind, 'client')
  assert.equal(resolved.rows.length, 3)
  assert.deepEqual(resolved.rows[0], {
    variable: 'NO2',
    time: '2024-06-01',
    stat: 'mean',
    value: 1.5,
    units: 'mol/cm2',
  })
  assert.deepEqual(resolved.rows.map(r => r.value), [1.5, 2.5, 3.5])
})

test('a chart with source handles can export NetCDF from the server', () => {
  // NetCDF is the right artifact for a gridded field: it keeps the grid
  // structure, CF metadata, units and attributes that a lat/lon/value CSV
  // throws away. The endpoint has existed since T10 and nothing ever called
  // it. Availability mirrors the backend's own gate (_chart_source_handles):
  // metadata.source_handles, falling back to provenance.source_handles.
  const chart = {
    type: 'heatmap',
    chart_id: 'abc123',
    metadata: { source_handles: ['cube_xyz'] },
  }

  const resolved = resolveNetcdfExport(chart)

  assert.equal(resolved.kind, 'server')
  assert.equal(resolved.url, '/api/chart/abc123/export.nc')
})

test('NetCDF export falls back to provenance source handles', () => {
  const chart = {
    type: 'heatmap',
    chart_id: 'abc123',
    provenance: { source_handles: ['cube_xyz'] },
  }

  assert.equal(resolveNetcdfExport(chart).kind, 'server')
})

test('a multi-source chart offers no NetCDF export', () => {
  // /chart/{id}/export.nc converts source_handles[0] and only that one. A
  // comparison carries [region_a, region_b, aligned], so offering the button
  // would hand the reader region A's granule named after the whole
  // comparison -- an artifact that misrepresents its own provenance, with
  // nothing on screen or in the filename to catch it.
  const chart = {
    type: 'heatmap_multi',
    chart_id: 'abc123',
    metadata: { source_handles: ['cube_a', 'cube_b', 'cube_aligned'] },
  }

  const resolved = resolveNetcdfExport(chart)

  assert.equal(resolved.kind, 'unavailable')
  assert.match(resolved.message, /3 sources/)
})

test('a chart with no source handles offers no NetCDF export', () => {
  // The backend 422s this case ("does not include a source handle to
  // export"), so offering the button would only produce a failed download.
  const chart = { type: 'heatmap', chart_id: 'abc123', metadata: { source_handles: [] } }

  const resolved = resolveNetcdfExport(chart)

  assert.equal(resolved.kind, 'unavailable')
  assert.match(resolved.message, /source handle/i)
})
