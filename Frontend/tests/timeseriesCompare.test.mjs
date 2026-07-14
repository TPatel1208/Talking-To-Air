import assert from 'node:assert/strict'
import test from 'node:test'
import {
  allRangesOverlap,
  buildOverlayTraces,
  seriesLabel,
  timeseriesOverlayCompatible,
  toOverlaySeries,
} from '../src/utils/timeseriesCompare.js'

const stationA = {
  type: 'timeseries',
  title: 'NO2 mean over Station A',
  units: 'mol/m^2',
  times: ['2024-01-01T00:00:00', '2024-01-02T00:00:00', '2024-01-03T00:00:00'],
  values: [1, 2, 3],
  provenance: { region_name: 'Station A', start_date: '2024-01-01', end_date: '2024-01-03' },
}

const stationB = {
  type: 'timeseries',
  title: 'NO2 mean over Station B',
  units: 'mol/m^2',
  times: ['2024-01-02T00:00:00', '2024-01-04T00:00:00'],
  values: [5, 6],
  provenance: { region_name: 'Station B', start_date: '2024-01-02', end_date: '2024-01-04' },
}

const windSpeed = {
  type: 'timeseries',
  title: 'Wind speed over Station A',
  units: 'm/s',
  times: ['2024-01-01T00:00:00', '2024-01-03T00:00:00'],
  values: [3, 4],
  provenance: { region_name: 'Station A', start_date: '2024-01-01', end_date: '2024-01-03' },
}

const futureStation = {
  type: 'timeseries',
  title: 'NO2 mean over Station C',
  units: 'mol/m^2',
  times: ['2025-06-01T00:00:00', '2025-06-02T00:00:00'],
  values: [9, 10],
  provenance: { region_name: 'Station C', start_date: '2025-06-01', end_date: '2025-06-02' },
}

test('matching units + overlapping ranges are compatible -- overlay', () => {
  const result = timeseriesOverlayCompatible([stationA, stationB])
  assert.equal(result.compatible, true)
  assert.equal(result.reason, null)
})

test('mismatched units are incompatible -- grid fallback', () => {
  const result = timeseriesOverlayCompatible([stationA, windSpeed])
  assert.equal(result.compatible, false)
  assert.match(result.reason, /different units/i)
})

test('non-overlapping time ranges are incompatible -- grid fallback', () => {
  const result = timeseriesOverlayCompatible([stationA, futureStation])
  assert.equal(result.compatible, false)
  assert.match(result.reason, /overlap/i)
})

test('a shared single point of overlap counts as compatible -- binary rule, no partial-overlap heuristic', () => {
  const touching = { ...stationB, times: ['2024-01-03T00:00:00', '2024-01-05T00:00:00'], values: [7, 8] }
  const result = timeseriesOverlayCompatible([stationA, touching])
  assert.equal(result.compatible, true)
})

test('every pair must overlap, not just adjacent ones', () => {
  // A and B overlap, B and C overlap, but A and C do not -- still incompatible.
  const early = { ...stationA, times: ['2024-01-01T00:00:00', '2024-01-02T00:00:00'] }
  const middle = { ...stationB, times: ['2024-01-02T00:00:00', '2024-01-05T00:00:00'] }
  const late = { ...futureStation, units: 'mol/m^2', times: ['2024-01-05T00:00:00', '2024-01-06T00:00:00'] }
  assert.equal(allRangesOverlap([early, middle, late]), false)
})

test('fewer than two filled charts is never compatible -- nothing to overlay', () => {
  assert.equal(timeseriesOverlayCompatible([]).compatible, false)
  assert.equal(timeseriesOverlayCompatible([stationA]).compatible, false)
  assert.equal(timeseriesOverlayCompatible([stationA, null]).compatible, false)
})

test('seriesLabel prefers the region name, then title, then a generic fallback', () => {
  assert.equal(seriesLabel(stationA, 0), 'Station A')
  assert.equal(seriesLabel({ title: 'Untitled', provenance: {} }, 1), 'Untitled')
  assert.equal(seriesLabel({}, 2), 'Series 3')
})

test('toOverlaySeries maps filled charts to {times, values, label, units}, dropping empty slots', () => {
  const series = toOverlaySeries([stationA, null, stationB])
  assert.equal(series.length, 2)
  assert.deepEqual(series[0], { times: stationA.times, values: stationA.values, label: 'Station A', units: 'mol/m^2' })
  assert.equal(series[1].label, 'Station B')
})

test('buildOverlayTraces produces one trace with one legend-visible name per series', () => {
  const series = toOverlaySeries([stationA, stationB])
  const traces = buildOverlayTraces(series)
  assert.equal(traces.length, 2)
  assert.deepEqual(traces.map(t => t.name), ['Station A', 'Station B'])
  assert.ok(traces.every(t => t.type === 'scatter'))
  // distinguishable colors -- no two series should share a line color
  assert.notEqual(traces[0].line.color, traces[1].line.color)
})
