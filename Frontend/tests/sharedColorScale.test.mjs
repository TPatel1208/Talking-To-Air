import assert from 'node:assert/strict'
import test from 'node:test'
import { computeSharedColorScale } from '../src/utils/sharedColorScale.js'

const NO2 = (overrides = {}) => ({
  type: 'heatmap',
  variable: 'NO2',
  units: 'mol/m^2',
  vmin: 0,
  vmax: 10,
  colormap: { name: 'viridis', lut: [[0, 0, 0, 255], [255, 255, 255, 255]] },
  ...overrides,
})

test('matching variable/units across panels yields the pooled min/max and the first panel\'s colormap', () => {
  const a = NO2({ vmin: 0, vmax: 10 })
  const b = NO2({ vmin: -2, vmax: 6 })
  const c = NO2({ vmin: 1, vmax: 20 })

  const result = computeSharedColorScale([a, b, c])

  assert.equal(result.available, true)
  assert.equal(result.vmin, -2)
  assert.equal(result.vmax, 20)
  assert.equal(result.colormap, a.colormap)
  assert.equal(result.reason, null)
})

test('reads variable/units from provenance when not present at the top level', () => {
  const a = { type: 'heatmap', provenance: { variable: 'O3', units: 'ppb' }, vmin: 0, vmax: 5, colormap: { name: 'x' } }
  const b = { type: 'heatmap', provenance: { variable: 'O3', units: 'ppb' }, vmin: 2, vmax: 9, colormap: { name: 'x' } }

  const result = computeSharedColorScale([a, b])

  assert.equal(result.available, true)
  assert.equal(result.vmin, 0)
  assert.equal(result.vmax, 9)
})

test('mismatched variables leave shared scaling unavailable and independent scales untouched', () => {
  const no2 = NO2()
  const wind = { type: 'heatmap', variable: 'wind_speed', units: 'm/s', vmin: 0, vmax: 15, colormap: { name: 'plasma' } }

  const result = computeSharedColorScale([no2, wind])

  assert.equal(result.available, false)
  assert.equal(result.vmin, null)
  assert.equal(result.vmax, null)
  assert.equal(result.colormap, null)
  assert.match(result.reason, /different variables/i)
})

test('mismatched units on the same variable name also forces independent scales', () => {
  const a = NO2({ units: 'mol/m^2' })
  const b = NO2({ units: 'molecules/cm^2' })

  const result = computeSharedColorScale([a, b])

  assert.equal(result.available, false)
  assert.match(result.reason, /different variables/i)
})

test('fewer than two filled panels is never "available" -- nothing to share a scale across', () => {
  assert.equal(computeSharedColorScale([]).available, false)
  assert.equal(computeSharedColorScale([NO2()]).available, false)
  assert.equal(computeSharedColorScale([NO2(), null]).available, false)
})
