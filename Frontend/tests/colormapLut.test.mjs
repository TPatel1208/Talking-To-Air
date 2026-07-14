import assert from 'node:assert/strict'
import test from 'node:test'
import { colorForValue } from '../src/utils/colormapLut.js'

const lut = [
  [68, 1, 84, 255],
  [59, 82, 139, 255],
  [33, 145, 140, 255],
  [94, 201, 98, 255],
  [253, 231, 37, 255],
]

test('maps a mid-range value to the interpolated LUT stop', () => {
  assert.deepEqual(colorForValue(5, { vmin: 0, vmax: 10, lut }), [33, 145, 140, 255])
})

test('clamps out-of-range values to the LUT end stops instead of extrapolating', () => {
  assert.deepEqual(colorForValue(-50, { vmin: 0, vmax: 10, lut }), [68, 1, 84, 255])
  assert.deepEqual(colorForValue(500, { vmin: 0, vmax: 10, lut }), [253, 231, 37, 255])
})

test('returns null for null/NaN/undefined values so no-data stays transparent, never invented', () => {
  assert.equal(colorForValue(null, { vmin: 0, vmax: 10, lut }), null)
  assert.equal(colorForValue(undefined, { vmin: 0, vmax: 10, lut }), null)
  assert.equal(colorForValue(NaN, { vmin: 0, vmax: 10, lut }), null)
})
