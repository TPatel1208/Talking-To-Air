import assert from 'node:assert/strict'
import test from 'node:test'
import { nearestCell } from '../src/utils/heatmapHover.js'

const payload = {
  points: {
    lats: [10, 20, 30],
    lons: [100, 110, 120],
    values: [1.5, 2.5, 3.5],
  },
}

test('returns the value/coords of the closest shipped point to a hover position', () => {
  assert.deepEqual(nearestCell(109, 19, payload), { lat: 20, lon: 110, value: 2.5 })
})

test('returns null when the payload has no usable points', () => {
  assert.equal(nearestCell(0, 0, { points: { lats: [], lons: [], values: [] } }), null)
  assert.equal(nearestCell(0, 0, {}), null)
})
