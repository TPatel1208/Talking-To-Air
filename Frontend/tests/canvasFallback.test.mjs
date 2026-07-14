import assert from 'node:assert/strict'
import test from 'node:test'
import { buildCanvasFallbackFrame } from '../src/utils/canvasFallback.js'

const lut = [
  [68, 1, 84, 255],
  [253, 231, 37, 255],
]

test('colors each valid cell from the LUT and leaves no-data cells fully transparent', () => {
  const frame = buildCanvasFallbackFrame({
    lats: [0, 1],
    lons: [0, 1],
    values: [
      [0, null],
      [10, 5],
    ],
    vmin: 0,
    vmax: 10,
    lut,
  })

  assert.equal(frame.width, 2)
  assert.equal(frame.height, 2)
  // row 0: [0, null] -> [vmin color, transparent]
  assert.deepEqual(Array.from(frame.pixels.slice(0, 4)), [68, 1, 84, 255])
  assert.deepEqual(Array.from(frame.pixels.slice(4, 8)), [0, 0, 0, 0])
  // row 1: [10, 5] -> [vmax color, midpoint color]
  assert.deepEqual(Array.from(frame.pixels.slice(8, 12)), [253, 231, 37, 255])
})
