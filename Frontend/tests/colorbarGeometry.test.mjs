import assert from 'node:assert/strict'
import test from 'node:test'
import { colorbarGeometry } from '../src/utils/colorbarGeometry.js'

const lut = [
  [68, 1, 84, 255],
  [59, 82, 139, 255],
  [253, 231, 37, 255],
]

test('gradient stops depict exactly the LUT colors, in order, so the legend never lies about the pixels', () => {
  const { gradientStops } = colorbarGeometry({ vmin: 0, vmax: 10, lut })

  assert.equal(gradientStops.length, lut.length)
  assert.deepEqual(gradientStops.map(s => s.color), [
    'rgba(68, 1, 84, 1.000)',
    'rgba(59, 82, 139, 1.000)',
    'rgba(253, 231, 37, 1.000)',
  ])
  assert.deepEqual(gradientStops.map(s => s.offset), [0, 0.5, 1])
})

test('ticks are evenly spaced from vmin to vmax, with the caps matching the range exactly', () => {
  const { ticks } = colorbarGeometry({ vmin: 0, vmax: 20, lut, tickCount: 5 })

  assert.deepEqual(ticks.map(t => t.value), [0, 5, 10, 15, 20])
  assert.deepEqual(ticks.map(t => t.position), [0, 0.25, 0.5, 0.75, 1])
})
