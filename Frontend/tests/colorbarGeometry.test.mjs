import assert from 'node:assert/strict'
import test from 'node:test'
import { colorbarGeometry, scaleClipNote } from '../src/utils/colorbarGeometry.js'

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

test('a percentile scale discloses the clip so saturated extremes are not read as the true range', () => {
  const note = scaleClipNote({ method: 'percentile', p: [2, 98] })

  assert.match(note, /clipped/i)
  assert.match(note, /2nd/)
  assert.match(note, /98th/)
})

test('an explicit (caller-imposed) scale has no clip to disclose', () => {
  assert.equal(scaleClipNote({ method: 'explicit' }), null)
})

test('a payload with no scale field (older charts) discloses nothing', () => {
  assert.equal(scaleClipNote(undefined), null)
  assert.equal(scaleClipNote(null), null)
})
