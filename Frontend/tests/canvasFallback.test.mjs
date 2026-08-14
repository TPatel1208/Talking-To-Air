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

test('renders a flat float32 view identically to the nested rows (T59)', () => {
  // The frame stack hands this a `subarray` VIEW over one flat Float32Array
  // (T59 D2/Phase 2) rather than the nested rows a heatmap payload ships. NaN
  // rides in the float32 natively -- D13's "NaN needs no sentinel" -- and must
  // land on the same fully transparent pixel a null does.
  const nested = buildCanvasFallbackFrame({
    lats: [0, 1], lons: [0, 1],
    values: [[0, null], [10, 5]],
    vmin: 0, vmax: 10, lut,
  })

  const stack = new Float32Array([0, NaN, 10, 5, 999, 999])
  const flat = buildCanvasFallbackFrame({
    lats: [0, 1], lons: [0, 1],
    values: stack.subarray(0, 4),
    vmin: 0, vmax: 10, lut,
  })

  assert.equal(flat.width, 2)
  assert.equal(flat.height, 2)
  assert.deepEqual(Array.from(flat.pixels), Array.from(nested.pixels))
})

test('a flat view reads only its own plane, never past the end of it', () => {
  // The view is a window into a stack of many frames. Reading by
  // `row * width + col` off the VIEW keeps frame 2's first row out of frame
  // 1's last one; reading off the whole array would not.
  const stack = new Float32Array([0, 10, 10, 0, 5, 5, 5, 5])
  const second = buildCanvasFallbackFrame({
    lats: [0, 1], lons: [0, 1],
    values: stack.subarray(4, 8),
    vmin: 0, vmax: 10, lut,
  })

  // Every cell of plane 2 is 5, so every pixel is the same colour -- and it is
  // a real colour, not the transparent zeros a read past the view would give.
  for (let px = 0; px < 4; px++) {
    assert.deepEqual(Array.from(second.pixels.slice(px * 4, px * 4 + 4)), [253, 231, 37, 255])
  }
})
