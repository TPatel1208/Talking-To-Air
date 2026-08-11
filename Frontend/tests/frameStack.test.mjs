import assert from 'node:assert/strict'
import test from 'node:test'
import { decodeFrameStack, selectFrame } from '../src/utils/frameStack.js'

// [1+N, ny, nx] = [3, 2, 3]: the period mean plus two buckets over a 2x3 grid.
const block = { shape: [3, 2, 3], dtype: 'float32', period_index: 0, cells_per_frame: 6 }

function stackBytes(planes) {
  return Float32Array.from(planes.flat()).buffer
}

const bytes = stackBytes([
  [1, 2, 3, 4, 5, 6],
  [10, 20, 30, 40, 50, 60],
  [100, 200, 300, 400, 500, 600],
])

test('the grid comes off the payload shape, never off the array length', () => {
  const stack = decodeFrameStack(bytes, block)

  assert.equal(stack.planeCount, 3)
  assert.equal(stack.height, 2)
  assert.equal(stack.width, 3)
})

test('a plane is a view over the one flat array, not a copy of it', () => {
  // Phase 2 measured this: 100 `subarray` views cost +0.00 MB, 100 `slice`
  // copies cost +7.64 MB -- a full duplicate of the stack.
  const stack = decodeFrameStack(bytes, block)
  const plane = stack.planeView(1)

  assert.equal(plane.buffer, stack.values.buffer)
  assert.equal(plane.length, 6)
  assert.deepEqual(Array.from(plane), [10, 20, 30, 40, 50, 60])
})

test('each plane starts where the payload says it does', () => {
  const stack = decodeFrameStack(bytes, block)

  assert.equal(stack.planeView(0).byteOffset, 0)
  assert.equal(stack.planeView(1).byteOffset, 6 * 4)
  assert.equal(stack.planeView(2).byteOffset, 12 * 4)
})

test('100 views over a real-sized stack cost nothing in arrayBuffers', () => {
  // Budget from `arrayBuffers`, never `heapUsed` -- a typed array's storage is
  // not on the JS heap, so a 100-frame stack reads as heapUsed +0.07 MB and
  // arrayBuffers +7.64 MB. A memory check written against heapUsed reports the
  // frame store as free, and would pass just as happily against `slice`.
  const cells = 19107
  const big = new ArrayBuffer(100 * cells * 4)
  const stack = decodeFrameStack(big, { shape: [100, 99, 193], dtype: 'float32', period_index: 0 })

  const before = process.memoryUsage().arrayBuffers
  // Retained, so a copying implementation cannot be rescued by GC running
  // before the sample.
  const views = []
  for (let i = 0; i < 100; i++) views.push(stack.planeView(i))
  const after = process.memoryUsage().arrayBuffers

  assert.equal(views.length, 100)
  assert.ok(after - before < 1024 * 1024, `views allocated ${(after - before) / 1e6} MB; expected ~0`)
})

test('a truncated blob is refused, not reshaped into a plausible field', () => {
  // Half a stack still reshapes cleanly into whole frames and renders as a
  // believable map with the tail of the scrub silently absent. That is the
  // failure the backend records a length and a digest against; the client must
  // not undo it by rendering what fits.
  const short = bytes.slice(0, 12 * 4)

  assert.throws(() => decodeFrameStack(short, block), /12 .*18|expected/i)
})

test('a stack the payload does not describe is refused', () => {
  assert.equal(decodeFrameStack(bytes, { shape: [3, 2], dtype: 'float32' }), null)
  assert.equal(decodeFrameStack(bytes, { shape: [3, 2, 3], dtype: 'float64' }), null)
  assert.equal(decodeFrameStack(null, block), null)
})

// ── Selecting the frame the canvas draws ─────────────────────────────────────

const chart = {
  type: 'heatmap',
  frames: { ...block, lats: [10, 11], lons: [20, 21, 22], url: '/chart/c1/frames.f32.gz' },
}

test('the selected stop draws its own plane, on the frame stack\'s own grid', () => {
  const stack = decodeFrameStack(bytes, block)
  const frame = selectFrame(chart, stack, { plane: 2 })

  assert.deepEqual(Array.from(frame.values), [100, 200, 300, 400, 500, 600])
  assert.equal(frame.values.buffer, stack.values.buffer)
  // The frame grid, NOT the payload's thinned lats/lons: the stack is
  // block-meaned to a 20,000-cell ceiling and has a grid of its own.
  assert.deepEqual(frame.lats, [10, 11])
  assert.deepEqual(frame.lons, [20, 21, 22])
})

test('a grid that disagrees with the blob shape draws nothing', () => {
  // The canvas sizes itself from lats/lons and indexes by row*width+col. If
  // the axis and the pixels disagree about the grid, every value lands at the
  // wrong coordinate -- a map that looks fine and is wrong everywhere.
  const stack = decodeFrameStack(bytes, block)
  const skewed = { frames: { ...chart.frames, lons: [20, 21] } }

  assert.equal(selectFrame(skewed, stack, { plane: 1 }), null)
})

test('nothing to draw before there is a stack or a stop', () => {
  const stack = decodeFrameStack(bytes, block)

  assert.equal(selectFrame(chart, null, { plane: 1 }), null)
  assert.equal(selectFrame(chart, stack, null), null)
  assert.equal(selectFrame(chart, stack, { plane: 99 }), null)
})
