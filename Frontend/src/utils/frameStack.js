// The frame blob, laid out for rendering (T59 Phase 6, deliverables 1 and 2).
//
// `GET /chart/{id}/frames.f32.gz` serves the stack with `Content-Encoding:
// gzip`, so the browser inflates it on the wire and what arrives here is raw
// little-endian float32 -- there is no JS decompressor in this path and must
// not be one.
//
// ONE flat `Float32Array` over the whole stack, and every frame is a
// `subarray` VIEW into it. Phase 2 measured the alternative: 100 views cost
// +0.00 MB, 100 `slice` copies cost +7.64 MB, a full duplicate. Hand the
// canvas a view, never a copy.
//
// The layout is read off the payload -- `shape` is [1+N, ny, nx] and
// `period_index` says which plane is the aggregate. Never by convention: the
// day plane 0 stops being the aggregate, a frontend that assumed it draws the
// wrong hour and says nothing.
export function decodeFrameStack(buffer, block) {
  if (!buffer || !block) return null
  if (block.dtype && block.dtype !== 'float32') return null

  const shape = block.shape
  if (!Array.isArray(shape) || shape.length !== 3) return null
  const [planeCount, height, width] = shape.map(Number)
  if (![planeCount, height, width].every(n => Number.isInteger(n) && n > 0)) return null

  const cells = height * width
  const expected = planeCount * cells
  const actual = Math.floor(buffer.byteLength / 4)
  // A short blob still reshapes cleanly into whole frames, so it renders as a
  // plausible field with the tail of the scrub silently missing. Refused
  // loudly instead -- this is the one failure that looks like success.
  if (actual !== expected) {
    throw new Error(`frame stack is ${actual} float32 values, expected ${expected} for shape ${shape.join('x')}`)
  }

  const values = new Float32Array(buffer)

  return {
    values,
    planeCount,
    height,
    width,
    cellsPerPlane: cells,
    // Zero-copy by construction. `subarray` shares the buffer; `slice` would
    // not, and nothing in this module may reach for it.
    planeView(index) {
      if (!Number.isInteger(index) || index < 0 || index >= planeCount) return null
      const start = index * cells
      return values.subarray(start, start + cells)
    },
  }
}

// What the canvas draws for the stop the slider points at: one plane's view,
// on the FRAME stack's own grid.
//
// The grid matters. The stack is block-meaned to a 20,000-cell ceiling and has
// lats/lons of its own, which are not the payload's thinned rendering grid.
// And it must agree with the blob's shape: the canvas sizes itself from
// lats/lons and indexes by `row * width + col`, so a disagreement puts every
// value at the wrong coordinate and produces a map that looks entirely
// plausible and is wrong everywhere. Refused rather than drawn.
export function selectFrame(chart, stack, stop) {
  const block = chart?.frames
  if (!block || !stack || !stop) return null

  const { lats, lons } = block
  if (!Array.isArray(lats) || !Array.isArray(lons)) return null
  if (lats.length !== stack.height || lons.length !== stack.width) return null

  const values = stack.planeView(stop.plane)
  return values ? { values, lats, lons } : null
}
