import { colorForValue } from './colormapLut.js'

// Builds an RGBA byte frame from the shipped lats/lons/values grid, for
// when the server-rendered overlay PNG is unavailable (T17 degrade-don't-die
// posture). No-data cells stay fully transparent (alpha 0) — the fallback
// never interpolates across a gap it has no measurement for.
export function buildCanvasFallbackFrame({ lats, lons, values, vmin, vmax, lut }) {
  const width = lons.length
  const height = lats.length
  const pixels = new Uint8ClampedArray(width * height * 4)

  // A heatmap payload ships nested rows; a T59 frame is a `subarray` VIEW over
  // one flat float32 stack, handed here without a copy (Phase 2: 100 views
  // cost +0.00 MB, 100 copies cost a duplicate of the whole stack). Indexing
  // off the VIEW rather than the underlying array is what keeps the next
  // frame's first row out of this frame's last one. NaN needs no sentinel --
  // it arrives in the float32 and falls through the same non-finite branch a
  // null does.
  const flat = ArrayBuffer.isView(values)
  const cellAt = flat
    ? (row, col) => values[row * width + col]
    : (row, col) => values[row][col]

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const color = colorForValue(cellAt(row, col), { vmin, vmax, lut })
      if (!color) continue // leave as zeros: fully transparent, never invented

      const offset = (row * width + col) * 4
      pixels[offset] = color[0]
      pixels[offset + 1] = color[1]
      pixels[offset + 2] = color[2]
      pixels[offset + 3] = color[3] ?? 255
    }
  }

  return { width, height, pixels }
}
