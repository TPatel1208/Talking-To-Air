// Maps a value into an RGBA color sampled from the shared colormap LUT
// (see Backend/utils/colormaps.py). The LUT is a flat array of [r,g,b,a]
// stops evenly spaced across [vmin, vmax] — the same array shipped in the
// payload and sampled by the server overlay renderer, so the map, the
// hover swatch, and the colorbar can never disagree on what a value looks like.
export function colorForValue(value, { vmin, vmax, lut }) {
  if (!Number.isFinite(value) || !Array.isArray(lut) || !lut.length) return null

  const span = vmax - vmin
  const t = span > 0 ? (value - vmin) / span : 0
  const clamped = Math.min(1, Math.max(0, t))
  const index = Math.round(clamped * (lut.length - 1))
  return lut[index]
}
