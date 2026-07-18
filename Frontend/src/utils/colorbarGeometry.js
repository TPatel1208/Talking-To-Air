// Pure geometry for the SVG scientific colorbar — gradient stops and tick
// positions derived directly from vmin/vmax/lut (the same LUT sampled onto
// the map and the overlay PNG) so the legend can never depict a color that
// isn't actually on the map.
export function colorbarGeometry({ vmin, vmax, lut, tickCount = 5 }) {
  const stops = Array.isArray(lut) ? lut : []
  const gradientStops = stops.map((rgba, i) => ({
    offset: stops.length > 1 ? i / (stops.length - 1) : 0,
    color: rgbaToCss(rgba),
  }))

  const steps = Math.max(1, tickCount - 1)
  const span = vmax - vmin
  const ticks = []
  for (let i = 0; i < tickCount; i++) {
    ticks.push({ value: vmin + (span * i) / steps, position: i / steps })
  }

  return { gradientStops, ticks }
}

function rgbaToCss([r, g, b, a = 255]) {
  return `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`
}

// A one-line disclosure for the legend when the color scale is a percentile
// clip (T43): the extreme tails are saturated, so a reader must not mistake
// the colored range for the data's true min/max. Returns null for a
// caller-imposed (explicit) scale — which is not a clip of this field — and
// for older payloads carrying no `scale` at all.
export function scaleClipNote(scale) {
  if (!scale || scale.method !== 'percentile') return null
  const p = Array.isArray(scale.p) && scale.p.length === 2 ? scale.p : [2, 98]
  return `Color scale clipped at ${ordinal(p[0])}–${ordinal(p[1])} percentile`
}

function ordinal(n) {
  const rem100 = n % 100
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`
  const suffix = { 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] || 'th'
  return `${n}${suffix}`
}
