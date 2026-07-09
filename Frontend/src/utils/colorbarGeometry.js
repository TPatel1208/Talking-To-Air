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
