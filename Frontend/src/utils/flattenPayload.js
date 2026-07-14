// Flatten 2D grid → parallel lat/lon/value arrays. Skips null cells.
// Optionally downsamples to MAX_POINTS for performance.
const MAX_POINTS = 8000
function flattenGrid(lats, lons, values) {
  const flatLat = [], flatLon = [], flatVal = []
  for (let ri = 0; ri < lats.length; ri++) {
    for (let ci = 0; ci < lons.length; ci++) {
      const v = values[ri][ci]
      if (v != null) {
        flatLat.push(lats[ri])
        flatLon.push(lons[ci])
        flatVal.push(v)
      }
    }
  }
  // Downsample evenly if too many points
  if (flatLat.length > MAX_POINTS) {
    const step = Math.ceil(flatLat.length / MAX_POINTS)
    return {
      lat: flatLat.filter((_, i) => i % step === 0),
      lon: flatLon.filter((_, i) => i % step === 0),
      val: flatVal.filter((_, i) => i % step === 0),
    }
  }
  return { lat: flatLat, lon: flatLon, val: flatVal }
}

export function flattenPayload(payload) {
  if (Array.isArray(payload.lats) && Array.isArray(payload.lons) && Array.isArray(payload.values)) {
    const grid = flattenGrid(payload.lats, payload.lons, payload.values)
    if (grid.val.length) return grid
  }

  const points = payload.points
  if (
    points &&
    Array.isArray(points.lats) &&
    Array.isArray(points.lons) &&
    Array.isArray(points.values) &&
    points.values.length
  ) {
    const lat = []
    const lon = []
    const val = []
    for (let i = 0; i < points.values.length; i++) {
      const value = points.values[i]
      if (!Number.isFinite(value)) continue
      lat.push(points.lats[i])
      lon.push(points.lons[i])
      val.push(value)
    }
    return {
      lat,
      lon,
      val,
    }
  }

  return { lat: [], lon: [], val: [] }
}
