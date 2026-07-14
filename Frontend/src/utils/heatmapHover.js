import { flattenPayload } from './flattenPayload.js'

// Nearest-cell hover lookup against the shipped (downsampled, <=8k-cell)
// grid/points arrays — no server round-trip. Visual fidelity (full-res
// overlay PNG) and interaction resolution (these arrays) are deliberately
// decoupled per the T23 design.
export function nearestCell(lng, lat, payload) {
  const { lat: lats, lon: lons, val } = flattenPayload(payload)
  if (!val.length) return null

  let bestIndex = -1
  let bestDist = Infinity
  for (let i = 0; i < val.length; i++) {
    const dLat = lats[i] - lat
    const dLon = lons[i] - lng
    const dist = dLat * dLat + dLon * dLon
    if (dist < bestDist) {
      bestDist = dist
      bestIndex = i
    }
  }

  return { lat: lats[bestIndex], lon: lons[bestIndex], value: val[bestIndex] }
}
