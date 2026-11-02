// HMS is a WGS84 vector layer over standard Web Mercator XYZ tiles.
// Keep this keyless source local to HMS; other maps retain their provider config.
export function smokeMapStyle() {
  return {
    version: 8,
    sources: {
      basemap: {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        scheme: 'xyz',
        maxzoom: 19,
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      },
    },
    layers: [
      { id: 'background', type: 'background', paint: { 'background-color': '#e4e1d8' } },
      { id: 'basemap', type: 'raster', source: 'basemap' },
    ],
  }
}

function validBounds(bounds) {
  if (!Array.isArray(bounds)) return null
  const flat = bounds.length === 2 && bounds.every(Array.isArray) ? bounds.flat() : bounds
  if (flat.length !== 4 || !flat.every(Number.isFinite)) return null
  const [west, south, east, north] = flat
  if (west < -180 || east > 180 || west >= east || south < -90 || north > 90 || south >= north) return null
  // Mercator cannot display the poles; do not change the underlying GeoJSON.
  return [[west, Math.max(south, -85.051129)], [east, Math.min(north, 85.051129)]]
}

export function smokeMapBounds(payload) {
  // Fit the requested area even when smoke occupies only a small part of it,
  // or the result is empty. Never swap latitude and longitude or reproject here.
  const requested = validBounds(payload.bounds)
  if (requested) return requested
  let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity
  function visit(coordinates) {
    if (!Array.isArray(coordinates)) return
    if (typeof coordinates[0] === 'number') {
      const [lon, lat] = coordinates
      if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lon) > 180 || Math.abs(lat) > 90) return
      west = Math.min(west, lon); east = Math.max(east, lon)
      south = Math.min(south, lat); north = Math.max(north, lat)
    } else coordinates.forEach(visit)
  }
  for (const feature of payload.geojson?.features || []) visit(feature.geometry?.coordinates)
  return validBounds([west, south, east, north])
}

export function fitSmokeMap(map, bounds, container) {
  if (!bounds || !container.clientWidth || !container.clientHeight) return
  map.fitBounds(bounds, {
    padding: Math.min(35, container.clientWidth / 8, container.clientHeight / 8),
    maxZoom: 10, duration: 0,
  })
}
