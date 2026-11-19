// HMS is a WGS84 vector layer over standard Web Mercator XYZ tiles.
// Keep this keyless source local to HMS; other maps retain their provider config.
export function smokeMapStyle(tileConfig = {}) {
  return {
    version: 8,
    sources: {
      basemap: {
        type: 'raster',
       tiles: [tileConfig.basemap_light_url || 'https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png'],
        tileSize: 256,
        scheme: 'xyz',
        maxzoom: 19,
        attribution: tileConfig.basemap_attribution || '© CARTO © OpenStreetMap contributors',
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

const normalized = new WeakMap()

// The harness stores smoke as a generic `polygon_map` (one FeatureCollection
// per polygons handle); the smoke panel reads the `smoke_map` shape. Other
// payloads pass through unchanged, and a given payload maps to one stable object.
export function asSmokeMap(chart) {
  if (chart?.type !== 'polygon_map') return chart
  if (normalized.has(chart)) return normalized.get(chart)
  const features = (chart.polygons || []).flatMap(entry => entry?.geojson?.features || [])
  const counts = { light: 0, medium: 0, heavy: 0, unknown: 0 }
  for (const feature of features) {
    const density = feature?.properties?.density
    counts[density in counts ? density : 'unknown'] += 1
  }
  const start = features.map(f => f?.properties?.start).find(Boolean)
  const result = {
    ...chart,
    type: 'smoke_map',
    geojson: { type: 'FeatureCollection', features },
    counts,
    feature_count: chart.feature_count ?? features.length,
    date: chart.date ?? (typeof start === 'string' ? start.slice(0, 10) : 'map'),
  }
  normalized.set(chart, result)
  return result
}
