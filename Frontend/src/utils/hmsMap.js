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

// The quantity the harness's HMS smoke source records on its polygons handles.
const SMOKE_QUANTITY = 'smoke_extent'

function polygonFeatures(chart) {
  return (chart?.polygons || []).flatMap(entry => entry?.geojson?.features || [])
}

// Whether a `polygon_map` holds smoke: its features carry a density, or it has
// none and every source behind it is the smoke extent.
export function isSmokePolygonMap(chart) {
  if (chart?.type !== 'polygon_map') return false
  const features = polygonFeatures(chart)
  if (features.length) return features.some(feature => feature?.properties?.density != null)
  const sources = chart.provenance?.sources || []
  return sources.length > 0 && sources.every(source => source?.quantity === SMOKE_QUANTITY || source?.kind === 'points')
    && sources.some(source => source?.quantity === SMOKE_QUANTITY)
}

// A chart's point overlay (`overlays.points`) as a GeoJSON FeatureCollection; empty when it has none.
export function overlayPointsGeoJSON(chart) {
  const points = chart?.overlays?.points?.points || []
  return {
    type: 'FeatureCollection',
    features: points
      .filter(point => Number.isFinite(point?.lon) && Number.isFinite(point?.lat))
      .map(point => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [point.lon, point.lat] },
        properties: { id: point.id, name: point.name ?? null, value: point.value ?? null },
      })),
  }
}

// A non-smoke `polygon_map` as the generic polygon panel draws it: its polygons
// as one FeatureCollection, its point overlay, and the bounds to fit.
export function polygonMapLayers(chart) {
  const polygons = { type: 'FeatureCollection', features: polygonFeatures(chart) }
  return {
    polygons,
    points: overlayPointsGeoJSON(chart),
    bounds: smokeMapBounds({ bounds: chart?.bounds, geojson: polygons }),
  }
}

// The harness stores smoke as a generic `polygon_map` (one FeatureCollection
// per polygons handle); the smoke panel reads the `smoke_map` shape. Only a
// polygon map of smoke is converted, keeping its `overlays.points`; any other
// payload passes through unchanged, and a given payload maps to one stable object.
export function asSmokeMap(chart) {
  if (!isSmokePolygonMap(chart)) return chart
  if (normalized.has(chart)) return normalized.get(chart)
  const features = polygonFeatures(chart)
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
