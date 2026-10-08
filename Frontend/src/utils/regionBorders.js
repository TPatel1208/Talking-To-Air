// US state border GeoJSON, fetched once per session and shared by every map
// panel (Plotly scattergeo traces previously, MapLibre GeoJSON layers now).
const STATES_URL = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

// How a state border is drawn, on the single map and on the comparison
// thumbnails alike, so a thumbnail looks like the map it expands into. A dark
// line on a light halo: the dark line alone vanished on viridis's dark purple,
// and a light one alone would vanish on its yellow. The halo is drawn first.
export const BORDER_LINE = {
  color: 'rgba(20,20,20,0.9)',
  width: 1.1,
  haloColor: 'rgba(255,255,255,0.7)',
  haloWidth: 2.8,
}

// The beforeId for a data overlay layer: under the border halo once the
// borders exist. The overlay is re-added on every recolor, and MapLibre puts a
// layer added without a beforeId on top, which buried the borders.
export function overlayBeforeId(map) {
  return map.getLayer('region-borders-halo') ? 'region-borders-halo' : undefined
}

let _bordersPromise = null
export function fetchUsStatesGeoJSON() {
  if (_bordersPromise) return _bordersPromise
  // 5s timeout so a slow/failed CDN response never hangs a chart.
  const timeout = new Promise(resolve => setTimeout(() => resolve(null), 5000))
  _bordersPromise = Promise.race([
    fetch(STATES_URL).then(r => r.ok ? r.json() : null).catch(() => null),
    timeout,
  ])
  return _bordersPromise
}

// True when the bounding box is substantially over the continental US --
// used to skip the state-border fetch for non-CONUS maps (global, Europe,
// etc.) and avoid a pointless cross-origin request.
export function isConusBounds(minx, miny, maxx, maxy) {
  const lonOverlap = Math.min(maxx, -65) - Math.max(minx, -130)
  const latOverlap = Math.min(maxy, 50) - Math.max(miny, 24)
  const mapArea = (maxx - minx) * (maxy - miny)
  if (mapArea <= 0) return false
  const overlap = Math.max(0, lonOverlap) * Math.max(0, latOverlap)
  return overlap / mapArea > 0.3 // >30% of the map must be within CONUS
}
