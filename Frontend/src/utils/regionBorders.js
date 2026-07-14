// US state border GeoJSON, fetched once per session and shared by every map
// panel (Plotly scattergeo traces previously, MapLibre GeoJSON layers now).
const STATES_URL = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

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
