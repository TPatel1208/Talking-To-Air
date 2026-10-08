// Geometry for a comparison thumbnail (HeatmapMultiPanel). The tile shows
// either the server png, which overlay_render.py warps to Web Mercator, or the
// canvas fallback, one pixel per lat/lon cell. Its shape and every outline
// drawn over it must follow whichever of the two is actually showing.

const toRad = (deg) => (deg * Math.PI) / 180

// Web Mercator's y for a latitude, in the same radian units as longitude.
function mercatorY(lat) {
  return Math.log(Math.tan(Math.PI / 4 + toRad(lat) / 2))
}

// The box the tile depicts, how to project into it, its width/height, and
// whether the image must be mirrored to come out north-up and west-left.
// mode is resolveThumbnailMode's 'native' | 'canvas'.
export function thumbnailFrame(panel, mode) {
  if (mode === 'native') {
    const bounds = panel.overlay?.bounds || panel.bounds
    const [minx, miny, maxx, maxy] = bounds
    const aspect = toRad(maxx - minx) / (mercatorY(maxy) - mercatorY(miny))
    return { bounds, projection: 'mercator', aspect, flipX: false, flipY: false }
  }
  // The canvas paints lats[0] as its top row and lons[0] as its left column,
  // in whatever order the source data had them.
  const { lats, lons } = panel
  if (!lats?.length || !lons?.length) {
    const bounds = panel.bounds || panel.overlay?.bounds || [0, 0, 1, 1]
    const aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
    return { bounds, projection: 'linear', aspect: aspect > 0 ? aspect : 1, flipX: false, flipY: false }
  }
  const firstLat = lats[0], lastLat = lats[lats.length - 1]
  const firstLon = lons[0], lastLon = lons[lons.length - 1]
  const bounds = [
    Math.min(firstLon, lastLon), Math.min(firstLat, lastLat),
    Math.max(firstLon, lastLon), Math.max(firstLat, lastLat),
  ]
  const aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
  return { bounds, projection: 'linear', aspect, flipX: firstLon > lastLon, flipY: firstLat < lastLat }
}

// A lon/lat as [x, y] fractions of the north-up tile: [0, 0] is its
// north-west corner, [1, 1] its south-east.
export function projectPoint(frame, lon, lat) {
  const [minx, miny, maxx, maxy] = frame.bounds
  const y = frame.projection === 'mercator'
    ? (mercatorY(maxy) - mercatorY(lat)) / (mercatorY(maxy) - mercatorY(miny))
    : (maxy - lat) / (maxy - miny)
  return [(lon - minx) / (maxx - minx), y]
}

function polygonsOf(geometry) {
  if (geometry?.type === 'Polygon') return [geometry.coordinates]
  if (geometry?.type === 'MultiPolygon') return geometry.coordinates
  return []
}

// By the ring's bounding box, not its vertices: a long straight border can
// cross the tile with every vertex outside it. Over-inclusion is harmless, as
// the svg clips whatever falls outside the tile.
function overlapsBounds(polygons, [minx, miny, maxx, maxy]) {
  return polygons.some(([outer = []]) => {
    const lons = outer.map(p => p[0])
    const lats = outer.map(p => p[1])
    return Math.min(...lons) <= maxx && Math.max(...lons) >= minx
      && Math.min(...lats) <= maxy && Math.max(...lats) >= miny
  })
}

// One ring as SVG path data in a 0..100 viewBox.
function ringPath(ring, frame) {
  const fmt = (v) => +(v * 100).toFixed(2)
  return ring.slice(0, -1).map(([lon, lat], i) => {
    const [x, y] = projectPoint(frame, lon, lat)
    return `${i ? 'L' : 'M'}${fmt(x)} ${fmt(y)}`
  }).join('') + 'Z'
}

// The state outlines in view of a thumbnail, as SVG path data for a 0..100
// viewBox stretched over the tile. A null geojson -- the CDN fetch failed or
// was skipped outside CONUS -- draws nothing.
export function thumbnailBorders(geojson, frame) {
  return (geojson?.features || [])
    .map(feature => polygonsOf(feature.geometry))
    .filter(polygons => overlapsBounds(polygons, frame.bounds))
    .map(polygons => polygons.map(rings => ringPath(rings[0], frame)).join(''))
    .join('')
}
