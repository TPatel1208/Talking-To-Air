import assert from 'node:assert/strict'
import test from 'node:test'
import { thumbnailFrame, projectPoint, thumbnailBorders } from '../src/utils/thumbnailGeometry.js'

// A comparison thumbnail is either the server png (warped to Web Mercator by
// overlay_render.py) or the canvas fallback (one pixel per lat/lon cell). The
// tile's shape, its orientation and every outline drawn over it have to follow
// whichever of the two is actually showing.

// cmp_11192fae0e05, the New Jersey panel: a 67x130 png over these bounds.
const NJ_OVERLAY_BOUNDS = [-75.53999874915606, 38.940000772476196, -73.9000036922502, 41.35999846458435]
const NJ_PANEL = {
  bounds: [-75.52999877929688, 38.95000076293945, -73.91000366210938, 41.349998474121094],
  overlay: { url: '/chart/cmp/overlay.png?panel=0', bounds: NJ_OVERLAY_BOUNDS },
  lats: [38.95, 39.75, 40.55, 41.35],
  lons: [-75.53, -74.72, -73.91],
}

test('a native tile takes the mercator png its own shape, so nothing is cropped', () => {
  // Sized in plain degrees (0.68) the tile was wider than the png (0.515), and
  // objectFit: cover cut about a quarter of the state off top and bottom.
  const frame = thumbnailFrame(NJ_PANEL, 'native')

  assert.ok(Math.abs(frame.aspect - 67 / 130) < 0.01, `aspect ${frame.aspect}`)
})

test('a canvas tile is one pixel per cell, so it keeps the grid extent in plain degrees', () => {
  const frame = thumbnailFrame(NJ_PANEL, 'canvas')

  assert.deepEqual(frame.bounds, [-75.53, 38.95, -73.91, 41.35])
  assert.ok(Math.abs(frame.aspect - (1.62 / 2.4)) < 1e-9, `aspect ${frame.aspect}`)
})

test('a canvas tile from south-first rows is flipped to put north at the top', () => {
  // Every stored payload ships lats ascending. The canvas paints row 0 at the
  // top, so without a flip each fallback thumbnail was drawn upside down.
  assert.equal(thumbnailFrame(NJ_PANEL, 'canvas').flipY, true)
  const northFirst = { ...NJ_PANEL, lats: [...NJ_PANEL.lats].reverse() }
  assert.equal(thumbnailFrame(northFirst, 'canvas').flipY, false)
})

test('a canvas tile from east-first columns is flipped to put west on the left', () => {
  assert.equal(thumbnailFrame(NJ_PANEL, 'canvas').flipX, false)
  const eastFirst = { ...NJ_PANEL, lons: [...NJ_PANEL.lons].reverse() }
  assert.equal(thumbnailFrame(eastFirst, 'canvas').flipX, true)
})

test('a point projects into the tile with north at the top, west at the left', () => {
  for (const mode of ['native', 'canvas']) {
    const frame = thumbnailFrame(NJ_PANEL, mode)
    const [minx, miny, maxx, maxy] = frame.bounds
    assert.deepEqual(projectPoint(frame, minx, maxy), [0, 0], `${mode} top-left`)
    const [x, y] = projectPoint(frame, maxx, miny)
    assert.ok(Math.abs(x - 1) < 1e-9 && Math.abs(y - 1) < 1e-9, `${mode} bottom-right ${x},${y}`)
  }
})

test('on the mercator png a latitude sits where mercator puts it, not halfway in degrees', () => {
  // Mercator stretches northward, so the degree midpoint of the box lies below
  // the middle row of the png. Over a state the gap is under a pixel; over a
  // CONUS-sized box it is 4% of the tile, and an outline drawn linearly would
  // sit visibly north of the data. Hand-computed from ln(tan(45° + lat/2)):
  // (1.0107 - 0.7070) / (1.0107 - 0.4509) = 0.5425.
  const conus = { overlay: { bounds: [-125, 25, -65, 50] }, lats: [25, 50], lons: [-125, -65] }

  const [, nativeY] = projectPoint(thumbnailFrame(conus, 'native'), -75, 37.5)
  const [, canvasY] = projectPoint(thumbnailFrame(conus, 'canvas'), -75, 37.5)

  assert.ok(Math.abs(nativeY - 0.5425) < 1e-3, `native ${nativeY}`)
  assert.ok(Math.abs(canvasY - 0.5) < 1e-9, `canvas ${canvasY}`)
})

// A square "state" filling the box's south-west quarter, one inside the box
// but elsewhere, and one a continent away; shaped like us-states.json.
const square = (name, [x0, y0, x1, y1], type = 'Polygon') => {
  const ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
  return {
    type: 'Feature',
    properties: { name },
    geometry: type === 'Polygon' ? { type, coordinates: [ring] } : { type, coordinates: [[ring]] },
  }
}
const BOX = { overlay: { bounds: [0, 0, 10, 10] }, lats: [0, 10], lons: [0, 10] }
const STATES = {
  type: 'FeatureCollection',
  features: [
    square('New Jersey', [0, 0, 5, 5]),
    square('Delaware', [5, 5, 10, 10], 'MultiPolygon'),
    square('Alaska', [100, 60, 110, 70]),
  ],
}

test('outlines are drawn for the states in the box, in tile coordinates', () => {
  const d = thumbnailBorders(STATES, thumbnailFrame(BOX, 'canvas'))

  // viewBox 0..100, north-up: the south-west quarter spans x 0..50, y 50..100.
  assert.match(d, /M0 100L50 100L50 50L0 50Z/)
  assert.match(d, /M50 50L100 50L100 0L50 0Z/, 'multipolygons are outlined too')
  assert.doesNotMatch(d, /1000/, 'a state outside the box is not drawn')
})

test('a border that crosses the box with every vertex outside it is still drawn', () => {
  // Western borders are long straight lines: Colorado/Wyoming is one edge
  // along 41N with its vertices hundreds of km apart. A tile around Cheyenne
  // contains none of them, and the border is exactly what it needs to show.
  const wyoming = square('Wyoming', [-111, 41, -104, 45])
  const cheyenne = { overlay: { bounds: [-105.5, 40.5, -104.5, 41.5] }, lats: [40.5, 41.5], lons: [-105.5, -104.5] }

  assert.notEqual(thumbnailBorders({ features: [wyoming] }, thumbnailFrame(cheyenne, 'canvas')), '')
})

test('no borders to draw is not an error', () => {
  // The border GeoJSON is fetched from a CDN with a 5s timeout and resolves
  // null on any failure; a tile outside CONUS never fetches it at all.
  assert.equal(thumbnailBorders(null, thumbnailFrame(BOX, 'canvas')), '')
})

test('a canvas tile with no grid arrays falls back to the panel bounds', () => {
  // ThumbnailCanvas draws nothing without lats/lons, but the tile still needs
  // a shape; this used to be the only thing every thumbnail rendered.
  const frame = thumbnailFrame({ bounds: [0, 0, 4, 2] }, 'canvas')

  assert.deepEqual(frame.bounds, [0, 0, 4, 2])
  assert.equal(frame.aspect, 2)
  assert.equal(frame.flipY, false)
})

test('the native png is already north-up and is never flipped', () => {
  const frame = thumbnailFrame(NJ_PANEL, 'native')

  assert.equal(frame.flipX, false)
  assert.equal(frame.flipY, false)
  assert.deepEqual(frame.bounds, NJ_OVERLAY_BOUNDS)
})
