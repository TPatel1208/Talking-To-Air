import test from 'node:test'
import assert from 'node:assert/strict'
import {
  smokeMapStyle, smokeMapBounds, fitSmokeMap, asSmokeMap, isSmokePolygonMap, overlayPointsGeoJSON, polygonMapLayers,
} from '../src/utils/hmsMap.js'

const geojson = { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: {
  type: 'MultiPolygon', coordinates: [[[[-80, 30], [-70, 30], [-70, 40], [-80, 30]]]],
} }] }


test('HMS basemap uses project tile configuration', () => {
  const config = {
    basemap_light_url: 'https://tiles.example.com/{z}/{x}/{y}.png',
    basemap_attribution: 'Example map provider',
  }

  const source = smokeMapStyle(config).sources.basemap

  assert.deepEqual(source.tiles, [config.basemap_light_url])
  assert.equal(source.attribution, config.basemap_attribution)
  assert.equal(source.scheme, 'xyz')
  assert.equal(source.tileSize, 256)
})

test('HMS basemap has a fallback configuration', () => {
  const source = smokeMapStyle().sources.basemap

  assert.deepEqual(source.tiles, [
    'https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png',
  ])
  assert.match(source.attribution, /CARTO/)
})


test('requested CONUS takes priority over plume extent, with lon/lat order preserved', () => {
  assert.deepEqual(smokeMapBounds({ bounds: [-125, 24, -66, 50], geojson }), [[-125, 24], [-66, 50]])
})

test('US bounds retain Alaska and Hawaii even with no observed smoke', () => {
  assert.deepEqual(smokeMapBounds({ bounds: [-171.791111, 18.91619, -66.96466, 71.357764] }),
    [[-171.791111, 18.91619], [-66.96466, 71.357764]])
})

test('supports corner-pair bounds and uses polygons only when bounds are absent or invalid', () => {
  assert.deepEqual(smokeMapBounds({ bounds: [[-125, 24], [-66, 50]] }), [[-125, 24], [-66, 50]])
  for (const bounds of [undefined, [], [NaN, 0, 1, 2], [100, 90, -100, -90]]) {
    assert.deepEqual(smokeMapBounds({ bounds, geojson }), [[-80, 30], [-70, 40]])
  }
  assert.equal(smokeMapBounds({ geojson: { features: [{ geometry: null }] } }), null)
})

test('bounds calculation does not modify smoke coordinates or properties', () => {
  const before = JSON.stringify(geojson)
  smokeMapBounds({ geojson })
  assert.equal(JSON.stringify(geojson), before)
})

test('fitBounds uses requested area and adapts padding to resized or hidden panels', () => {
  const calls = []
  const map = { fitBounds: (...args) => calls.push(args) }
  const bounds = [[-125, 24], [-66, 50]]
  fitSmokeMap(map, bounds, { clientWidth: 800, clientHeight: 480 })
  fitSmokeMap(map, bounds, { clientWidth: 160, clientHeight: 480 })
  fitSmokeMap(map, bounds, { clientWidth: 0, clientHeight: 480 })
  assert.equal(calls.length, 2)
  assert.deepEqual(calls[0], [bounds, { padding: 35, maxZoom: 10, duration: 0 }])
  assert.equal(calls[1][1].padding, 20)
})

test('a harness polygon_map becomes the smoke map shape the panel reads', () => {
  const feature = density => ({ type: 'Feature', geometry: null, properties: { density, start: '2024-06-01T11:00:00Z' } })
  const chart = {
    type: 'polygon_map', bounds: [-80, 30, -70, 40], feature_count: 3,
    polygons: [
      { handle: 'polygons_a', geojson: { type: 'FeatureCollection', features: [feature('light'), feature('heavy')] } },
      { handle: 'polygons_b', geojson: { type: 'FeatureCollection', features: [feature('odd')] } },
    ],
  }
  const smoke = asSmokeMap(chart)
  assert.equal(smoke.type, 'smoke_map')
  assert.equal(smoke.geojson.features.length, 3)
  assert.deepEqual(smoke.counts, { light: 1, medium: 0, heavy: 1, unknown: 1 })
  assert.equal(smoke.date, '2024-06-01')
  assert.deepEqual(smokeMapBounds(smoke), [[-80, 30], [-70, 40]])
  assert.equal(asSmokeMap(chart), smoke)
  const other = { type: 'heatmap' }
  assert.equal(asSmokeMap(other), other)
})

const OVERLAY = { points: { handle: 'points_1', points: [
  { id: 'site-1', name: 'Trenton', lat: 40.2, lon: -74.7, value: 12.5 },
  { id: 'site-2', name: null, lat: Number.NaN, lon: -74.0 },
] } }

test('a smoke polygon map keeps its point overlay when it becomes the smoke shape', () => {
  const chart = {
    type: 'polygon_map', bounds: [-80, 30, -70, 40], overlays: OVERLAY,
    polygons: [{ handle: 'polygons_a', geojson: { type: 'FeatureCollection', features: [
      { type: 'Feature', geometry: null, properties: { density: 'medium' } },
    ] } }],
  }
  const smoke = asSmokeMap(chart)
  assert.equal(smoke.type, 'smoke_map')
  assert.deepEqual(smoke.overlays, OVERLAY)
  assert.deepEqual(overlayPointsGeoJSON(smoke).features.map(f => f.geometry.coordinates), [[-74.7, 40.2]])
})

test('a polygon map that is not smoke keeps its type and draws its polygons and points', () => {
  const feature = { type: 'Feature', geometry: { type: 'Polygon', coordinates: [[[-75, 39], [-74, 39], [-74, 40], [-75, 39]]] },
    properties: { name: 'A district' } }
  const chart = {
    type: 'polygon_map', bounds: null, overlays: OVERLAY,
    polygons: [{ handle: 'polygons_b', geojson: { type: 'FeatureCollection', features: [feature] } }],
  }
  assert.equal(isSmokePolygonMap(chart), false)
  assert.equal(asSmokeMap(chart), chart)
  const layers = polygonMapLayers(chart)
  assert.deepEqual(layers.polygons.features, [feature])
  assert.equal(layers.points.features.length, 1)
  assert.deepEqual(layers.points.features[0].properties, { id: 'site-1', name: 'Trenton', value: 12.5 })
  assert.deepEqual(layers.bounds, [[-75, 39], [-74, 40]])
})

test('an empty polygon map is smoke only when its sources are the smoke extent', () => {
  const empty = sources => ({ type: 'polygon_map', polygons: [], feature_count: 0, provenance: { sources } })
  assert.equal(asSmokeMap(empty([{ handle: 'polygons_a', kind: 'polygons', quantity: 'smoke_extent' }])).type, 'smoke_map')
  assert.equal(asSmokeMap(empty([{ handle: 'polygons_c', kind: 'polygons' }])).type, 'polygon_map')
  assert.equal(asSmokeMap(empty([])).type, 'polygon_map')
})
