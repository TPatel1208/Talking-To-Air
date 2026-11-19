import test from 'node:test'
import assert from 'node:assert/strict'
import { smokeMapStyle, smokeMapBounds, fitSmokeMap, asSmokeMap } from '../src/utils/hmsMap.js'

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
