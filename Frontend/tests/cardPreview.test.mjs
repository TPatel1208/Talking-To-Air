import test from 'node:test'
import assert from 'node:assert/strict'

import { cardPreview, sparklinePath, profilePath } from '../src/utils/cardPreview.js'

const chart = (data) => ({ kind: 'chart', data })
const artifact = (data) => ({ kind: 'artifact', data })

test('a map previews its own server-rendered image', () => {
  const preview = cardPreview(chart({ type: 'heatmap', overlay: { url: '/chart/map_1/overlay.png' } }))
  assert.equal(preview.kind, 'map')
  assert.equal(preview.overlayUrl, '/chart/map_1/overlay.png?thumb=96')
})

test('a comparison previews its first panel', () => {
  const preview = cardPreview(chart({
    type: 'heatmap_multi',
    panels: [{ overlay: { url: '/chart/a/overlay.png' } }, { overlay: { url: '/chart/b/overlay.png' } }],
  }))
  assert.equal(preview.kind, 'map')
  assert.equal(preview.overlayUrl, '/chart/a/overlay.png?thumb=96')
})

test('a map with no image falls back to the map icon', () => {
  assert.deepEqual(cardPreview(chart({ type: 'heatmap' })), { kind: 'icon', icon: 'map' })
})

test('a time series previews as a sparkline of its values', () => {
  const preview = cardPreview(chart({ type: 'timeseries', values: [1, 3, 2] }))
  assert.equal(preview.kind, 'sparkline')
  assert.match(preview.path, /^M/)
})

test('a time series with fewer than two values falls back to its icon', () => {
  assert.deepEqual(cardPreview(chart({ type: 'timeseries', values: [5, null] })), { kind: 'icon', icon: 'timeseries' })
})

test('a profile previews as a curve against its vertical axis', () => {
  const preview = cardPreview(chart({ type: 'profile', values: [1, 2, 3], layers: [0, 1, 2], layer_order: 'bottom_up' }))
  assert.equal(preview.kind, 'profile')
  assert.match(preview.path, /^M/)
})

test('a table shows its row count', () => {
  assert.deepEqual(cardPreview(artifact({ type: 'table', row_count: 24 })), { kind: 'table', rowLabel: '24 rows' })
  assert.deepEqual(cardPreview(artifact({ type: 'table', row_count: 1 })), { kind: 'table', rowLabel: '1 row' })
  assert.deepEqual(cardPreview(artifact({ type: 'table' })), { kind: 'table', rowLabel: null })
})

test('other outputs get an icon for their type', () => {
  assert.deepEqual(cardPreview(artifact({ type: 'timeseries' })), { kind: 'icon', icon: 'timeseries' })
  assert.deepEqual(cardPreview(artifact({ type: 'something_new' })), { kind: 'icon', icon: 'generic' })
  assert.deepEqual(cardPreview(chart({ type: 'something_new' })), { kind: 'icon', icon: 'generic' })
})

test('a sparkline spans the box, high values at the top', () => {
  assert.equal(sparklinePath([0, 10], { width: 40, height: 20, pad: 0 }), 'M0 20 L40 0')
})

test('a flat series sits mid-height', () => {
  assert.equal(sparklinePath([5, 5], { width: 40, height: 20, pad: 0 }), 'M0 10 L40 10')
})

test('a gap in the series breaks the line', () => {
  assert.equal(sparklinePath([0, null, 10, 0], { width: 30, height: 10, pad: 0 }), 'M0 10 M20 0 L30 10')
})

test('a bottom-up profile puts the last layer at the top', () => {
  assert.equal(profilePath([0, 10], [0, 1], { width: 20, height: 20, pad: 0 }), 'M0 20 L20 0')
})

test('a reversed axis (pressure, or a top-down index) puts the first level at the top', () => {
  assert.equal(profilePath([0, 10], [0, 1], { width: 20, height: 20, pad: 0, reversed: true }), 'M0 0 L20 20')
})

test('a map preview asks the server for a small thumbnail', () => {
  assert.equal(
    cardPreview(chart({ type: 'heatmap', overlay: { url: '/chart/map_1/overlay.png' } })).overlayUrl,
    '/chart/map_1/overlay.png?thumb=96',
  )
  assert.equal(
    cardPreview(chart({ type: 'heatmap_multi', panels: [{ overlay: { url: '/chart/c/overlay.png?panel=0' } }] })).overlayUrl,
    '/chart/c/overlay.png?panel=0&thumb=96',
  )
})
