import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveOverlayMode } from '../src/utils/overlayMode.js'

test('no override with a native overlay url resolves to native', () => {
  assert.equal(resolveOverlayMode(null, '/chart/abc.png'), 'native')
})

test('an active override always resolves to canvas, even with a native url available', () => {
  assert.equal(resolveOverlayMode({ vmin: 0, vmax: 1 }, '/chart/abc.png'), 'canvas')
})

test('no override and no native url falls back to canvas', () => {
  assert.equal(resolveOverlayMode(null, undefined), 'canvas')
})

test('toggling an override off while a native url exists flips the mode back to native', () => {
  // Regression: compare mode's "auto-scale each" toggle going back on used to
  // leave the map showing a stale canvas frame because the recolor effect
  // short-circuited on override-is-falsy instead of re-resolving the mode.
  assert.equal(resolveOverlayMode({ vmin: 0, vmax: 1 }, '/chart/abc.png'), 'canvas')
  assert.equal(resolveOverlayMode(null, '/chart/abc.png'), 'native')
})
