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

test('a selected frame forces canvas, whatever the scale and url say (T59)', () => {
  // The server PNG is the period aggregate at native resolution -- it cannot
  // show an hour. A stack whose pooled scale came back null (nothing survived
  // masking) still has frames to draw, so the frame alone has to be enough to
  // take the canvas path.
  const frame = { values: new Float32Array(4), lats: [0, 1], lons: [0, 1] }

  assert.equal(resolveOverlayMode(null, '/chart/abc.png', frame), 'canvas')
  assert.equal(resolveOverlayMode({ vmin: 0, vmax: 1 }, '/chart/abc.png', frame), 'canvas')
})

test('leaving scrubber mode hands the native overlay back', () => {
  assert.equal(resolveOverlayMode(null, '/chart/abc.png', null), 'native')
})
