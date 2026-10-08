import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { BORDER_LINE, overlayBeforeId } from '../src/utils/regionBorders.js'

// The comparison thumbnails draw their state outlines as svg, the single map
// as MapLibre line layers. Both must take their style from BORDER_LINE so a
// thumbnail looks like the map it expands into. There is no jsdom to render
// either, so the wiring is checked in the source text.
const src = (name) => readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'components', name), 'utf8')

test('a border is a dark line on a light halo, so it reads across the whole colormap', () => {
  // A dark line alone vanished on viridis's dark purple, which is most of any
  // low-signal field; a light line alone would vanish on its yellow.
  const { color, width, haloColor, haloWidth } = BORDER_LINE
  assert.match(color, /^rgba\(\s*[0-4]?\d,\s*[0-4]?\d,\s*[0-4]?\d,/, 'dark core')
  assert.match(haloColor, /^rgba\(255,\s*255,\s*255,/, 'light halo')
  assert.ok(haloWidth > width + 1, 'the halo shows on both sides of the line')
})

test('the single map and the thumbnails both draw the halo under the line', () => {
  const map = src('MapLibreHeatmapPanel.jsx')
  const halo = map.indexOf("'line-color': BORDER_LINE.haloColor, 'line-width': BORDER_LINE.haloWidth")
  const line = map.indexOf("'line-color': BORDER_LINE.color, 'line-width': BORDER_LINE.width")
  assert.ok(halo !== -1 && line !== -1 && halo < line, 'map: halo layer added before the line layer')

  const thumbs = src('HeatmapMultiPanel.jsx')
  const svgHalo = thumbs.indexOf('stroke={BORDER_LINE.haloColor} strokeWidth={BORDER_LINE.haloWidth}')
  const svgLine = thumbs.indexOf('stroke={BORDER_LINE.color} strokeWidth={BORDER_LINE.width}')
  assert.ok(svgHalo !== -1 && svgLine !== -1 && svgHalo < svgLine, 'thumbnail: halo path painted before the line')
})

test('a redrawn overlay goes under the borders, not on top of them', () => {
  // MapLibre stacks a layer added without a beforeId on top. The overlay is
  // re-added on every recolor (a scrubber frame, compare's auto-scale) after
  // the borders already exist, which buried them under the data.
  const mapWith = (...ids) => ({ getLayer: (id) => (ids.includes(id) ? { id } : undefined) })

  assert.equal(overlayBeforeId(mapWith('region-borders-halo', 'region-borders')), 'region-borders-halo')
  assert.equal(overlayBeforeId(mapWith()), undefined, 'before the borders load, the overlay simply goes on top')
})

test('every overlay layer the single map adds is placed with it', () => {
  // Each `id: 'overlay'` layer spec, up to the next statement that adds or
  // removes anything, must close with the beforeId.
  const map = src('MapLibreHeatmapPanel.jsx')
  const starts = [...map.matchAll(/id: 'overlay',/g)].map(m => m.index)
  assert.equal(starts.length, 2, 'native and canvas overlays')
  for (const start of starts) {
    const call = map.slice(start, start + 400).split(/map\.(?:add|remove)/)[0]
    assert.match(call, /\},\s*overlayBeforeId\(map\)\)/)
  }
})
