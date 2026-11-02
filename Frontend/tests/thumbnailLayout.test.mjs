import assert from 'node:assert/strict'
import test from 'node:test'
import { fitThumbnails } from '../src/utils/thumbnailLayout.js'

// A comparison has to fit on one screen, like the single map does: the user
// compares the panels by looking from one to the other, which scrolling breaks.
// Every tile shares one image height -- the largest that fits both the width
// and the height the output panel has -- and is as wide as its own shape asks.

const NJ = 0.518
const DE = 0.426
const near = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-6, `${msg}: ${a} vs ${b}`)

test('in a wide, short window the tiles are as tall as the panel allows, and no taller', () => {
  // The reported window: about 1700px of width and 500px of height to spend.
  // Sized from the width alone, NJ came out about 1,640px tall.
  const { tiles } = fitThumbnails({ width: 1700, height: 500, aspects: [NJ, DE], gap: 12, labelHeight: 30 })

  near(tiles[0].height, 470, 'image height = budget - label')
  near(tiles[1].height, 470, 'tiles share one height')
  near(tiles[0].width, 470 * NJ, 'NJ width follows its shape')
  near(tiles[1].width, 470 * DE, 'DE width follows its shape')
})

test('in a narrow window the row fills the width exactly and the tiles shrink to match', () => {
  // 300px across two tiles: the width runs out long before the height does.
  const { tiles } = fitThumbnails({ width: 300, height: 900, aspects: [NJ, DE], gap: 12, labelHeight: 30 })

  near(tiles[0].width + 12 + tiles[1].width, 300, 'row width')
  near(tiles[0].height, tiles[1].height, 'still one shared height')
  near(tiles[0].width / tiles[0].height, NJ, 'shape kept')
  assert.ok(tiles[0].height < 900 - 30)
})

test('more than three panels wrap into rows of three that share the height', () => {
  // Four states in one row would each be a sliver; two rows of 3 + 1 split the
  // height instead, and every row still fits the width.
  const aspects = [0.5, 0.5, 0.5, 0.5]
  const { tiles, rows } = fitThumbnails({ width: 2000, height: 600, aspects, gap: 12, labelHeight: 30 })

  assert.deepEqual(rows, [[0, 1, 2], [3]])
  // Two rows, two labels, one gap between: (600 - 2 * 30 - 12) / 2.
  for (const tile of tiles) near(tile.height, 264, 'height split across rows')
})

test('a width limit in any row binds every row, so all tiles keep one height', () => {
  const aspects = [0.5, 0.5, 2.5, 0.5]
  const { tiles } = fitThumbnails({ width: 600, height: 2000, aspects, gap: 10, labelHeight: 30 })

  // The first row (0.5 + 0.5 + 2.5 = 3.5) is the widest: (600 - 20) / 3.5.
  for (const tile of tiles) near(tile.height, 580 / 3.5, 'shared height')
})

test('too little room gives tiles a minimum height rather than nothing', () => {
  // A very short window, or the first render before anything is measured.
  // Below the floor a tile is unreadable; scrolling is the lesser evil there.
  for (const [width, height] of [[0, 0], [300, 40], [-5, 500]]) {
    const { tiles } = fitThumbnails({ width, height, aspects: [NJ, DE], gap: 12, labelHeight: 30, minHeight: 120 })
    for (const tile of tiles) near(tile.height, 120, `floor at ${width}x${height}`)
  }
})
