import assert from 'node:assert/strict'
import test from 'node:test'
import { thumbnailLegends } from '../src/utils/thumbnailLegend.js'

// The thumbnails are already-rendered images, so a legend can only describe
// the scale each one was drawn with -- it cannot recolor them onto a common
// one the way compare mode's computeSharedColorScale does.

const LUT = [[68, 1, 84, 255], [253, 231, 37, 255]]
const panel = (overrides = {}) => ({
  vmin: 2.0957e13,
  vmax: 1.4178e16,
  units: 'molecules/cm^2',
  colormap: { name: 'viridis', lut: LUT },
  scale: { method: 'percentile', p: [2, 98] },
  ...overrides,
})

test('panels drawn on one scale share one legend', () => {
  // cmp_11192fae0e05: both panels carry the same vmin/vmax, so purple Delaware
  // against yellow New Jersey is a real difference -- and nothing said so.
  const { shared, perPanel } = thumbnailLegends([panel(), panel()])

  assert.deepEqual(shared, {
    vmin: 2.0957e13,
    vmax: 1.4178e16,
    lut: LUT,
    units: 'molecules/cm^2',
    clipNote: 'Color scale clipped at 2nd–98th percentile',
  })
  assert.deepEqual(perPanel, [null, null])
})

test('panels drawn on different scales each get their own legend', () => {
  // One shared legend here would invite reading the same color as the same
  // value across tiles when it is not.
  const { shared, perPanel } = thumbnailLegends([panel(), panel({ vmax: 9e15 })])

  assert.equal(shared, null)
  assert.equal(perPanel[0].vmax, 1.4178e16)
  assert.equal(perPanel[1].vmax, 9e15)
})

test('different units or colormaps are different scales', () => {
  assert.equal(thumbnailLegends([panel(), panel({ units: 'DU' })]).shared, null)
  assert.equal(thumbnailLegends([panel(), panel({ colormap: { name: 'magma', lut: LUT } })]).shared, null)
})

test('a panel with no usable scale gets no legend', () => {
  const { shared, perPanel } = thumbnailLegends([panel(), panel({ vmax: null }), panel({ colormap: null })])

  assert.equal(shared, null)
  assert.equal(perPanel[0].vmax, 1.4178e16)
  assert.deepEqual(perPanel.slice(1), [null, null])
})
