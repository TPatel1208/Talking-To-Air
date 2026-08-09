import test from 'node:test'
import assert from 'node:assert/strict'

import { levelResolutionFields } from '../src/utils/metadataDisplay.js'

// Mirrors Backend/preprocessing/level_resolver.py::LevelResolution as it
// arrives on chart.provenance.level_resolution, with the numbers the Phase 1
// spike measured on a real TEMPO_O3PROF retrieval over New Jersey.
function chartWithLevel(overrides = {}) {
  return {
    provenance: {
      level_resolution: {
        index: 19,
        selector_value: 19,
        kind: 'pressure',
        units: 'hPa',
        requested: 300.0,
        resolved_level: 260.0476,
        level_error: 39.9524,
        dominant_fraction: 0.831,
        runner_up: 20,
        runner_up_fraction: 0.169,
        margin: 0.662,
        n_pixels: 2688,
        excluded_fraction: 0,
        resolved_level_spread: 0,
        axis_variable: 'ozone_profile_pressure',
        ...overrides,
      },
    },
  }
}

test('a chart that selected no physical level has no level section', () => {
  assert.equal(levelResolutionFields({ provenance: {} }), null)
  assert.equal(levelResolutionFields(null), null)
})

test('the requested level and the level actually delivered are both named', () => {
  // The whole point of the disclosure: "nearest available layer to 300 hPa" is
  // a different claim from "a 300 hPa map", and only showing both makes that
  // visible.
  const fields = levelResolutionFields(chartWithLevel())

  assert.equal(fields.requested, '300 hPa')
  assert.equal(fields.resolved, '260.05 hPa (layer 19)')
  assert.equal(fields.levelError, '39.95 hPa from the level requested')
})

test('agreement is reported with its runner-up, not as a bare percentage', () => {
  // A lone "83.1%" does not say what the other 16.9% did. Naming the runner-up
  // layer is what lets a reader judge whether the map would have looked
  // different done properly.
  const fields = levelResolutionFields(chartWithLevel())

  assert.equal(
    fields.agreement,
    '83.1% of the analyzed area resolves to layer 19 (2,688 pixel-columns sampled)',
  )
  assert.equal(fields.runnerUp, '16.9% resolves to layer 20 instead')
})

test('unanimous agreement says so instead of naming an absent runner-up', () => {
  const fields = levelResolutionFields(chartWithLevel({
    dominant_fraction: 1.0, runner_up: null, runner_up_fraction: 0.0, margin: 1.0,
  }))

  assert.equal(
    fields.agreement,
    '100% of the analyzed area resolves to layer 19 (2,688 pixel-columns sampled)',
  )
  assert.equal(fields.runnerUp, null)
})

test('a level error of zero is still stated rather than hidden', () => {
  // "Not shown" and "exact" read identically to someone scanning the panel,
  // and they are different facts.
  const fields = levelResolutionFields(chartWithLevel({
    requested: 260.0476, resolved_level: 260.0476, level_error: 0.0,
  }))

  assert.equal(fields.levelError, 'exactly the level requested')
})

test('an altitude selection is described in its own units', () => {
  const fields = levelResolutionFields(chartWithLevel({
    kind: 'altitude', units: 'km', requested: 26, resolved_level: 26.9749,
    level_error: 0.9749, index: 12, runner_up: null, runner_up_fraction: 0, dominant_fraction: 1,
  }))

  assert.equal(fields.requested, '26 km')
  assert.equal(fields.resolved, '26.97 km (layer 12)')
  assert.equal(fields.levelError, '0.97 km from the level requested')
  assert.equal(fields.kind, 'altitude')
})

test('the axis variable the level was resolved against is named', () => {
  // Which of a product's vertical axes answered the question is provenance:
  // TEMPO_O3PROF publishes two, and the answer differs between them.
  const fields = levelResolutionFields(chartWithLevel())

  assert.equal(fields.axisVariable, 'ozone_profile_pressure')
})

test('an error too small to display is reported as a band, not rounded to zero', () => {
  // "0 hPa from the level requested" contradicts itself, and it collapses the
  // exact-vs-approximate distinction the zero branch exists to preserve.
  const fields = levelResolutionFields(chartWithLevel({ level_error: 0.004 }))

  assert.equal(fields.levelError, 'less than 0.005 hPa from the level requested')
})

test('a partial payload degrades to missing lines rather than throwing', () => {
  // This block renders outside fmt()/NOT_AVAILABLE, so an unguarded field takes
  // the whole Metadata tab down instead of leaving one row blank.
  const fields = levelResolutionFields({ provenance: { level_resolution: { index: 1 } } })

  assert.equal(fields.resolved, null)
  assert.equal(fields.levelError, null)
  assert.equal(fields.agreement, null)
})

test('the excluded fraction of the region is surfaced when it is non-zero', () => {
  assert.equal(levelResolutionFields(chartWithLevel()).excluded, null)

  const sparse = levelResolutionFields(chartWithLevel({ excluded_fraction: 0.42 }))
  assert.equal(sparse.excluded, '42% of the region had no usable vertical coordinate')
})

test('a layer that varies across the region says so', () => {
  assert.equal(levelResolutionFields(chartWithLevel()).spread, null)

  const varying = levelResolutionFields(chartWithLevel({ resolved_level_spread: 38.4 }))
  assert.equal(varying.spread, 'This layer itself ranges 38.4 hPa across the region')
})
