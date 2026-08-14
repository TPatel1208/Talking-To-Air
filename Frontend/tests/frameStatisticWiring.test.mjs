import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The seams the toggle rides on live in JSX, and there is no jsdom here, so
// they are asserted against the source — the idiom Phase 10 established for
// `ScrubTrack` and `SharedDeltaThresholdTests` for `_DELTA_HIGH`. Every one of
// these is a link a util test cannot see, because a util test hands itself the
// argument the component forgot: Phase 8's falsified sentence was true in
// `frameDelta.js` and false on screen for two phases for exactly that reason.
const PANEL = readFileSync(new URL('../src/components/OutputPanel.jsx', import.meta.url), 'utf-8')
const SCRUBBER = readFileSync(new URL('../src/components/MapScrubber.jsx', import.meta.url), 'utf-8')

test('everything that speaks keys off the RENDERED statistic, never the selected one', () => {
  // The whole of design tension 3 in one assertion. `selected` is what the
  // reader clicked; `rendered` is what the canvas actually holds. Wiring the
  // scale, the sentences or the Statistics tab to `selected` puts a max legend,
  // a max anchor and a max scoping over the mean's pixels for the length of a
  // fetch — which is Phase 13's statistic guard defeated on the client side,
  // and a wrong plane renders believably.
  for (const call of [
    /resolveScrubberScale\(chart, scrubbing, statistic\.rendered\)/,
    /resolveFrameDelta\(chart, statistic\.rendered\)/,
    /statsForStop\(chart, currentStop, statistic\.rendered\)/,
    /extentOverstatementNote\(chart, statistic\.rendered\)/,
    /<StatisticsTab[^>]*statistic=\{statistic\.rendered\}/s,
  ]) {
    assert.match(PANEL, call, `not keyed to statistic.rendered: ${call}`)
  }
})

test('the bytes are fetched for what was ASKED for, so a failure is discoverable', () => {
  // The mirror of the rule above. `rendered` is derived from the load state, so
  // fetching `rendered` would be circular: it can never leave the mean and the
  // plane is never requested at all.
  assert.match(PANEL, /resolveStatisticSource\(chart, askedStatistic\)/)
  assert.match(PANEL, /useFrameStack\(frameSource,/)
})

test('scrubbing does not silently reset the statistic, and the toggle does not reset the stop', () => {
  // Two halves of one bug. `selectStop` dropping `statistic` would send the map
  // back to the mean on the first drag with the max button still lit; the
  // toggle dropping `index` would cost the reader their place on every switch,
  // punishing exactly the person who scrubbed to a peak and asked to see it.
  const stop = PANEL.match(/const selectStop = [^\n]*\n/)?.[0] || ''
  const statistic = PANEL.match(/const selectStatistic = [^\n]*\n/)?.[0] || ''

  assert.match(stop, /\.\.\.scrubChoice/, 'selectStop rebuilds the choice and drops the statistic')
  assert.match(statistic, /\.\.\.scrubChoice/, 'selectStatistic rebuilds the choice and drops the stop')
  assert.doesNotMatch(statistic, /index:/, 'selectStatistic overwrites the remembered stop')
})

test('the degradation sentence names the statistic whose bytes are missing', () => {
  // "The frame values for this map have expired" is false of a chart whose mean
  // is sitting there intact — `store_frame_stack` protects it from its own
  // planes' evictions and degrades one statistic at a time.
  assert.match(PANEL, /resolveFrameState\(chart, frameLoad\.loadState, statistic\.selected\)/)
})

test('the scrubber renders the toggle, its refusal and the overstatement', () => {
  assert.match(PANEL, /<MapScrubber[\s\S]*?choice=\{statistic\}/, 'MapScrubber is not given the choice')
  assert.match(PANEL, /<MapScrubber[\s\S]*?overstatement=\{overstatement\}/, 'MapScrubber is not given the overstatement')
  assert.match(SCRUBBER, /<StatisticChoice choice=\{choice\}/, 'the toggle is never rendered')
  assert.match(SCRUBBER, /\{overstatement && /, 'the overstatement is never rendered')
  // Phase 5 decision 2, one level in: a reader who finds no toggle and no
  // reason is the thing the disclosure exists to prevent.
  assert.match(SCRUBBER, /choice\.refusal\?\.detail/, 'the plane refusal is never shown')
})

test('the toggle offers the payload’s own options rather than a list written here', () => {
  // A hardcoded ['mean','max','min'] renders a button for a plane that failed
  // to store, and the reader watches it 404.
  assert.match(SCRUBBER, /choice\.options\.map\(/)
  assert.doesNotMatch(SCRUBBER, /\[\s*'mean'\s*,\s*'max'/)
})

test('no component reaches for the store key, in any statistic', () => {
  // `_key` addresses the blob store directly and is not the frontend's to
  // hold. It sits beside `url` in all four blocks now, not just the mean's.
  for (const [name, source] of [['OutputPanel', PANEL], ['MapScrubber', SCRUBBER]]) {
    assert.doesNotMatch(source, /_key/, `${name} reads the store key`)
  }
})

test('leaving scrubber mode hands the Map tab back with no statistic attached', () => {
  // D11's exception again, from the other side. Re-entering the scrubber on a
  // chart last left in max mode must not put a max plane back under a Map tab
  // that spent the interval drawing a mean, and the scale must be released:
  // `resolveScrubberScale` returns null outside the mode whatever the toggle
  // last said, because `scrubbing` is its own first gate.
  const toggle = PANEL.match(/const toggleScrub = [^\n]*\n/)?.[0] || ''

  assert.match(toggle, /statistic: 'mean'/, 'the mode toggle carries the last statistic back in')
  assert.match(toggle, /index: 0/, 'the mode toggle carries the last stop back in')
  assert.match(PANEL, /resolveScrubberScale\(chart, scrubbing,/, 'the scale no longer gates on the mode')
})

test('the Map tab is never handed the statistic', () => {
  // D11's explicit exception. `plot_singular` has no statistic parameter and
  // the Map tab is always a period mean; making it follow the toggle would move
  // the export, methods.md, the Statistics tab, the comparison panels and the
  // agent's tool contract, and is a T-number of its own.
  const map = PANEL.match(/<MapLibreHeatmapPanel[\s\S]*?\/>/)?.[0] || ''

  assert.ok(map, 'MapLibreHeatmapPanel is not rendered')
  assert.doesNotMatch(map, /statistic/)
})
