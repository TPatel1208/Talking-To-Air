import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { PANEL_MIN_WIDTH, PANEL_PADDING_X, SCRUB_TRACK_MIN_WIDTH } from '../src/utils/panelLayout.js'

// There is no jsdom here and a width is a DOM measurement, so the layout is
// pinned the only way this repo can pin a component: by reading the source, the
// same idiom as SharedDeltaThresholdTests and FinishedRowStatusContractTests.
//
// The defect these exist for was measured live at a 556 px viewport with the
// sessions, chat and jobs panels open: the output panel rendered 0 px wide and
// was pushed off-screen at left: 634, taking the map with it -- .maplibregl-map
// measured 0 px too. Phase 9 recorded it as a slider defect because the slider
// was where it showed up.
const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf-8')
const OUTPUT_PANEL = read('../src/components/OutputPanel.jsx')
const APP = read('../src/App.jsx')
const SIDEBAR = read('../src/components/SessionSidebar.jsx')
const RIGHT_PANEL = read('../src/components/RightPanel.jsx')

test('no output-panel root can be squeezed to nothing', () => {
  // `minWidth: 0` on a `flex: 1` child is what defeats the automatic minimum
  // content size, and it was on all five render branches -- compare, empty,
  // artifact, chart and the artifact-tabs panel. One branch keeping it is the
  // whole bug back on whichever screen happens to hit that branch, which is
  // why this bans the shape rather than checking the ones that exist today.
  assert.ok(PANEL_MIN_WIDTH > 0)

  const zeroFloorRoots = OUTPUT_PANEL.match(/flex: 1,\s*minWidth: 0\b/g) || []
  assert.deepEqual(zeroFloorRoots, [],
    'a flex: 1 root still declares minWidth: 0, so the output column can be '
    + 'squeezed to 0 px and pushed off-screen again')
})

test('the floor is one number, not five copies of it', () => {
  // Five branches with five literals is five chances for one of them to drift,
  // and the one that drifts is the branch nobody opens on a narrow screen.
  assert.ok(Number.isFinite(PANEL_MIN_WIDTH) && PANEL_MIN_WIDTH > 0)

  // The shared style is the only place a panel floor is written.
  assert.match(OUTPUT_PANEL, /const panelRootStyle = \{[^}]*minWidth: `\$\{PANEL_MIN_WIDTH\}px`/s,
    'panelRootStyle does not take its floor from PANEL_MIN_WIDTH')

  // ...and every branch reaches for it rather than re-declaring the shape. The
  // count is asserted so a NEW branch added later with its own inline root
  // fails here instead of silently reintroducing the collapse on one screen.
  const roots = OUTPUT_PANEL.match(/<div style=\{panelRoot(?:Style|Clipped)\}>/g) || []
  assert.equal(roots.length, 5,
    `expected all 5 render branches to use the shared root style, found ${roots.length}`)
})

test('the floor clears the scrubber it has to contain', () => {
  // The panel exists to contain the track, so its floor must clear the track's.
  // Derived rather than asserted where possible -- but pinned here too, because
  // the derivation is the thing that would get "simplified" into a literal.
  assert.ok(
    PANEL_MIN_WIDTH >= SCRUB_TRACK_MIN_WIDTH + PANEL_PADDING_X * 2,
    `the output panel floor (${PANEL_MIN_WIDTH}px) is under the scrubber floor `
    + `plus its padding, so the control it exists to keep usable no longer fits`,
  )
})

test('the scrubber takes its floor from the shared constant, not a literal', () => {
  // Two independently-written widths is how the inner one quietly stops fitting
  // inside the outer one.
  const scrubber = read('../src/components/MapScrubber.jsx')

  assert.match(scrubber, /SCRUB_TRACK_MIN_WIDTH/,
    'MapScrubber hardcodes its min-width instead of sharing one with the panel '
    + 'that has to contain it')
  assert.doesNotMatch(scrubber, /minWidth: '\d+px'/,
    'MapScrubber still carries a literal min-width')
})

test('when the panels cannot all fit, the app scrolls rather than hiding one', () => {
  // A floor alone only moves the defect: with `overflow: hidden` on the row, a
  // panel that no longer fits is simply invisible and unreachable, which is
  // what "0 px and off-screen at left: 634" actually was. The row has to admit
  // it has more content than width.
  const root = APP.match(/<div style=\{\{[^}]*display:\s*'flex',[^}]*height:\s*'100%',[^}]*\}\}>/s)
  assert.ok(root, 'could not find the app root flex row')

  assert.doesNotMatch(root[0], /overflow:\s*'hidden'/,
    'the app root still hides its overflow, so a panel squeezed out of the row '
    + 'is unreachable rather than scrollable')
  assert.match(root[0], /overflowX:\s*'auto'/,
    'the app root does not allow horizontal scrolling')
})

test('the side panels absorb nothing, which is why the output panel needs the floor', () => {
  // The reasoning the fix rests on, pinned so it fails if someone makes a side
  // panel flexible: all three are fixed-width and flexShrink: 0, so the output
  // panel is the ONLY child that can absorb a narrow viewport, and without a
  // floor it absorbs all of it.
  assert.match(SIDEBAR, /flexShrink:\s*0/)
  assert.match(RIGHT_PANEL, /flexShrink:\s*0/)
  assert.match(APP, /width: '380px', flexShrink: 0/)
})
