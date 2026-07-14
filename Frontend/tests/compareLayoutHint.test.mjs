import assert from 'node:assert/strict'
import test from 'node:test'
import { shouldShowCollapseHint } from '../src/utils/compareLayoutHint.js'

const expanded = { compareMode: 'active', sessionsCollapsed: false, chatCollapsed: false, rightPanelCollapsed: false }

test('shows the hint once compare mode is active and every side panel is still expanded', () => {
  assert.equal(shouldShowCollapseHint(expanded), true)
})

test('hides once any one side panel is already collapsed', () => {
  assert.equal(shouldShowCollapseHint({ ...expanded, sessionsCollapsed: true }), false)
  assert.equal(shouldShowCollapseHint({ ...expanded, chatCollapsed: true }), false)
  assert.equal(shouldShowCollapseHint({ ...expanded, rightPanelCollapsed: true }), false)
})

test('never shows outside active compare mode', () => {
  assert.equal(shouldShowCollapseHint({ ...expanded, compareMode: 'off' }), false)
  assert.equal(shouldShowCollapseHint({ ...expanded, compareMode: 'choosing-count' }), false)
})
