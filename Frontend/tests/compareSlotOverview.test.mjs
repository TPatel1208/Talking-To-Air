import assert from 'node:assert/strict'
import test from 'node:test'
import { focusChartPayload, toggleExpanded, slotScaleNote } from '../src/utils/compareSlotOverview.js'

test('focusChartPayload wraps a chart into the same {kind, data} shape the single-chart focus path expects', () => {
  const chart = { type: 'heatmap', title: 'A' }
  const result = focusChartPayload(chart)
  assert.equal(result.kind, 'chart')
  assert.equal(result.data, chart) // identity, not a copy -- App.jsx matches active state by reference
})

test('toggleExpanded opens a collapsed slot without affecting other already-open slots', () => {
  const expanded = new Set([1])
  const result = toggleExpanded(expanded, 0)
  assert.deepEqual([...result].sort(), [0, 1])
  assert.deepEqual([...expanded], [1]) // original set untouched
})

test('toggleExpanded closes an already-open slot, leaving the rest untouched', () => {
  const expanded = new Set([0, 1, 2])
  const result = toggleExpanded(expanded, 1)
  assert.deepEqual([...result].sort(), [0, 2])
})

test('slotScaleNote surfaces the shared-scale mismatch reason when panels are on independent scales', () => {
  const shared = { available: false, reason: 'Different variables — showing independent scales' }
  assert.equal(slotScaleNote(shared), shared.reason)
})

test('slotScaleNote is null when a shared scale is in use -- nothing to explain', () => {
  const shared = { available: true, reason: null }
  assert.equal(slotScaleNote(shared), null)
})

test('slotScaleNote is null when there is no shared-scale context at all (e.g. timeseries small-multiple grid)', () => {
  assert.equal(slotScaleNote(null), null)
  assert.equal(slotScaleNote(undefined), null)
})
