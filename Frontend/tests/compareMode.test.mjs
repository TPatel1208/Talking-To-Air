import assert from 'node:assert/strict'
import test from 'node:test'
import {
  activeCompareKind,
  compareBadgeLabel,
  createEmptySelection,
  filledCharts,
  isChartComparable,
  isSelectionFull,
  slotIndexOf,
  toggleSlot,
} from '../src/utils/compareMode.js'

const heatmapA = { type: 'heatmap', title: 'A' }
const heatmapB = { type: 'heatmap', title: 'B' }
const heatmapC = { type: 'heatmap', title: 'C' }
const timeseries = { type: 'timeseries', title: 'D' }

test('createEmptySelection makes an all-null array of the requested length', () => {
  assert.deepEqual(createEmptySelection(3), [null, null, null])
})

test('toggleSlot fills slots in click order, left to right', () => {
  let selection = createEmptySelection(3)
  let result = toggleSlot(selection, heatmapA)
  assert.equal(result.status, 'added')
  assert.equal(result.index, 0)
  selection = result.selection

  result = toggleSlot(selection, heatmapB)
  assert.equal(result.status, 'added')
  assert.equal(result.index, 1)
  selection = result.selection

  assert.deepEqual(selection, [heatmapA, heatmapB, null])
})

test('toggleSlot removes a chart already present, freeing its own slot without shifting others', () => {
  let selection = [heatmapA, heatmapB, null]
  const result = toggleSlot(selection, heatmapA)
  assert.equal(result.status, 'removed')
  assert.equal(result.index, 0)
  assert.deepEqual(result.selection, [null, heatmapB, null])
})

test('toggleSlot re-adds a removed chart into the next empty slot, not necessarily its old one', () => {
  let selection = [null, heatmapB, null]
  const result = toggleSlot(selection, heatmapA)
  assert.equal(result.status, 'added')
  assert.equal(result.index, 0)
  assert.deepEqual(result.selection, [heatmapA, heatmapB, null])
})

test('toggleSlot rejects a new chart when every slot is already filled', () => {
  const full = [heatmapA, heatmapB, heatmapC]
  const result = toggleSlot(full, timeseries)
  assert.equal(result.status, 'full')
  assert.equal(result.index, -1)
  assert.deepEqual(result.selection, full) // unchanged, no silent eviction
})

test('isSelectionFull is true only once every slot holds a chart', () => {
  assert.equal(isSelectionFull([heatmapA, null]), false)
  assert.equal(isSelectionFull([heatmapA, heatmapB]), true)
})

test('slotIndexOf matches by object identity, not by value equality', () => {
  const selection = [heatmapA, heatmapB, null]
  assert.equal(slotIndexOf(selection, heatmapB), 1)
  assert.equal(slotIndexOf(selection, { type: 'heatmap', title: 'B' }), -1)
  assert.equal(slotIndexOf(selection, null), -1)
})

test('filledCharts drops empty slots', () => {
  assert.deepEqual(filledCharts([heatmapA, null, heatmapC]), [heatmapA, heatmapC])
})

test('isChartComparable accepts both heatmap and timeseries when no kind is committed yet', () => {
  assert.equal(isChartComparable(heatmapA, []), true)
  assert.equal(isChartComparable(timeseries, []), true)
  assert.equal(isChartComparable({ type: 'heatmap_multi' }, []), false)
  assert.equal(isChartComparable(null, []), false)
})

test('activeCompareKind is null until the first slot is filled, then locks to that chart\'s type', () => {
  assert.equal(activeCompareKind([null, null]), null)
  assert.equal(activeCompareKind([heatmapA, null]), 'heatmap')
  assert.equal(activeCompareKind([null, timeseries]), 'timeseries')
})

test('isChartComparable gates on the kind the first-added slot established (T29) -- same-kind-only, mixed comparisons impossible', () => {
  const heatmapSession = [heatmapA, null]
  assert.equal(isChartComparable(heatmapB, heatmapSession), true)
  assert.equal(isChartComparable(timeseries, heatmapSession), false)

  const timeseriesSession = [timeseries, null]
  assert.equal(isChartComparable(heatmapA, timeseriesSession), false)
  assert.equal(isChartComparable({ type: 'timeseries', title: 'E' }, timeseriesSession), true)
})

test('isChartComparable defaults selection to empty (both kinds open) when the caller omits it', () => {
  assert.equal(isChartComparable(heatmapA), true)
  assert.equal(isChartComparable(timeseries), true)
})

test('compareBadgeLabel reports the 1-indexed slot a chart occupies', () => {
  const selection = [null, heatmapB, heatmapA]
  assert.equal(compareBadgeLabel(selection, heatmapA), 'In comparison — Slot 3')
  assert.equal(compareBadgeLabel(selection, heatmapB), 'In comparison — Slot 2')
})

test('compareBadgeLabel is null when the chart is not in the selection', () => {
  const selection = [null, heatmapB, null]
  assert.equal(compareBadgeLabel(selection, heatmapA), null)
  assert.equal(compareBadgeLabel(selection, heatmapC), null)
})
