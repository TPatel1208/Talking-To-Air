// Pure decision helpers behind CompareGrid's per-slot metadata affordance (T34).
// Kept separate from the CompareGrid component so they're testable with the
// repo's existing node:test pattern -- no React-rendering infra required.

export function focusChartPayload(chart) {
  return { kind: 'chart', data: chart }
}

// Immutable toggle so each SlotGrid cell's expand state is independent --
// opening/closing one slot's info toggle never touches another slot's set
// membership.
export function toggleExpanded(expandedIndices, index) {
  const next = new Set(expandedIndices)
  if (next.has(index)) {
    next.delete(index)
  } else {
    next.add(index)
  }
  return next
}

// Reuses HeatmapCompareBody's existing computeSharedColorScale() result --
// null/unavailable means nothing to explain, mismatched means the note that
// already renders above the grid should also sit next to this slot's
// Source-dataset line.
export function slotScaleNote(shared) {
  if (!shared || shared.available) return null
  return shared.reason || null
}
