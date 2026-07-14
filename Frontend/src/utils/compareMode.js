// Pure state helpers for compare mode (T28). No React here -- AuthenticatedApp
// (App.jsx) owns the actual compareMode/compareCount/compareSelection state;
// these functions just compute the next value so the state machine is
// testable without rendering anything.

// Map and timeseries charts are both comparable (T28 + T29), but never
// mixed within one comparison session -- whichever kind the first-added
// slot establishes is the only kind selectable for the rest of that session.
export const COMPARABLE_CHART_TYPES = ['heatmap', 'timeseries']

// The kind (heatmap|timeseries) the current selection has committed to, or
// null when every slot is still empty and either kind may start it.
export function activeCompareKind(selection) {
  const first = (selection || []).find(Boolean)
  return first ? first.type : null
}

export function isChartComparable(chart, selection = []) {
  if (!chart || !COMPARABLE_CHART_TYPES.includes(chart.type)) return false
  const activeKind = activeCompareKind(selection)
  return activeKind === null || chart.type === activeKind
}

export function createEmptySelection(count) {
  return new Array(count).fill(null)
}

// Slot membership is tracked by object identity -- the same chart object
// reference that lives on the message (msg.charts[i]), matching how
// Chat.jsx already compares focusedOutput.data === chart elsewhere.
export function slotIndexOf(selection, chart) {
  if (!chart) return -1
  return selection.findIndex(slot => slot === chart)
}

export function isSelectionFull(selection) {
  return selection.every(slot => slot !== null)
}

// Click-to-toggle: already-added chart is removed from its slot (freeing
// that specific slot, not shifting others down); a new chart fills the
// first empty slot in order; if every slot is full, the selection is
// returned unchanged with status 'full' so the caller can surface a hint.
export function toggleSlot(selection, chart) {
  const existingIndex = slotIndexOf(selection, chart)
  if (existingIndex !== -1) {
    const next = selection.slice()
    next[existingIndex] = null
    return { selection: next, status: 'removed', index: existingIndex }
  }

  const emptyIndex = selection.findIndex(slot => slot === null)
  if (emptyIndex === -1) {
    return { selection, status: 'full', index: -1 }
  }

  const next = selection.slice()
  next[emptyIndex] = chart
  return { selection: next, status: 'added', index: emptyIndex }
}

export function filledCharts(selection) {
  return (selection || []).filter(Boolean)
}

// The chat output card's slot badge (Chat.jsx OutputCard) -- null when the
// chart isn't in the selection, "In comparison — Slot N" (1-indexed) when it is.
export function compareBadgeLabel(selection, chart) {
  const index = slotIndexOf(selection, chart)
  return index === -1 ? null : `In comparison — Slot ${index + 1}`
}
