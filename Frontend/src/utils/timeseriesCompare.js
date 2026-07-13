// Compatibility/fallback decision for compare mode's chart-overlay path (T29).
// Given the filled timeseries payloads in a comparison session, decides
// whether they can be overlaid onto one shared Plotly figure (matching units,
// and every pair of time ranges overlapping by at least one point -- no
// partial-overlap heuristics, the rule is binary) or must fall back to T28's
// small-multiple grid. No dual-axis mode: any mismatch means "fall back".

function unitsOf(chart) {
  return chart?.units ?? chart?.provenance?.units ?? ''
}

// Null when a chart carries no parseable timestamps -- callers treat that as
// "can't prove overlap", which resolves to the safe fallback (grid).
function timeRangeOf(chart) {
  const times = chart?.times || []
  const parsed = times.map(t => new Date(t).getTime()).filter(Number.isFinite)
  if (!parsed.length) return null
  return { start: Math.min(...parsed), end: Math.max(...parsed) }
}

function rangesOverlap(a, b) {
  return a.start <= b.end && b.start <= a.end
}

// True only when every pair of filled charts' time ranges overlaps by at
// least one point -- a chain of pairwise-adjacent overlaps isn't enough.
export function allRangesOverlap(charts) {
  const ranges = charts.map(timeRangeOf)
  if (ranges.some(r => r === null)) return false
  for (let i = 0; i < ranges.length; i++) {
    for (let j = i + 1; j < ranges.length; j++) {
      if (!rangesOverlap(ranges[i], ranges[j])) return false
    }
  }
  return true
}

const UNITS_MISMATCH_REASON = 'Different units — showing separate charts'
const RANGE_MISMATCH_REASON = "Time ranges don't overlap — showing separate charts"

// charts: array of filled (non-null) timeseries chart payloads.
export function timeseriesOverlayCompatible(charts) {
  const filled = (charts || []).filter(Boolean)

  if (filled.length < 2) {
    return { compatible: false, reason: null }
  }

  const units = unitsOf(filled[0])
  const unitsMatch = filled.every(c => unitsOf(c) === units)
  if (!unitsMatch) {
    return { compatible: false, reason: UNITS_MISMATCH_REASON }
  }

  if (!allRangesOverlap(filled)) {
    return { compatible: false, reason: RANGE_MISMATCH_REASON }
  }

  return { compatible: true, reason: null }
}

// Legend/series label: most distinguishing identifying metadata the payload
// already carries -- region name, then title, then the date range -- falling
// back to a generic "Series N" if nothing better is available.
export function seriesLabel(chart, index) {
  const p = chart?.provenance || {}
  if (p.region_name) return p.region_name
  if (chart?.title) return chart.title
  if (p.start_date && p.end_date) return `${p.start_date} – ${p.end_date}`
  return `Series ${index + 1}`
}

export function toOverlaySeries(charts) {
  return (charts || []).filter(Boolean).map((chart, i) => ({
    times: chart.times || [],
    values: chart.values || [],
    label: seriesLabel(chart, i),
    units: unitsOf(chart),
  }))
}

// Distinguishable, colorblind-tolerant palette for overlay traces -- cycles
// if there are ever more series than colors (compare mode caps at 4).
const OVERLAY_COLORS = ['#1D9E75', '#D97706', '#2563EB', '#DB2777', '#7C3AED', '#DC2626', '#0891B2', '#65A30D']

export function overlayColor(index) {
  return OVERLAY_COLORS[index % OVERLAY_COLORS.length]
}

// Pure Plotly trace builder -- one trace (and therefore one legend entry)
// per series, kept separate from the React component so it's testable
// without a DOM.
export function buildOverlayTraces(series) {
  return (series || []).map((s, i) => {
    const color = overlayColor(i)
    return {
      type: 'scatter',
      mode: 'lines+markers',
      name: s.label,
      x: s.times,
      y: s.values,
      line: { color, width: 2 },
      marker: { color, size: 5 },
      hovertemplate: `%{x|%Y-%m-%d %H:%M}<br>${s.label}: %{y:.3e} ${s.units || ''}<extra></extra>`,
    }
  })
}
