// Real summary stats derived from a chart payload's own values — never invented.
export function computeChartStats(chart) {
  if (!chart) return null

  if (chart.type === 'heatmap' || chart.type === 'heatmap_multi') {
    const payload = chart.type === 'heatmap_multi' ? chart.panels?.[0] : chart
    if (!payload) return null
    return statsFromPayload(payload) || statsFromValues(rawCellValues(payload), payload.units, 'rendered-grid')
  }

  if (chart.type === 'timeseries') {
    return statsFromValues(chart.values || [], chart.units, 'series')
  }

  return null
}

// Statistics the backend computed on the FULL-resolution field, before the
// grid was thinned for rendering (plot_tools._field_statistics). Recomputing
// them here from `values` would describe the thumbnail instead: the grid is
// capped at _MAX_GRID_CELLS by a uniform stride, which steps over exactly the
// small-area features that set the extremes. Returns null for a payload that
// predates this field, so an older artifact still renders its own numbers.
function statsFromPayload(payload) {
  const s = payload.statistics
  if (!s || !Number.isFinite(s.count) || s.count <= 0) return null
  if (![s.mean, s.min, s.max].every(Number.isFinite)) return null

  return {
    mean: s.mean,
    max: s.max,
    min: s.min,
    units: payload.units,
    basis: 'analyzed-region',
    count: s.count,
    validPct: Number.isFinite(s.valid_fraction) ? s.valid_fraction * 100 : 0,
  }
}

// Every cell of the payload's own grid, nulls included. Masked pixels (QA
// drops, fill values, out-of-region cells) are serialized as null by the
// backend and must count against validPct — a flatten that skips nulls is
// right for rendering but would make validPct read 100% no matter how much
// of the field was masked away.
function rawCellValues(payload) {
  if (Array.isArray(payload.values) && Array.isArray(payload.values[0])) {
    return payload.values.flat()
  }
  // Legacy read path only. `points` was a flattened duplicate of the same
  // field that the backend stopped emitting (plot_tools._build_heatmap_payload)
  // once nothing rendered from it. Charts are durable rows in Postgres, so
  // payloads that shipped it — and predate `statistics`, or this branch is
  // never reached — are still openable and this is the only value array they
  // carry. Note it holds finite cells only, so an old chart's validPct reads
  // 100%; that is what those payloads have always reported here.
  const points = payload.points
  if (points && Array.isArray(points.values)) return points.values
  return []
}

function statsFromValues(values, units, basis) {
  const finite = values.filter(Number.isFinite)
  if (!finite.length) return null

  const sum = finite.reduce((acc, v) => acc + v, 0)
  const mean = sum / finite.length
  const max = Math.max(...finite)
  const min = Math.min(...finite)
  const validPct = values.length ? (finite.length / values.length) * 100 : 0

  return {
    mean, max, min, units, basis,
    count: finite.length,
    validPct,
  }
}
