import { flattenPayload } from './flattenPayload.js'

// Real summary stats derived from a chart payload's own values — never invented.
export function computeChartStats(chart) {
  if (!chart) return null

  if (chart.type === 'heatmap' || chart.type === 'heatmap_multi') {
    const payload = chart.type === 'heatmap_multi' ? chart.panels?.[0] : chart
    if (!payload) return null
    return statsFromValues(rawCellValues(payload), payload.units)
  }

  if (chart.type === 'timeseries') {
    return statsFromValues(chart.values || [], chart.units)
  }

  return null
}

// Every cell of the payload's own grid, nulls included. Masked pixels (QA
// drops, fill values, out-of-region cells) are serialized as null by the
// backend and must count against validPct — flattenPayload strips them,
// which is right for rendering but would make validPct always 100%.
function rawCellValues(payload) {
  if (Array.isArray(payload.values) && Array.isArray(payload.values[0])) {
    return payload.values.flat()
  }
  const points = payload.points
  if (points && Array.isArray(points.values)) return points.values
  return []
}

function statsFromValues(values, units) {
  const finite = values.filter(Number.isFinite)
  if (!finite.length) return null

  const sum = finite.reduce((acc, v) => acc + v, 0)
  const mean = sum / finite.length
  const max = Math.max(...finite)
  const min = Math.min(...finite)
  const validPct = values.length ? (finite.length / values.length) * 100 : 0

  return {
    mean, max, min, units,
    count: finite.length,
    validPct,
  }
}

// Bins real values into a fixed number of buckets for a histogram — no
// server round-trip, computed from the same array the map/chart already has.
export function computeHistogram(chart, bucketCount = 12) {
  if (!chart) return null
  let values
  if (chart.type === 'heatmap' || chart.type === 'heatmap_multi') {
    const payload = chart.type === 'heatmap_multi' ? chart.panels?.[0] : chart
    if (!payload) return null
    values = flattenPayload(payload).val
  } else if (chart.type === 'timeseries') {
    values = chart.values || []
  } else {
    return null
  }

  const finite = values.filter(Number.isFinite)
  if (!finite.length) return null

  const min = Math.min(...finite)
  const max = Math.max(...finite)
  if (min === max) return { min, max, buckets: [{ from: min, to: max, count: finite.length }] }

  const width = (max - min) / bucketCount
  const counts = new Array(bucketCount).fill(0)
  for (const v of finite) {
    const idx = Math.min(bucketCount - 1, Math.floor((v - min) / width))
    counts[idx] += 1
  }

  const maxCount = Math.max(...counts)
  const buckets = counts.map((count, i) => ({
    from: min + i * width,
    to: min + (i + 1) * width,
    count,
    pct: maxCount ? (count / maxCount) * 100 : 0,
  }))

  return { min, max, buckets }
}
