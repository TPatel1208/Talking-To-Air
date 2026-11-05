/**
 * What the small square on a chat output card shows: a miniature of the map,
 * a sparkline of a time series, a profile curve, a table's row count, or an
 * icon for the output type when there is nothing to draw. Pure, so it is
 * unit-testable under plain `node --test`.
 */
import { profileAxis } from './verticalProfile.js'

const BOX = { width: 40, height: 40, pad: 4 }

const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const round = (v) => Math.round(v * 10) / 10

function normalizer(values) {
  const finite = values.filter(isNum)
  const min = Math.min(...finite)
  const max = Math.max(...finite)
  return (v) => (max === min ? 0.5 : (v - min) / (max - min))
}

/** An SVG path through `values` left to right, high values at the top. A
 * missing value breaks the line rather than bridging the gap. */
export function sparklinePath(values, { width, height, pad } = BOX) {
  const norm = normalizer(values)
  const w = width - 2 * pad
  const h = height - 2 * pad
  const step = values.length > 1 ? w / (values.length - 1) : 0
  let path = ''
  let drawing = false
  values.forEach((v, i) => {
    if (!isNum(v)) {
      drawing = false
      return
    }
    const point = `${round(pad + i * step)} ${round(pad + (1 - norm(v)) * h)}`
    path += path ? ` ${drawing ? 'L' : 'M'}${point}` : `M${point}`
    drawing = true
  })
  return path
}

/** An SVG path of `values` (across) against `axis` (up). `reversed` puts
 * the first axis value at the top: pressure, or a top-down layer index. */
export function profilePath(values, axis, { width, height, pad, reversed = false, log = false } = BOX) {
  const pairs = values
    .map((v, i) => [v, log && axis[i] > 0 ? Math.log10(axis[i]) : axis[i]])
    .filter(([v, a]) => isNum(v) && isNum(a))
  const normX = normalizer(pairs.map(([v]) => v))
  const normY = normalizer(pairs.map(([, a]) => a))
  const w = width - 2 * pad
  const h = height - 2 * pad
  return pairs
    .map(([v, a], i) => {
      const y = reversed ? normY(a) : 1 - normY(a)
      return `${i ? 'L' : 'M'}${round(pad + normX(v) * w)} ${round(pad + y * h)}`
    })
    .join(' ')
}

// The card draws at 48px; 96 keeps it sharp on a high-density screen. The
// server scales the overlay down rather than sending it full size.
const THUMBNAIL_SIDE = 96

function withThumbnailSize(url) {
  return `${url}${url.includes('?') ? '&' : '?'}thumb=${THUMBNAIL_SIDE}`
}

const CHART_ICON = { heatmap: 'map', heatmap_multi: 'map', timeseries: 'timeseries', profile: 'profile' }
const ARTIFACT_ICON = { map: 'map', comparison: 'map', timeseries: 'timeseries', table: 'table' }

function chartPreview(chart) {
  const icon = { kind: 'icon', icon: CHART_ICON[chart.type] || 'generic' }
  if (chart.type === 'heatmap' || chart.type === 'heatmap_multi') {
    const overlayUrl = chart.overlay?.url || chart.panels?.find(panel => panel?.overlay?.url)?.overlay.url
    return overlayUrl ? { kind: 'map', overlayUrl: withThumbnailSize(overlayUrl) } : icon
  }
  const values = Array.isArray(chart.values) ? chart.values : []
  if (values.filter(isNum).length < 2) return icon
  if (chart.type === 'timeseries') return { kind: 'sparkline', path: sparklinePath(values) }
  if (chart.type === 'profile') {
    const axis = profileAxis(chart)
    const reversed = axis.kind === 'pressure' || (axis.kind === null && axis.order === 'top_down')
    const log = axis.kind === 'pressure' && axis.values.every(v => isNum(v) && v > 0)
    return { kind: 'profile', path: profilePath(values, axis.values, { ...BOX, reversed, log }) }
  }
  return icon
}

export function cardPreview(item) {
  const data = item?.data || {}
  if (item?.kind === 'chart') return chartPreview(data)
  if (data.type === 'table') {
    const n = data.row_count
    return { kind: 'table', rowLabel: isNum(n) ? `${n} ${n === 1 ? 'row' : 'rows'}` : null }
  }
  return { kind: 'icon', icon: ARTIFACT_ICON[data.type] || 'generic' }
}
