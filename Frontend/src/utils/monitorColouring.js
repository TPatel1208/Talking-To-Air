// How the ground-monitor map colours its points. A monitor map from the
// harness gives each row its AQI category and colour, and the table's
// metadata carries the legend; both are used as given. Only rows with neither
// (the older daily summaries) fall back to a min-max scale.

const NO_COLOUR = '#94a3b8'

function finite(value) {
  if (value == null || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

// The value a monitor is mapped and listed by.
export function monitorValue(row) {
  return finite(row?.mean) ?? finite(row?.max) ?? finite(row?.value) ?? finite(row?.peak?.value)
}

function legendRange(entry, unit) {
  const low = finite(entry.concentration_low)
  const high = finite(entry.concentration_high)
  const suffix = unit ? ` ${unit}` : ''
  if (low != null && high != null) return `${low}–${high}${suffix}`
  if (low != null) return `above ${low}${suffix}`
  if (high != null) return `below ${high}${suffix}`
  return ''
}

// The legend the table's metadata carries: `{ title, unit, entries }`, or null
// for a table without one.
export function colouringLegend(metadata) {
  const colouring = metadata?.colouring
  const entries = Array.isArray(colouring?.legend)
    ? colouring.legend.filter(e => e && typeof e.color === 'string' && typeof e.category === 'string')
    : []
  if (!entries.length) return null
  const period = colouring.averaging_period ? ` (${colouring.averaging_period})` : ''
  return {
    title: colouring.method === 'aqi_category' ? `AQI category${period}` : 'Legend',
    unit: colouring.unit || '',
    entries: entries.map(e => ({ category: e.category, color: e.color, range: legendRange(e, colouring.unit) })),
  }
}

function scaleColour(value, min, max) {
  if (value == null) return NO_COLOUR
  const range = max - min
  const ratio = range > 0 ? Math.max(0, Math.min(1, (value - min) / range)) : 0.5
  if (ratio < 0.25) return '#10b981'
  if (ratio < 0.5) return '#f59e0b'
  if (ratio < 0.75) return '#f97316'
  return '#ef4444'
}

// A point's colour: the row's own, else its category's from the legend, else
// the min-max scale.
export function monitorColour(row, legend, min, max) {
  if (typeof row?.color === 'string' && row.color) return row.color
  const entry = legend?.entries.find(e => e.category === row?.aqi_category)
  if (entry) return entry.color
  return scaleColour(monitorValue(row), min, max)
}

// The rows a station popup shows, as `{ label, value, color? }`.
export function popupLines(row) {
  const units = row?.units || ''
  if (row?.aqi_category != null || (row?.mean == null && row?.value != null)) {
    const value = finite(row.value)
    const lines = [{ label: 'Value', value: value == null ? '—' : `${value} ${units}`.trim() }]
    if (row.aqi_category) lines.push({ label: 'AQI category', value: row.aqi_category, color: row.color })
    if (row.above_standard_level === true) lines.push({ label: 'NAAQS', value: 'Above the standard level' })
    if (row.above_standard_level === false) lines.push({ label: 'NAAQS', value: 'At or below the standard level' })
    return lines
  }
  const mean = finite(row?.mean)
  const lines = [{ label: 'Daily Mean', value: `${mean != null ? mean.toFixed(2) : '—'} ${units}`.trim() }]
  const max = finite(row?.max)
  if (max != null) lines.push({ label: 'Daily Max', value: `${max.toFixed(2)} ${units}`.trim() })
  return lines
}
