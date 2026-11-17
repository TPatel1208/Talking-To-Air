// The lines of a `timeseries` payload. A payload with `series` (from
// render.timeseries, including monitor readings) has one entry per line, each
// naming the y axis it uses from `axes`; an older payload is the one line in
// `times`/`values`.

const LINE_COLOURS = ['#1D9E75', '#3B6FD4', '#D9822B', '#8E5BC0', '#C0392B', '#2F9AA8']

function validSeries(payload) {
  const series = Array.isArray(payload?.series) ? payload.series : []
  return series.filter(s => Array.isArray(s?.times) && Array.isArray(s?.values) && s.times.length)
}

// Whether the payload draws more than the single `times`/`values` line.
export function hasMultipleLines(payload) {
  return validSeries(payload).length > 1
}

// `{ label, units, axis, times, values, color }` for every line to draw.
export function chartLines(payload) {
  const series = validSeries(payload)
  if (!series.length) {
    return [{
      label: payload?.title || '', units: payload?.units || '', axis: 'y',
      times: payload?.times || [], values: payload?.values || [], color: LINE_COLOURS[0],
    }]
  }
  return series.map((line, i) => ({
    label: line.label || `Series ${i + 1}`,
    units: line.units || '',
    axis: line.axis === 'y2' ? 'y2' : 'y',
    times: line.times,
    values: line.values,
    color: LINE_COLOURS[i % LINE_COLOURS.length],
  }))
}

// The distinct units on each axis, in `y`, `y2` order.
export function axisUnits(lines) {
  const units = { y: '', y2: '' }
  for (const line of lines) {
    if (line.units && !units[line.axis]) units[line.axis] = line.units
  }
  return units
}
