/**
 * Display formatting for science values and units: `molecules/cm^2` reads as
 * molecules/cm², and 1.406e+15 reads as 1.406 × 10¹⁵. Pure, so it is
 * unit-testable under plain `node --test`.
 */

const SUPERSCRIPT = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻',
}
const MINUS = '−'

function superscript(exponent) {
  return [...String(exponent)].map(ch => SUPERSCRIPT[ch] ?? ch).join('')
}

// Unit symbols a CF-style exponent can follow (`cm-2`, `m3`, `s-1`). Only
// these take a superscript: a units string may also name the species, and the
// digits in `NO2` or `O3` are a formula's subscripts, not exponents.
const UNIT_SYMBOLS = new Set([
  'm', 'cm', 'mm', 'km', 's', 'h', 'g', 'kg', 'mg', 'ug', 'µg', 'μg',
  'mol', 'molec', 'molecules', 'sr', 'Pa', 'hPa', 'K', 'W', 'J', 'L', 'Hz', 'N',
])

/** `cm^2`, `m^-2` and CF-style `cm-2`, `m3` written with superscript
 * exponents. A bare digit run is an exponent only on a unit symbol, and not
 * when more of a word or a decimal point follows, so `PM2.5` and `NO2` stay. */
export function formatUnits(units) {
  if (units == null) return ''
  return String(units)
    .replace(/\^(-?\d+)/g, (_, exp) => superscript(exp))
    .replace(/(?<![A-Za-zµμ])([A-Za-zµμ]+)(-?\d+)(?![\w.])/g, (match, unit, exp) => (
      UNIT_SYMBOLS.has(unit) ? unit + superscript(exp) : match
    ))
}

/** `n` with `digits` mantissa decimals as `m × 10ⁿ` when it is very large or
 * very small, and as a plain 4-significant-figure number otherwise. */
export function formatScientific(n, digits = 3) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '—'
  const abs = Math.abs(n)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) {
    const [mantissa, exponent] = n.toExponential(digits).split('e')
    return `${mantissa.replace('-', MINUS)} × 10${superscript(Number(exponent))}`
  }
  return String(Number(n.toPrecision(4))).replace('-', MINUS)
}

/** A statistic with its units, both formatted for display. */
export function formatStatValue(n, units) {
  const value = formatScientific(n)
  if (value === '—') return value
  const shown = formatUnits(units)
  return shown ? `${value} ${shown}` : value
}

/** Colorbar tick labels sharing one power of ten, so long labels like
 * 4.5 × 10¹⁴ do not crowd a narrow bar. `multiplier` names that power once
 * (empty for ordinary magnitudes) and belongs beside the units. Labels take
 * the fewest significant figures that keep every tick distinct, so a narrow
 * range far from zero (101000–101500 Pa) does not read as five 1s. */
export function scaledTicks(values) {
  const finite = values.filter(v => typeof v === 'number' && Number.isFinite(v))
  const maxAbs = Math.max(0, ...finite.map(Math.abs))
  const needsScale = maxAbs !== 0 && (maxAbs < 1e-3 || maxAbs >= 1e4)
  const exponent = needsScale ? Math.floor(Math.log10(maxAbs)) : 0
  const labelsAt = (precision) => values.map((v) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return '—'
    const scaled = needsScale ? v / 10 ** exponent : v
    return String(Number(scaled.toPrecision(precision))).replace('-', MINUS)
  })
  const distinct = new Set(finite).size
  let precision = needsScale ? 2 : 3
  let labels = labelsAt(precision)
  while (precision < 6 && new Set(labels.filter(l => l !== '—')).size < distinct) {
    precision += 1
    labels = labelsAt(precision)
  }
  return {
    labels,
    multiplier: needsScale ? `× 10${superscript(exponent)}` : '',
  }
}

export function scaledUnitsLabel(multiplier, units) {
  return [multiplier, formatUnits(units)].filter(Boolean).join(' ')
}

const FROM_SUPERSCRIPT = Object.fromEntries(Object.entries(SUPERSCRIPT).map(([plain, sup]) => [sup, plain]))
const SUPERSCRIPT_RUN = new RegExp(`[${Object.values(SUPERSCRIPT).join('')}]+`, 'g')

/** `text` split into plain and exponent runs, exponents back in plain digits.
 * Components render the exponent runs in <sup>: most fonts carry ¹²³ but not
 * ⁴–⁹ or ⁰, so Unicode exponents mix two typefaces. */
export function superscriptRuns(text) {
  const runs = []
  let last = 0
  for (const match of String(text ?? '').matchAll(SUPERSCRIPT_RUN)) {
    if (match.index > last) runs.push({ text: text.slice(last, match.index), sup: false })
    runs.push({ text: [...match[0]].map(ch => FROM_SUPERSCRIPT[ch]).join(''), sup: true })
    last = match.index + match[0].length
  }
  if (last < String(text ?? '').length) runs.push({ text: String(text).slice(last), sup: false })
  return runs
}

/** `text` with its exponents as the <sup> markup Plotly renders. */
export function plotlySuperscript(text) {
  return superscriptRuns(text).map(run => (run.sup ? `<sup>${run.text}</sup>` : run.text)).join('')
}

const ISO_START = /^(\d{4}-\d{2}-\d{2})(?:T(\d{2}):(\d{2}))?/

function dateParts(value) {
  if (!value) return null
  const match = ISO_START.exec(String(value))
  return match ? { date: match[1], time: match[2] == null ? null : `${match[2]}:${match[3]}` } : { date: String(value), time: null }
}

/** A start/end pair as compact text: dates for whole days or spans across
 * days, and the hours for a span inside one day, where the date alone would
 * make two scans of the same day read alike. */
export function compactDateRange(start, end, separator = ' – ') {
  const a = dateParts(start)
  const b = dateParts(end)
  if (!a || !b) return (a || b)?.date ?? ''
  if (a.date !== b.date) return `${a.date}${separator}${b.date}`
  const wholeDay = (time) => time == null || time === '00:00' || time === '23:59'
  if (wholeDay(a.time) && wholeDay(b.time)) return a.date
  if (a.time === b.time) return `${a.date} ${a.time} UTC`
  return `${a.date} ${a.time ?? '00:00'}–${b.time ?? '23:59'} UTC`
}
