import test from 'node:test'
import assert from 'node:assert/strict'

import { formatUnits, formatScientific, formatStatValue, scaledTicks, scaledUnitsLabel, superscriptRuns, plotlySuperscript, compactDateRange } from '../src/utils/displayFormat.js'

test('caret exponents become superscripts', () => {
  assert.equal(formatUnits('molecules/cm^2'), 'molecules/cm²')
  assert.equal(formatUnits('mol m^-2'), 'mol m⁻²')
  assert.equal(formatUnits('10^15 molecules/cm^2'), '10¹⁵ molecules/cm²')
})

test('CF-style trailing exponents become superscripts', () => {
  assert.equal(formatUnits('molecules cm-2'), 'molecules cm⁻²')
  assert.equal(formatUnits('W m-2 sr-1'), 'W m⁻² sr⁻¹')
  assert.equal(formatUnits('µg/m3'), 'µg/m³')
  assert.equal(formatUnits('kg m-2 s-1'), 'kg m⁻² s⁻¹')
})

test('units with no exponent pass through untouched', () => {
  for (const units of ['ppb', 'DU', '1', 'K', 'degrees_north', 'Parts per billion', '']) {
    assert.equal(formatUnits(units), units)
  }
  assert.equal(formatUnits(null), '')
  assert.equal(formatUnits(undefined), '')
})

test('a hyphenated word is not read as an exponent', () => {
  assert.equal(formatUnits('dimension-less'), 'dimension-less')
  assert.equal(formatUnits('PM2.5'), 'PM2.5')
})

test('a chemical formula in a units string keeps its subscript digits', () => {
  assert.equal(formatUnits('NO2 ppb'), 'NO2 ppb')
  assert.equal(formatUnits('mol O3 / m^2'), 'mol O3 / m²')
  assert.equal(formatUnits('molecules NO2 cm-2'), 'molecules NO2 cm⁻²')
  assert.equal(formatUnits('ug CO2 m-3'), 'ug CO2 m⁻³')
})

test('large and small magnitudes use ×10 with a superscript exponent', () => {
  assert.equal(formatScientific(1.406e15), '1.406 × 10¹⁵')
  assert.equal(formatScientific(-1.897e14), '−1.897 × 10¹⁴')
  assert.equal(formatScientific(2.5e-5), '2.500 × 10⁻⁵')
  assert.equal(formatScientific(1.4e15, 1), '1.4 × 10¹⁵')
})

test('ordinary magnitudes stay plain', () => {
  assert.equal(formatScientific(42.1234), '42.12')
  assert.equal(formatScientific(0.04), '0.04')
  assert.equal(formatScientific(0), '0')
  assert.equal(formatScientific(-3.5), '−3.5')
  assert.equal(formatScientific(1234), '1234')
})

test('a mantissa that rounds to 10 moves to the next power', () => {
  assert.equal(formatScientific(9.9996e15), '1.000 × 10¹⁶')
})

test('non-finite values render as a dash', () => {
  for (const n of [NaN, Infinity, null, undefined]) assert.equal(formatScientific(n), '—')
})

test('a stat value carries its formatted units', () => {
  assert.equal(formatStatValue(1.406e15, 'molecules/cm^2'), '1.406 × 10¹⁵ molecules/cm²')
  assert.equal(formatStatValue(42, ''), '42')
  assert.equal(formatStatValue(NaN, 'ppb'), '—')
})

test('colorbar ticks share one power of ten, named once beside the units', () => {
  const scaled = scaledTicks([4.5e14, 1.2e15, 2.0e15, 2.7e15, 3.5e15])
  assert.deepEqual(scaled.labels, ['0.45', '1.2', '2', '2.7', '3.5'])
  assert.equal(scaled.multiplier, '× 10¹⁵')
})

test('ordinary colorbar ticks need no multiplier', () => {
  const scaled = scaledTicks([0, 25, 50, 75])
  assert.deepEqual(scaled.labels, ['0', '25', '50', '75'])
  assert.equal(scaled.multiplier, '')
})

test('small colorbar ticks scale down the same way', () => {
  const scaled = scaledTicks([-2e-5, 0, 2e-5])
  assert.deepEqual(scaled.labels, ['−2', '0', '2'])
  assert.equal(scaled.multiplier, '× 10⁻⁵')
})

test('the units label carries the multiplier', () => {
  assert.equal(scaledUnitsLabel('× 10¹⁵', 'molecules/cm^2'), '× 10¹⁵ molecules/cm²')
  assert.equal(scaledUnitsLabel('', 'ppb'), 'ppb')
  assert.equal(scaledUnitsLabel('× 10¹⁵', ''), '× 10¹⁵')
})

test('exponent runs split out as plain digits to render in <sup>', () => {
  assert.deepEqual(superscriptRuns('1.406 × 10¹⁵ molecules/cm²'), [
    { text: '1.406 × 10', sup: false },
    { text: '15', sup: true },
    { text: ' molecules/cm', sup: false },
    { text: '2', sup: true },
  ])
  assert.deepEqual(superscriptRuns('× 10⁻⁵'), [{ text: '× 10', sup: false }, { text: '-5', sup: true }])
  assert.deepEqual(superscriptRuns('ppb'), [{ text: 'ppb', sup: false }])
  assert.deepEqual(superscriptRuns(''), [])
})

test('plotly text carries exponents as <sup> markup', () => {
  assert.equal(plotlySuperscript('molecules/cm²'), 'molecules/cm<sup>2</sup>')
  assert.equal(plotlySuperscript('mean (× 10¹⁵ mol m⁻²)'), 'mean (× 10<sup>15</sup> mol m<sup>-2</sup>)')
})

test('ticks on a narrow range far from zero stay distinct', () => {
  const scaled = scaledTicks([101000, 101125, 101250, 101375, 101500])
  assert.equal(new Set(scaled.labels).size, 5, scaled.labels.join(' '))
  assert.equal(scaled.multiplier, '× 10⁵')
  const plain = scaledTicks([290.1, 290.3, 290.5, 290.7, 290.9])
  assert.equal(new Set(plain.labels).size, 5, plain.labels.join(' '))
})

test('a span within one day keeps its hours', () => {
  assert.equal(compactDateRange('2026-06-05T14:00:00Z', '2026-06-05T18:30:00Z'), '2026-06-05 14:00–18:30 UTC')
  assert.equal(compactDateRange('2026-06-05T14:00:00', '2026-06-05T14:00:00'), '2026-06-05 14:00 UTC')
})

test('a span across days, or of whole days, shows dates only', () => {
  assert.equal(compactDateRange('2026-06-04T10:26:42.031514624', '2026-06-05T22:07:24.04'), '2026-06-04 – 2026-06-05')
  assert.equal(compactDateRange('2026-06-05T00:00:00', '2026-06-05T23:59:59'), '2026-06-05')
  assert.equal(compactDateRange('2026-06-05', '2026-06-05'), '2026-06-05')
  assert.equal(compactDateRange('2026-06-05', ''), '2026-06-05')
  assert.equal(compactDateRange('', ''), '')
})

test('the separator between two dates is the caller\'s', () => {
  assert.equal(compactDateRange('2026-06-01', '2026-06-30', ' → '), '2026-06-01 → 2026-06-30')
})
