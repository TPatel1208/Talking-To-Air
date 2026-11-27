import assert from 'node:assert/strict'
import test from 'node:test'
import {
  colouringLegend, monitorColour, monitorValue, popupLines,
} from '../src/utils/monitorColouring.js'

const LEGEND_METADATA = {
  colouring: {
    method: 'aqi_category', pollutant: 'o3', averaging_period: '8-hour', unit: 'ppm',
    legend: [
      { category: 'Good', color: '#00E400', concentration_low: 0, concentration_high: 0.054 },
      { category: 'Moderate', color: '#FFFF00', concentration_low: 0.055, concentration_high: 0.07 },
      { category: 'Beyond the AQI table', color: '#3F0011', concentration_low: 0.6 },
    ],
  },
}

const ROWS = [
  { site_id: 'a', latitude: 40.1, longitude: -74.2, value: 0.031, units: 'ppm', aqi_category: 'Good', color: '#00E400', above_standard_level: false },
  { site_id: 'b', latitude: 40.2, longitude: -74.3, value: 0.071, units: 'ppm', aqi_category: 'Moderate', color: '#FFFF00', above_standard_level: true },
]

test('a monitor row is coloured by the colour the payload gives, not by its place among the values', () => {
  const legend = colouringLegend(LEGEND_METADATA)
  const values = ROWS.map(monitorValue)
  const min = Math.min(...values)
  const max = Math.max(...values)

  assert.equal(monitorColour(ROWS[0], legend, min, max), '#00E400')
  assert.equal(monitorColour(ROWS[1], legend, min, max), '#FFFF00')
  // The same row against a very different range keeps its colour.
  assert.equal(monitorColour(ROWS[0], legend, 0, 0.001), '#00E400')
})

test('a row with a category but no colour takes its category colour from the legend', () => {
  const legend = colouringLegend(LEGEND_METADATA)
  assert.equal(monitorColour({ value: 0.06, aqi_category: 'Moderate' }, legend, 0, 1), '#FFFF00')
})

test('a row with neither colour nor category falls back to the min-max scale', () => {
  assert.equal(monitorColour({ mean: 1 }, null, 1, 10), '#10b981')
  assert.equal(monitorColour({ mean: 10 }, null, 1, 10), '#ef4444')
  assert.equal(monitorColour({}, null, 1, 10), '#94a3b8')
})

test('the legend carries each category with its colour and concentration range', () => {
  const legend = colouringLegend(LEGEND_METADATA)

  assert.equal(legend.title, 'AQI category (8-hour)')
  assert.deepEqual(legend.entries.map(e => [e.category, e.color, e.range]), [
    ['Good', '#00E400', '0–0.054 ppm'],
    ['Moderate', '#FFFF00', '0.055–0.07 ppm'],
    ['Beyond the AQI table', '#3F0011', 'above 0.6 ppm'],
  ])
  assert.equal(colouringLegend({}), null)
  assert.equal(colouringLegend({ colouring: { legend: [] } }), null)
})

test('a monitor map row carries its value under `value`, and older rows under mean or max', () => {
  assert.equal(monitorValue({ value: 0.031 }), 0.031)
  assert.equal(monitorValue({ mean: 4, value: 9 }), 4)
  assert.equal(monitorValue({ max: 7 }), 7)
  assert.equal(monitorValue({ peak: { value: 3 } }), 3)
  assert.equal(monitorValue({ value: null }), null)
})

test('the station popup shows the value, its AQI category and the standard comparison', () => {
  assert.deepEqual(popupLines(ROWS[1]), [
    { label: 'Value', value: '0.071 ppm' },
    { label: 'AQI category', value: 'Moderate', color: '#FFFF00' },
    { label: 'NAAQS', value: 'Above the standard level' },
  ])
  assert.deepEqual(popupLines({ mean: 4.5, max: 9, units: 'ppb' }), [
    { label: 'Daily Mean', value: '4.50 ppb' },
    { label: 'Daily Max', value: '9.00 ppb' },
  ])
})
