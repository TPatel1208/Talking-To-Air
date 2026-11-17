import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { chartMode, productUsed, provenanceUrl } from '../src/utils/chartProvenance.js'
import { modeBadge } from '../src/utils/resultMode.js'
import { axisUnits, chartLines, hasMultipleLines } from '../src/utils/timeseriesSeries.js'
import { resolveCsvExport } from '../src/utils/chartExport.js'

const PROVENANCE = {
  mode: { kind: 'validated', label: 'no2_regional_trend v2', workflow: { name: 'no2_regional_trend', version: 2 } },
  run_id: 'run-wf-1',
  product: {
    key: 'TEMPO_NO2_L3',
    rule: 'newest_version',
    rule_text: 'The newest TEMPO NO2 Level 3 product that covers the requested dates was used.',
  },
  handle: 'grid_1',
}

test('the metadata tab reads the product and the sentence for the rule that chose it', () => {
  assert.deepEqual(productUsed(PROVENANCE), {
    key: 'TEMPO_NO2_L3',
    rule: 'newest_version',
    ruleText: 'The newest TEMPO NO2 Level 3 product that covers the requested dates was used.',
  })
  assert.equal(productUsed({}), null)
  assert.equal(productUsed({ product: null }), null)
})

test('the chart frame mode wins over the endpoint, which fills in when the frame has none', () => {
  const frameMode = { kind: 'exploratory', label: 'Exploratory', workflow: null }
  assert.equal(chartMode({ mode: frameMode }, PROVENANCE), frameMode)
  assert.equal(chartMode({}, PROVENANCE), PROVENANCE.mode)
  assert.equal(chartMode({}, null), null)
  assert.equal(modeBadge(chartMode({}, PROVENANCE)).kind, 'validated')
})

test('provenance is requested by chart id', () => {
  assert.equal(provenanceUrl('/api', 'chart 1'), '/api/chart/chart%201/provenance')
})

const MONITOR_SERIES = {
  type: 'timeseries',
  title: 'Ozone over Essex County',
  variable: 'Ozone',
  units: 'ppm',
  stat: 'value',
  times: ['2025-07-01T00:00:00', '2025-07-01T01:00:00'],
  values: [0.031, 0.036],
  axes: [{ id: 'y', units: 'ppm' }, { id: 'y2', units: 'ug/m3' }],
  series: [
    { label: 'Site A · EPA AQS', units: 'ppm', axis: 'y', times: ['2025-07-01T00:00:00', '2025-07-01T01:00:00'], values: [0.031, 0.036], handle: 'series_1', series: 'Site A' },
    { label: 'Site B · EPA AQS', units: 'ppm', axis: 'y', times: ['2025-07-01T00:00:00'], values: [0.028], handle: 'series_1', series: 'Site B' },
    { label: 'PM2.5', units: 'ug/m3', axis: 'y2', times: ['2025-07-01T00:00:00'], values: [8.2], handle: 'series_2', series: 'PM2.5' },
  ],
}

test('a time series built from monitor readings draws one line per series on its axis', () => {
  assert.equal(hasMultipleLines(MONITOR_SERIES), true)
  const lines = chartLines(MONITOR_SERIES)

  assert.deepEqual(lines.map(l => [l.label, l.axis]), [
    ['Site A · EPA AQS', 'y'], ['Site B · EPA AQS', 'y'], ['PM2.5', 'y2'],
  ])
  assert.equal(new Set(lines.map(l => l.color)).size, 3)
  assert.deepEqual(axisUnits(lines), { y: 'ppm', y2: 'ug/m3' })
})

test('a one-line payload is still the single times/values line', () => {
  const single = { type: 'timeseries', title: 'NO2', units: 'molec/cm2', times: ['t1', 't2'], values: [1, 2] }

  assert.equal(hasMultipleLines(single), false)
  const lines = chartLines(single)
  assert.equal(lines.length, 1)
  assert.deepEqual(lines[0].values, [1, 2])
  assert.equal(lines[0].axis, 'y')
})

test('the browser CSV of a multi-line series carries every line, labelled', () => {
  const resolved = resolveCsvExport(MONITOR_SERIES)

  assert.equal(resolved.kind, 'client')
  assert.equal(resolved.rows.length, 4)
  assert.deepEqual(resolved.rows.map(r => r.series), ['Site A · EPA AQS', 'Site A · EPA AQS', 'Site B · EPA AQS', 'PM2.5'])
})

const srcDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'src')
const read = (name) => readFileSync(join(srcDir, name), 'utf8')

test('the chat no longer reads tool_call frames', () => {
  assert.doesNotMatch(read('hooks/useChat.js'), /tool_call|toolCalls/)
  assert.doesNotMatch(read('components/Chat.jsx'), /tool_call|toolCalls|toolSteps/)
})

test('the result archive is fetched through apiFetch, so it carries the session token', () => {
  const component = read('components/ResultActions.jsx')
  assert.match(component, /fetcher: apiFetch/)
  assert.doesNotMatch(component, /\bfetch\(/)
})

test('a connector notice opens the connectors view for its connector', () => {
  assert.match(read('components/ResultActions.jsx'), /onOpenConnectors\(notice\.connector\)/)
  assert.match(read('App.jsx'), /onOpenConnectors=\{openConnectors\}/)
  assert.match(read('App.jsx'), /connectorsRequest=\{connectorsRequest\}/)
})
