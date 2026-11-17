import assert from 'node:assert/strict'
import test from 'node:test'
import {
  answerBadge, connectorNotices, downloadableResults, hydrateHistoryMessage, labelBadge,
  modeBadge, normalizeResults, resultExportUrl,
} from '../src/utils/resultMode.js'
import { extractSuggestedFollowups } from '../src/utils/followups.js'
import { fetchResultArchive } from '../src/utils/resultDownload.js'

const VALIDATED = { kind: 'validated', label: 'no2_regional_trend v2', workflow: { name: 'no2_regional_trend', version: 2 } }
const DRAFT = { kind: 'draft', label: 'Draft workflow: no2_regional_trend v3', workflow: { name: 'no2_regional_trend', version: 3 } }
const EXPLORATORY = { kind: 'exploratory', label: 'Exploratory', workflow: null }

test('a chart or artifact mode becomes a validated, draft or exploratory badge', () => {
  assert.deepEqual(modeBadge(VALIDATED), { kind: 'validated', text: 'Validated: no2_regional_trend v2' })
  assert.deepEqual(modeBadge(DRAFT), { kind: 'draft', text: 'Draft workflow: no2_regional_trend v3' })
  assert.deepEqual(modeBadge(EXPLORATORY), { kind: 'exploratory', text: 'Exploratory' })
  assert.equal(modeBadge(null), null)
  assert.equal(modeBadge({ kind: 'other' }), null)
})

test('the answer label string alone gives the same three badges', () => {
  assert.deepEqual(labelBadge('Exploratory'), { kind: 'exploratory', text: 'Exploratory' })
  assert.deepEqual(labelBadge('Draft workflow: no2_regional_trend v3'), { kind: 'draft', text: 'Draft workflow: no2_regional_trend v3' })
  assert.deepEqual(labelBadge('no2_regional_trend v2'), { kind: 'validated', text: 'Validated: no2_regional_trend v2' })
  assert.equal(labelBadge(null), null)
  assert.equal(labelBadge('  '), null)
})

test('an answer is badged from its first result, else its label', () => {
  const done = { label: 'Exploratory', results: [{ run_id: 'r1', status: 'done', mode: VALIDATED }] }
  assert.equal(answerBadge(done).kind, 'validated')
  assert.equal(answerBadge({ label: 'Exploratory', results: [] }).kind, 'exploratory')
  assert.equal(answerBadge({ label: null }), null)
})

test('a history message keeps its label, results and follow-ups under the names the chat reads', () => {
  const stored = {
    role: 'assistant',
    content: 'Here is the trend.',
    label: 'no2_regional_trend v2',
    results: [{ run_id: 'r1', status: 'done', mode: VALIDATED, reason: null, connector: null }],
    suggested_followups: ['Show the map', 7, 'Export it'],
  }
  const hydrated = hydrateHistoryMessage(stored, extractSuggestedFollowups)

  assert.equal(hydrated.label, 'no2_regional_trend v2')
  assert.equal(hydrated.results.length, 1)
  assert.deepEqual(hydrated.suggestedFollowups, ['Show the map', 'Export it'])
  assert.deepEqual(hydrated.artifacts, [])
  assert.equal(answerBadge(hydrated).kind, 'validated')

  const user = hydrateHistoryMessage({ role: 'user', content: 'hi' }, extractSuggestedFollowups)
  assert.equal(user.label, null)
  assert.deepEqual(user.results, [])
})

test('normalizeResults drops anything that is not a result object', () => {
  assert.deepEqual(normalizeResults(undefined), [])
  assert.deepEqual(normalizeResults([null, 'x', { run_id: 'r' }]), [{ run_id: 'r' }])
})

test('a result stopped on a connector yields a notice with its reason and connector', () => {
  const results = [
    { run_id: 'r1', status: 'done', mode: VALIDATED, reason: null, connector: null },
    {
      run_id: 'r2', status: 'failed', mode: VALIDATED, connector: 'earthdata',
      reason: 'Your Earthdata connection has expired. Reconnect it to retrieve satellite data.',
    },
    { run_id: 'r3', status: 'failed', mode: EXPLORATORY, connector: 'epa_aqs', reason: null },
  ]
  const notices = connectorNotices(results)

  assert.equal(notices.length, 2)
  assert.deepEqual(notices[0], {
    runId: 'r2', connector: 'earthdata', connectorName: 'NASA Earthdata',
    reason: 'Your Earthdata connection has expired. Reconnect it to retrieve satellite data.',
  })
  assert.equal(notices[1].connector, 'epa_aqs')
  assert.match(notices[1].reason, /EPA AQS/)
})

test('each finished result gets a download link to its archive', () => {
  const results = [
    { run_id: 'aaaaaaaa-1', status: 'done', mode: VALIDATED },
    { run_id: 'bbbbbbbb-2', status: 'failed', mode: EXPLORATORY, connector: 'earthdata' },
    { run_id: 'cccccccc-3', status: 'done', mode: EXPLORATORY },
  ]
  const downloads = downloadableResults(results)

  assert.deepEqual(downloads.map(d => d.runId), ['aaaaaaaa-1', 'cccccccc-3'])
  assert.equal(downloads[0].badge.kind, 'validated')
  assert.equal(resultExportUrl('/api', 'aaaaaaaa-1'), '/api/result/aaaaaaaa-1/export.zip')
  assert.deepEqual(downloadableResults([]), [])
})

test('the archive is fetched through the authenticated fetcher at the export url', async () => {
  const calls = []
  const fetcher = async (url) => {
    calls.push(url)
    return {
      ok: true,
      blob: async () => 'BLOB',
      headers: { get: () => 'attachment; filename="result-abc.zip"' },
    }
  }
  const out = await fetchResultArchive({ apiBase: '/api', runId: 'run-1', fallbackName: 'fallback.zip', fetcher })

  assert.deepEqual(calls, ['/api/result/run-1/export.zip'])
  assert.equal(out.blob, 'BLOB')
  assert.equal(out.filename, 'result-abc.zip')
})

test('a failed archive fetch throws the server detail, or the status', async () => {
  const withDetail = async () => ({ ok: false, status: 404, json: async () => ({ detail: 'No such result' }) })
  await assert.rejects(
    fetchResultArchive({ apiBase: '', runId: 'x', fallbackName: 'f.zip', fetcher: withDetail }),
    /No such result/,
  )
  const noBody = async () => ({ ok: false, status: 500, json: async () => { throw new Error('no body') } })
  await assert.rejects(
    fetchResultArchive({ apiBase: '', runId: 'x', fallbackName: 'f.zip', fetcher: noBody }),
    /status 500/,
  )
})
