// How a result was made, and what the answer's results offer: the
// validated / draft / exploratory badge, a missing-connector notice and the
// result archive link. Every function takes the plain frame or history shape.

const KIND_NAMES = {
  validated: 'Validated',
  draft: 'Draft',
  exploratory: 'Exploratory',
}

export const CONNECTOR_NAMES = {
  earthdata: 'NASA Earthdata',
  epa_aqs: 'EPA AQS',
}

const EXPORTABLE = new Set(['done', 'incomplete'])

function badgeText(kind, label, workflow) {
  if (kind === 'exploratory') return 'Exploratory'
  const named = label || (workflow ? `${workflow.name} v${workflow.version}` : '')
  if (kind === 'draft') return /^draft workflow/i.test(named) ? named : `Draft workflow: ${named}`
  return named ? `Validated: ${named}` : KIND_NAMES.validated
}

// A badge from a frame's `mode` ({kind, label, workflow}); null without one.
export function modeBadge(mode) {
  if (!mode || typeof mode !== 'object' || !KIND_NAMES[mode.kind]) return null
  return { kind: mode.kind, text: badgeText(mode.kind, mode.label, mode.workflow) }
}

// A badge from the answer's `label` string alone, as `done` and history carry it.
export function labelBadge(label) {
  if (typeof label !== 'string' || !label.trim()) return null
  const text = label.trim()
  if (text === 'Exploratory') return { kind: 'exploratory', text }
  if (/^draft workflow/i.test(text)) return { kind: 'draft', text }
  return { kind: 'validated', text: `Validated: ${text}` }
}

// The answer's badge: its first result's mode when present, else its label.
export function answerBadge(msg) {
  return modeBadge(msg?.results?.[0]?.mode) || labelBadge(msg?.label)
}

// The results a `done` frame or history message carries, as an array of objects.
export function normalizeResults(results) {
  return Array.isArray(results) ? results.filter(r => r && typeof r === 'object') : []
}

// One notice per result that stopped on a missing, expired or rejected
// connector, or on a connector whose service is down. A down service is
// marked `unavailable`: reconnecting would not help, so it offers no connect.
export function connectorNotices(results) {
  return normalizeResults(results)
    .filter(r => typeof r.connector === 'string' && r.connector)
    .map(r => ({
      runId: r.run_id ?? null,
      connector: r.connector,
      connectorName: CONNECTOR_NAMES[r.connector] || r.connector,
      reason: r.reason || `${CONNECTOR_NAMES[r.connector] || r.connector} is not connected.`,
      ...(r.unavailable === true ? { unavailable: true } : {}),
    }))
}

// Whether a notice offers to connect its connector.
export function offersConnect(notice) {
  return !notice?.unavailable
}

export function resultExportUrl(apiBase, runId) {
  return `${apiBase}/result/${encodeURIComponent(runId)}/export.zip`
}

// The results that have something to download, each with its archive's name.
export function downloadableResults(results) {
  const rows = normalizeResults(results).filter(r => r.run_id && EXPORTABLE.has(r.status))
  return rows.map((r, i) => ({
    runId: r.run_id,
    filename: `result-${rows.length > 1 ? `${i + 1}-` : ''}${String(r.run_id).slice(0, 8)}.zip`,
    badge: modeBadge(r.mode),
  }))
}

// A history message as the chat state holds it: camel-cased follow-ups and
// normalised results beside the fields the server sends.
export function hydrateHistoryMessage(message, extractFollowups) {
  return {
    ...message,
    artifacts: message.artifacts || [],
    suggestedFollowups: extractFollowups(message),
    results: normalizeResults(message.results),
    label: typeof message.label === 'string' ? message.label : null,
  }
}
