// The harness fields of `/chart/{id}/provenance`: how the chart was made and
// which product was used, with the rule that chose it.

// `{ key, rule, ruleText }` for a product block; null when the payload names none.
export function productUsed(provenance) {
  const product = provenance?.product
  if (!product || typeof product !== 'object') return null
  const key = typeof product.key === 'string' ? product.key : ''
  const ruleText = typeof product.rule_text === 'string' ? product.rule_text : ''
  if (!key && !ruleText) return null
  return { key, rule: typeof product.rule === 'string' ? product.rule : '', ruleText }
}

// The mode a chart's metadata tab shows: the frame's own, else the endpoint's.
export function chartMode(chart, provenance) {
  return chart?.mode || provenance?.mode || null
}

export function provenanceUrl(apiBase, chartId) {
  return `${apiBase}/chart/${encodeURIComponent(chartId)}/provenance`
}
