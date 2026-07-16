/**
 * utils/evidenceSummary.js
 * -------------------------
 * The deterministic "Supporting information" facts that ride along on chart
 * payloads as `provenance.evidence` (PRD T36 Phase 2 --
 * Backend/tools/satellite_tools/plot_tools.py::_evidence). Each fact is a
 * co-located summary of a companion band already present in the opened file --
 * QA pass rate, retrieval uncertainty, cloud fraction, aerosol index -- with an
 * honest `coverage` valid-fraction. This turns that raw fact list into the
 * render-ready rows the Statistics tab shows; there is NO narrative here, and
 * no fact is invented -- an empty evidence list yields no rows, so the section
 * omits itself entirely (the "disclose, don't hide" empty-state convention,
 * kin to maskingProvenance.js).
 *
 * Kept pure and outside the component tree so formatting is unit-testable
 * without a DOM (this repo's frontend runner is plain `node --test`).
 */

// Compact numeric formatting: exponential for very small / very large science
// magnitudes (e.g. 2.8e15 molecules/cm^2), plain otherwise (0.04, 0.25, 300).
function formatNumber(n) {
  if (!Number.isFinite(n)) return '—'
  const abs = Math.abs(n)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return n.toExponential(2)
  return String(Number(n.toPrecision(4)))
}

function humanize(name) {
  return String(name || '').replace(/_/g, ' ')
}

// The human label for a fact: QA gets a fixed name, an uncertainty band reads
// as "Retrieval uncertainty", every other band shows its (humanized) name.
export function evidenceLabel(fact) {
  if (fact.stat === 'pass_rate') return 'QA pass rate'
  if (String(fact.name || '').toLowerCase().includes('uncertainty')) return 'Retrieval uncertainty'
  return humanize(fact.name)
}

// The value text: a pass rate is a percentage; everything else is the mean in
// the band's own units, with the uncertainty's fraction-of-science appended
// when the backend computed one.
export function evidenceValueText(fact) {
  if (fact.stat === 'pass_rate') return `${(fact.value * 100).toFixed(1)}%`
  const units = fact.units ? ` ${fact.units}` : ''
  let text = `${formatNumber(fact.value)}${units}`
  if (Number.isFinite(fact.pct_of_science)) {
    text += ` (~${(fact.pct_of_science * 100).toFixed(1)}% of value)`
  }
  return text
}

// Honest coverage disclosure -- "over 87% valid pixels" -- so a fact computed
// over mostly-fill data never masquerades as solid.
export function evidenceCoverageText(coverage) {
  if (!Number.isFinite(coverage)) return ''
  return `over ${Math.round(coverage * 100)}% valid pixels`
}

// Evidence lives in `provenance.evidence` for every chart type (plot_tools
// always attaches it through _attach_reproducibility -> _provenance); a
// top-level `evidence` is accepted too for symmetry with maskingProvenance.js.
// Rows carry only finite-valued facts, so a malformed entry can't render a
// bogus number.
export function evidenceRows(chart) {
  const list = chart?.provenance?.evidence || chart?.evidence
  if (!Array.isArray(list)) return []
  return list
    .filter(f => f && f.name && Number.isFinite(f.value))
    .map(f => ({
      key: `${f.name}-${f.stat}`,
      name: f.name,
      role: f.role || '',
      label: evidenceLabel(f),
      valueText: evidenceValueText(f),
      coverageText: evidenceCoverageText(f.coverage),
    }))
}

// Whether the Supporting-information section has anything to render. When
// false the section omits itself entirely -- no "Not available" grid.
export function hasEvidence(chart) {
  return evidenceRows(chart).length > 0
}
