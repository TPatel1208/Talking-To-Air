// Shared-scale computation for compare mode (T28). Checks whether every
// filled-slot heatmap payload reports the same variable/units (the field
// the payload already carries -- see chartMetaChips in OutputPanel.jsx for
// the same provenance.variable / units fallback convention), and if so
// computes one vmin/vmax/colormap to recolor every panel onto. When they
// don't match, shared scaling is skipped entirely and callers should fall
// back to each panel's own natively-computed scale.

const MISMATCH_REASON = 'Different variables — showing independent scales'

function variableKey(chart) {
  return chart?.provenance?.variable ?? chart?.variable ?? null
}

function unitsKey(chart) {
  return chart?.units ?? chart?.provenance?.units ?? null
}

// charts: array of filled (non-null) heatmap chart payloads.
export function computeSharedColorScale(charts) {
  const filled = (charts || []).filter(Boolean)

  if (filled.length < 2) {
    return { available: false, reason: null, vmin: null, vmax: null, colormap: null }
  }

  const variable = variableKey(filled[0])
  const units = unitsKey(filled[0])
  const allMatch = filled.every(c => variableKey(c) === variable && unitsKey(c) === units)

  if (!allMatch) {
    return { available: false, reason: MISMATCH_REASON, vmin: null, vmax: null, colormap: null }
  }

  const vmins = filled.map(c => c.vmin).filter(Number.isFinite)
  const vmaxs = filled.map(c => c.vmax).filter(Number.isFinite)
  if (vmins.length !== filled.length || vmaxs.length !== filled.length) {
    return { available: false, reason: null, vmin: null, vmax: null, colormap: null }
  }

  return {
    available: true,
    reason: null,
    vmin: Math.min(...vmins),
    vmax: Math.max(...vmaxs),
    colormap: filled[0].colormap,
  }
}
