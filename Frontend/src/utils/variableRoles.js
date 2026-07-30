// Variable-role taxonomy display helpers (PRD T35). Pure functions that turn
// the backend's classified inventory (services/discovery_service.py's additive
// `inventory` key) and the chart provenance's `related_variables` view into the
// shapes the inventory panel and the chart-page related-variables panel render.
// The classification itself lives backend-side (datasets/variable_roles.py) —
// this is display only, so there is one canonical interpretation, never two.

export const ROLE_SCIENCE = 'science'
export const ROLE_QUALITY = 'quality'
export const ROLE_CONTEXT = 'context'
export const ROLE_RETRIEVAL_METADATA = 'retrieval-metadata'
export const ROLE_UNCLASSIFIED = 'unclassified'

// Science first, unclassified last — the honest-incompleteness bucket sits at
// the bottom, one clean group rather than scattered guesses.
export const ROLE_META = {
  [ROLE_SCIENCE]: { label: 'Science', blurb: 'The geophysical product', accent: 'var(--teal)' },
  [ROLE_QUALITY]: { label: 'Quality', blurb: 'Trust signals — flags, uncertainties', accent: 'var(--amber, #b7791f)' },
  [ROLE_CONTEXT]: { label: 'Context', blurb: 'Atmosphere, surface & geometry bands', accent: 'var(--sky, #2b6cb0)' },
  [ROLE_RETRIEVAL_METADATA]: { label: 'Retrieval metadata', blurb: 'Algorithm intermediates', accent: 'var(--violet, #6b46c1)' },
  [ROLE_UNCLASSIFIED]: { label: 'Unclassified', blurb: 'No positive evidence — not guessed', accent: 'var(--text-hint)' },
}

export const ROLE_ORDER = [
  ROLE_SCIENCE, ROLE_QUALITY, ROLE_CONTEXT, ROLE_RETRIEVAL_METADATA, ROLE_UNCLASSIFIED,
]

export function roleLabel(role) {
  return ROLE_META[role]?.label || role
}

// A confidence tier is worth surfacing only when it's a hedge (low / none) —
// High/Medium decisions read as plain assertions, a Low or unclassified entry
// gets a visible qualifier so the UI never asserts a keyword guess.
export function confidenceHint(confidence) {
  if (confidence === 'low') return 'heuristic — low confidence'
  if (confidence === null || confidence === undefined) return 'unclassified'
  return null
}

// Group a classified inventory into ordered, non-empty role sections, each a
// name-first list. Preserves input order within a section. `inventory` is the
// backend `inventory.variables` list (or a bare array of the same records).
export function groupInventoryByRole(inventory) {
  const variables = Array.isArray(inventory)
    ? inventory
    : Array.isArray(inventory?.variables) ? inventory.variables : []
  const buckets = new Map()
  for (const entry of variables) {
    if (!entry || !entry.name) continue
    const role = entry.role || ROLE_UNCLASSIFIED
    if (!buckets.has(role)) buckets.set(role, [])
    buckets.get(role).push(entry)
  }
  return ROLE_ORDER
    .filter(role => buckets.get(role)?.length)
    .map(role => ({
      role,
      label: roleLabel(role),
      blurb: ROLE_META[role]?.blurb || '',
      accent: ROLE_META[role]?.accent || 'var(--text-hint)',
      variables: buckets.get(role),
    }))
}

// The chart-page related-variables view: the plotted role plus the sibling
// sections that actually have content. Returns { role, confidence, sections }
// where sections is [] when a product carries no companions (e.g. MODIS AOD) —
// so the panel renders nothing spurious rather than empty headers.
export function relatedVariableSections(related) {
  if (!related) return null
  const sections = []
  if (related.qa_sibling) {
    sections.push({ key: 'qa', label: 'QA flag', names: [related.qa_sibling] })
  }
  if (related.uncertainty_sibling) {
    sections.push({ key: 'uncertainty', label: 'Uncertainty', names: [related.uncertainty_sibling] })
  }
  if (Array.isArray(related.context_siblings) && related.context_siblings.length) {
    sections.push({ key: 'context', label: 'Context bands', names: related.context_siblings })
  }
  return {
    variable: related.variable || null,
    role: related.role || null,
    confidence: related.confidence ?? null,
    sections,
  }
}

// Whether the chart-page panel has anything worth rendering: a known role or at
// least one sibling section. An unclassified plotted variable with no siblings
// yields nothing, which is correct.
export function hasRelatedVariables(related) {
  const view = relatedVariableSections(related)
  if (!view) return false
  const hasRole = view.role && view.role !== ROLE_UNCLASSIFIED
  return Boolean(hasRole || view.sections.length)
}
