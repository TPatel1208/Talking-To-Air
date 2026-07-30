// T49: the deterministic variable-choice picker payload the backend attaches
// to a chat 'done' event when the T48 resolver's confidence was medium or low.
// This module is the pure, testable core (extraction + threshold + grouping +
// filtering); VariableChoicePicker.jsx is the thin rendering shell over it,
// following the codebase pattern of keeping logic in a util (see followups.js /
// companionActions.js) so JSX stays declarative and the behavior is unit-tested.

// <=6 candidates render as inline chips (like FollowupChips); more than that
// opens a searchable, grouped modal instead of an unscrollable wall of chips.
export const VARIABLE_CHOICE_MODAL_THRESHOLD = 6

// Taxonomy category order for the modal's grouped list (PRD): the distinct
// products a researcher most likely wants first, equivalent variants next, and
// the deprioritized-but-not-hidden implementation/plumbing fields last.
const CATEGORY_ORDER = ['distinct', 'equivalent', 'implementation']
const CATEGORY_LABELS = {
  distinct: 'Distinct products',
  equivalent: 'Equivalent variables',
  implementation: 'Diagnostic / plumbing',
}

export function extractVariableChoice(doneData) {
  const vc = doneData?.variable_choice
  if (!vc || !Array.isArray(vc.candidates)) return null
  const candidates = vc.candidates
    .filter(c => c && typeof c.name === 'string' && typeof c.prompt === 'string' && c.prompt)
    .map(c => ({
      name: c.name,
      category: typeof c.category === 'string' ? c.category : 'distinct',
      units: c.units == null ? null : String(c.units),
      validFraction: typeof c.valid_fraction === 'number' ? c.valid_fraction : null,
      reasons: Array.isArray(c.reasons) ? c.reasons.filter(r => typeof r === 'string') : [],
      prompt: c.prompt,
    }))
  if (candidates.length === 0) return null
  return { message: typeof vc.message === 'string' ? vc.message : '', candidates }
}

export function variableChoiceUsesModal(candidates) {
  return (candidates?.length || 0) > VARIABLE_CHOICE_MODAL_THRESHOLD
}

export function groupCandidatesByCategory(candidates) {
  const buckets = new Map()
  for (const c of candidates || []) {
    const key = CATEGORY_ORDER.includes(c.category) ? c.category : 'distinct'
    if (!buckets.has(key)) buckets.set(key, [])
    buckets.get(key).push(c)
  }
  return CATEGORY_ORDER
    .filter(key => buckets.has(key))
    .map(key => ({ category: key, label: CATEGORY_LABELS[key], items: buckets.get(key) }))
}

export function filterCandidates(candidates, query) {
  const q = String(query || '').trim().toLowerCase()
  if (!q) return candidates || []
  return (candidates || []).filter(c => {
    const haystack = [c.name, c.units || '', ...(c.reasons || [])].join(' ').toLowerCase()
    return haystack.includes(q)
  })
}

// A compact valid-data label for a picker row ("82% valid"), or "" when the
// backend didn't compute a fraction. Never fabricated.
export function formatValidFraction(validFraction) {
  if (typeof validFraction !== 'number') return ''
  return `${Math.round(validFraction * 100)}% valid`
}
