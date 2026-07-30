/**
 * utils/companionActions.js
 * --------------------------
 * Prompt builders behind the chart-page related-variables panel's one-click
 * actions (PRD T36 Phase 1). Turns a companion sibling name plus the current
 * chart into a prefilled natural-language string dispatched through the same
 * `sendMessage` path as T22's FollowupChips and App.jsx's other prefilled
 * prompts -- no new execution engine, no new failure surface.
 *
 * Kept pure and outside the component tree so the capability/QA-redundancy
 * gates are unit-testable without a DOM (this repo's frontend test runner is
 * plain `node --test`, no jsdom/RTL -- see metadataDisplay.js).
 */
import { reproducibilityFields, regionLabel, qaMethodologyFields, resolveMaskingRaw } from './metadataDisplay.js'
import { relatedVariableSections } from './variableRoles.js'

// Region+time phrase templated from the plotted chart's query (T36: "context
// comes from the plotted chart's query"). Omits a field that's absent rather
// than fabricating one -- an under-specified prompt lets the agent resolve
// it, a fabricated bbox would silently narrow the companion's request.
function regionTimePhrase(chart) {
  const { dataset, startDate, endDate } = reproducibilityFields(chart)
  const region = regionLabel(chart)
  const parts = []
  if (dataset) parts.push(`from ${dataset}`)
  if (region) parts.push(`over ${region}`)
  if (startDate && endDate) parts.push(`for ${startDate} to ${endDate}`)
  return parts.join(' ')
}

export function buildPlotPrompt(companion, chart) {
  const phrase = regionTimePhrase(chart)
  return phrase ? `Plot ${companion} ${phrase}.` : `Plot ${companion}.`
}

export function buildComparePrompt(companion, chart) {
  const sciVariable = chart?.provenance?.related_variables?.variable
  const subject = sciVariable ? `${sciVariable} and ${companion}` : companion
  const phrase = regionTimePhrase(chart)
  return phrase
    ? `Compare ${subject} on the map ${phrase}.`
    : `Compare ${subject} on the map.`
}

export function buildQaFilterPrompt(chart) {
  const { qualityFlagVar } = qaMethodologyFields(chart)
  const base = qualityFlagVar
    ? `Re-plot this chart with QA filtering applied using ${qualityFlagVar}`
    : `Apply QA filtering to this chart`
  const phrase = regionTimePhrase(chart)
  return phrase ? `${base} ${phrase}.` : `${base}.`
}

// Per-companion actions, gated by which related-variables section it came
// from (T36 capability gate). Every context/uncertainty sibling can be
// plotted; only context siblings (the atmosphere/surface/geometry bands) get
// a map-compare action, since that's what T28's compare grid is built for.
// The QA sibling itself never gets a button here -- it backs the separate
// chart-level "Apply QA filtering" action instead (qaFilterAction below), not
// a plot of its own. Overlay is never offered: T29 (chart-overlay) is
// unbuilt, so there is deliberately no code path that can emit one.
export function sectionActions(sectionKey, companion, chart) {
  if (sectionKey === 'uncertainty') {
    return [
      { key: `plot-${companion}`, label: `Plot ${companion}`, prompt: buildPlotPrompt(companion, chart) },
    ]
  }
  if (sectionKey === 'context') {
    return [
      { key: `plot-${companion}`, label: `Plot ${companion}`, prompt: buildPlotPrompt(companion, chart) },
      { key: `compare-${companion}`, label: `Compare with ${companion}`, prompt: buildComparePrompt(companion, chart) },
    ]
  }
  return []
}

// The chart-level QA-filter action (not per-companion). Suppressed when a
// deterministic/pinned mask already ran (masking.applied -- e.g. TEMPO's
// pinned qa_good_values, T25): offering it there would be a redundant
// recommendation. Offered only when a quality_flag_var sibling exists to
// filter on (e.g. OMI's ambiguous cases) -- with no flag var there's nothing
// actionable, so the capability gate suppresses it rather than showing a
// dead button.
export function qaFilterAction(chart) {
  const masking = resolveMaskingRaw(chart)
  if (masking?.applied) return null
  const { qualityFlagVar } = qaMethodologyFields(chart)
  if (!qualityFlagVar) return null
  return { key: 'qa-filter', label: 'Re-plot with QA filtering applied', prompt: buildQaFilterPrompt(chart) }
}

// The full action tree for the related-variables panel: one entry per
// section (qa/uncertainty/context) with each companion's gated actions,
// plus the chart-level QA action. The single composition point both the
// component and its tests use, so "does clicking a rendered action dispatch
// the right prompt" is verifiable without a DOM (chart.provenance.related_
// variables) -- section/name shape mirrors variableRoles.relatedVariable
// Sections exactly, just with actions attached per companion.
export function buildRelatedVariableActions(chart) {
  const related = chart?.provenance?.related_variables
  const view = relatedVariableSections(related)
  if (!view) return { sections: [], qaAction: null }
  const sections = view.sections.map(section => ({
    key: section.key,
    label: section.label,
    items: section.names.map(name => ({ name, actions: sectionActions(section.key, name, chart) })),
  }))
  return { sections, qaAction: qaFilterAction(chart) }
}
