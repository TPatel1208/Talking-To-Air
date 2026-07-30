import assert from 'node:assert/strict'
import test from 'node:test'
import {
  extractVariableChoice,
  variableChoiceUsesModal,
  groupCandidatesByCategory,
  filterCandidates,
  VARIABLE_CHOICE_MODAL_THRESHOLD,
} from '../src/utils/variableChoice.js'

function candidate(name, over = {}) {
  return {
    name,
    category: 'distinct',
    units: '1',
    valid_fraction: 0.8,
    reasons: ['geophysical quantity'],
    prompt: `plot AOD over New Jersey using ${name}`,
    ...over,
  }
}

test('extracts and normalizes a variable_choice off a done event', () => {
  const done = {
    thread_id: 'a',
    variable_choice: {
      message: 'This dataset has 2 candidate variables — pick one:',
      candidates: [candidate('DT_AOD_550_AVG'), candidate('COMBINE_AOD_550_AVG')],
    },
  }
  const vc = extractVariableChoice(done)
  assert.equal(vc.message, 'This dataset has 2 candidate variables — pick one:')
  assert.equal(vc.candidates.length, 2)
  assert.equal(vc.candidates[0].name, 'DT_AOD_550_AVG')
  assert.equal(vc.candidates[0].prompt, 'plot AOD over New Jersey using DT_AOD_550_AVG')
  assert.equal(vc.candidates[0].validFraction, 0.8)
})

test('returns null when the done event omits the field or has no candidates', () => {
  assert.equal(extractVariableChoice({ thread_id: 'a' }), null)
  assert.equal(extractVariableChoice(null), null)
  assert.equal(extractVariableChoice({ variable_choice: { message: 'x', candidates: [] } }), null)
})

test('drops candidates missing a name or prompt rather than rendering an unclickable row', () => {
  const done = {
    variable_choice: {
      message: 'm',
      candidates: [candidate('good'), { category: 'distinct' }, candidate('no_prompt', { prompt: '' })],
    },
  }
  const vc = extractVariableChoice(done)
  assert.deepEqual(vc.candidates.map(c => c.name), ['good'])
})

test('threshold: <=6 candidates renders inline chips, >6 opens a modal', () => {
  assert.equal(VARIABLE_CHOICE_MODAL_THRESHOLD, 6)
  const six = Array.from({ length: 6 }, (_, i) => candidate(`v${i}`))
  const seven = Array.from({ length: 7 }, (_, i) => candidate(`v${i}`))
  assert.equal(variableChoiceUsesModal(six), false)
  assert.equal(variableChoiceUsesModal(seven), true)
})

test('groups candidates by taxonomy category, distinct first and implementation last', () => {
  const candidates = [
    candidate('impl', { category: 'implementation' }),
    candidate('dist', { category: 'distinct' }),
    candidate('equiv', { category: 'equivalent' }),
  ]
  const groups = groupCandidatesByCategory(candidates)
  assert.deepEqual(groups.map(g => g.category), ['distinct', 'equivalent', 'implementation'])
  assert.equal(groups[0].items[0].name, 'dist')
})

test('filter matches candidate name, units, and reasons case-insensitively', () => {
  const candidates = [
    candidate('Terra_AOD_550', { reasons: ['aggregated mean field'] }),
    candidate('Aqua_NDVI', { units: 'index', reasons: [] }),
  ]
  assert.deepEqual(filterCandidates(candidates, 'aod').map(c => c.name), ['Terra_AOD_550'])
  assert.deepEqual(filterCandidates(candidates, 'MEAN').map(c => c.name), ['Terra_AOD_550'])
  assert.deepEqual(filterCandidates(candidates, 'index').map(c => c.name), ['Aqua_NDVI'])
  assert.equal(filterCandidates(candidates, '').length, 2)
})
