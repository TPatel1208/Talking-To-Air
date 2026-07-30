import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildPlotPrompt, buildComparePrompt, buildQaFilterPrompt,
  sectionActions, qaFilterAction, buildRelatedVariableActions,
} from '../src/utils/companionActions.js'

// A TEMPO O3 chart shape: deterministic pinned QA mask already applied
// (T25), real context bands, one uncertainty sibling -- matches the live-
// verification fixture in PRD T36.
const tempoO3Chart = {
  type: 'timeseries',
  query: {
    dataset: 'TEMPO_O3TOT_L3', start_date: '2026-07-01', end_date: '2026-07-07',
    bbox: [-75.5, 40.0, -74.0, 41.0],
  },
  provenance: {
    region_name: 'New Jersey',
    masking: { applied: true, qa_status: 'verified' },
    qa_methodology: { quality_flag_var: 'main_data_quality_flag' },
    related_variables: {
      variable: 'column_amount_o3', role: 'science', confidence: 'high',
      qa_sibling: 'main_data_quality_flag',
      uncertainty_sibling: 'column_amount_o3_uncertainty',
      context_siblings: ['radiative_cloud_frac', 'uv_aerosol_index'],
    },
  },
}

// An OMI-shaped chart: ambiguous QA, no mask applied yet, a quality_flag_var
// sibling to filter on -- the case T36's QA-redundancy gate is meant to
// surface an action for.
const omiAmbiguousChart = {
  type: 'heatmap',
  query: { dataset: 'OMI_NO2_L3', start_date: '2026-06-01', end_date: '2026-06-02' },
  provenance: {
    masking: { applied: false, qa_status: 'ambiguous — awaiting classification' },
    qa_methodology: { quality_flag_var: 'ground_pixel_quality_flags' },
    related_variables: {
      variable: 'tropospheric_no2', role: 'science', confidence: 'medium',
      qa_sibling: 'ground_pixel_quality_flags', uncertainty_sibling: null, context_siblings: [],
    },
  },
}

// A MODIS AOD chart: no context bands, no uncertainty sibling, no quality
// flag var to filter on -- the data-gate fixture (T36 Testing Decisions).
const modisAodChart = {
  type: 'heatmap',
  query: { dataset: 'MODIS_AOD_TERRA', start_date: '2026-05-01', end_date: '2026-05-01' },
  provenance: {
    masking: { applied: true },
    related_variables: {
      variable: 'COMBINE_AOD_550_AVG', role: 'science', confidence: 'high',
      qa_sibling: null, uncertainty_sibling: null, context_siblings: [],
    },
  },
}

test('buildPlotPrompt templates the companion, dataset, region and time', () => {
  for (const companion of ['radiative_cloud_frac', 'uv_aerosol_index', 'column_amount_o3_uncertainty']) {
    const prompt = buildPlotPrompt(companion, tempoO3Chart)
    assert.match(prompt, new RegExp(companion))
    assert.match(prompt, /TEMPO_O3TOT_L3/)
    assert.match(prompt, /New Jersey/)
    assert.match(prompt, /2026-07-01/)
    assert.match(prompt, /2026-07-07/)
  }
})

test('buildPlotPrompt degrades gracefully when query fields are absent (no fabricated bbox)', () => {
  const bare = { provenance: { related_variables: { variable: 'x' } } }
  const prompt = buildPlotPrompt('radiative_cloud_frac', bare)
  assert.equal(prompt, 'Plot radiative_cloud_frac.')
})

test('buildComparePrompt names both the plotted science variable and the companion', () => {
  const prompt = buildComparePrompt('uv_aerosol_index', tempoO3Chart)
  assert.match(prompt, /column_amount_o3/)
  assert.match(prompt, /uv_aerosol_index/)
  assert.match(prompt, /New Jersey/)
  assert.match(prompt, /map/i)
})

test('buildQaFilterPrompt names the quality_flag_var sibling', () => {
  const prompt = buildQaFilterPrompt(omiAmbiguousChart)
  assert.match(prompt, /ground_pixel_quality_flags/)
  assert.match(prompt, /OMI_NO2_L3/)
})

test('sectionActions offers Plot for uncertainty siblings, no compare', () => {
  const actions = sectionActions('uncertainty', 'column_amount_o3_uncertainty', tempoO3Chart)
  assert.equal(actions.length, 1)
  assert.equal(actions[0].label, 'Plot column_amount_o3_uncertainty')
})

test('sectionActions offers Plot and Compare for context siblings', () => {
  const actions = sectionActions('context', 'radiative_cloud_frac', tempoO3Chart)
  const labels = actions.map(a => a.label)
  assert.deepEqual(labels, ['Plot radiative_cloud_frac', 'Compare with radiative_cloud_frac'])
})

test('sectionActions offers nothing for the qa section (backs the chart-level action instead)', () => {
  assert.deepEqual(sectionActions('qa', 'main_data_quality_flag', tempoO3Chart), [])
})

test('capability gate: no action emitted by any section is ever an overlay action', () => {
  for (const sectionKey of ['qa', 'uncertainty', 'context']) {
    const actions = sectionActions(sectionKey, 'some_companion', tempoO3Chart)
    for (const action of actions) {
      assert.doesNotMatch(action.label.toLowerCase(), /overlay/)
      assert.doesNotMatch(action.key.toLowerCase(), /overlay/)
    }
  }
})

test('QA-redundancy gate: a deterministically masked chart gets no QA-filter action', () => {
  assert.equal(qaFilterAction(tempoO3Chart), null)
})

test('QA-redundancy gate: an unmasked chart with a quality_flag_var sibling offers one', () => {
  const action = qaFilterAction(omiAmbiguousChart)
  assert.ok(action)
  assert.equal(action.label, 'Re-plot with QA filtering applied')
  assert.match(action.prompt, /ground_pixel_quality_flags/)
})

test('QA-redundancy gate: no action when neither masked nor a quality_flag_var sibling exists', () => {
  const chart = {
    query: { dataset: 'X' },
    provenance: { masking: { applied: false }, related_variables: { variable: 'x' } },
  }
  assert.equal(qaFilterAction(chart), null)
})

test('data gate: MODIS AOD (no context companions) yields no context actions', () => {
  const { sections, qaAction } = buildRelatedVariableActions(modisAodChart)
  assert.deepEqual(sections, [])
  assert.equal(qaAction, null)
})

test('data gate: a no-companions fixture yields no sections and no QA action', () => {
  const chart = { provenance: { related_variables: null } }
  assert.deepEqual(buildRelatedVariableActions(chart), { sections: [], qaAction: null })
})

test('wiring: buildRelatedVariableActions produces clickable actions that dispatch the built prompt exactly once', () => {
  const { sections, qaAction } = buildRelatedVariableActions(tempoO3Chart)
  const contextSection = sections.find(s => s.key === 'context')
  const cloudFrac = contextSection.items.find(i => i.name === 'radiative_cloud_frac')
  const plotAction = cloudFrac.actions.find(a => a.key === 'plot-radiative_cloud_frac')

  const sent = []
  const stubSendMessage = (text) => sent.push(text)

  // Simulates the component's onClick={() => onSend(action.prompt)}.
  stubSendMessage(plotAction.prompt)

  assert.equal(sent.length, 1)
  assert.equal(sent[0], buildPlotPrompt('radiative_cloud_frac', tempoO3Chart))
  // TEMPO O3 is already masked -- confirms the gate suppresses the redundant action end-to-end.
  assert.equal(qaAction, null)
})

test('wiring: the OMI QA-filter action dispatches through the same path', () => {
  const { qaAction } = buildRelatedVariableActions(omiAmbiguousChart)
  const sent = []
  const stubSendMessage = (text) => sent.push(text)
  stubSendMessage(qaAction.prompt)
  assert.equal(sent.length, 1)
  assert.match(sent[0], /ground_pixel_quality_flags/)
})
