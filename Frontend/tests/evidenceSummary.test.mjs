import test from 'node:test'
import assert from 'node:assert/strict'

import {
  evidenceRows, hasEvidence, evidenceLabel, evidenceValueText, evidenceCoverageText,
} from '../src/utils/evidenceSummary.js'

// A TEMPO-O3-shaped payload: context means + a QA pass-rate fact, each with
// coverage -- the rich Supporting-information case.
test('builds rows for context and QA facts with coverage', () => {
  const chart = {
    provenance: {
      evidence: [
        { name: 'main_data_quality_flag', role: 'quality', stat: 'pass_rate', value: 1.0, units: '', coverage: 0.87 },
        { name: 'radiative_cloud_frac', role: 'context', stat: 'mean', value: 0.04, units: '1', coverage: 0.92 },
      ],
    },
  }
  const rows = evidenceRows(chart)
  assert.equal(rows.length, 2)
  assert.equal(rows[0].label, 'QA pass rate')
  assert.equal(rows[0].valueText, '100.0%')
  assert.equal(rows[0].coverageText, 'over 87% valid pixels')
  assert.equal(rows[1].label, 'radiative cloud frac')
  assert.equal(rows[1].valueText, '0.04 1')
  assert.equal(rows[1].coverageText, 'over 92% valid pixels')
  assert.equal(hasEvidence(chart), true)
})

// An uncertainty fact reads as "Retrieval uncertainty" and appends its
// fraction-of-science when the backend computed one.
test('formats an uncertainty fact with pct-of-science', () => {
  const fact = {
    name: 'vertical_column_troposphere_uncertainty', role: 'quality', stat: 'mean',
    value: 0.08, units: 'molecules/cm^2', coverage: 1.0, pct_of_science: 0.08,
  }
  assert.equal(evidenceLabel(fact), 'Retrieval uncertainty')
  assert.equal(evidenceValueText(fact), '0.08 molecules/cm^2 (~8.0% of value)')
})

// Large/small science magnitudes format as exponential; coverage rounds.
test('formats large magnitudes as exponential', () => {
  assert.equal(evidenceValueText({ stat: 'mean', value: 2.8e15, units: 'molecules/cm^2' }), '2.80e+15 molecules/cm^2')
  assert.equal(evidenceCoverageText(0.876), 'over 88% valid pixels')
})

// MODIS-AOD-shaped payload: empty evidence -> no rows, section omits itself.
test('yields no rows for an empty evidence list', () => {
  assert.deepEqual(evidenceRows({ provenance: { evidence: [] } }), [])
  assert.equal(hasEvidence({ provenance: { evidence: [] } }), false)
})

// Missing / malformed evidence never throws and never renders.
test('is safe when evidence is absent or malformed', () => {
  assert.deepEqual(evidenceRows(null), [])
  assert.deepEqual(evidenceRows({}), [])
  assert.deepEqual(evidenceRows({ provenance: {} }), [])
  assert.equal(hasEvidence({ provenance: { evidence: 'nope' } }), false)
  // A fact with a non-finite value is dropped, not rendered as NaN.
  assert.deepEqual(evidenceRows({ provenance: { evidence: [{ name: 'x', value: null }] } }), [])
})

// Evidence at the top level is accepted too (symmetry with maskingProvenance).
test('reads top-level evidence', () => {
  const chart = { evidence: [{ name: 'uv_aerosol_index', role: 'context', stat: 'mean', value: 1.5, units: '1', coverage: 1.0 }] }
  const rows = evidenceRows(chart)
  assert.equal(rows.length, 1)
  assert.equal(rows[0].valueText, '1.5 1')
})
