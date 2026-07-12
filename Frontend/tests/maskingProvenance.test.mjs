import test from 'node:test'
import assert from 'node:assert/strict'

import { resolveMasking } from '../src/utils/maskingProvenance.js'

// Timeseries payloads carry masking at the top level
// (plot_tools ts_payload["masking"]).
test('reads top-level masking (timeseries payload)', () => {
  const chart = {
    type: 'timeseries',
    masking: { qa_status: 'cf-deterministic', qa_source: 'cf_flag_meanings' },
  }
  const masking = resolveMasking(chart)
  assert.deepEqual(masking, {
    qaStatus: 'cf-deterministic',
    qaSource: 'cf_flag_meanings',
    qaNote: '',
  })
})

// Heatmap payloads carry masking under provenance
// (plot_tools._provenance copies agg_meta["masking"] there).
test('reads masking under provenance (heatmap payload)', () => {
  const chart = {
    type: 'heatmap',
    provenance: { masking: { qa_status: 'verified', qa_source: 'collections_yaml' } },
  }
  const masking = resolveMasking(chart)
  assert.equal(masking.qaStatus, 'verified')
  assert.equal(masking.qaSource, 'collections_yaml')
})

// aggregation_meta.masking is the third location the backend uses.
test('falls back to aggregation_meta.masking', () => {
  const chart = {
    type: 'heatmap',
    aggregation_meta: { masking: { qa_status: 'not applied — semantics unknown', qa_source: 'none' } },
  }
  const masking = resolveMasking(chart)
  assert.equal(masking.qaStatus, 'not applied — semantics unknown')
})

// qa_note is surfaced when the backend includes it (ambiguous/pending cases).
test('includes qa_note when present', () => {
  const chart = {
    type: 'heatmap',
    provenance: {
      masking: {
        qa_status: 'ambiguous — awaiting classification',
        qa_source: 'cf_flag_meanings',
        qa_note: 'every flag_meanings token classifies as bad-quality; no good class to key a mask on -- no mask applied',
      },
    },
  }
  const masking = resolveMasking(chart)
  assert.equal(masking.qaStatus, 'ambiguous — awaiting classification')
  assert.match(masking.qaNote, /no good class/)
})

// No masking record -> null, so the disclosure renders nothing.
test('returns null when no masking present', () => {
  assert.equal(resolveMasking({ type: 'heatmap', provenance: {} }), null)
  assert.equal(resolveMasking({ type: 'timeseries' }), null)
})

// A masking object without qa_status is not a usable disclosure.
test('returns null when qa_status missing', () => {
  assert.equal(resolveMasking({ type: 'heatmap', masking: { qa_source: 'none' } }), null)
})

// Guards against non-object input.
test('returns null for nullish or non-object chart', () => {
  assert.equal(resolveMasking(null), null)
  assert.equal(resolveMasking(undefined), null)
})

// Top-level masking wins over provenance when both exist.
test('prefers top-level masking over provenance', () => {
  const chart = {
    type: 'timeseries',
    masking: { qa_status: 'inferred, not verified' },
    provenance: { masking: { qa_status: 'verified' } },
  }
  assert.equal(resolveMasking(chart).qaStatus, 'inferred, not verified')
})
