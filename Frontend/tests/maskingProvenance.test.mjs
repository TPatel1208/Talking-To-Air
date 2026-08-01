import test from 'node:test'
import assert from 'node:assert/strict'

import { resolveMasking, resolveRegionFidelity, formatQaPassRate } from '../src/utils/maskingProvenance.js'

// Timeseries payloads carry masking at the top level
// (plot_tools ts_payload["masking"]).
test('reads top-level masking (timeseries payload)', () => {
  const chart = {
    type: 'timeseries',
    masking: { qa_status: 'cf-deterministic', qa_source: 'cf_flag_meanings' },
  }
  const masking = resolveMasking(chart)
  // The full resolved shape: three other consumers read it, so the T55
  // pass-rate fields being present-but-null on a payload that carries no rate
  // is part of the contract, not an accident.
  assert.deepEqual(masking, {
    qaStatus: 'cf-deterministic',
    qaSource: 'cf_flag_meanings',
    qaNote: '',
    qaPassRate: null,
    qaCheckedPixels: null,
    qaPassingPixels: null,
    qaFlagMissingPixels: null,
    qaPassRateByTime: null,
    qaPassRateTimes: null,
    qaPassRateBasis: '',
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

// ── Realized QA pass rate (T55) ───────────────────────────────────────────

// The rate the mask actually applied, counted backend-side from the same
// boolean condition that gutted the data, plus the pixel counts that are its
// denominator.
test('surfaces the realized QA pass rate and its pixel basis', () => {
  const chart = {
    type: 'timeseries',
    masking: {
      qa_status: 'verified',
      qa_pass_rate: 0.7532,
      qa_checked_pixels: 15000,
      qa_passing_pixels: 11298,
      qa_flag_missing_pixels: 120,
      qa_pass_rate_by_time: [0.0, 1.0],
      qa_pass_rate_times: ['2024-01-01T00:00:00', '2024-01-02T00:00:00'],
      qa_pass_rate_basis: 'cos(latitude)-weighted fraction ...',
    },
  }
  const masking = resolveMasking(chart)
  assert.equal(masking.qaPassRate, 0.7532)
  assert.equal(masking.qaCheckedPixels, 15000)
  assert.equal(masking.qaPassingPixels, 11298)
  assert.equal(masking.qaFlagMissingPixels, 120)
  assert.deepEqual(masking.qaPassRateByTime, [0.0, 1.0])
  assert.deepEqual(masking.qaPassRateTimes, ['2024-01-01T00:00:00', '2024-01-02T00:00:00'])
  assert.match(masking.qaPassRateBasis, /weighted fraction/)
})

// Absent keys mean QA never ran -> null, so the card can say "Not applied"
// rather than render a fabricated 0%.
test('qaPassRate is null when the backend reported no rate', () => {
  const masking = resolveMasking({ masking: { qa_status: 'not applied — semantics unknown' } })
  assert.equal(masking.qaPassRate, null)
  assert.equal(masking.qaCheckedPixels, null)
})

// A genuine 0.0 is a real, terrible answer -- it must survive the resolver.
// Every check has to be `!= null`, never truthy.
test('a genuine 0.0 pass rate survives as 0, not null', () => {
  const masking = resolveMasking({
    masking: { qa_status: 'verified', qa_pass_rate: 0, qa_checked_pixels: 400, qa_passing_pixels: 0 },
  })
  assert.equal(masking.qaPassRate, 0)
  assert.notEqual(masking.qaPassRate, null)
})

// A fully fill- or cloud-covered scene really had nothing to check. That is a
// different, diagnosable state from "QA never ran" -- the count survives as a
// real 0 while the rate stays null.
test('a scene with zero checked pixels keeps the count and drops the rate', () => {
  const masking = resolveMasking({
    masking: { qa_status: 'verified', qa_checked_pixels: 0, qa_passing_pixels: 0 },
  })
  assert.equal(masking.qaCheckedPixels, 0)
  assert.equal(masking.qaPassRate, null)
})

// Rounding must not manufacture a perfect score. One discarded pixel in 15,000
// is 99.993%, and rendering that as "100.0%" is the same class of lie as the
// valid-pct tautology this repo already fixed once.
test('formatQaPassRate floors just short of 100% when any pixel failed', () => {
  assert.equal(
    formatQaPassRate({ qaPassRate: 0.999933, qaCheckedPixels: 15000, qaPassingPixels: 14999 }),
    '99.9%',
  )
})

// ...and symmetrically at the bottom: one surviving pixel is not "0.0%".
test('formatQaPassRate ceils just above 0% when any pixel passed', () => {
  assert.equal(
    formatQaPassRate({ qaPassRate: 0.000067, qaCheckedPixels: 15000, qaPassingPixels: 1 }),
    '0.1%',
  )
})

// The ends are only floored when they are not the truth. A genuinely perfect
// scene reads 100.0%, and a genuinely wiped-out one reads 0.0%.
test('formatQaPassRate renders true 100% and true 0% verbatim', () => {
  assert.equal(
    formatQaPassRate({ qaPassRate: 1, qaCheckedPixels: 400, qaPassingPixels: 400 }),
    '100.0%',
  )
  assert.equal(
    formatQaPassRate({ qaPassRate: 0, qaCheckedPixels: 400, qaPassingPixels: 0 }),
    '0.0%',
  )
})

// No rate -> no percentage. The caller renders "Not applied" instead of a
// fabricated number.
test('formatQaPassRate returns null when there is no rate', () => {
  assert.equal(formatQaPassRate({ qaPassRate: null, qaCheckedPixels: 0 }), null)
  assert.equal(formatQaPassRate(null), null)
})

// ── Region fidelity (T42): region_type / display_name disclosure ──────────

// A bounding-box region is worth disclosing: the "region" was a rectangle,
// not the named place's real boundary.
test('resolveRegionFidelity surfaces a bounding_box region', () => {
  const chart = {
    type: 'heatmap',
    provenance: { region_type: 'bounding_box', display_name: 'United States' },
  }
  assert.deepEqual(resolveRegionFidelity(chart), {
    regionType: 'bounding_box',
    displayName: 'United States',
  })
})

// A real polygon is the faithful case -> nothing to warn about, render nothing.
test('resolveRegionFidelity returns null for a polygon region', () => {
  const chart = { type: 'heatmap', provenance: { region_type: 'polygon', display_name: 'Paris, France' } }
  assert.equal(resolveRegionFidelity(chart), null)
})

// point_buffer and boundary_cells are the other disclose-worthy kinds.
test('resolveRegionFidelity surfaces point_buffer and boundary_cells', () => {
  assert.equal(
    resolveRegionFidelity({ provenance: { region_type: 'point_buffer', display_name: 'X' } }).regionType,
    'point_buffer',
  )
  assert.equal(
    resolveRegionFidelity({ provenance: { region_type: 'boundary_cells', display_name: 'Y' } }).regionType,
    'boundary_cells',
  )
})

// No region_type -> null, so nothing renders for older/plain payloads.
test('resolveRegionFidelity returns null when region_type absent', () => {
  assert.equal(resolveRegionFidelity({ type: 'heatmap', provenance: {} }), null)
  assert.equal(resolveRegionFidelity(null), null)
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
