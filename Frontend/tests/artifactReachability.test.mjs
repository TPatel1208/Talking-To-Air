import test from 'node:test'
import assert from 'node:assert/strict'

import { isReachableArtifact, reachableArtifacts } from '../src/utils/artifactReachability.js'

// ── table artifacts: always reachable, no chart to compare against ─────────

test('a table artifact is reachable regardless of msg.charts', () => {
  const artifact = { id: 'tbl_1', type: 'table' }
  assert.equal(isReachableArtifact(artifact, []), true)
  assert.equal(isReachableArtifact(artifact, [{ chart_id: 'ts_other' }]), true)
})

// ── ground-validation timeseries: reachable only when no chart shares its id ─

test('a ground-validation timeseries artifact (no matching chart) is reachable', () => {
  // validate_against_ground/exceedance_overlay mint their own chart_id and
  // never call emit_chart, so msg.charts never has a matching entry.
  const artifact = { id: 'ts_overlay1', type: 'timeseries' }
  assert.equal(isReachableArtifact(artifact, []), true)
  assert.equal(isReachableArtifact(artifact, [{ chart_id: 'ts_unrelated' }]), true)
})

test('a chart-backed timeseries artifact (matching chart_id) is NOT reachable', () => {
  // Regression guard: a plain satellite timeseries calls both emit_chart and
  // build_artifact_reference with the same chart_id -- msg.charts already
  // covers it, so it must not also get its own card (would duplicate).
  const artifact = { id: 'ts_abc123', type: 'timeseries' }
  const charts = [{ chart_id: 'ts_abc123', type: 'timeseries' }]
  assert.equal(isReachableArtifact(artifact, charts), false)
})

test('a chart also matches via chart.id (not just chart_id)', () => {
  const artifact = { id: 'ts_xyz', type: 'timeseries' }
  assert.equal(isReachableArtifact(artifact, [{ id: 'ts_xyz' }]), false)
})

// ── map/comparison: never reachable -- explicitly out of scope (T33) ───────

test('map and comparison artifacts are never reachable', () => {
  assert.equal(isReachableArtifact({ id: 'map_1', type: 'map' }, []), false)
  assert.equal(isReachableArtifact({ id: 'cmp_1', type: 'comparison' }, []), false)
})

test('isReachableArtifact is null-safe', () => {
  assert.equal(isReachableArtifact(null, []), false)
  assert.equal(isReachableArtifact(undefined, undefined), false)
})

// ── reachableArtifacts: the message-level filter Chat.jsx/App.jsx use ──────

test('reachableArtifacts keeps table and orphan-timeseries, drops chart-backed and map/comparison', () => {
  const msg = {
    charts: [{ chart_id: 'ts_chart1', type: 'timeseries' }],
    artifacts: [
      { id: 'tbl_1', type: 'table' },
      { id: 'ts_chart1', type: 'timeseries' },   // chart-backed -- excluded
      { id: 'ts_overlay1', type: 'timeseries' }, // ground-validation -- included
      { id: 'map_1', type: 'map' },
      { id: 'cmp_1', type: 'comparison' },
    ],
  }
  const result = reachableArtifacts(msg)
  assert.deepEqual(result.map(a => a.id), ['tbl_1', 'ts_overlay1'])
})

test('reachableArtifacts guards a plain heatmap turn: exactly its one chart, zero extra artifact cards', () => {
  // A plain heatmap or region-comparison query still produces exactly one
  // card (the chart card) -- no duplicate from its parallel map/comparison
  // artifact stub.
  const msg = {
    charts: [{ chart_id: 'map_1', type: 'heatmap' }],
    artifacts: [{ id: 'map_1', type: 'map' }],
  }
  assert.deepEqual(reachableArtifacts(msg), [])
})

test('reachableArtifacts is null-safe for a message with no artifacts/charts', () => {
  assert.deepEqual(reachableArtifacts({}), [])
  assert.deepEqual(reachableArtifacts(undefined), [])
})
