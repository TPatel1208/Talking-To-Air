import assert from 'node:assert/strict'
import test from 'node:test'
import { isGroundMapArtifact } from '../src/utils/groundMap.js'
import { isReachableArtifact } from '../src/utils/artifactReachability.js'

test('a table the backend flagged as having coordinates is a ground map', () => {
  assert.equal(isGroundMapArtifact({ type: 'table', metadata: { has_coordinates: true } }), true)
})

test('a table whose loaded rows carry coordinates is a ground map, even without the flag', () => {
  const artifact = { type: 'table', rows: [{ latitude: null, longitude: null }, { latitude: 40.7, longitude: -74.1 }] }
  assert.equal(isGroundMapArtifact(artifact), true)
})

test('a table with no coordinates is not a ground map', () => {
  assert.equal(isGroundMapArtifact({ type: 'table', metadata: { has_coordinates: false } }), false)
  assert.equal(isGroundMapArtifact({ type: 'table', rows: [{ latitude: null, longitude: null }] }), false)
  assert.equal(isGroundMapArtifact({ type: 'table' }), false)
})

test('a heatmap "map" artifact is never a ground map', () => {
  assert.equal(isGroundMapArtifact({ id: 'map_abc', type: 'map', metadata: { has_coordinates: true } }), false)
})

test('nothing is not a ground map', () => {
  assert.equal(isGroundMapArtifact(null), false)
  assert.equal(isGroundMapArtifact(undefined), false)
})

test('a ground map stays reachable from chat, like any table', () => {
  assert.equal(isReachableArtifact({ id: 'tbl_1', type: 'table', metadata: { has_coordinates: true } }, []), true)
})
