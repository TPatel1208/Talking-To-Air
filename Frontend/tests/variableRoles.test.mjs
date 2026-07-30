import assert from 'node:assert/strict'
import test from 'node:test'
import {
  groupInventoryByRole,
  relatedVariableSections,
  hasRelatedVariables,
  ROLE_ORDER,
} from '../src/utils/variableRoles.js'

// A slice of a real classified inventory (backend datasets/variable_roles.py,
// TEMPO O3TOT), the shape services/discovery_service.py attaches as `inventory`.
const tempoO3Inventory = {
  variables: [
    { name: 'product/column_amount_o3', leaf: 'column_amount_o3', role: 'science', confidence: 'high', units: 'DU' },
    { name: 'product/radiative_cloud_frac', leaf: 'radiative_cloud_frac', role: 'context', confidence: 'medium' },
    { name: 'product/o3_below_cloud', leaf: 'o3_below_cloud', role: 'retrieval-metadata', confidence: 'medium' },
    { name: 'qa_statistics/num_column_samples', leaf: 'num_column_samples', role: 'quality', confidence: 'high' },
    { name: 'weight', leaf: 'weight', role: 'unclassified', confidence: null },
  ],
}

test('groups the inventory into ordered, name-first role sections', () => {
  const groups = groupInventoryByRole(tempoO3Inventory)
  const roles = groups.map(g => g.role)
  // Science first; the order follows ROLE_ORDER and skips empty roles.
  assert.equal(roles[0], 'science')
  assert.deepEqual(roles, ROLE_ORDER.filter(r => roles.includes(r)))
  const science = groups.find(g => g.role === 'science')
  assert.deepEqual(science.variables.map(v => v.leaf), ['column_amount_o3'])
})

test('renders one clean Unclassified group when non-empty', () => {
  const groups = groupInventoryByRole(tempoO3Inventory)
  const unclassified = groups.find(g => g.role === 'unclassified')
  assert.ok(unclassified, 'unclassified group should be present')
  assert.equal(unclassified.label, 'Unclassified')
  assert.deepEqual(unclassified.variables.map(v => v.leaf), ['weight'])
  // ...and sits last.
  assert.equal(groups[groups.length - 1].role, 'unclassified')
})

test('accepts a bare variables array as well as the {variables} envelope', () => {
  const groups = groupInventoryByRole(tempoO3Inventory.variables)
  assert.equal(groups[0].role, 'science')
})

test('returns no groups for an empty/absent inventory', () => {
  assert.deepEqual(groupInventoryByRole(undefined), [])
  assert.deepEqual(groupInventoryByRole({ variables: [] }), [])
})

test('related panel surfaces plotted role plus QA / uncertainty / context siblings', () => {
  const view = relatedVariableSections({
    variable: 'vertical_column',
    role: 'science',
    confidence: 'high',
    qa_sibling: 'main_data_quality_flag',
    uncertainty_sibling: 'vertical_column_uncertainty',
    context_siblings: ['radiative_cloud_frac', 'uv_aerosol_index'],
  })
  assert.equal(view.role, 'science')
  const keys = view.sections.map(s => s.key)
  assert.deepEqual(keys, ['qa', 'uncertainty', 'context'])
  const context = view.sections.find(s => s.key === 'context')
  assert.deepEqual(context.names, ['radiative_cloud_frac', 'uv_aerosol_index'])
})

test('related panel renders nothing spurious when a product has no context bands (MODIS AOD)', () => {
  const modis = {
    variable: 'COMBINE_AOD_550_AVG', role: 'science', confidence: 'high',
    qa_sibling: null, uncertainty_sibling: null, context_siblings: [],
  }
  const view = relatedVariableSections(modis)
  assert.deepEqual(view.sections, [])
  // A science role with no siblings still renders (shows the role); the panel's
  // "no companion variables" state, not empty headers.
  assert.equal(hasRelatedVariables(modis), true)
})

test('hasRelatedVariables is false for an unclassified plot with no siblings', () => {
  assert.equal(hasRelatedVariables(null), false)
  assert.equal(hasRelatedVariables({ role: 'unclassified', context_siblings: [] }), false)
})
