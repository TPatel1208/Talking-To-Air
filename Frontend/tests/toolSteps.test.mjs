import assert from 'node:assert/strict'
import test from 'node:test'
import { toolStepsTitle } from '../src/utils/toolSteps.js'

// T64 Phase 4: the collapsible tool card named every turn "ask_earthdata_agent()",
// a tool the supervisor no longer has.

const dispatch = (...locations) => ({
  name: 'dispatch',
  args: { requests: locations.map(location => ({ agent: 'satellite', intent: 'visualize', location })) },
})

test('a supervisor turn is titled by its dispatch calls and the requests they carried', () => {
  assert.equal(toolStepsTitle([dispatch('NJ', 'NY'), dispatch('Newark')]), 'dispatch · 3 requests')
  assert.equal(toolStepsTitle([dispatch('NJ')]), 'dispatch · 1 request')
})

test('a failed supervisor turn says so', () => {
  assert.equal(toolStepsTitle([dispatch('NJ')], 'boom'), 'dispatch · error')
})

test('a fast-path turn counts the sub-agent\'s own steps without naming a supervisor tool', () => {
  const steps = [{ name: 'search_datasets', args: {} }, { name: 'plot_singular', args: {} }]

  assert.equal(toolStepsTitle(steps), '2 steps')
  assert.equal(toolStepsTitle(steps.slice(0, 1)), '1 step')
  assert.equal(toolStepsTitle(steps, 'boom'), 'error')
})
