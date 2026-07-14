import assert from 'node:assert/strict'
import test from 'node:test'
import { starterMessage } from '../src/utils/starterPrompts.js'

// T22 story #3/#6: clicking a starter must be exactly like typing and
// sending its full prompt — never the short display label.
test('starterMessage sends the full prompt verbatim, not the label', () => {
  const starter = {
    id: 'discovery_no2_dataset',
    label: 'Find an NO2 dataset',
    prompt: 'What NASA datasets are available for NO2 column density over New Jersey?',
    category: 'discovery',
  }

  assert.equal(starterMessage(starter), starter.prompt)
  assert.notEqual(starterMessage(starter), starter.label)
})
