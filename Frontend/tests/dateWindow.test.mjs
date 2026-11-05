import test from 'node:test'
import assert from 'node:assert/strict'

import {
  parseWindow, windowString, windowError, cmrTemporal, windowPhrase,
} from '../src/utils/dateWindow.js'

test('a range is written as start/end, the form the discovery endpoints take', () => {
  assert.equal(windowString('2026-06-01', '2026-06-30'), '2026-06-01/2026-06-30')
})

test('a single day is a window that starts and ends on it', () => {
  assert.equal(windowString('2026-06-05', '2026-06-05'), '2026-06-05/2026-06-05')
  assert.equal(windowString('2026-06-05', ''), '2026-06-05/2026-06-05')
})

test('no start date means no window', () => {
  assert.equal(windowString('', '2026-06-30'), '')
  assert.equal(windowString('', ''), '')
})

test('a window parses back into its two dates', () => {
  assert.deepEqual(parseWindow('2026-06-01/2026-06-30'), { start: '2026-06-01', end: '2026-06-30' })
  assert.deepEqual(parseWindow('2026-06-05'), { start: '2026-06-05', end: '2026-06-05' })
  assert.deepEqual(parseWindow(''), { start: '', end: '' })
  assert.deepEqual(parseWindow(null), { start: '', end: '' })
})

test('text that is not a date is not read as one', () => {
  assert.deepEqual(parseWindow('last summer'), { start: '', end: '' })
})

test('an end before the start is reported', () => {
  assert.match(windowError('2026-06-30', '2026-06-01'), /end date is before the start/i)
  assert.equal(windowError('2026-06-01', '2026-06-30'), null)
  assert.equal(windowError('2026-06-01', ''), null)
  assert.equal(windowError('', ''), null)
})

test('the window becomes a CMR temporal filter spanning whole days', () => {
  assert.equal(cmrTemporal('2026-06-01/2026-06-30'), '2026-06-01T00:00:00Z,2026-06-30T23:59:59Z')
  assert.equal(cmrTemporal('2026-06-05/2026-06-05'), '2026-06-05T00:00:00Z,2026-06-05T23:59:59Z')
  assert.equal(cmrTemporal(''), null)
  assert.equal(cmrTemporal('2026-06-30/2026-06-01'), null)
})

test('the retrieve request reads naturally for a day and for a range', () => {
  assert.equal(windowPhrase('2026-06-05/2026-06-05'), 'on 2026-06-05')
  assert.equal(windowPhrase('2026-06-01/2026-06-30'), 'from 2026-06-01 to 2026-06-30')
  assert.equal(windowPhrase(''), '')
})
