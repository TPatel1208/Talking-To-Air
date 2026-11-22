import assert from 'node:assert/strict'
import test from 'node:test'
import {
  extractQuestion, normalizeQuestion, optionText, pickOneCandidates, pickOneRequest,
  togglePick, pickManyRequest, isIsoDate, itemType, addItem, removeItem,
  confirmListMessage, confirmListRequest, textRequest,
} from '../src/utils/pausedQuestion.js'

const RUN = 'run_' + 'a'.repeat(32)

function question(over) {
  return { run_id: RUN, step: 'data', kind: 'pick_one', question: 'Go ahead?', options: ['confirm', 'narrow'], ...over }
}

test('a done event carries the question or none', () => {
  const q = extractQuestion({ question: question() })
  assert.deepEqual(
    { runId: q.runId, step: q.step, kind: q.kind, text: q.text, options: q.options },
    { runId: RUN, step: 'data', kind: 'pick_one', text: 'Go ahead?', options: ['confirm', 'narrow'] },
  )
  assert.equal(extractQuestion({}), null)
  assert.equal(extractQuestion(null), null)
})

test('a question the app cannot answer is not normalised', () => {
  for (const bad of [
    question({ kind: 'inputs' }), question({ run_id: '' }), question({ step: undefined }),
    question({ options: [] }), question({ options: undefined }),
    question({ kind: 'confirm_list', items: undefined, options: undefined }), 'text', [],
  ]) {
    assert.equal(normalizeQuestion(bad), null, JSON.stringify(bad))
  }
  assert.notEqual(normalizeQuestion(question({ kind: 'text', options: undefined })), null)
  assert.notEqual(normalizeQuestion(question({ kind: 'confirm_list', items: [], options: undefined })), null)
})

test('an option reads by its label, name or value', () => {
  assert.equal(optionText('confirm'), 'confirm')
  assert.equal(optionText({ label: 'L', name: 'N' }), 'L')
  assert.equal(optionText({ value: 'V' }), 'V')
  assert.equal(optionText({ id: 1 }), '{"id":1}')
})

test('pick_one answers with the option itself and its text as the message', () => {
  const q = normalizeQuestion(question({ options: ['confirm', { label: 'Narrow it', id: 2 }] }))
  assert.deepEqual(pickOneRequest(q, 'confirm'), {
    message: 'confirm', answer: { run_id: RUN, step: 'data', value: 'confirm' },
  })
  assert.deepEqual(pickOneRequest(q, 'Narrow it').answer.value, { label: 'Narrow it', id: 2 })
  assert.equal(pickOneRequest(q, 'other'), null)
})

test('pick_one candidates are the options by text, a repeat once', () => {
  const q = normalizeQuestion(question({ options: ['a', 'b', 'a'] }))
  assert.deepEqual(pickOneCandidates(q).map(c => [c.name, c.prompt]), [['a', 'a'], ['b', 'b']])
})

test('pick_many toggles picks in option order and answers with the picked options', () => {
  const q = normalizeQuestion(question({ kind: 'pick_many', options: ['no2', 'o3', 'pm25'] }))
  let picked = []
  picked = togglePick(picked, 2)
  picked = togglePick(picked, 0)
  assert.deepEqual(picked, [0, 2])
  assert.deepEqual(pickManyRequest(q, picked), {
    message: 'no2, pm25', answer: { run_id: RUN, step: 'data', value: ['no2', 'pm25'] },
  })
  assert.deepEqual(togglePick(picked, 0), [2])
  assert.equal(pickManyRequest(q, []), null)
  assert.equal(pickManyRequest(q, [7]), null)
})

test('ISO dates are checked as calendar dates', () => {
  assert.equal(isIsoDate('2024-02-29'), true)
  assert.equal(isIsoDate('2023-02-29'), false)
  assert.equal(isIsoDate('2024-6-1'), false)
  assert.equal(isIsoDate('2024-13-01'), false)
  assert.equal(isIsoDate(20240601), false)
})

test('the offered items give the type an added item must have', () => {
  assert.equal(itemType(['2024-06-01', '2024-06-02']), 'date')
  assert.equal(itemType([1, 2.5]), 'number')
  assert.equal(itemType(['a', '2024-06-01']), 'text')
  assert.equal(itemType([true]), 'fixed')
  assert.equal(itemType([{ a: 1 }]), 'fixed')
  assert.equal(itemType([]), 'text')
})

test('a date is added in order, and a bad or repeated one is refused', () => {
  const items = ['2024-06-01', '2024-06-03']
  assert.deepEqual(addItem(items, 'date', '2024-06-02'), { items: ['2024-06-01', '2024-06-02', '2024-06-03'] })
  assert.deepEqual(addItem(items, 'date', ' 2024-06-05 '), { items: [...items, '2024-06-05'] })
  assert.ok(addItem(items, 'date', '2024-06-03').error)
  assert.ok(addItem(items, 'date', '2024-02-30').error)
  assert.ok(addItem(items, 'date', '').error)
  assert.deepEqual(items, ['2024-06-01', '2024-06-03'])
})

test('numbers and texts are added once; fixed items only come out', () => {
  assert.deepEqual(addItem([1, 2], 'number', '3.5'), { items: [1, 2, 3.5] })
  assert.ok(addItem([1, 2], 'number', 'x').error)
  assert.ok(addItem([1, 2], 'number', '2').error)
  assert.deepEqual(addItem(['a'], 'text', 'b'), { items: ['a', 'b'] })
  assert.ok(addItem(['a'], 'text', ' A ').error)
  assert.ok(addItem([true], 'fixed', 'x').error)
})

test('removing an item leaves the others in order', () => {
  assert.deepEqual(removeItem(['a', 'b', 'c'], 1), ['a', 'c'])
})

test('confirm_list sends the list and says how many days were confirmed', () => {
  const q = normalizeQuestion(question({
    kind: 'confirm_list', options: undefined, items: ['2024-06-01', '2024-06-02', '2024-06-03'],
  }))
  const items = ['2024-06-01', '2024-06-03']
  assert.deepEqual(confirmListRequest(q, items), {
    message: 'Confirmed 2 days: 2024-06-01, 2024-06-03',
    answer: { run_id: RUN, step: 'data', value: items },
  })
  assert.equal(confirmListMessage(['2024-06-01'], 'date'), 'Confirmed 1 day: 2024-06-01')
  assert.equal(confirmListMessage([1, 2], 'number'), 'Confirmed 2 values: 1, 2')
  assert.equal(confirmListRequest(q, []), null)
})

test('a long list is clipped in the message but sent whole', () => {
  const days = Array.from({ length: 366 }, (_, i) => new Date(Date.UTC(2024, 0, 1 + i)).toISOString().slice(0, 10))
  const q = normalizeQuestion(question({ kind: 'confirm_list', options: undefined, items: days }))
  const request = confirmListRequest(q, days)
  assert.ok(request.message.length <= 2000)
  assert.equal(request.answer.value.length, 366)
})

test('text answers are trimmed, and empty or too long ones are not sent', () => {
  const q = normalizeQuestion(question({ kind: 'text', options: undefined }))
  assert.deepEqual(textRequest(q, '  New Jersey  '), {
    message: 'New Jersey', answer: { run_id: RUN, step: 'data', value: 'New Jersey' },
  })
  assert.equal(textRequest(q, '   '), null)
  assert.equal(textRequest(q, 'x'.repeat(1001)), null)
  assert.equal(textRequest(q, undefined), null)
})
