// The question a paused run waits on, and the answer the app sends for it.
//
// The chat `done` event and a row of `GET /runs` carry the same payload:
// {run_id, step, kind, question, options | items}. Four kinds exist. This is
// the pure core of QuestionPicker.jsx: it normalises the payload, builds the
// `answer` ({run_id, step, value}) and the short `message` the transcript
// records, and edits a confirm_list's items. The backend checks every answer
// again in code, so nothing here is the only guard.

export const QUESTION_KINDS = ['pick_one', 'pick_many', 'confirm_list', 'text']

// The most message text one answer sends; the request allows far more.
const MAX_MESSAGE_CHARS = 2000
// The longest text a text question takes, as the backend checks it.
export const MAX_TEXT_ANSWER_CHARS = 1000

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/

// An option as the user reads it: a string as it is, an object by its label,
// name or value.
export function optionText(option) {
  if (typeof option === 'string') return option
  if (option && typeof option === 'object') {
    for (const key of ['label', 'name', 'value']) {
      if (typeof option[key] === 'string') return option[key]
    }
  }
  return JSON.stringify(option)
}

// The question as the picker reads it, or null for anything the app cannot
// answer: an unknown kind, no run or step, or nothing to pick from.
export function normalizeQuestion(raw) {
  if (!raw || typeof raw !== 'object') return null
  if (!QUESTION_KINDS.includes(raw.kind)) return null
  if (typeof raw.run_id !== 'string' || !raw.run_id) return null
  if (typeof raw.step !== 'string' || !raw.step) return null
  const question = {
    runId: raw.run_id,
    step: raw.step,
    kind: raw.kind,
    text: typeof raw.question === 'string' ? raw.question : '',
    options: [],
    items: [],
  }
  if (raw.kind === 'pick_one' || raw.kind === 'pick_many') {
    if (!Array.isArray(raw.options) || raw.options.length === 0) return null
    question.options = raw.options
  } else if (raw.kind === 'confirm_list') {
    if (!Array.isArray(raw.items)) return null
    question.items = raw.items
  }
  return question
}

export function extractQuestion(doneData) {
  return normalizeQuestion(doneData?.question)
}

// The answer request for `value`: the body fields POST /chat takes beside the message.
export function answerRequest(question, value, message) {
  return {
    message: clip(message),
    answer: { run_id: question.runId, step: question.step, value },
  }
}

function clip(text) {
  const clean = String(text).trim()
  return clean.length <= MAX_MESSAGE_CHARS ? clean : `${clean.slice(0, MAX_MESSAGE_CHARS - 1)}…`
}

// pick_one

// The options as candidates for the chip or modal picker, named by their text.
// A repeated text is shown once.
export function pickOneCandidates(question) {
  const seen = new Set()
  const candidates = []
  for (const option of question.options) {
    const text = optionText(option)
    if (seen.has(text)) continue
    seen.add(text)
    candidates.push({
      name: text, category: 'distinct', units: null, validFraction: null, reasons: [], prompt: text,
    })
  }
  return candidates
}

// The answer for the option whose text was picked; null when none has it.
export function pickOneRequest(question, text) {
  const option = question.options.find(o => optionText(o) === text)
  if (option === undefined) return null
  return answerRequest(question, option, optionText(option))
}

// pick_many

// The picks after toggling the option at `index`, kept in the options' order.
export function togglePick(picked, index) {
  const next = new Set(picked)
  if (next.has(index)) next.delete(index)
  else next.add(index)
  return [...next].sort((a, b) => a - b)
}

export function pickManyRequest(question, picked) {
  const chosen = [...new Set(picked)].sort((a, b) => a - b)
    .filter(i => Number.isInteger(i) && i >= 0 && i < question.options.length)
    .map(i => question.options[i])
  if (chosen.length === 0) return null
  return answerRequest(question, chosen, chosen.map(optionText).join(', '))
}

// confirm_list

export function isIsoDate(value) {
  if (typeof value !== 'string' || !ISO_DATE.test(value)) return false
  const [year, month, day] = value.split('-').map(Number)
  const date = new Date(Date.UTC(year, month - 1, day))
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day
}

// The type every offered item has, which an added item must have too: 'date'
// (ISO dates), 'number', 'text', or 'fixed' for items that can only be
// removed (booleans, objects, mixed).
export function itemType(items) {
  if (!items.length) return 'text'
  if (items.every(isIsoDate)) return 'date'
  if (items.every(i => typeof i === 'number' && Number.isFinite(i))) return 'number'
  if (items.every(i => typeof i === 'string')) return 'text'
  return 'fixed'
}

export const ITEM_NOUNS = { date: ['day', 'days'], number: ['value', 'values'], text: ['item', 'items'] }

function noun(type, count) {
  const [one, many] = ITEM_NOUNS[type] || ITEM_NOUNS.text
  return count === 1 ? one : many
}

export function itemLabel(item) {
  return typeof item === 'string' ? item : optionText(item)
}

export function removeItem(items, index) {
  return items.filter((_, i) => i !== index)
}

function sameItem(type, a, b) {
  if (type === 'text') return String(a).trim().toLowerCase() === String(b).trim().toLowerCase()
  return a === b
}

// `items` with `raw` added, or the reason it was not: {items} or {error}.
// Dates must be ISO dates and a repeat of any item is refused.
export function addItem(items, type, raw) {
  const text = String(raw ?? '').trim()
  if (!text) return { error: 'Enter a value to add.' }
  let value = text
  if (type === 'date') {
    if (!isIsoDate(text)) return { error: 'Enter a valid date.' }
  } else if (type === 'number') {
    value = Number(text)
    if (!Number.isFinite(value)) return { error: 'Enter a number.' }
  } else if (type !== 'text') {
    return { error: 'Items of this question can only be removed.' }
  }
  if (items.some(item => sameItem(type, item, value))) return { error: 'That one is already in the list.' }
  return { items: type === 'date' ? [...items, value].sort() : [...items, value] }
}

// "Confirmed 3 days: 2024-06-01, 2024-06-02, 2024-06-03"
export function confirmListMessage(items, type) {
  return `Confirmed ${items.length} ${noun(type, items.length)}: ${items.map(itemLabel).join(', ')}`
}

export function confirmListRequest(question, items) {
  if (!items.length) return null
  return answerRequest(question, items, confirmListMessage(items, itemType(question.items)))
}

// text

export function textRequest(question, text) {
  const value = String(text ?? '').trim()
  if (!value || value.length > MAX_TEXT_ANSWER_CHARS) return null
  return answerRequest(question, value, value)
}
