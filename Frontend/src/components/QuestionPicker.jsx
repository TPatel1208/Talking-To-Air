import { useState } from 'react'
import VariableChoicePicker from './VariableChoicePicker'
import {
  addItem, confirmListRequest, itemLabel, itemType, optionText, pickManyRequest, pickOneCandidates,
  pickOneRequest, removeItem, textRequest, togglePick, MAX_TEXT_ANSWER_CHARS,
} from '../utils/pausedQuestion'

// The control for the question a paused run waits on. `question` is the
// normalised payload (utils/pausedQuestion.js); `onAnswer` receives the
// request to send ({message, answer}). The logic for each kind lives in the
// util; this is the thin rendering shell.

const buttonStyle = {
  padding: '6px 12px', borderRadius: '100px', fontSize: '12px', fontWeight: 600,
  fontFamily: 'var(--font)', background: 'var(--bg-card)', border: '1px solid var(--teal)',
  color: 'var(--teal-text)', cursor: 'pointer',
}
const disabledStyle = { ...buttonStyle, opacity: 0.5, cursor: 'not-allowed' }
const inputStyle = {
  padding: '6px 10px', borderRadius: '8px', fontSize: '12px', fontFamily: 'var(--font)',
  background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)',
}
const rowStyle = { display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' }

function PickMany({ question, onAnswer }) {
  const [picked, setPicked] = useState([])
  const request = pickManyRequest(question, picked)
  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {question.options.map((option, index) => (
          <label key={index} style={{ display: 'flex', gap: '6px', alignItems: 'center', fontSize: '12px', color: 'var(--text-primary)' }}>
            <input
              type="checkbox"
              checked={picked.includes(index)}
              onChange={() => setPicked(prev => togglePick(prev, index))}
            />
            {optionText(option)}
          </label>
        ))}
      </div>
      <div>
        <button disabled={!request} onClick={() => onAnswer(request)} style={request ? buttonStyle : disabledStyle}>
          Confirm
        </button>
      </div>
    </>
  )
}

function ConfirmList({ question, onAnswer }) {
  const type = itemType(question.items)
  const [items, setItems] = useState(question.items)
  const [draft, setDraft] = useState('')
  const [problem, setProblem] = useState('')
  const request = confirmListRequest(question, items)

  const add = () => {
    const next = addItem(items, type, draft)
    if (next.error) {
      setProblem(next.error)
      return
    }
    setItems(next.items)
    setDraft('')
    setProblem('')
  }

  return (
    <>
      <div style={rowStyle}>
        {items.map((item, index) => (
          <span
            key={`${itemLabel(item)}-${index}`}
            style={{
              display: 'inline-flex', gap: '6px', alignItems: 'center', padding: '3px 8px', borderRadius: '100px',
              fontSize: '12px', fontFamily: 'var(--font-mono)', background: 'var(--bg-card)',
              border: '1px solid var(--border)', color: 'var(--text-primary)',
            }}
          >
            {itemLabel(item)}
            <button
              onClick={() => setItems(prev => removeItem(prev, index))}
              aria-label={`Remove ${itemLabel(item)}`}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-hint)', fontSize: '14px', lineHeight: 1, padding: 0 }}
            >
              ×
            </button>
          </span>
        ))}
        {items.length === 0 && (
          <span style={{ fontSize: '12px', color: 'var(--text-hint)', fontStyle: 'italic' }}>Nothing left to confirm.</span>
        )}
      </div>
      {type !== 'fixed' && (
        <div style={rowStyle}>
          <input
            type={type === 'date' ? 'date' : type === 'number' ? 'number' : 'text'}
            value={draft}
            onChange={e => { setDraft(e.target.value); setProblem('') }}
            onKeyDown={e => { if (e.key === 'Enter') add() }}
            aria-label="Add an item"
            style={inputStyle}
          />
          <button onClick={add} style={buttonStyle}>Add</button>
        </div>
      )}
      {problem && <span style={{ fontSize: '11.5px', color: 'var(--error)' }}>{problem}</span>}
      <div>
        <button disabled={!request} onClick={() => onAnswer(request)} style={request ? buttonStyle : disabledStyle}>
          Confirm {items.length}
        </button>
      </div>
    </>
  )
}

function TextAnswer({ question, onAnswer }) {
  const [text, setText] = useState('')
  const request = textRequest(question, text)
  return (
    <div style={rowStyle}>
      <input
        value={text}
        maxLength={MAX_TEXT_ANSWER_CHARS}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && request) onAnswer(request) }}
        aria-label="Your answer"
        style={{ ...inputStyle, flex: 1, minWidth: '160px' }}
      />
      <button disabled={!request} onClick={() => onAnswer(request)} style={request ? buttonStyle : disabledStyle}>
        Send
      </button>
    </div>
  )
}

export default function QuestionPicker({ question, onAnswer }) {
  if (!question || !onAnswer) return null

  if (question.kind === 'pick_one') {
    // Today's chips, or the searchable list for many options.
    return (
      <VariableChoicePicker
        variableChoice={{ message: question.text, candidates: pickOneCandidates(question) }}
        onSend={text => {
          const request = pickOneRequest(question, text)
          if (request) onAnswer(request)
        }}
      />
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '4px 2px 0' }}>
      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{question.text}</span>
      {question.kind === 'pick_many' && <PickMany question={question} onAnswer={onAnswer} />}
      {question.kind === 'confirm_list' && <ConfirmList question={question} onAnswer={onAnswer} />}
      {question.kind === 'text' && <TextAnswer question={question} onAnswer={onAnswer} />}
    </div>
  )
}
