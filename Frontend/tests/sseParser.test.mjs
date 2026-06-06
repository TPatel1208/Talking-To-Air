import assert from 'node:assert/strict'
import test from 'node:test'
import { createSseParser } from '../src/utils/sseParser.js'

function parseChunks(chunks) {
  const events = []
  const parser = createSseParser(event => events.push(event))

  chunks.forEach(chunk => parser.feed(chunk))
  parser.end()

  return events
}

test('reconstructs multiline data fields exactly', () => {
  const events = parseChunks([
    'event: done\n',
    'data: {"response":"line one\\n\n',
    'data: line two","thread_id":"abc"}\n\n',
  ])

  assert.deepEqual(events, [{
    event: 'done',
    data: '{"response":"line one\\n\nline two","thread_id":"abc"}',
    id: '',
  }])
})

test('handles event frames split across chunks', () => {
  const events = parseChunks([
    'event: cha',
    'rt\ndata: {"type":"bar","values":[1,',
    '2,3]}\n\n',
  ])

  assert.deepEqual(events, [{
    event: 'chart',
    data: '{"type":"bar","values":[1,2,3]}',
    id: '',
  }])
})

test('ignores comments and supports CRLF line endings', () => {
  const events = parseChunks([
    ': keepalive\r\nevent: image\r\ndata: {"url":"/x.png"}\r\n\r\n',
  ])

  assert.deepEqual(events, [{
    event: 'image',
    data: '{"url":"/x.png"}',
    id: '',
  }])
})

test('dispatches a final frame without a trailing blank line', () => {
  const events = parseChunks([
    'event: tool_call\n',
    'data: {"name":"search","args":{"q":"soil"}}',
  ])

  assert.deepEqual(events, [{
    event: 'tool_call',
    data: '{"name":"search","args":{"q":"soil"}}',
    id: '',
  }])
})
