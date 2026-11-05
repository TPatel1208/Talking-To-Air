import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

// plotly.js 3 dropped string titles: `title: 'Time'` renders no title at all,
// with no error. Every layout title must be `title: { text: ... }`. The
// time-series layouts live inline in JSX with no runnable test, so this reads
// the source.

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..', 'src')
const PLOTLY_LAYOUT_FILES = ['components/ChartMessage.jsx', 'utils/verticalProfile.js']

test('no plotly layout passes a bare string as a title', () => {
  for (const file of PLOTLY_LAYOUT_FILES) {
    const source = readFileSync(join(SRC, file), 'utf8')
      .split('\n')
      .filter(line => !/^\s*(\/\/|\*|\/\*)/.test(line))
      .join('\n')
    const titles = [...source.matchAll(/\btitle:\s*([^\s,}])/g)]
    assert.ok(titles.length > 0, `${file} has no title: -- re-point this guard`)
    for (const [match, first] of titles) {
      assert.equal(first, '{', `${file}: "${match}" -- plotly 3+ ignores a string title; use { text: ... }`)
    }
  }
})
