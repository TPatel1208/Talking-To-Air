import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

// T47 finding #1: the QA session showed a DELETE /session/{id} fire in the
// middle of a 404 cascade. Deleting a session is destructive and must only
// ever happen on a deliberate user action — never auto-fired by a history
// load, a session switch, or any error handler. This pins that invariant at
// the source level so a future refactor can't quietly introduce an auto-delete.

const here = dirname(fileURLToPath(import.meta.url))
const srcDir = join(here, '..', 'src')

function collectSources(dir) {
  const out = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...collectSources(full))
    else if (/\.(js|jsx)$/.test(entry.name)) out.push(full)
  }
  return out
}

// A fetch to a /session/ URL issued with method DELETE — the destructive
// "remove this conversation" call. Other DELETEs (e.g. disconnecting a
// connector) are unrelated and out of scope for this guard.
const SESSION_DELETE = /fetch\(`[^`]*\/session\/[^`]*`,\s*\{[^}]*method:\s*['"]DELETE['"]/g

test('the only DELETE /session request in the client is the explicit deleteSession action', () => {
  const offenders = []
  let deleteSites = 0

  for (const file of collectSources(srcDir)) {
    const source = readFileSync(file, 'utf8')
    const matches = [...source.matchAll(SESSION_DELETE)]
    if (matches.length === 0) continue

    deleteSites += matches.length
    const isUseChat = file.endsWith(join('hooks', 'useChat.js'))
    const declIndex = source.indexOf('const deleteSession')

    for (const match of matches) {
      // Every session DELETE must live inside useChat's deleteSession — i.e.
      // after its declaration and before the next callback declaration.
      const nextDecl = source.indexOf('const clearError', declIndex)
      const insideDeleteSession =
        isUseChat && declIndex !== -1 && match.index > declIndex &&
        (nextDecl === -1 || match.index < nextDecl)
      if (!insideDeleteSession) offenders.push(`${file} @ ${match.index}`)
    }
  }

  assert.equal(offenders.length, 0, `unexpected DELETE /session request site(s): ${offenders.join(', ')}`)
  assert.equal(deleteSites, 1, 'expected exactly one DELETE /session request site (deleteSession)')
})
