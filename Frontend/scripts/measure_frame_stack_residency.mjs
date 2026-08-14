// T59 Phase 15, design tension 4: what holding three statistics' stacks costs,
// and what dropping two of them costs to get back.
//
// BUDGET FROM `arrayBuffers`, NEVER `heapUsed`. A typed array's storage is not
// on the JS heap, so Phase 6 measured one 100-frame stack as `heapUsed +0.07 MB`
// and `arrayBuffers +7.64 MB` — a budget written against `heapUsed` reports the
// frame store as free.
//
//   node --expose-gc scripts/measure_frame_stack_residency.mjs
import { decodeFrameStack } from '../src/utils/frameStack.js'

const FRAMES = 100
const CELLS = 20_000 // D5's rendering ceiling, which bounds a real stack
const HEIGHT = 100
const WIDTH = 200
const PLANES = FRAMES + 1 // the period aggregate rides the same entry

const block = { shape: [PLANES, HEIGHT, WIDTH], dtype: 'float32', period_index: 0 }

const MB = (bytes) => (bytes / 1024 / 1024).toFixed(2)

// Twice, deliberately. One `gc()` reliably reclaims the JS objects but leaves
// backing stores from the same cycle still accounted — measured here at 15.41 MB
// after dropping two of three stacks, i.e. one uncollected stack reported as
// resident. A single pass would have made "evict all but the active statistic"
// look half as effective as it is.
function settle() {
  if (!global.gc) return
  global.gc()
  global.gc()
}

function arrayBuffers() {
  settle()
  return process.memoryUsage().arrayBuffers
}

// A fresh blob per statistic, as three separate fetches would produce.
function blob() {
  const buffer = new ArrayBuffer(PLANES * CELLS * 4)
  const view = new Float32Array(buffer)
  for (let i = 0; i < view.length; i += 997) view[i] = i % 1000
  return buffer
}

console.log(`layout: ${PLANES} planes x ${HEIGHT}x${WIDTH} float32 (${MB(PLANES * CELLS * 4)} MB of values per statistic)\n`)

const base = arrayBuffers()

// ── One statistic resident, which is what `useFrameStack` holds today ────────
let held = [decodeFrameStack(blob(), block)]
const one = arrayBuffers()
console.log(`1 statistic resident  arrayBuffers +${MB(one - base)} MB`)

// ── Three resident, the "keep all three" alternative ─────────────────────────
held.push(decodeFrameStack(blob(), block))
held.push(decodeFrameStack(blob(), block))
const three = arrayBuffers()
console.log(`3 statistics resident arrayBuffers +${MB(three - base)} MB  (${((three - base) / (one - base)).toFixed(2)}x)`)

// ── Dropping back to one, which is what keying the hook by url already does ──
held = [held[0]]
const dropped = arrayBuffers()
console.log(`evicted back to 1     arrayBuffers +${MB(dropped - base)} MB`)

// ── What coming back costs: a re-decode, never a re-download ─────────────────
// `_FRAME_CACHE_CONTROL` is `private, immutable, max-age=31536000`, so a plane
// the reader has already seen is served from the HTTP cache.
const bytes = blob()
const started = process.hrtime.bigint()
const again = decodeFrameStack(bytes, block)
const ms = Number(process.hrtime.bigint() - started) / 1e6
console.log(`\nre-decode of a cached blob: ${ms.toFixed(3)} ms (${again.planeCount} planes)`)

// ── Zero-copy, still. `subarray` views cost nothing; `slice` doubles. ────────
const beforeViews = arrayBuffers()
const views = Array.from({ length: FRAMES }, (_, i) => again.planeView(i + 1))
console.log(`${views.length} planeView() views: arrayBuffers +${MB(arrayBuffers() - beforeViews)} MB`)
console.log(`  (views share the buffer: ${views[0].buffer === again.values.buffer})`)
