import test from 'node:test'
import assert from 'node:assert/strict'
import { SMOKE_LEGEND, smokeMapPng, waitForSmokeMap } from '../src/utils/hmsMapExport.js'

function fakeMap(ready = true) {
  const listeners = new Map()
  return {
    ready,
    on(name, fn) { if (!listeners.has(name)) listeners.set(name, new Set()); listeners.get(name).add(fn) },
    off(name, fn) { listeners.get(name)?.delete(fn) },
    emit(name) { for (const fn of listeners.get(name) || []) fn() },
    loaded() { return this.ready }, areTilesLoaded() { return this.ready }, isMoving() { return false },
    triggerRepaint() {},
    getCanvas() { return { width: 1600, height: 960, clientWidth: 800 } },
    listenerCount() { return [...listeners.values()].reduce((sum, set) => sum + set.size, 0) },
  }
}

function canvasFixture(encoding = 'success') {
  const calls = []
  const ctx = { fillRect: (...args) => calls.push(['rect', ...args]),
    drawImage: (...args) => calls.push(['image', ...args]), scale: (...args) => calls.push(['scale', ...args]),
    fillText: (...args) => calls.push(['text', ...args]) }
  const canvas = { getContext: () => ctx, toBlob(callback, type) {
    assert.equal(type, 'image/png')
    if (encoding === 'cors') throw new DOMException('Tainted canvas', 'SecurityError')
    callback(encoding === 'null' ? null : new Blob(['png'], { type }))
  } }
  return { canvas, calls }
}

test('legend labels and colors are identical for UI and PNG, without counts', () => {
  assert.deepEqual(SMOKE_LEGEND.map(item => item.label), ['Light', 'Medium', 'Heavy', 'Unknown'])
  assert.deepEqual(SMOKE_LEGEND.map(item => item.color), ['#e6bc46', '#da7734', '#943b60', '#777777'])
})

test('PNG preserves full native framebuffer and adds legend and attribution outside map', async () => {
  const map = fakeMap()
  const { canvas, calls } = canvasFixture()
  const blob = await smokeMapPng(map, { createCanvas: () => canvas })
  assert.equal(blob.type, 'image/png')
  assert.equal(canvas.width, 1600)
  assert.equal(canvas.height, 1136)
  assert.deepEqual(calls.find(call => call[0] === 'image').slice(2), [0, 0])
  assert.deepEqual(calls.find(call => call[0] === 'scale'), ['scale', 2, 2])
  assert.deepEqual(calls.filter(call => call[0] === 'text').map(call => call[1]),
    ['Light', 'Medium', 'Heavy', 'Unknown', '© OpenStreetMap contributors', 'Smoke data: NOAA HMS'])
  assert.equal(map.listenerCount(), 0)
  // fakeMap intentionally has no camera setters, resize or fitBounds.
})

test('waits for fully loaded stationary map and cleans up listeners', async () => {
  const map = fakeMap(false)
  const waiting = waitForSmokeMap(map, undefined, 100)
  map.ready = true
  map.emit('idle')
  await waiting
  assert.equal(map.listenerCount(), 0)
})

test('timeout, tile failure and cancellation do not export a partial map', async () => {
  const map = fakeMap(false)
  await assert.rejects(waitForSmokeMap(map, undefined, 5), /still loading/)
  assert.equal(map.listenerCount(), 0)
  const failed = waitForSmokeMap(map)
  map.emit('error')
  await assert.rejects(failed, /tiles could not be loaded/)
  const controller = new AbortController()
  const cancelled = waitForSmokeMap(map, controller.signal)
  controller.abort()
  await assert.rejects(cancelled, { name: 'AbortError' })
  assert.equal(map.listenerCount(), 0)
})

test('reports cross-origin restrictions and failed PNG encoding explicitly', async () => {
  for (const [encoding, pattern] of [['cors', /cross-origin/], ['null', /encoding failed/]]) {
    const { canvas } = canvasFixture(encoding)
    await assert.rejects(smokeMapPng(fakeMap(), { createCanvas: () => canvas }), pattern)
  }
})

test('legend wraps on narrow maps without changing the viewport', async () => {
  const map = fakeMap()
  map.getCanvas = () => ({ width: 360, height: 960, clientWidth: 180 })
  const { canvas, calls } = canvasFixture()
  await smokeMapPng(map, { createCanvas: () => canvas })
  assert.equal(canvas.width, 360)
  assert.equal(canvas.height, 1304)
  const labels = calls.filter(call => call[0] === 'text').slice(0, 4)
  assert.ok(labels.every(call => call[2] === 33))
  assert.ok(labels[3][3] > labels[0][3])
})
