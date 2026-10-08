import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveObjectURL } from 'node:buffer'
import { loadThumbnail } from '../src/utils/thumbnailSource.js'

// A compare thumbnail cannot be a bare <img> pointed at the API: the browser
// fetches an <img> itself and cannot add the bearer header, so the auth
// middleware 401s every one. loadThumbnail fetches through an authed fetcher
// and hands the <img> a blob: URL, which needs no request at all.

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

function serving(status, body = PNG) {
  const calls = []
  const fetcher = async (url) => {
    calls.push(url)
    return new Response(body, { status, headers: { 'Content-Type': 'image/png' } })
  }
  fetcher.calls = calls
  return fetcher
}

// Resolves with the object URL onLoad reports, or null if nothing is reported
// before the fetch settles.
function load(url, fetcher) {
  return new Promise((resolve) => {
    const cancel = loadThumbnail(url, { fetcher, onLoad: (objectUrl) => resolve({ objectUrl, cancel }) })
    setTimeout(() => resolve({ objectUrl: null, cancel }), 50)
  })
}

const bytesAt = async (objectUrl) => new Uint8Array(await resolveObjectURL(objectUrl).arrayBuffer())

test('the png is fetched through the authed fetcher and handed back as an object url', async () => {
  const fetcher = serving(200)

  const { objectUrl, cancel } = await load('/api/chart/abc/overlay.png?panel=0', fetcher)

  assert.deepEqual(fetcher.calls, ['/api/chart/abc/overlay.png?panel=0'])
  assert.match(objectUrl, /^blob:/)
  assert.deepEqual(await bytesAt(objectUrl), PNG)
  cancel()
})

test('an error response hands back nothing, so the thumbnail keeps its canvas', async () => {
  // 401 is a lapsed session, 404 an overlay the store's LRU cap evicted. The
  // body is an error document, not a png, and must not become an image.
  for (const status of [401, 404]) {
    const { objectUrl } = await load('/api/chart/abc/overlay.png?panel=0', serving(status, '{"detail":"no"}'))
    assert.equal(objectUrl, null, `status ${status}`)
  }
})

test('cancelling releases the object url', async () => {
  // The component cancels when the panel unmounts or its overlay url changes;
  // an unrevoked blob holds the png in memory for the life of the page.
  const { objectUrl, cancel } = await load('/api/chart/abc/overlay.png?panel=0', serving(200))
  assert.ok(resolveObjectURL(objectUrl), 'loaded blob is live')

  cancel()

  assert.equal(resolveObjectURL(objectUrl), undefined)
})

test('a response that lands after cancelling is dropped and leaks nothing', async () => {
  // The slow-network race: the panel is gone (or shows a newer url) before its
  // png arrives. Reporting it would set state on a stale panel, and a url
  // minted after cleanup ran would never be revoked.
  let respond
  const pending = new Promise((resolve) => { respond = resolve })
  const fetcher = () => pending
  const created = []
  const realCreate = URL.createObjectURL
  URL.createObjectURL = (blob) => { const u = realCreate(blob); created.push(u); return u }
  try {
    let reported = null
    const cancel = loadThumbnail('/api/chart/abc/overlay.png?panel=0', {
      fetcher,
      onLoad: (objectUrl) => { reported = objectUrl },
    })

    cancel()
    respond(new Response(PNG, { status: 200 }))
    await new Promise((resolve) => setTimeout(resolve, 20))

    assert.equal(reported, null)
    assert.ok(created.every((u) => resolveObjectURL(u) === undefined), 'no live blob left behind')
  } finally {
    URL.createObjectURL = realCreate
  }
})

test('a fetch that throws hands back nothing', async () => {
  const unreachable = async () => { throw new TypeError('Failed to fetch') }

  const { objectUrl } = await load('/api/chart/abc/overlay.png?panel=0', unreachable)

  assert.equal(objectUrl, null)
})
