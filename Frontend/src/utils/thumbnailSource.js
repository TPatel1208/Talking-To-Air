// Loads a comparison thumbnail's server-rendered png for an <img>.
//
// A bare <img src="/api/chart/..."> cannot authenticate: the browser fetches
// it itself and has no way to add the bearer header the auth middleware
// requires. So the png is fetched through `fetcher` (apiFetch in the app) and
// handed back as a blob: URL, which loads without a request.
//
// A failure -- a lapsed session, an overlay the store evicted, an unreachable
// backend -- reports nothing: the caller keeps drawing the canvas fallback
// from the grid the chart payload already carries.
//
// Returns a cancel function, for an effect's cleanup: it revokes the object
// URL so the blob does not outlive the panel, and drops a response that lands
// afterwards rather than minting a URL nothing would ever revoke.
export function loadThumbnail(url, { fetcher, onLoad }) {
  let cancelled = false
  let objectUrl = null
  fetcher(url)
    .then((res) => (res.ok ? res.blob() : null))
    .then((blob) => {
      if (!blob || cancelled) return
      objectUrl = URL.createObjectURL(blob)
      onLoad(objectUrl)
    })
    .catch(() => {})
  return () => {
    cancelled = true
    if (objectUrl) URL.revokeObjectURL(objectUrl)
  }
}
