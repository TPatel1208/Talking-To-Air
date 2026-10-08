// Decides which overlay source MapLibreHeatmapPanel should be showing: the
// server-rendered native PNG (no color-scale override, and one exists) or
// the client-built canvas fallback (an override is active, or there's no
// native overlay to fall back to). Extracted so the recolor effect can
// re-derive this on every override change instead of special-casing "no
// override" as "leave whatever's already drawn alone" -- that shortcut is
// what let a canvas frame painted under compare mode's shared scale survive
// a toggle back to each panel's own native scale.
// A selected T59 frame (a flat float32 view over one interval of the frame
// stack) also forces canvas, and does so on its own: the server PNG is the
// period aggregate warped at native resolution and cannot show an hour, and a
// stack whose pooled scale came back null -- nothing survived masking -- still
// has frames to draw with no override to carry them.
export function resolveOverlayMode(override, overlayUrl, frame = null) {
  if (frame) return 'canvas'
  return !override && overlayUrl ? 'native' : 'canvas'
}

// The same decision for a comparison thumbnail, which is an <img> rather than
// a map source. Its png is fetched with auth into a blob (loadThumbnail), so
// `loaded` is `{ url, objectUrl }` for the overlay url that blob came from, or
// null. Native only once a blob for THIS url is in hand; until then, and when
// the load fails (an evicted overlay, a lapsed session), the canvas draws the
// grid the chart payload already carries.
export function resolveThumbnailMode(overlayUrl, loaded) {
  return overlayUrl && loaded?.url === overlayUrl && loaded.objectUrl ? 'native' : 'canvas'
}
