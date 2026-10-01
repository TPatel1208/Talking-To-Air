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

// The same decision for a comparison thumbnail, which is a plain <img> rather
// than a map source. The native png can fail to load: the overlay store evicts
// old entries, and an <img> cannot send the bearer token. The chart payload
// still carries the grid, so a failed load falls back to the canvas.
// `failedUrl` is the url whose load failed, so a different url is tried again.
export function resolveThumbnailMode(overlayUrl, failedUrl) {
  return overlayUrl && overlayUrl !== failedUrl ? 'native' : 'canvas'
}
