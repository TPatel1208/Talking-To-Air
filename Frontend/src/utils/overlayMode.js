// Decides which overlay source MapLibreHeatmapPanel should be showing: the
// server-rendered native PNG (no color-scale override, and one exists) or
// the client-built canvas fallback (an override is active, or there's no
// native overlay to fall back to). Extracted so the recolor effect can
// re-derive this on every override change instead of special-casing "no
// override" as "leave whatever's already drawn alone" -- that shortcut is
// what let a canvas frame painted under compare mode's shared scale survive
// a toggle back to each panel's own native scale.
export function resolveOverlayMode(override, overlayUrl) {
  return !override && overlayUrl ? 'native' : 'canvas'
}
